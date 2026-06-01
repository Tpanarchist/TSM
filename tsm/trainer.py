from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DatasetConfig, TrainConfig, TsmConfig
from .data import make_dataset
from .self_field import Self


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_run_dir(cfg: TrainConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.runs_dir) / f"{stamp}_{cfg.run_name}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    return run_dir


def append_metrics(path: Path, metrics: dict[str, float | int]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def scalar_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def scalar_tensors(values: dict[str, torch.Tensor], prefix: str = "") -> dict[str, float]:
    return {f"{prefix}{key}": float(value.detach().cpu()) for key, value in values.items()}


def save_checkpoint(path: Path, model: Self, optimizer: torch.optim.Optimizer | None, cfg: TrainConfig, step: int, best: float) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg.to_dict(),
        "step": step,
        "best_loss": best,
    }
    torch.save(payload, path)


def load_model_from_checkpoint(checkpoint: str | Path, device: torch.device) -> tuple[Self, TrainConfig, dict]:
    payload = torch.load(checkpoint, map_location=device)
    cfg = TrainConfig.from_dict(payload["config"])
    model = Self(cfg.model).to(device)
    model.load_state_dict(payload["model_state"])
    return model, cfg, payload


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.shape[0] == 1:
        array = (tensor[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_sample_grid(path: Path, image_t: torch.Tensor, recon: torch.Tensor, image_tp1: torch.Tensor, pred: torch.Tensor) -> None:
    rows = min(6, image_t.shape[0])
    cells = []
    for row in range(rows):
        cells.extend([
            _tensor_to_image(image_t[row]),
            _tensor_to_image(recon[row]),
            _tensor_to_image(image_tp1[row]),
            _tensor_to_image(pred[row]),
        ])
    width, height = cells[0].size
    grid = Image.new(cells[0].mode, (width * 4, height * rows))
    for index, cell in enumerate(cells):
        x = (index % 4) * width
        y = (index // 4) * height
        grid.paste(cell, (x, y))
    grid.save(path)


def _infinite(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def train(cfg: TrainConfig, device_name: str = "cuda", resume: str | None = None) -> Path:
    set_seed(cfg.dataset.seed)
    device = resolve_device(device_name)
    run_dir = make_run_dir(cfg)
    cfg.write_yaml(run_dir / "config.yaml")
    dataset = make_dataset(cfg.dataset, cfg.model)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    model = Self(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start_step = 0
    best = math.inf
    if resume:
        payload = torch.load(resume, map_location=device)
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload.get("step", 0))
        best = float(payload.get("best_loss", math.inf))
    metrics_path = run_dir / "metrics.jsonl"
    iterator = _infinite(loader)
    model.train()
    progress = tqdm(range(start_step + 1, cfg.max_steps + 1), desc="train", unit="step")
    last_batch: dict[str, torch.Tensor] | None = None
    last_output = None
    first_metrics: dict[str, float | int] | None = None
    last_metrics: dict[str, float | int] | None = None
    for step in progress:
        batch = move_batch(next(iterator), device)
        last_batch = batch
        should_record = step % cfg.log_interval == 0 or step == 1 or step == cfg.max_steps
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_train(batch, include_label_diagnostics=should_record)
        output.total_loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_output = output
        metrics = {
            "step": step,
            "total": float(output.total_loss.detach().cpu()),
            **scalar_losses(output.losses),
            **scalar_tensors(output.diagnostics),
        }
        first_metrics = first_metrics or metrics
        last_metrics = metrics
        if should_record:
            append_metrics(metrics_path, metrics)
            progress.set_postfix(total=f"{metrics['total']:.4f}")
        if metrics["total"] < best:
            best = metrics["total"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, cfg, step, best)
        if step % cfg.checkpoint_interval == 0 or step == cfg.max_steps:
            save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, cfg, step, best)
        if step % cfg.sample_interval == 0 and last_batch is not None:
            save_sample_grid(
                run_dir / "samples" / f"step_{step:06d}.png",
                last_batch["image_t"],
                output.recon_image,
                last_batch["image_tp1"],
                output.next_image,
            )
    if last_batch is not None and last_output is not None:
        save_sample_grid(
            run_dir / "samples" / "final.png",
            last_batch["image_t"],
            last_output.recon_image,
            last_batch["image_tp1"],
            last_output.next_image,
        )
    with open(run_dir / "run_summary.md", "w", encoding="utf-8") as handle:
        handle.write(f"# {cfg.run_name}\n\n")
        handle.write(f"- steps: {cfg.max_steps}\n")
        handle.write(f"- best_total_loss: {best:.6f}\n")
        handle.write(f"- dataset: {cfg.dataset.name}\n")
        if first_metrics and last_metrics:
            handle.write(f"- first_total_loss: {first_metrics['total']:.6f}\n")
            handle.write(f"- final_total_loss: {last_metrics['total']:.6f}\n")
            handle.write(f"- final_context_used_count: {last_metrics['context_used_count']:.1f}\n")
            handle.write(f"- final_context_effective_count: {last_metrics['context_effective_count']:.3f}\n")
            handle.write(f"- final_ternary_zero_fraction: {last_metrics['ternary_zero_fraction']:.3f}\n")
            handle.write(f"- final_ternary_nonzero_fraction: {last_metrics['ternary_nonzero_fraction']:.3f}\n")
            if "context_mode_purity" in last_metrics:
                handle.write(f"- final_mode_context_consistency: {last_metrics['mode_context_consistency']:.3f}\n")
                handle.write(f"- final_context_mode_purity: {last_metrics['context_mode_purity']:.3f}\n")
                handle.write(f"- final_mode_context_separation: {last_metrics['mode_context_separation']:.3f}\n")
                handle.write(f"- final_mode_context_used_count: {last_metrics['mode_context_used_count']:.1f}\n")
                handle.write(f"- final_mode_count: {last_metrics['mode_count']:.1f}\n")
            if "ternary_mode_probe_accuracy" in last_metrics:
                handle.write(f"- final_ternary_mode_probe_accuracy: {last_metrics['ternary_mode_probe_accuracy']:.3f}\n")
                handle.write(f"- final_ternary_mode_mutual_information: {last_metrics['ternary_mode_mutual_information']:.3f}\n")
                handle.write(f"- final_ternary_context_mutual_information: {last_metrics['ternary_context_mutual_information']:.3f}\n")
                handle.write(f"- final_ternary_axis_usage_count: {last_metrics['ternary_axis_usage_count']:.1f}\n")
                handle.write(f"- final_ternary_axis_stability: {last_metrics['ternary_axis_stability']:.3f}\n")
    return run_dir


@torch.no_grad()
def evaluate(checkpoint: str | Path, device_name: str = "cuda", split: str = "test", limit: int | None = None) -> dict[str, float]:
    device = resolve_device(device_name)
    model, cfg, _payload = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    data_cfg = DatasetConfig(**cfg.dataset.__dict__)
    data_cfg.split = split
    if limit is not None:
        data_cfg.limit = limit
    dataset = make_dataset(data_cfg, cfg.model)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="eval", unit="batch"):
        batch = move_batch(batch, device)
        output = model.forward_train(batch, include_label_diagnostics=True)
        values = {
            "total": float(output.total_loss.cpu()),
            **scalar_losses(output.losses),
            **scalar_tensors(output.diagnostics),
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def sample(checkpoint: str | Path, out_dir: str | Path, device_name: str = "cuda", split: str = "test") -> Path:
    device = resolve_device(device_name)
    model, cfg, _payload = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_cfg = DatasetConfig(**cfg.dataset.__dict__)
    data_cfg.split = split
    data_cfg.limit = min(data_cfg.limit or 32, 32)
    dataset = make_dataset(data_cfg, cfg.model)
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, 16), shuffle=False, num_workers=0)
    batch = move_batch(next(iter(loader)), device)
    output = model.forward_train(batch)
    save_sample_grid(out_path / "samples.png", batch["image_t"], output.recon_image, batch["image_tp1"], output.next_image)
    return out_path / "samples.png"


def smoke(device_name: str = "cuda", steps: int = 20) -> Path:
    cfg = TrainConfig(
        run_name="smoke",
        model=TsmConfig(d_model=64, workspace_latents=16, contexts=4, definitions_per_context=8, attention_heads=4),
        dataset=DatasetConfig(name="synthetic", limit=128, seed=3),
        batch_size=16,
        max_steps=steps,
        learning_rate=1e-3,
        log_interval=1,
        checkpoint_interval=max(1, steps),
        sample_interval=max(1, steps),
    )
    return train(cfg, device_name=device_name)
