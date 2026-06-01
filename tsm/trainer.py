from __future__ import annotations

import copy
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
from .diagnostics import ternary_axis_specialization, ternary_label_diagnostics
from .hardening import DefinitionEvidenceTracker
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


def read_metrics(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def _load_model_state(model: Self, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {"defs.file_query.weight"}
    allowed_missing_prefixes = ("active_file_gate.", "active_file_expectation.")
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    disallowed_missing = {
        key for key in missing - allowed_missing if not key.startswith(allowed_missing_prefixes)
    }
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"checkpoint state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def load_model_from_checkpoint(checkpoint: str | Path, device: torch.device) -> tuple[Self, TrainConfig, dict]:
    payload = torch.load(checkpoint, map_location=device)
    cfg = TrainConfig.from_dict(payload["config"])
    model = Self(cfg.model).to(device)
    _load_model_state(model, payload["model_state"])
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
    shuffle = not bool(getattr(dataset, "sequential", False))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers)
    model = Self(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    definition_tracker = (
        DefinitionEvidenceTracker(cfg.definition_hardening, cfg.model.definitions_per_context)
        if cfg.definition_hardening.enabled
        else None
    )
    start_step = 0
    best = math.inf
    if resume:
        payload = torch.load(resume, map_location=device)
        _load_model_state(model, payload["model_state"])
        if payload.get("optimizer_state"):
            try:
                optimizer.load_state_dict(payload["optimizer_state"])
            except ValueError:
                if "defs.file_query.weight" not in payload["model_state"]:
                    pass
                else:
                    raise
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
        hardening_metrics: dict[str, float] = {}
        if should_record and definition_tracker is not None:
            mode = batch.get("mode", batch.get("label"))
            if mode is not None:
                impacts = model.ternary_prediction_impacts(output, batch["image_tp1"])
                hardening_metrics = definition_tracker.update(
                    step,
                    output.ternary,
                    mode,
                    impacts,
                    prediction_loss=output.losses["prediction"],
                )
        optimizer.step()
        last_output = output
        metrics = {
            "step": step,
            "total": float(output.total_loss.detach().cpu()),
            **scalar_losses(output.losses),
            **scalar_tensors(output.diagnostics),
            **hardening_metrics,
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
            if "temporal_visible_fraction" in last_metrics:
                handle.write(f"- final_temporal_visible_fraction: {last_metrics['temporal_visible_fraction']:.3f}\n")
                handle.write(f"- final_temporal_occluded_fraction: {last_metrics['temporal_occluded_fraction']:.3f}\n")
                handle.write(f"- final_temporal_reappeared_fraction: {last_metrics['temporal_reappeared_fraction']:.3f}\n")
                handle.write(f"- final_temporal_sae_occlusion_delta: {last_metrics['temporal_sae_occlusion_delta']:.6f}\n")
                handle.write(f"- final_temporal_prediction_visible_mean: {last_metrics['temporal_prediction_visible_mean']:.6f}\n")
                handle.write(f"- final_temporal_prediction_occluded_mean: {last_metrics['temporal_prediction_occluded_mean']:.6f}\n")
                handle.write(f"- final_temporal_prediction_reappeared_mean: {last_metrics['temporal_prediction_reappeared_mean']:.6f}\n")
                handle.write(f"- final_temporal_context_visible_used_count: {last_metrics['temporal_context_visible_used_count']:.1f}\n")
                handle.write(f"- final_temporal_context_occluded_used_count: {last_metrics['temporal_context_occluded_used_count']:.1f}\n")
                handle.write(f"- final_temporal_memory_occluded_hit_fraction: {last_metrics['temporal_memory_occluded_hit_fraction']:.3f}\n")
                handle.write(f"- final_temporal_memory_occluded_confidence_mean: {last_metrics['temporal_memory_occluded_confidence_mean']:.3f}\n")
                handle.write(f"- final_memory_prediction_occluded_impact_mean: {last_metrics['memory_prediction_occluded_impact_mean']:.6f}\n")
                handle.write(f"- final_memory_total_prediction_occluded_impact_mean: {last_metrics['memory_total_prediction_occluded_impact_mean']:.6f}\n")
                handle.write(f"- final_memory_definition_prediction_occluded_impact_mean: {last_metrics['memory_definition_prediction_occluded_impact_mean']:.6f}\n")
                handle.write(f"- final_memory_condition_norm: {last_metrics['memory_condition_norm']:.6f}\n")
                handle.write(f"- final_memory_definition_condition_norm: {last_metrics['memory_definition_condition_norm']:.6f}\n")
                handle.write(f"- final_temporal_memory_definition_occluded_flip_fraction: {last_metrics['temporal_memory_definition_occluded_flip_fraction']:.3f}\n")
            if "phase_ternary_mode_probe_accuracy" in last_metrics:
                handle.write(f"- final_phase_ternary_probe_accuracy: {last_metrics['phase_ternary_mode_probe_accuracy']:.3f}\n")
                handle.write(f"- final_phase_ternary_mode_mutual_information: {last_metrics['phase_ternary_mode_mutual_information']:.3f}\n")
                handle.write(f"- final_phase_ternary_axis_usage_count: {last_metrics['phase_ternary_axis_usage_count']:.1f}\n")
            if "object_ternary_mode_probe_accuracy" in last_metrics:
                handle.write(f"- final_object_ternary_probe_accuracy: {last_metrics['object_ternary_mode_probe_accuracy']:.3f}\n")
                handle.write(f"- final_object_ternary_mode_mutual_information: {last_metrics['object_ternary_mode_mutual_information']:.3f}\n")
                handle.write(f"- final_object_context_consistency: {last_metrics['object_mode_context_consistency']:.3f}\n")
            if "memory_object_feature_probe_accuracy" in last_metrics:
                handle.write(f"- final_memory_object_probe_accuracy: {last_metrics['memory_object_feature_probe_accuracy']:.3f}\n")
                handle.write(f"- final_memory_object_centroid_separation: {last_metrics['memory_object_feature_centroid_separation']:.3f}\n")
            if "occluded_object_ternary_mode_probe_accuracy" in last_metrics:
                handle.write(f"- final_occluded_object_ternary_probe_accuracy: {last_metrics['occluded_object_ternary_mode_probe_accuracy']:.3f}\n")
                handle.write(f"- final_occluded_object_ternary_mode_mutual_information: {last_metrics['occluded_object_ternary_mode_mutual_information']:.3f}\n")
            if "occluded_base_ternary_mode_probe_accuracy" in last_metrics:
                handle.write(f"- final_occluded_base_ternary_probe_accuracy: {last_metrics['occluded_base_ternary_mode_probe_accuracy']:.3f}\n")
                handle.write(f"- final_occluded_memory_definition_object_probe_delta: {last_metrics['occluded_memory_definition_object_probe_delta']:.3f}\n")
            if "reappeared_feature_match_accuracy" in last_metrics:
                handle.write(f"- final_reappeared_ternary_match_accuracy: {last_metrics['reappeared_feature_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_base_ternary_match_accuracy: {last_metrics['reappeared_base_feature_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_memory_definition_match_delta: {last_metrics['reappeared_memory_definition_match_delta']:.3f}\n")
                handle.write(f"- final_reappeared_ternary_match_margin: {last_metrics['reappeared_feature_match_margin']:.6f}\n")
                handle.write(f"- final_reappeared_paired_ternary_match_accuracy: {last_metrics['reappeared_paired_feature_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_paired_base_ternary_match_accuracy: {last_metrics['reappeared_base_paired_feature_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_paired_memory_definition_match_delta: {last_metrics['reappeared_paired_memory_definition_match_delta']:.3f}\n")
            if "reappeared_file_instance_match_accuracy" in last_metrics:
                handle.write(f"- final_reappeared_file_instance_match_accuracy: {last_metrics['reappeared_file_instance_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_file_hard_instance_match_accuracy: {last_metrics['reappeared_file_instance_hard_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_target_file_instance_match_accuracy: {last_metrics['reappeared_target_file_instance_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_target_file_hard_instance_match_accuracy: {last_metrics['reappeared_target_file_instance_hard_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_query_file_instance_match_accuracy: {last_metrics['reappeared_query_file_instance_match_accuracy']:.3f}\n")
                handle.write(f"- final_reappeared_query_file_hard_instance_match_accuracy: {last_metrics['reappeared_query_file_instance_hard_match_accuracy']:.3f}\n")
                if "reappeared_expected_file_instance_match_accuracy" in last_metrics:
                    handle.write(f"- final_reappeared_expected_file_instance_match_accuracy: {last_metrics['reappeared_expected_file_instance_match_accuracy']:.3f}\n")
                    handle.write(f"- final_reappeared_expected_file_hard_instance_match_accuracy: {last_metrics['reappeared_expected_file_instance_hard_match_accuracy']:.3f}\n")
                    if "reappeared_expected_state_instance_match_accuracy" in last_metrics:
                        handle.write(f"- final_reappeared_expected_state_instance_match_accuracy: {last_metrics['reappeared_expected_state_instance_match_accuracy']:.3f}\n")
                        handle.write(f"- final_reappeared_expected_state_hard_instance_match_accuracy: {last_metrics['reappeared_expected_state_instance_hard_match_accuracy']:.3f}\n")
                    if "reappeared_trajectory_position_error" in last_metrics:
                        handle.write(f"- final_reappeared_trajectory_position_error: {last_metrics['reappeared_trajectory_position_error']:.6f}\n")
                        handle.write(f"- final_reappeared_trajectory_valid_fraction: {last_metrics['reappeared_trajectory_valid_fraction']:.3f}\n")
                    if "active_file_expectation_pair" in last_metrics:
                        handle.write(f"- final_active_file_expectation_pair_loss: {last_metrics['active_file_expectation_pair']:.6f}\n")
                        handle.write(f"- final_active_file_expectation_hard_loss: {last_metrics['active_file_expectation_hard']:.6f}\n")
                if "reappeared_active_query_file_candidate_instance_match_accuracy" in last_metrics:
                    handle.write(f"- final_reappeared_active_query_file_match_accuracy: {last_metrics['reappeared_active_query_file_candidate_instance_match_accuracy']:.3f}\n")
                    handle.write(f"- final_reappeared_active_query_file_hard_match_accuracy: {last_metrics['reappeared_active_query_file_candidate_instance_hard_match_accuracy']:.3f}\n")
                    handle.write(f"- final_reappeared_active_query_file_candidate_mean_count: {last_metrics['reappeared_active_query_file_candidate_mean_count']:.3f}\n")
                    handle.write(f"- final_reappeared_active_query_file_row_coverage: {last_metrics['reappeared_active_query_file_candidate_row_coverage_fraction']:.3f}\n")
                    handle.write(f"- final_reappeared_active_query_file_target_recall: {last_metrics['reappeared_active_query_file_candidate_target_recall_fraction']:.3f}\n")
                if "reappeared_learned_active_query_file_candidate_instance_match_accuracy" in last_metrics:
                    handle.write(f"- final_reappeared_learned_active_query_file_match_accuracy: {last_metrics['reappeared_learned_active_query_file_candidate_instance_match_accuracy']:.3f}\n")
                    handle.write(f"- final_reappeared_learned_active_query_file_hard_match_accuracy: {last_metrics['reappeared_learned_active_query_file_candidate_instance_hard_match_accuracy']:.3f}\n")
                    handle.write(f"- final_reappeared_learned_active_query_file_candidate_mean_count: {last_metrics['reappeared_learned_active_query_file_candidate_mean_count']:.3f}\n")
                    handle.write(f"- final_reappeared_learned_active_query_file_row_coverage: {last_metrics['reappeared_learned_active_query_file_candidate_row_coverage_fraction']:.3f}\n")
                    handle.write(f"- final_reappeared_learned_active_query_file_target_recall: {last_metrics['reappeared_learned_active_query_file_candidate_target_recall_fraction']:.3f}\n")
                    handle.write(f"- final_reappeared_learned_active_gate_scaffold_recall: {last_metrics['reappeared_learned_active_file_gate_scaffold_recall']:.3f}\n")
            if "occluded_memory_object_feature_probe_accuracy" in last_metrics:
                handle.write(f"- final_occluded_memory_object_probe_accuracy: {last_metrics['occluded_memory_object_feature_probe_accuracy']:.3f}\n")
                handle.write(f"- final_occluded_memory_object_centroid_separation: {last_metrics['occluded_memory_object_feature_centroid_separation']:.3f}\n")
            if "definition_hardened_count" in last_metrics:
                handle.write(f"- final_definition_candidate_count: {last_metrics['definition_candidate_count']:.1f}\n")
                handle.write(f"- final_definition_hardened_count: {last_metrics['definition_hardened_count']:.1f}\n")
                handle.write(f"- final_definition_softened_count: {last_metrics['definition_softened_count']:.1f}\n")
                handle.write(f"- final_definition_rejected_count: {last_metrics['definition_rejected_count']:.1f}\n")
                handle.write(f"- final_definition_ready_window_count: {last_metrics['definition_ready_window_count']:.1f}\n")
                handle.write(f"- final_definition_counter_window_count: {last_metrics['definition_counter_window_count']:.1f}\n")
                handle.write(f"- final_definition_quarantined_update_count: {last_metrics['definition_quarantined_update_count']:.1f}\n")
                handle.write(f"- final_definition_mean_ready_mode_mi: {last_metrics['definition_mean_ready_mode_mi']:.3f}\n")
                handle.write(f"- final_definition_mean_ready_prediction_impact: {last_metrics['definition_mean_ready_prediction_impact']:.6f}\n")
    if definition_tracker is not None:
        with open(run_dir / "definition_evidence.json", "w", encoding="utf-8") as handle:
            json.dump(definition_tracker.to_dict(), handle, indent=2, sort_keys=True)
    return run_dir


@torch.no_grad()
def evaluate(checkpoint: str | Path, device_name: str = "cuda", split: str = "test", limit: int | None = None) -> dict[str, float]:
    device = resolve_device(device_name)
    model, cfg, _payload = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    data_cfg = DatasetConfig(**cfg.dataset.__dict__)
    data_cfg.split = split
    data_cfg.variant = split
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
def axis_report(
    checkpoint: str | Path,
    out_path: str | Path | None = None,
    device_name: str = "cuda",
    split: str = "test",
    limit: int | None = None,
    min_usage: float = 1e-6,
    label_key: str = "mode",
) -> Path:
    device = resolve_device(device_name)
    model, cfg, _payload = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    data_cfg = DatasetConfig(**cfg.dataset.__dict__)
    data_cfg.split = split
    data_cfg.variant = split
    if limit is not None:
        data_cfg.limit = limit
    dataset = make_dataset(data_cfg, cfg.model)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    ternary_chunks: list[torch.Tensor] = []
    mode_chunks: list[torch.Tensor] = []
    context_chunks: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="axis-report", unit="batch"):
        batch = move_batch(batch, device)
        output = model.forward_train(batch, include_label_diagnostics=False)
        labels = batch.get(label_key)
        if labels is None and label_key == "mode":
            labels = batch.get("label")
        if labels is None:
            raise KeyError(f"batch does not contain label key {label_key!r}")
        ternary_chunks.append(output.ternary.detach().cpu())
        mode_chunks.append(labels.detach().cpu())
        context_chunks.append(output.context.probs.argmax(dim=-1).detach().cpu())
    if not ternary_chunks:
        raise RuntimeError("axis report has no batches to inspect")

    ternary = torch.cat(ternary_chunks, dim=0)
    modes = torch.cat(mode_chunks, dim=0)
    contexts = torch.cat(context_chunks, dim=0)
    diagnostics = {
        key: float(value.detach().cpu())
        for key, value in ternary_label_diagnostics(ternary, modes, contexts).items()
    }
    axes = ternary_axis_specialization(ternary, modes, min_usage=min_usage)
    report = {
        "checkpoint": str(checkpoint),
        "split": split,
        "label_key": label_key,
        "rows": int(ternary.shape[0]),
        "min_usage": min_usage,
        "diagnostics": diagnostics,
        "axes": axes,
    }
    if out_path is None:
        checkpoint_path = Path(checkpoint)
        suffix = "" if label_key == "mode" else f"_{label_key.replace('/', '_')}"
        out_path = checkpoint_path.parent.parent / f"axis_specialization{suffix}_{split}.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return out


def _metric(metrics: dict[str, float | int], key: str) -> float:
    return float(metrics.get(key, float("nan")))


def run_seed_sweep(
    cfg: TrainConfig,
    device_name: str = "cuda",
    seeds: list[int] | None = None,
    steps: int | None = None,
    disabled_cfg: TrainConfig | None = None,
    comparison_cfgs: list[tuple[str, TrainConfig]] | None = None,
    eval_split: str = "test",
    eval_limit: int | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    seeds = seeds or [cfg.dataset.seed]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(out_dir) if out_dir is not None else Path(cfg.runs_dir) / f"{stamp}_ternary_seed_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    configs = [("enabled", cfg)]
    if disabled_cfg is not None:
        configs.append(("disabled", disabled_cfg))
    configs.extend(comparison_cfgs or [])
    for seed in seeds:
        for condition, base_cfg in configs:
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg.dataset.seed = seed
            if steps is not None:
                run_cfg.max_steps = steps
            run_cfg.run_name = f"{base_cfg.run_name}_{condition}_seed_{seed}"
            run_cfg.runs_dir = str(sweep_dir / "runs")
            run_dir = train(run_cfg, device_name=device_name)
            checkpoint = run_dir / "checkpoints" / "best.pt"
            heldout = evaluate(checkpoint, device_name=device_name, split=eval_split, limit=eval_limit)
            report_path = axis_report(
                checkpoint,
                device_name=device_name,
                split=eval_split,
                limit=eval_limit,
            )
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            metrics = read_metrics(run_dir / "metrics.jsonl")
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "axis_report": str(report_path),
                    "train_final": metrics[-1] if metrics else {},
                    "heldout": heldout,
                    "axis_diagnostics": report["diagnostics"],
                }
            )

    with open(sweep_dir / "sweep_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"seeds": seeds, "eval_split": eval_split, "runs": rows}, handle, indent=2, sort_keys=True)
    with open(sweep_dir / "sweep_summary.md", "w", encoding="utf-8") as handle:
        handle.write("# Ternary Seed Sweep\n\n")
        handle.write(f"- eval_split: {eval_split}\n")
        handle.write(f"- seeds: {', '.join(str(seed) for seed in seeds)}\n\n")
        handle.write(
            "| seed | condition | heldout_total | heldout_prediction | probe | "
            "occluded_probe | occluded_base | bridge_delta | reappear_match | "
            "reappear_pair | file_instance | file_hard | target_file_instance | query_file | query_hard | "
            "active_query | active_count | learned_active | learned_count | learned_recall | "
            "memory_def_impact | mi | nonzero | axis_usage | run |\n"
        )
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            heldout = row["heldout"]
            axis = row["axis_diagnostics"]
            handle.write(
                "| "
                f"{row['seed']} | {row['condition']} | "
                f"{_metric(heldout, 'total'):.6f} | "
                f"{_metric(heldout, 'prediction'):.6f} | "
                f"{_metric(axis, 'ternary_mode_probe_accuracy'):.3f} | "
                f"{_metric(heldout, 'occluded_object_ternary_mode_probe_accuracy'):.3f} | "
                f"{_metric(heldout, 'occluded_base_ternary_mode_probe_accuracy'):.3f} | "
                f"{_metric(heldout, 'occluded_memory_definition_object_probe_delta'):.3f} | "
                f"{_metric(heldout, 'reappeared_feature_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_paired_feature_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_file_instance_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_file_instance_hard_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_target_file_instance_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_query_file_instance_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_query_file_instance_hard_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_active_query_file_candidate_instance_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_active_query_file_candidate_mean_count'):.3f} | "
                f"{_metric(heldout, 'reappeared_learned_active_query_file_candidate_instance_match_accuracy'):.3f} | "
                f"{_metric(heldout, 'reappeared_learned_active_query_file_candidate_mean_count'):.3f} | "
                f"{_metric(heldout, 'reappeared_learned_active_file_gate_scaffold_recall'):.3f} | "
                f"{_metric(heldout, 'memory_definition_prediction_occluded_impact_mean'):.6f} | "
                f"{_metric(axis, 'ternary_mode_mutual_information'):.3f} | "
                f"{_metric(heldout, 'ternary_nonzero_fraction'):.3f} | "
                f"{_metric(axis, 'ternary_axis_usage_count'):.1f} | "
                f"{row['run_dir']} |\n"
            )
    return sweep_dir


@torch.no_grad()
def sample(checkpoint: str | Path, out_dir: str | Path, device_name: str = "cuda", split: str = "test") -> Path:
    device = resolve_device(device_name)
    model, cfg, _payload = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_cfg = DatasetConfig(**cfg.dataset.__dict__)
    data_cfg.split = split
    data_cfg.variant = split
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
