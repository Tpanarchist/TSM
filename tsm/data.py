from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DatasetConfig, TsmConfig


DATASET_ALIASES = {
    "mnist": "ylecun/mnist",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
    "fashion-mnist": "zalando-datasets/fashion_mnist",
}


def canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def load_public_dataset(name: str, split: str, cache_dir: str):
    from datasets import load_dataset

    return load_dataset(canonical_dataset_name(name), split=split, cache_dir=cache_dir)


def _dataset_id(name: str) -> int:
    return zlib.crc32(canonical_dataset_name(name).encode("utf-8")) & 0x7FFFFFFF


def _find_image(row: dict[str, Any]) -> Image.Image:
    for key in ("image", "img"):
        value = row.get(key)
        if value is not None:
            if isinstance(value, Image.Image):
                return value
            return Image.fromarray(np.asarray(value))
    raise KeyError("row does not contain an image/img field")


def pil_to_tensor(image: Image.Image, cfg: TsmConfig) -> torch.Tensor:
    mode = "L" if cfg.image_channels == 1 else "RGB"
    image = image.convert(mode).resize((cfg.image_size, cfg.image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if cfg.image_channels == 1:
        array = array[None, :, :]
    else:
        array = array.transpose(2, 0, 1)
    return torch.from_numpy(array)


def deterministic_next_image(image: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    shift_y = int(torch.randint(-2, 3, (1,), generator=gen).item())
    shift_x = int(torch.randint(-2, 3, (1,), generator=gen).item())
    out = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
    noise = torch.randn(out.shape, generator=gen, dtype=out.dtype) * 0.03
    out = (out + noise).clamp(0.0, 1.0)
    if out.shape[-1] >= 12 and out.shape[-2] >= 12:
        box = max(2, out.shape[-1] // 7)
        y = int(torch.randint(0, out.shape[-2] - box + 1, (1,), generator=gen).item())
        x = int(torch.randint(0, out.shape[-1] - box + 1, (1,), generator=gen).item())
        out[..., y : y + box, x : x + box] *= 0.25
    return out


class ImageStreamDataset(Dataset):
    def __init__(
        self,
        data_cfg: DatasetConfig,
        model_cfg: TsmConfig,
        hf_dataset: Any | None = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.name = canonical_dataset_name(data_cfg.name)
        self.dataset_id = _dataset_id(self.name)
        self.ds = hf_dataset if hf_dataset is not None else load_public_dataset(self.name, data_cfg.split, data_cfg.cache_dir)
        if data_cfg.limit is not None and hasattr(self.ds, "select"):
            self.ds = self.ds.select(range(min(data_cfg.limit, len(self.ds))))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.ds[idx]
        image = pil_to_tensor(_find_image(row), self.model_cfg)
        return {
            "image_t": image,
            "image_tp1": deterministic_next_image(image, self.data_cfg.seed + idx),
            "label": torch.tensor(int(row.get("label", -1)), dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }


class SyntheticImageStreamDataset(Dataset):
    def __init__(self, model_cfg: TsmConfig, length: int = 512, seed: int = 17) -> None:
        self.model_cfg = model_cfg
        self.length = length
        self.seed = seed
        self.dataset_id = _dataset_id("synthetic")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        c = self.model_cfg.image_channels
        h = self.model_cfg.image_size
        image = torch.zeros(c, h, h)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(h), indexing="ij")
        for _ in range(3):
            cy = torch.randint(4, h - 4, (1,), generator=gen).item()
            cx = torch.randint(4, h - 4, (1,), generator=gen).item()
            radius = torch.randint(2, max(3, h // 5), (1,), generator=gen).item()
            value = torch.rand(1, generator=gen).item() * 0.8 + 0.2
            mask = ((yy - cy).square() + (xx - cx).square()) <= radius * radius
            image[:, mask] = value
        image = image.clamp(0.0, 1.0)
        return {
            "image_t": image,
            "image_tp1": deterministic_next_image(image, self.seed * 10 + idx),
            "label": torch.tensor(-1, dtype=torch.long),
            "mode": torch.tensor(-1, dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }


def _synthetic_base_image(model_cfg: TsmConfig, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    c = model_cfg.image_channels
    h = model_cfg.image_size
    image = torch.zeros(c, h, h)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(h), indexing="ij")
    for _ in range(3):
        cy = torch.randint(5, max(6, h - 5), (1,), generator=gen).item()
        cx = torch.randint(5, max(6, h - 5), (1,), generator=gen).item()
        radius = torch.randint(2, max(3, h // 5), (1,), generator=gen).item()
        value = torch.rand(1, generator=gen).item() * 0.8 + 0.2
        mask = ((yy - cy).square() + (xx - cx).square()) <= radius * radius
        image[:, mask] = value
    return image.clamp(0.0, 1.0)


def _apply_mode_marker(image: torch.Tensor, mode: int) -> torch.Tensor:
    out = image.clone()
    marker_size = max(2, min(4, image.shape[-1] // 8))
    value = float(mode + 1) / 4.0
    out[..., :marker_size, :marker_size] = value
    out[..., :marker_size, marker_size : marker_size * 2] = 0.0
    return out


def apply_controlled_mode(image: torch.Tensor, mode: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if mode == 0:
        out = image.clone()
    elif mode == 1:
        out = torch.roll(image, shifts=3, dims=-1)
    elif mode == 2:
        out = torch.roll(image, shifts=3, dims=-2)
    elif mode == 3:
        out = 1.0 - image
        box = max(3, image.shape[-1] // 5)
        y = int(torch.randint(0, image.shape[-2] - box + 1, (1,), generator=gen).item())
        x = int(torch.randint(0, image.shape[-1] - box + 1, (1,), generator=gen).item())
        out[..., y : y + box, x : x + box] *= 0.15
    else:
        raise ValueError(f"unsupported synthetic mode: {mode}")
    return _apply_mode_marker(out.clamp(0.0, 1.0), mode)


class MultiModeSyntheticImageStreamDataset(Dataset):
    """Synthetic stream with known transformation regimes for ContextRouter validation."""

    def __init__(self, model_cfg: TsmConfig, length: int = 1024, seed: int = 31, modes: int = 4) -> None:
        if modes != 4:
            raise ValueError("MultiModeSyntheticImageStreamDataset currently supports exactly 4 modes")
        self.model_cfg = model_cfg
        self.length = length
        self.seed = seed
        self.modes = modes
        self.dataset_id = _dataset_id("synthetic_modes")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mode = idx % self.modes
        base = _synthetic_base_image(self.model_cfg, self.seed + idx)
        image = _apply_mode_marker(base, mode)
        next_image = apply_controlled_mode(base, mode, self.seed * 100 + idx)
        return {
            "image_t": image,
            "image_tp1": next_image,
            "label": torch.tensor(mode, dtype=torch.long),
            "mode": torch.tensor(mode, dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }


def make_dataset(data_cfg: DatasetConfig, model_cfg: TsmConfig) -> Dataset:
    if data_cfg.name == "synthetic":
        return SyntheticImageStreamDataset(model_cfg, length=data_cfg.limit or 512, seed=data_cfg.seed)
    if data_cfg.name in {"synthetic_modes", "synthetic_multimode"}:
        return MultiModeSyntheticImageStreamDataset(model_cfg, length=data_cfg.limit or 1024, seed=data_cfg.seed)
    Path(data_cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    return ImageStreamDataset(data_cfg, model_cfg)
