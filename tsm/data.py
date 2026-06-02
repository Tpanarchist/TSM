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
        label = torch.tensor(int(row.get("label", -1)), dtype=torch.long)
        return {
            "image_t": image,
            "image_tp1": deterministic_next_image(image, self.data_cfg.seed + idx),
            "label": label,
            "mode": label,
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


def _mode_variant(split: str | None) -> str:
    if split in {"test", "validation", "val", "heldout"}:
        return "heldout"
    return "train"


def apply_controlled_mode(image: torch.Tensor, mode: int, seed: int, split: str | None = "train") -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    variant = _mode_variant(split)
    if mode == 0:
        out = image.clone()
    elif mode == 1:
        shift = -5 if variant == "heldout" else 3
        out = torch.roll(image, shifts=shift, dims=-1)
    elif mode == 2:
        shift = 5 if variant == "heldout" else 3
        out = torch.roll(image, shifts=shift, dims=-2)
    elif mode == 3:
        out = 1.0 - image
        box = max(4, image.shape[-1] // 4) if variant == "heldout" else max(3, image.shape[-1] // 5)
        y = int(torch.randint(0, image.shape[-2] - box + 1, (1,), generator=gen).item())
        x = int(torch.randint(0, image.shape[-1] - box + 1, (1,), generator=gen).item())
        out[..., y : y + box, x : x + box] *= 0.05 if variant == "heldout" else 0.15
    else:
        raise ValueError(f"unsupported synthetic mode: {mode}")
    return _apply_mode_marker(out.clamp(0.0, 1.0), mode)


class MultiModeSyntheticImageStreamDataset(Dataset):
    """Synthetic stream with known transformation regimes for ContextRouter validation."""

    def __init__(
        self,
        model_cfg: TsmConfig,
        length: int = 1024,
        seed: int = 31,
        modes: int = 4,
        split: str = "train",
    ) -> None:
        if modes != 4:
            raise ValueError("MultiModeSyntheticImageStreamDataset currently supports exactly 4 modes")
        self.model_cfg = model_cfg
        self.length = length
        self.seed = seed
        self.modes = modes
        self.split = split
        self.dataset_id = _dataset_id("synthetic_modes")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mode = idx % self.modes
        base = _synthetic_base_image(self.model_cfg, self.seed + idx)
        image = _apply_mode_marker(base, mode)
        next_image = apply_controlled_mode(base, mode, self.seed * 100 + idx, self.split)
        return {
            "image_t": image,
            "image_tp1": next_image,
            "label": torch.tensor(mode, dtype=torch.long),
            "mode": torch.tensor(mode, dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }


def _temporal_variant(split: str | None) -> str:
    if split in {"test", "validation", "val", "heldout"}:
        return "heldout"
    return "train"


def _temporal_position(model_cfg: TsmConfig, sequence_id: int, object_id: int, phase: int, split: str) -> tuple[int, int]:
    h = model_cfg.image_size
    margin = max(4, h // 7)
    span = max(1, h - 2 * margin)
    speed = 4 if _temporal_variant(split) == "heldout" else 3
    base_x = (sequence_id * 5 + object_id * 3) % span
    base_y = (object_id * max(3, span // 4) + (sequence_id % 3) * 2) % span
    x = margin + ((base_x + phase * speed) % span)
    y = margin + ((base_y + (phase // 2)) % span)
    return int(x), int(y)


def _draw_temporal_object(image: torch.Tensor, object_id: int, x: int, y: int, value: float) -> None:
    h = image.shape[-1]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(h), indexing="ij")
    radius = max(2, h // 12)
    if object_id == 0:
        mask = (xx - x).abs().maximum((yy - y).abs()) <= radius
    elif object_id == 1:
        mask = (xx - x).square() + (yy - y).square() <= radius * radius
    elif object_id == 2:
        mask = (xx - x).abs() + (yy - y).abs() <= radius + 1
    else:
        mask = ((xx - x).abs() <= 1) | ((yy - y).abs() <= 1)
        mask = mask & ((xx - x).abs().maximum((yy - y).abs()) <= radius + 1)
    image[:, mask] = value


def _draw_occluder(image: torch.Tensor, x: int, y: int, split: str, opening: bool = False) -> None:
    h = image.shape[-1]
    half = max(3, h // 7) if _temporal_variant(split) == "heldout" else max(3, h // 8)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)
    x0 = max(0, x - half)
    x1 = min(h, x + half + 1)
    value = 0.48 if _temporal_variant(split) == "heldout" else 0.55
    image[..., y0:y1, x0:x1] = value
    image[..., y0:y1, x0] = 0.8
    image[..., y0:y1, x1 - 1] = 0.8
    image[..., y0, x0:x1] = 0.8
    image[..., y1 - 1, x0:x1] = 0.8
    if opening and x1 - x0 > 4:
        slit = x0 + (x1 - x0) // 2
        image[..., y0:y1, slit : min(x1, slit + 2)] = 0.15


def _render_temporal_frame(
    model_cfg: TsmConfig,
    object_id: int,
    sequence_id: int,
    phase: int,
    visible: bool,
    occluded: bool,
    split: str,
) -> torch.Tensor:
    c = model_cfg.image_channels
    h = model_cfg.image_size
    image = torch.full((c, h, h), 0.02)
    x, y = _temporal_position(model_cfg, sequence_id, object_id, phase, split)
    value = 0.45 + 0.13 * object_id
    if phase == 1 and visible:
        prev_x, prev_y = _temporal_position(model_cfg, sequence_id, object_id, max(0, phase - 1), split)
        _draw_temporal_object(image, object_id, prev_x, prev_y, value * 0.35)
    if phase == 2:
        _draw_occluder(image, x + 2, y, split, opening=False)
    if visible:
        _draw_temporal_object(image, object_id, x, y, value)
    if occluded:
        _draw_occluder(image, x, y, split, opening=phase >= 4)
    return image.clamp(0.0, 1.0)


def _contested_position(
    model_cfg: TsmConfig,
    scene_id: int,
    track_id: int,
    phase: int,
    split: str,
) -> tuple[int, int]:
    h = model_cfg.image_size
    margin = max(4, h // 7)
    span = max(1, h - 2 * margin)
    speed = 4 if _temporal_variant(split) == "heldout" else 3
    base = (scene_id * 5) % span
    y_base = ((scene_id % 4) * max(2, span // 6)) % span
    if track_id == 0:
        x = margin + ((base + phase * speed) % span)
        y = margin + ((y_base + 2 + phase // 2) % span)
    else:
        x = margin + ((base + span // 2 - phase * speed) % span)
        y = margin + ((y_base + max(6, span // 2) + phase // 2) % span)
    return int(x), int(y)


def _contested_curved_position(
    model_cfg: TsmConfig,
    scene_id: int,
    track_id: int,
    phase: int,
    split: str,
    track_count: int = 2,
) -> tuple[int, int]:
    h = model_cfg.image_size
    margin = max(4, h // 7)
    span = max(1, h - 2 * margin)
    phase = int(phase) % 5
    base = (scene_id * 5) % span
    y_base = ((scene_id % 4) * max(2, span // 6)) % span
    y_sep = max(6, span // 2)
    if track_count <= 2:
        # Preserve the original two-track curved benchmark exactly.
        x_offsets = {
            0: (1, 4, 7, 11, 14),
            1: (span // 2 + 1, span // 2 - 2, span // 2 - 5, span // 2 - 9, span // 2 - 12),
        }
        y_offsets = {
            0: (2, 5, 11, 13, 10),
            1: (y_sep + 2, y_sep + 5, y_sep + 11, y_sep + 13, y_sep + 10),
        }
        track = 0 if track_id == 0 else 1
        x = margin + ((base + x_offsets[track][phase]) % span)
        y = margin + ((y_base + y_offsets[track][phase]) % span)
        return int(x), int(y)

    # Phase offsets make the visible 1->2 velocity an intentionally bad
    # predictor of the hidden 2->0 continuation while preserving separability.
    x_offsets = {
        0: (1, 4, 7, 11, 14),
        1: (span // 2 + 1, span // 2 - 2, span // 2 - 5, span // 2 - 9, span // 2 - 12),
        2: (2, 5, 11, 13, 10),
        3: (span // 2 + 2, span // 2 + 5, span + 1, span + 3, span),
    }
    y_offsets = {
        0: (2, 5, 11, 13, 10),
        1: (y_sep + 2, y_sep + 5, y_sep + 11, y_sep + 13, y_sep + 10),
        2: (y_sep + 4, y_sep + 1, y_sep - 2, y_sep - 6, y_sep - 9),
        3: (1, 4, 7, 11, 14),
    }
    track = int(track_id) % min(max(2, track_count), 4)
    x = margin + ((base + x_offsets[track][phase]) % span)
    y = margin + ((y_base + y_offsets[track][phase]) % span)
    return int(x), int(y)


def _contested_motion_position(
    model_cfg: TsmConfig,
    scene_id: int,
    track_id: int,
    phase: int,
    split: str,
    motion: str = "linear",
    track_count: int = 2,
) -> tuple[int, int]:
    if motion == "curved":
        return _contested_curved_position(model_cfg, scene_id, track_id, phase, split, track_count)
    return _contested_position(model_cfg, scene_id, track_id, phase, split)


def _render_contested_frame(
    model_cfg: TsmConfig,
    scene_id: int,
    target_track_id: int,
    phase: int,
    target_visible: bool,
    target_occluded: bool,
    split: str,
    motion: str = "linear",
    track_count: int = 2,
) -> torch.Tensor:
    c = model_cfg.image_channels
    h = model_cfg.image_size
    image = torch.full((c, h, h), 0.02)
    object_id = 1
    value = 0.62
    for track_id in range(track_count):
        x, y = _contested_motion_position(model_cfg, scene_id, track_id, phase, split, motion, track_count)
        is_target = track_id == target_track_id
        if is_target:
            if phase == 1 and target_visible:
                prev_x, prev_y = _contested_motion_position(
                    model_cfg,
                    scene_id,
                    track_id,
                    max(0, phase - 1),
                    split,
                    motion,
                    track_count,
                )
                _draw_temporal_object(image, object_id, prev_x, prev_y, value * 0.35)
            if phase == 2:
                _draw_occluder(image, x + 2, y, split, opening=False)
            if target_visible:
                _draw_temporal_object(image, object_id, x, y, value)
            if target_occluded:
                _draw_occluder(image, x, y, split, opening=phase >= 4)
        else:
            if phase == 1:
                prev_x, prev_y = _contested_motion_position(
                    model_cfg,
                    scene_id,
                    track_id,
                    max(0, phase - 1),
                    split,
                    motion,
                    track_count,
                )
                _draw_temporal_object(image, object_id, prev_x, prev_y, value * 0.25)
            _draw_temporal_object(image, object_id, x, y, value * 0.92)
    return image.clamp(0.0, 1.0)


class TemporalObjectPermanenceDataset(Dataset):
    """Temporal stream for object permanence before any embodied/game adapter."""

    phase_count = 5
    sequential = True

    def __init__(
        self,
        model_cfg: TsmConfig,
        length: int = 1024,
        seed: int = 41,
        split: str = "train",
        object_count: int = 4,
    ) -> None:
        if object_count != 4:
            raise ValueError("TemporalObjectPermanenceDataset currently supports exactly 4 object identities")
        self.model_cfg = model_cfg
        self.length = length
        self.seed = seed
        self.split = split
        self.object_count = object_count
        self.dataset_id = _dataset_id("temporal_objects")

    def __len__(self) -> int:
        return self.length

    def _state(self, sequence_id: int, phase: int) -> tuple[int, bool, bool]:
        object_id = (sequence_id + self.seed) % self.object_count
        visible = phase in {0, 1, 2}
        occluded = phase in {3, 4}
        return object_id, visible, occluded

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence_id = idx // self.phase_count
        phase = idx % self.phase_count
        next_phase = (phase + 1) % self.phase_count
        next_sequence_id = sequence_id
        object_id, visible_t, occluded_t = self._state(sequence_id, phase)
        next_object_id, visible_tp1, occluded_tp1 = self._state(next_sequence_id, next_phase)

        mode = 2 if phase in {2, 3} else phase
        if phase == 4:
            mode = 3

        image_t = _render_temporal_frame(
            self.model_cfg,
            object_id,
            sequence_id,
            phase,
            visible_t,
            occluded_t,
            self.split,
        )
        image_tp1 = _render_temporal_frame(
            self.model_cfg,
            next_object_id,
            next_sequence_id,
            next_phase,
            visible_tp1,
            occluded_tp1,
            self.split,
        )
        x_t, y_t = _temporal_position(self.model_cfg, sequence_id, object_id, phase, self.split)
        x_tp1, y_tp1 = _temporal_position(self.model_cfg, next_sequence_id, next_object_id, next_phase, self.split)
        return {
            "image_t": image_t,
            "image_tp1": image_tp1,
            "label": torch.tensor(mode, dtype=torch.long),
            "mode": torch.tensor(mode, dtype=torch.long),
            "phase": torch.tensor(phase, dtype=torch.long),
            "object_id": torch.tensor(object_id, dtype=torch.long),
            "sequence_id": torch.tensor(sequence_id, dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
            "visible_t": torch.tensor(float(visible_t), dtype=torch.float32),
            "visible_tp1": torch.tensor(float(visible_tp1), dtype=torch.float32),
            "occluded_t": torch.tensor(float(occluded_t), dtype=torch.float32),
            "occluded_tp1": torch.tensor(float(occluded_tp1), dtype=torch.float32),
            "moved": torch.tensor(float(phase == 1), dtype=torch.float32),
            "identity_preserved": torch.tensor(float(next_object_id == object_id), dtype=torch.float32),
            "unexpected_disappearance": torch.tensor(float(visible_t and occluded_tp1), dtype=torch.float32),
            "object_position_t": torch.tensor([x_t, y_t], dtype=torch.float32),
            "object_position_tp1": torch.tensor([x_tp1, y_tp1], dtype=torch.float32),
        }


class ContestedTemporalObjectPermanenceDataset(Dataset):
    """Same-class object tracks where the target is occluded by row."""

    phase_count = 5
    sequential = True

    def __init__(
        self,
        model_cfg: TsmConfig,
        length: int = 1024,
        seed: int = 41,
        split: str = "train",
        motion: str = "linear",
        track_count: int = 2,
        dataset_name: str | None = None,
    ) -> None:
        if motion not in {"linear", "curved"}:
            raise ValueError(f"unsupported contested motion: {motion}")
        if track_count < 2 or track_count > 4:
            raise ValueError("ContestedTemporalObjectPermanenceDataset supports 2 to 4 tracks")
        self.model_cfg = model_cfg
        self.length = length
        self.seed = seed
        self.split = split
        self.motion = motion
        self.track_count = track_count
        default_name = (
            f"temporal_objects_contested_curved_{track_count}"
            if motion == "curved" and track_count != 2
            else ("temporal_objects_contested_curved" if motion == "curved" else "temporal_objects_contested_position")
        )
        self.dataset_name = dataset_name or default_name
        self.dataset_id = _dataset_id(self.dataset_name)

    def __len__(self) -> int:
        return self.length

    def _state(self, phase: int) -> tuple[bool, bool]:
        visible = phase in {0, 1, 2}
        occluded = phase in {3, 4}
        return visible, occluded

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rows_per_scene = self.phase_count * self.track_count
        scene_id = idx // rows_per_scene
        phase = (idx // self.track_count) % self.phase_count
        track_id = idx % self.track_count
        next_phase = (phase + 1) % self.phase_count
        object_id = 1
        object_file_id = scene_id * self.track_count + track_id

        visible_t, occluded_t = self._state(phase)
        visible_tp1, occluded_tp1 = self._state(next_phase)
        mode = 2 if phase in {2, 3} else phase
        if phase == 4:
            mode = 3

        image_t = _render_contested_frame(
            self.model_cfg,
            scene_id,
            track_id,
            phase,
            visible_t,
            occluded_t,
            self.split,
            self.motion,
            self.track_count,
        )
        image_tp1 = _render_contested_frame(
            self.model_cfg,
            scene_id,
            track_id,
            next_phase,
            visible_tp1,
            occluded_tp1,
            self.split,
            self.motion,
            self.track_count,
        )
        x_t, y_t = _contested_motion_position(
            self.model_cfg,
            scene_id,
            track_id,
            phase,
            self.split,
            self.motion,
            self.track_count,
        )
        x_tp1, y_tp1 = _contested_motion_position(
            self.model_cfg,
            scene_id,
            track_id,
            next_phase,
            self.split,
            self.motion,
            self.track_count,
        )
        other_track_id = (track_id + 1) % self.track_count
        other_x_t, other_y_t = _contested_motion_position(
            self.model_cfg,
            scene_id,
            other_track_id,
            phase,
            self.split,
            self.motion,
            self.track_count,
        )
        other_x_tp1, other_y_tp1 = _contested_motion_position(
            self.model_cfg,
            scene_id,
            other_track_id,
            next_phase,
            self.split,
            self.motion,
            self.track_count,
        )
        all_positions_t = []
        all_positions_tp1 = []
        all_file_ids = []
        for local_track in range(self.track_count):
            local_x_t, local_y_t = _contested_motion_position(
                self.model_cfg,
                scene_id,
                local_track,
                phase,
                self.split,
                self.motion,
                self.track_count,
            )
            local_x_tp1, local_y_tp1 = _contested_motion_position(
                self.model_cfg,
                scene_id,
                local_track,
                next_phase,
                self.split,
                self.motion,
                self.track_count,
            )
            all_positions_t.append([local_x_t, local_y_t])
            all_positions_tp1.append([local_x_tp1, local_y_tp1])
            all_file_ids.append(scene_id * self.track_count + local_track)
        return {
            "image_t": image_t,
            "image_tp1": image_tp1,
            "label": torch.tensor(mode, dtype=torch.long),
            "mode": torch.tensor(mode, dtype=torch.long),
            "phase": torch.tensor(phase, dtype=torch.long),
            "object_id": torch.tensor(object_id, dtype=torch.long),
            "track_id": torch.tensor(track_id, dtype=torch.long),
            "sequence_id": torch.tensor(scene_id, dtype=torch.long),
            "object_file_id": torch.tensor(object_file_id, dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
            "visible_t": torch.tensor(float(visible_t), dtype=torch.float32),
            "visible_tp1": torch.tensor(float(visible_tp1), dtype=torch.float32),
            "occluded_t": torch.tensor(float(occluded_t), dtype=torch.float32),
            "occluded_tp1": torch.tensor(float(occluded_tp1), dtype=torch.float32),
            "moved": torch.tensor(float(phase == 1), dtype=torch.float32),
            "identity_preserved": torch.tensor(1.0, dtype=torch.float32),
            "unexpected_disappearance": torch.tensor(float(visible_t and occluded_tp1), dtype=torch.float32),
            "object_position_t": torch.tensor([x_t, y_t], dtype=torch.float32),
            "object_position_tp1": torch.tensor([x_tp1, y_tp1], dtype=torch.float32),
            "distractor_position_t": torch.tensor([other_x_t, other_y_t], dtype=torch.float32),
            "distractor_position_tp1": torch.tensor([other_x_tp1, other_y_tp1], dtype=torch.float32),
            "all_object_positions_t": torch.tensor(all_positions_t, dtype=torch.float32),
            "all_object_positions_tp1": torch.tensor(all_positions_tp1, dtype=torch.float32),
            "all_object_file_ids": torch.tensor(all_file_ids, dtype=torch.long),
            "contested_track_count": torch.tensor(self.track_count, dtype=torch.long),
            "same_class_contested": torch.tensor(1.0, dtype=torch.float32),
            "nonlinear_contested_motion": torch.tensor(float(self.motion == "curved"), dtype=torch.float32),
            "identity_swap_violation": torch.tensor(0.0, dtype=torch.float32),
        }


def make_dataset(data_cfg: DatasetConfig, model_cfg: TsmConfig) -> Dataset:
    if data_cfg.name == "synthetic":
        return SyntheticImageStreamDataset(model_cfg, length=data_cfg.limit or 512, seed=data_cfg.seed)
    if data_cfg.name in {"synthetic_modes", "synthetic_multimode"}:
        split = data_cfg.variant or data_cfg.split
        return MultiModeSyntheticImageStreamDataset(model_cfg, length=data_cfg.limit or 1024, seed=data_cfg.seed, split=split)
    if data_cfg.name in {"temporal_objects", "object_permanence", "synthetic_temporal_objects"}:
        split = data_cfg.variant or data_cfg.split
        return TemporalObjectPermanenceDataset(model_cfg, length=data_cfg.limit or 1024, seed=data_cfg.seed, split=split)
    if data_cfg.name in {
        "temporal_objects_two_object",
        "temporal_objects_contested_position",
        "temporal_objects_contested_curved",
        "temporal_objects_contested_curved_2",
        "temporal_objects_contested_curved_3",
        "temporal_objects_contested_curved_4",
        "temporal_objects_contested_curved_4_wide",
        "temporal_objects_contested_nonlinear",
        "object_permanence_contested",
    }:
        split = data_cfg.variant or data_cfg.split
        motion = "curved" if data_cfg.name in {
            "temporal_objects_contested_curved",
            "temporal_objects_contested_curved_2",
            "temporal_objects_contested_curved_3",
            "temporal_objects_contested_curved_4",
            "temporal_objects_contested_curved_4_wide",
            "temporal_objects_contested_nonlinear",
        } else "linear"
        track_count = 2
        if data_cfg.name.endswith("_3"):
            track_count = 3
        elif data_cfg.name.endswith("_4") or data_cfg.name.endswith("_4_wide"):
            track_count = 4
        return ContestedTemporalObjectPermanenceDataset(
            model_cfg,
            length=data_cfg.limit or 1024,
            seed=data_cfg.seed,
            split=split,
            motion=motion,
            track_count=track_count,
            dataset_name=data_cfg.name if data_cfg.name == "temporal_objects_contested_curved_4_wide" else None,
        )
    Path(data_cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    return ImageStreamDataset(data_cfg, model_cfg)
