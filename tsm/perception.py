from __future__ import annotations

import torch
from torch import nn

from .config import TsmConfig
from .types import PerceptionBatch, PerceptionMeta


def _patch_position_features(
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack(
        [
            xx,
            yy,
            xx * yy,
            xx.square(),
            yy.square(),
            torch.sin(torch.pi * xx),
            torch.cos(torch.pi * xx),
            torch.sin(torch.pi * yy),
            torch.cos(torch.pi * yy),
        ],
        dim=-1,
    ).view(1, height * width, -1)
    repeats = (channels + base.shape[-1] - 1) // base.shape[-1]
    return base.repeat(1, 1, repeats)[..., :channels]


def _patch_coordinates(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(height * width, 2)


def _image_salience_position(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1)
    weights = (gray - gray.amin(dim=(1, 2), keepdim=True)).clamp_min(0.0)
    weight_sum = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    height, width = gray.shape[-2:]
    y = torch.arange(height, device=image.device, dtype=image.dtype) / float(max(height, 1))
    x = torch.arange(width, device=image.device, dtype=image.dtype) / float(max(width, 1))
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pos_x = (weights * xx).sum(dim=(1, 2), keepdim=True) / weight_sum
    pos_y = (weights * yy).sum(dim=(1, 2), keepdim=True) / weight_sum
    position = torch.cat([pos_x, pos_y], dim=-1).view(image.shape[0], 2)
    empty = weights.sum(dim=(1, 2)) <= 1e-6
    if bool(empty.any().item()):
        position = position.clone()
        position[empty] = 0.5
    return position


class ImagePatchTokenizer(nn.Module):
    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(
            cfg.image_channels,
            cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grid = self.proj(image)
        tokens = grid.flatten(2).transpose(1, 2)
        if self.cfg.use_patch_position_features:
            position = _patch_position_features(
                grid.shape[-2],
                grid.shape[-1],
                grid.shape[1],
                image.device,
                grid.dtype,
            )
            tokens = tokens + float(self.cfg.patch_position_scale) * position
        return self.norm(tokens)


class PerceptionSurface(nn.Module):
    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ImagePatchTokenizer(cfg)
        self.latents = nn.Parameter(torch.randn(cfg.workspace_latents, cfg.d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, image: torch.Tensor, dataset_id: torch.Tensor | None = None) -> PerceptionBatch:
        tokens = self.tokenizer(image)
        batch_size = image.shape[0]
        queries = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        latents, attn_weights = self.cross_attn(
            queries,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        latents = self.norm(latents + queries)
        confidence = torch.ones(batch_size, self.cfg.workspace_latents, 1, device=image.device)
        time_tag = torch.zeros(batch_size, self.cfg.workspace_latents, 1, device=image.device)
        patch_grid = self.cfg.image_size // self.cfg.patch_size
        patch_positions = _patch_coordinates(patch_grid, patch_grid, image.device, latents.dtype)
        latent_positions = torch.matmul(attn_weights.mean(dim=1).to(latents.dtype), patch_positions)
        return PerceptionBatch(
            latents=latents,
            meta=PerceptionMeta(
                source_confidence=confidence,
                time_tag=time_tag,
                dataset_id=dataset_id,
                position=latent_positions,
                binding_position=_image_salience_position(image),
            ),
        )
