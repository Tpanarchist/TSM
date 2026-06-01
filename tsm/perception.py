from __future__ import annotations

import torch
from torch import nn

from .config import TsmConfig
from .types import PerceptionBatch, PerceptionMeta


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
        tokens = self.proj(image).flatten(2).transpose(1, 2)
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
        latents, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        latents = self.norm(latents + queries)
        confidence = torch.ones(batch_size, self.cfg.workspace_latents, 1, device=image.device)
        time_tag = torch.zeros(batch_size, self.cfg.workspace_latents, 1, device=image.device)
        return PerceptionBatch(
            latents=latents,
            meta=PerceptionMeta(source_confidence=confidence, time_tag=time_tag, dataset_id=dataset_id),
        )
