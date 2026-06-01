from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .config import TsmConfig


@dataclass
class Foundation:
    definitions: list[str] = field(default_factory=list)
    postulates: list[str] = field(default_factory=lambda: ["compare", "remember", "update"])
    common_notions: list[str] = field(default_factory=lambda: ["contradiction_blocks_promotion"])
    proven_truths: list[str] = field(default_factory=list)


class Reality(nn.Module):
    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.foundation = Foundation()
        self.prior = nn.Parameter(torch.zeros(cfg.contexts, cfg.d_model))
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.transition = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.definition_conditioner = nn.Linear(cfg.definitions_per_context, cfg.d_model * 2, bias=False)
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.image_channels
        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, patch_dim),
        )

    @property
    def patches_per_side(self) -> int:
        return self.cfg.image_size // self.cfg.patch_size

    def context_prior(self, ctx_probs: torch.Tensor) -> torch.Tensor:
        return ctx_probs @ self.prior

    def predict_latents(self, q: torch.Tensor, ctx_embedding: torch.Tensor) -> torch.Tensor:
        ctx = ctx_embedding.unsqueeze(1).expand(-1, q.shape[1], -1)
        return self.predictor(torch.cat([q, ctx], dim=-1))

    def condition_latents(
        self,
        q: torch.Tensor,
        ternary: torch.Tensor | None = None,
        memory_feature: torch.Tensor | None = None,
        memory_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = q
        if ternary is None or not self.cfg.use_ternary_conditioning:
            pass
        else:
            gamma_raw, beta_raw = self.definition_conditioner(ternary).chunk(2, dim=-1)
            scale = self.cfg.ternary_condition_scale
            gamma = 1.0 + scale * torch.tanh(gamma_raw).unsqueeze(1)
            beta = scale * torch.tanh(beta_raw).unsqueeze(1)
            out = out * gamma + beta
        if memory_feature is not None and self.cfg.use_memory_conditioning:
            confidence = memory_confidence if memory_confidence is not None else torch.ones(
                memory_feature.shape[0],
                1,
                dtype=memory_feature.dtype,
                device=memory_feature.device,
            )
            memory_delta = self.cfg.memory_condition_scale * torch.tanh(memory_feature).unsqueeze(1)
            out = out + confidence.unsqueeze(1).clamp(0.0, 1.0) * memory_delta
        return out

    def transition_latents(
        self,
        q: torch.Tensor,
        ctx_embedding: torch.Tensor,
        ternary: torch.Tensor | None = None,
        memory_feature: torch.Tensor | None = None,
        memory_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioned_q = self.condition_latents(q, ternary, memory_feature, memory_confidence)
        ctx = ctx_embedding.unsqueeze(1).expand(-1, q.shape[1], -1)
        return q + self.transition(torch.cat([conditioned_q, ctx], dim=-1))

    def decode_image(self, q: torch.Tensor) -> torch.Tensor:
        grid = self.patches_per_side
        needed = grid * grid
        if q.shape[1] < needed:
            repeats = (needed + q.shape[1] - 1) // q.shape[1]
            q = q.repeat(1, repeats, 1)
        q = q[:, :needed]
        patches = self.patch_decoder(q)
        bsz = q.shape[0]
        p = self.cfg.patch_size
        c = self.cfg.image_channels
        patches = patches.view(bsz, grid, grid, c, p, p)
        image = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return torch.sigmoid(image.view(bsz, c, self.cfg.image_size, self.cfg.image_size))

    def reconstruct_image(self, q: torch.Tensor, ternary: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode_image(self.condition_latents(q, ternary))

    def predict_next_image(
        self,
        q: torch.Tensor,
        ctx_embedding: torch.Tensor,
        ternary: torch.Tensor | None = None,
        memory_feature: torch.Tensor | None = None,
        memory_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decode_image(self.transition_latents(q, ctx_embedding, ternary, memory_feature, memory_confidence))
