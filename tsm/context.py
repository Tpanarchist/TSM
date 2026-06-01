from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import TsmConfig
from .types import ContextOutput


class ContextRouter(nn.Module):
    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.contexts),
        )
        self.context_embeddings = nn.Parameter(torch.randn(cfg.contexts, cfg.d_model) * 0.02)

    def forward(self, q: torch.Tensor, o: torch.Tensor) -> ContextOutput:
        q_mean = q.mean(dim=1)
        o_mean = o.mean(dim=1)
        logits = self.router(torch.cat([q_mean, o_mean], dim=-1))
        probs = F.softmax(logits, dim=-1)
        embedding = probs @ self.context_embeddings
        return ContextOutput(logits=logits, probs=probs, embedding=embedding)


def context_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()


def context_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    mean_probs = probs.mean(dim=0)
    uniform_log = torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device, dtype=probs.dtype))
    return (mean_probs * mean_probs.clamp_min(1e-8).log()).sum() + uniform_log
