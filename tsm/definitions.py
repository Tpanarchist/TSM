from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from .config import TsmConfig
from .ternary import TernaryProject
from .types import HardeningState


@dataclass
class TsmDefinition:
    name: str
    kind: str
    boundary_conditions: list[str] = field(default_factory=list)
    permitted_relations: list[str] = field(default_factory=list)
    incompatible_relations: list[str] = field(default_factory=list)
    construction_rules: list[str] = field(default_factory=list)
    hardening_state: HardeningState = HardeningState.SOFT


class DefinitionBank(nn.Module):
    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        scale = cfg.d_model**-0.5
        self.axes = nn.Parameter(torch.randn(cfg.contexts, cfg.definitions_per_context, cfg.d_model) * scale)
        self.memory_axes = nn.Parameter(torch.randn(cfg.contexts, cfg.definitions_per_context, cfg.d_model) * scale)
        self.log_tau = nn.Parameter(torch.full((cfg.contexts, cfg.definitions_per_context), -2.0))
        self.log_alpha = nn.Parameter(torch.zeros(cfg.contexts, cfg.definitions_per_context))
        self.file_query = nn.Linear(cfg.definitions_per_context, cfg.definitions_per_context, bias=False)
        nn.init.eye_(self.file_query.weight)
        self.records = [
            TsmDefinition(name=f"ctx{k}_def{j}", kind="learned")
            for k in range(cfg.contexts)
            for j in range(cfg.definitions_per_context)
        ]

    def forward(
        self,
        eps: torch.Tensor,
        ctx_probs: torch.Tensor,
        memory_feature: torch.Tensor | None = None,
        memory_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.project(eps, ctx_probs, memory_feature, memory_confidence)

    def project(
        self,
        eps: torch.Tensor,
        ctx_probs: torch.Tensor,
        memory_feature: torch.Tensor | None = None,
        memory_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self.raw_scores(eps, ctx_probs, memory_feature, memory_confidence)
        tau = torch.einsum("bk,kj->bj", ctx_probs, self.log_tau.exp())
        alpha = torch.einsum("bk,kj->bj", ctx_probs, self.log_alpha.exp())
        return TernaryProject.apply(raw, tau, alpha)

    def raw_scores(
        self,
        eps: torch.Tensor,
        ctx_probs: torch.Tensor,
        memory_feature: torch.Tensor | None = None,
        memory_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = eps.mean(dim=1)
        axes = F.normalize(self.axes, dim=-1)
        per_ctx = torch.einsum("bd,kjd->bkj", pooled, axes)
        raw = torch.einsum("bk,bkj->bj", ctx_probs, per_ctx)
        if memory_feature is not None and self.cfg.use_memory_conditioning:
            confidence = memory_confidence if memory_confidence is not None else torch.ones(
                memory_feature.shape[0],
                1,
                dtype=memory_feature.dtype,
                device=memory_feature.device,
            )
            memory_axes = F.normalize(self.memory_axes, dim=-1)
            memory_per_ctx = torch.einsum("bd,kjd->bkj", memory_feature, memory_axes)
            memory_raw = torch.einsum("bk,bkj->bj", ctx_probs, memory_per_ctx)
            raw = raw + self.cfg.memory_definition_scale * confidence.clamp(0.0, 1.0) * memory_raw
        return raw

    def memory_scores(
        self,
        memory_feature: torch.Tensor,
        ctx_probs: torch.Tensor,
        memory_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        confidence = memory_confidence if memory_confidence is not None else torch.ones(
            memory_feature.shape[0],
            1,
            dtype=memory_feature.dtype,
            device=memory_feature.device,
        )
        memory_axes = F.normalize(self.memory_axes, dim=-1)
        memory_per_ctx = torch.einsum("bd,kjd->bkj", memory_feature, memory_axes)
        memory_raw = torch.einsum("bk,bkj->bj", ctx_probs, memory_per_ctx)
        return self.cfg.memory_definition_scale * confidence.clamp(0.0, 1.0) * memory_raw

    def file_query_scores(self, definition_scores: torch.Tensor) -> torch.Tensor:
        return self.file_query(definition_scores)

    def bit_cost(self) -> torch.Tensor:
        tau_cost = self.log_tau.exp().reciprocal().mean()
        alpha_cost = self.log_alpha.exp().mean()
        axis_cost = self.axes.square().mean() + self.memory_axes.square().mean()
        query_cost = self.file_query.weight.square().mean()
        return tau_cost + alpha_cost + axis_cost + query_cost
