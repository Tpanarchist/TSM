from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import TsmConfig
from .types import AppraisalGroup, DriveState, SaeOutput


class SAE(nn.Module):
    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_precision = nn.Parameter(torch.zeros(cfg.contexts, cfg.d_model))
        self.seek_gain = nn.Parameter(torch.tensor(0.1))
        self.fear_gain = nn.Parameter(torch.tensor(0.1))
        self.aversive_mask = nn.Parameter(torch.zeros(cfg.d_model), requires_grad=False)

    def forward(
        self,
        observation: torch.Tensor,
        expected: torch.Tensor,
        source_confidence: torch.Tensor,
        attach_power: torch.Tensor,
        drives: DriveState,
        ctx_probs: torch.Tensor,
    ) -> SaeOutput:
        raw = observation - expected
        base = (ctx_probs @ self.base_precision).unsqueeze(1)
        confidence = source_confidence.clamp_min(1e-4).log()
        seek = drives.seeking.unsqueeze(1) * self.seek_gain
        fear = drives.fear.unsqueeze(1) * self.fear_gain * self.aversive_mask.view(1, 1, -1)
        precision = F.softplus(base + confidence + attach_power + seek + fear) + 1e-4
        eps = precision * raw
        severity = eps.norm(dim=-1)
        coherence = torch.exp(-0.5 * (raw.square() * precision).mean(dim=-1))
        group = self._classify(raw, severity)
        return SaeOutput(raw=raw, precision=precision, eps=eps, severity=severity, coherence=coherence, group=group)

    def _classify(self, raw: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        sign = raw.mean(dim=-1)
        high = severity > severity.detach().mean().clamp_min(1e-4)
        group = torch.full_like(severity, int(AppraisalGroup.CONFIRMATION), dtype=torch.long)
        group = torch.where(sign.abs() < 0.02, torch.full_like(group, int(AppraisalGroup.NOISE)), group)
        group = torch.where((sign > 0.02) & high, torch.full_like(group, int(AppraisalGroup.NOVELTY)), group)
        group = torch.where((sign < -0.02) & high, torch.full_like(group, int(AppraisalGroup.CONTRADICTION)), group)
        return group

    @staticmethod
    def iteration_budget(severity: torch.Tensor, seeking: torch.Tensor, base: int = 1, maximum: int = 4) -> int:
        score = severity.detach().mean().item() + seeking.detach().mean().item()
        return max(base, min(maximum, base + int(score > 1.0) + int(score > 2.0)))
