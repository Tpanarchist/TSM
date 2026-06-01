from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import torch


class AppraisalGroup(IntEnum):
    CONFIRMATION = 0
    NOVELTY = 1
    THREAT = 2
    CONTRADICTION = 3
    OPPORTUNITY = 4
    BOUNDARY = 5
    NOISE = 6


class HardeningState(str, Enum):
    SOFT = "soft"
    MIXED = "mixed"
    PROJECTED = "projected"
    HARDENED = "hardened"


class Stage(str, Enum):
    PRE_INCARNATION = "pre_incarnation"
    SENSORIMOTOR = "sensorimotor"
    SYMBOLIC = "symbolic"
    CONCRETE_OPERATIONAL = "concrete_operational"
    FORMAL_OPERATIONAL = "formal_operational"
    METACOGNITIVE = "metacognitive"


@dataclass
class DriveState:
    seeking: torch.Tensor
    fear: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, device: torch.device | str) -> "DriveState":
        return cls(
            seeking=torch.zeros(batch_size, 1, device=device),
            fear=torch.zeros(batch_size, 1, device=device),
        )


@dataclass
class PerceptionMeta:
    source_confidence: torch.Tensor
    time_tag: torch.Tensor
    dataset_id: torch.Tensor | None = None


@dataclass
class PerceptionBatch:
    latents: torch.Tensor
    meta: PerceptionMeta


@dataclass
class ContextOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    embedding: torch.Tensor


@dataclass
class SaeOutput:
    raw: torch.Tensor
    precision: torch.Tensor
    eps: torch.Tensor
    severity: torch.Tensor
    coherence: torch.Tensor
    group: torch.Tensor


@dataclass
class CandidateTruth:
    name: str
    evidence: float
    coherence: float
    risk: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    candidate: CandidateTruth


@dataclass
class TrainOutput:
    total_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]
    recon_image: torch.Tensor
    next_image: torch.Tensor
    context: ContextOutput
    sae: SaeOutput
    ternary: torch.Tensor


@dataclass
class TickOutput:
    reconstruction: torch.Tensor
    next_prediction: torch.Tensor
    context_probs: torch.Tensor
    sae: SaeOutput
    ternary: torch.Tensor
    action: None = None
