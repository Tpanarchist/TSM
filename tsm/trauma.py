from __future__ import annotations

import torch

from .types import GateDecision


class TraumaMonitor:
    def __init__(self, severity_threshold: float = 10.0) -> None:
        self.severity_threshold = severity_threshold
        self.events: list[str] = []

    def scan(self, eps: torch.Tensor, rejections: list[GateDecision]) -> list[str]:
        if float(eps.detach().norm(dim=-1).mean().cpu()) > self.severity_threshold:
            self.events.append("persistent_high_prediction_error")
        if rejections:
            self.events.append(rejections[-1].reason)
        return self.events
