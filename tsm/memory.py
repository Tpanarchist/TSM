from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class MemoryRecord:
    severity: float
    coherence: float


class Memory:
    def __init__(self, max_records: int = 1024) -> None:
        self.records: deque[MemoryRecord] = deque(maxlen=max_records)

    def write(self, eps: torch.Tensor, coherence: torch.Tensor) -> None:
        self.records.append(
            MemoryRecord(
                severity=float(eps.detach().norm(dim=-1).mean().cpu()),
                coherence=float(coherence.detach().mean().cpu()),
            )
        )

    def short_term_decay(self) -> None:
        return None
