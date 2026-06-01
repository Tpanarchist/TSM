from __future__ import annotations

import torch

from .types import CandidateTruth


class EvidenceAccumulator:
    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold
        self.count = 0

    def accumulate(self, delta_q: torch.Tensor, coherence: torch.Tensor) -> list[CandidateTruth]:
        self.count += 1
        score = float(coherence.detach().mean().cpu())
        if score >= self.threshold:
            return [CandidateTruth(name=f"coherent_impression_{self.count}", evidence=score, coherence=score)]
        return []
