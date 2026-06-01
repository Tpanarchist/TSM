from __future__ import annotations

from .types import CandidateTruth, GateDecision


class MutationGate:
    def __init__(self, min_evidence: float = 0.75, min_coherence: float = 0.6, max_risk: float = 0.5) -> None:
        self.min_evidence = min_evidence
        self.min_coherence = min_coherence
        self.max_risk = max_risk
        self.decisions: list[GateDecision] = []

    @property
    def rejections(self) -> list[GateDecision]:
        return [decision for decision in self.decisions if not decision.accepted]

    def consider(self, candidate: CandidateTruth) -> GateDecision:
        if candidate.evidence < self.min_evidence:
            decision = GateDecision(False, "insufficient evidence", candidate)
        elif candidate.coherence < self.min_coherence:
            decision = GateDecision(False, "incoherent", candidate)
        elif candidate.risk > self.max_risk:
            decision = GateDecision(False, "risk", candidate)
        else:
            decision = GateDecision(True, "accepted", candidate)
        self.decisions.append(decision)
        return decision
