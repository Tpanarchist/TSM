from __future__ import annotations

from .types import Stage


class DevelopmentalScheduler:
    def __init__(self) -> None:
        self.stage = Stage.PRE_INCARNATION

    def active_terms(self) -> set[str]:
        return {"reconstruction", "prediction", "free_energy"}
