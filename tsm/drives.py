from __future__ import annotations

import torch

from .types import DriveState


class DriveDynamics:
    def initial(self, batch_size: int, device: torch.device | str) -> DriveState:
        return DriveState.zeros(batch_size, device)

    def update(self, drives: DriveState, severity: torch.Tensor) -> DriveState:
        signal = severity.detach().mean(dim=1, keepdim=True)
        seeking = 0.95 * drives.seeking + 0.05 * signal.clamp(0.0, 2.0)
        fear = 0.98 * drives.fear
        return DriveState(seeking=seeking, fear=fear)
