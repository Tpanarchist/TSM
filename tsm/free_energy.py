from __future__ import annotations

import torch
import torch.nn.functional as F


def variational_free_energy(raw: torch.Tensor, precision: torch.Tensor, q: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    accuracy = 0.5 * (raw.square() * precision).mean()
    complexity = F.mse_loss(q.mean(dim=1), prior)
    return accuracy + complexity


def expected_free_energy(policy_latents: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
    uncertainty = policy_latents.var(dim=1).mean(dim=-1)
    pragmatic = F.mse_loss(policy_latents.mean(dim=1), preference, reduction="none").mean(dim=-1)
    return pragmatic - uncertainty
