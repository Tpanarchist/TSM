from __future__ import annotations

import torch
from torch import nn


class TernaryProject(torch.autograd.Function):
    """Forward projects to {-alpha, 0, +alpha}; backward uses a clipped STE."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tau: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, tau)
        out = torch.zeros_like(x)
        out = torch.where(x > tau, torch.ones_like(out), out)
        out = torch.where(x < -tau, -torch.ones_like(out), out)
        return out * alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, _tau = ctx.saved_tensors
        passthrough = (x.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * passthrough, None, None


class TernaryProjection(nn.Module):
    def __init__(self, features: int, tau: float = 0.15, alpha: float = 1.0) -> None:
        super().__init__()
        self.log_tau = nn.Parameter(torch.full((features,), torch.log(torch.tensor(tau))))
        self.log_alpha = nn.Parameter(torch.full((features,), torch.log(torch.tensor(alpha))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().view(*([1] * (x.ndim - 1)), -1)
        alpha = self.log_alpha.exp().view(*([1] * (x.ndim - 1)), -1)
        return TernaryProject.apply(x, tau, alpha)

    def bit_cost(self) -> torch.Tensor:
        return self.log_alpha.exp().mean() + self.log_tau.exp().reciprocal().mean()
