from __future__ import annotations

import torch
from torch import nn

from .config import TsmConfig


class Mind(nn.Module):
    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.initial_state = nn.Parameter(torch.randn(cfg.workspace_latents, cfg.d_model) * 0.02)
        self.cell = nn.GRUCell(cfg.d_model * 3, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def initial(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return self.initial_state.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def infer(
        self,
        observation: torch.Tensor,
        eps: torch.Tensor,
        ctx_embedding: torch.Tensor,
        steps: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.initial(observation.shape[0], observation.device) if state is None else state
        ctx = ctx_embedding.unsqueeze(1).expand(-1, observation.shape[1], -1)
        inputs = torch.cat([observation, eps, ctx], dim=-1)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        for _ in range(max(1, steps)):
            previous = q
            q = self.cell(flat_inputs, q.reshape(-1, q.shape[-1])).view_as(q)
            q = self.norm(q + previous)
        return q, q - (state if state is not None else self.initial(observation.shape[0], observation.device))
