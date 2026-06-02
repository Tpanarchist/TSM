from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import TsmConfig


@dataclass
class ObjectSlotOutput:
    state: torch.Tensor
    position: torch.Tensor
    occupancy: torch.Tensor
    valid: torch.Tensor
    local_images: torch.Tensor


class ObjectSlotReadout(nn.Module):
    """Unlabeled object-local percept slots from image salience."""

    def __init__(self, cfg: TsmConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, image: torch.Tensor) -> ObjectSlotOutput:
        slot_count = int(self.cfg.object_slot_count)
        bsz, channels, height, width = image.shape
        if slot_count <= 0:
            empty_state = image.new_zeros((bsz, 0, 8))
            empty_position = image.new_zeros((bsz, 0, 2))
            empty_slot = image.new_zeros((bsz, 0))
            empty_local = image.new_zeros((bsz, 0, channels, height, width))
            return ObjectSlotOutput(
                state=empty_state,
                position=empty_position,
                occupancy=empty_slot,
                valid=empty_slot.to(torch.bool),
                local_images=empty_local,
            )

        gray = image.mean(dim=1)
        salience = (gray - gray.amin(dim=(1, 2), keepdim=True)).clamp_min(0.0)
        salience_sum = salience.sum(dim=(1, 2)).clamp_min(1e-6)
        working = salience.clone()
        yy, xx = torch.meshgrid(
            torch.arange(height, device=image.device, dtype=image.dtype),
            torch.arange(width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        background = image.amin(dim=(-3, -2, -1), keepdim=True).view(bsz, 1, 1, 1)
        states: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        occupancies: list[torch.Tensor] = []
        valids: list[torch.Tensor] = []
        local_images: list[torch.Tensor] = []
        sigma = float(max(1e-3, self.cfg.object_slot_sigma))
        nms_radius = float(max(0.0, self.cfg.object_slot_nms_radius))
        for _slot in range(slot_count):
            flat = working.flatten(1)
            peak_value, peak_index = flat.max(dim=1)
            peak_y = torch.div(peak_index, width, rounding_mode="floor").to(image.dtype)
            peak_x = (peak_index % width).to(image.dtype)
            dx = xx.unsqueeze(0) - peak_x.view(-1, 1, 1)
            dy = yy.unsqueeze(0) - peak_y.view(-1, 1, 1)
            distance_sq = dx.square() + dy.square()
            soft_mask = torch.exp(-distance_sq / (2.0 * sigma * sigma))
            weighted = salience * soft_mask
            mass = weighted.sum(dim=(1, 2)).clamp_min(1e-6)
            pos_x = (weighted * xx).sum(dim=(1, 2)) / mass
            pos_y = (weighted * yy).sum(dim=(1, 2)) / mass
            position = torch.stack([pos_x, pos_y], dim=-1)
            local = background + (image - background) * soft_mask.unsqueeze(1)
            local_gray = local.mean(dim=1)
            local_mean = local_gray.flatten(1).mean(dim=1)
            local_max = local_gray.flatten(1).amax(dim=1)
            local_std = local_gray.flatten(1).std(dim=1, unbiased=False)
            occupancy = (mass / salience_sum).clamp(0.0, 1.0)
            valid = peak_value > float(self.cfg.object_slot_salience_threshold)
            state = torch.stack(
                [
                    pos_x / float(max(1, width)),
                    pos_y / float(max(1, height)),
                    occupancy,
                    (peak_value / salience_sum).clamp(0.0, 1.0),
                    mass / float(max(1, height * width)),
                    local_mean,
                    local_max,
                    local_std,
                ],
                dim=-1,
            )
            states.append(state)
            positions.append(position)
            occupancies.append(occupancy)
            valids.append(valid)
            local_images.append(local)
            if nms_radius > 0.0:
                suppress = distance_sq <= nms_radius * nms_radius
                working = working.masked_fill(suppress, -1.0)

        return ObjectSlotOutput(
            state=torch.stack(states, dim=1),
            position=torch.stack(positions, dim=1),
            occupancy=torch.stack(occupancies, dim=1),
            valid=torch.stack(valids, dim=1),
            local_images=torch.stack(local_images, dim=1),
        )
