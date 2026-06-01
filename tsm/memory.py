from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class MemoryRecord:
    severity: float
    coherence: float


@dataclass
class ObjectFile:
    feature: torch.Tensor
    last_position: tuple[float, float] | None
    last_step: int
    velocity: tuple[float, float] | None = None
    last_phase: int | None = None
    updates: int = 1


@dataclass
class ObjectMemoryRead:
    feature: torch.Tensor
    confidence: torch.Tensor
    hit: torch.Tensor
    age: torch.Tensor
    write: torch.Tensor
    position: torch.Tensor
    position_valid: torch.Tensor
    velocity: torch.Tensor
    velocity_valid: torch.Tensor
    phase: torch.Tensor
    phase_valid: torch.Tensor


class Memory:
    def __init__(self, max_records: int = 1024, object_decay_steps: float = 32.0) -> None:
        self.records: deque[MemoryRecord] = deque(maxlen=max_records)
        self.object_files: dict[int, ObjectFile] = {}
        self.object_decay_steps = object_decay_steps
        self.object_read_count = 0
        self.object_write_count = 0

    def write(self, eps: torch.Tensor, coherence: torch.Tensor) -> None:
        self.records.append(
            MemoryRecord(
                severity=float(eps.detach().norm(dim=-1).mean().cpu()),
                coherence=float(coherence.detach().mean().cpu()),
            )
        )

    def read_write_object_files(
        self,
        batch: dict[str, torch.Tensor],
        feature: torch.Tensor,
        step: int,
    ) -> ObjectMemoryRead:
        bsz, width = feature.shape
        device = feature.device
        dtype = feature.dtype
        memory_feature = torch.zeros(bsz, width, dtype=dtype, device=device)
        confidence = torch.zeros(bsz, 1, dtype=dtype, device=device)
        hit = torch.zeros(bsz, dtype=torch.bool, device=device)
        age = torch.zeros(bsz, 1, dtype=dtype, device=device)
        write = torch.zeros(bsz, dtype=torch.bool, device=device)
        position = torch.zeros(bsz, 2, dtype=dtype, device=device)
        position_valid = torch.zeros(bsz, dtype=torch.bool, device=device)
        velocity = torch.zeros(bsz, 2, dtype=dtype, device=device)
        velocity_valid = torch.zeros(bsz, dtype=torch.bool, device=device)
        phase = torch.zeros(bsz, 1, dtype=dtype, device=device)
        phase_valid = torch.zeros(bsz, dtype=torch.bool, device=device)

        if "sequence_id" not in batch or "visible_t" not in batch:
            return ObjectMemoryRead(
                memory_feature,
                confidence,
                hit,
                age,
                write,
                position,
                position_valid,
                velocity,
                velocity_valid,
                phase,
                phase_valid,
            )

        sequence_ids = batch["sequence_id"].detach().cpu().to(torch.long)
        visible = batch["visible_t"].detach().cpu().to(torch.float32)
        positions = batch.get("object_position_t")
        positions_cpu = positions.detach().cpu().to(torch.float32) if torch.is_tensor(positions) else None
        phases = batch.get("phase")
        phases_cpu = phases.detach().cpu().to(torch.long) if torch.is_tensor(phases) else None
        detached_feature = feature.detach().cpu()
        for row in range(bsz):
            key = int(sequence_ids[row].item())
            stored = self.object_files.get(key)
            if stored is not None:
                object_age = max(0, step - stored.last_step)
                memory_feature[row] = stored.feature.to(device=device, dtype=dtype)
                hit[row] = True
                age[row] = float(object_age)
                confidence[row] = 1.0 / (1.0 + float(object_age) / self.object_decay_steps)
                if stored.last_position is not None:
                    position[row, 0] = stored.last_position[0]
                    position[row, 1] = stored.last_position[1]
                    position_valid[row] = True
                if stored.velocity is not None:
                    velocity[row, 0] = stored.velocity[0]
                    velocity[row, 1] = stored.velocity[1]
                    velocity_valid[row] = True
                if stored.last_phase is not None:
                    phase[row] = float(stored.last_phase)
                    phase_valid[row] = True
                self.object_read_count += 1

            if float(visible[row].item()) > 0.5:
                stored_position = None
                if positions_cpu is not None:
                    stored_position = (float(positions_cpu[row, 0].item()), float(positions_cpu[row, 1].item()))
                stored_velocity = stored.velocity if stored is not None else None
                if stored is not None and stored.last_position is not None and stored_position is not None:
                    stored_velocity = (
                        stored_position[0] - stored.last_position[0],
                        stored_position[1] - stored.last_position[1],
                    )
                stored_phase = int(phases_cpu[row].item()) if phases_cpu is not None else None
                updates = (stored.updates + 1) if stored is not None else 1
                self.object_files[key] = ObjectFile(
                    feature=detached_feature[row].clone(),
                    last_position=stored_position,
                    last_step=step,
                    velocity=stored_velocity,
                    last_phase=stored_phase,
                    updates=updates,
                )
                write[row] = True
                self.object_write_count += 1

        return ObjectMemoryRead(
            memory_feature,
            confidence,
            hit,
            age,
            write,
            position,
            position_valid,
            velocity,
            velocity_valid,
            phase,
            phase_valid,
        )

    def short_term_decay(self) -> None:
        return None
