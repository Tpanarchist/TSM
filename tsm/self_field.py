from __future__ import annotations

from itertools import permutations

import torch
from torch import nn
import torch.nn.functional as F

from .config import TsmConfig
from .context import ContextRouter, context_balance_loss, context_entropy
from .diagnostics import (
    candidate_error_match_diagnostics,
    candidate_instance_match_diagnostics,
    feature_label_diagnostics,
    feature_match_diagnostics,
    grouped_instance_match_diagnostics,
    paired_feature_match_diagnostics,
    position_recoverability_diagnostics,
    ternary_label_diagnostics,
)
from .definitions import DefinitionBank
from .develop import DevelopmentalScheduler
from .drives import DriveDynamics
from .evidence import EvidenceAccumulator
from .free_energy import variational_free_energy
from .gate import MutationGate
from .memory import Memory
from .mind import Mind
from .perception import PerceptionSurface
from .reality import Reality
from .sae import SAE
from .slots import ObjectSlotReadout
from .trauma import TraumaMonitor
from .types import DriveState, TickOutput, TrainOutput


ORACLE_POSITION_NOISE_SWEEP_PX = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0)
ORACLE_POSITION_NOISE_SWEEP_TRIALS = 8


def _mode_context_stats(
    mode: torch.Tensor,
    context_hard: torch.Tensor,
    context_count: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    valid = mode >= 0
    if not bool(valid.any().item()):
        zero = torch.zeros((), dtype=dtype, device=context_hard.device)
        return {
            "mode_context_consistency": zero,
            "context_mode_purity": zero,
            "mode_context_separation": zero,
            "mode_context_used_count": zero,
        }
    mode = mode[valid]
    context_hard = context_hard[valid]
    total = mode.numel()
    mode_majority = torch.zeros((), dtype=dtype, device=context_hard.device)
    for mode_id in mode.unique():
        routed = context_hard[mode == mode_id]
        counts = torch.bincount(routed, minlength=context_count).to(dtype)
        mode_majority = mode_majority + counts.max()

    context_majority = torch.zeros((), dtype=dtype, device=context_hard.device)
    for ctx_id in context_hard.unique():
        routed_modes = mode[context_hard == ctx_id]
        counts = torch.bincount(routed_modes).to(dtype)
        context_majority = context_majority + counts.max()

    mode_count = mode.unique().numel()
    used_count = context_hard.unique().numel()
    possible = max(1, min(int(mode_count), context_count))
    return {
        "mode_context_consistency": mode_majority / total,
        "context_mode_purity": context_majority / total,
        "mode_context_separation": torch.tensor(used_count / possible, dtype=dtype, device=context_hard.device),
        "mode_context_used_count": torch.tensor(used_count, dtype=dtype, device=context_hard.device),
    }


def _prefix_metrics(prefix: str, metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = mask.to(device=values.device).bool()
    if not bool(mask.any().item()):
        return torch.zeros((), dtype=dtype, device=values.device)
    return values[mask].mean()


def _masked_context_used(
    context_hard: torch.Tensor,
    mask: torch.Tensor,
    context_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = mask.to(device=context_hard.device).bool()
    if not bool(mask.any().item()):
        return torch.zeros((), dtype=dtype, device=context_hard.device)
    return torch.bincount(context_hard[mask], minlength=context_count).gt(0).to(dtype).sum()


def _zero_like_scalar(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=reference.dtype, device=reference.device)


def _object_instance_ids(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor | None:
    ids = batch.get("object_file_id", batch.get("sequence_id"))
    if ids is None:
        return None
    return ids.to(device=device, dtype=torch.long)


def _normalized_pixel_position(position: torch.Tensor, cfg: TsmConfig) -> torch.Tensor:
    return position / float(max(1, cfg.image_size))


def _binding_position_features(
    features: torch.Tensor,
    normalized_position: torch.Tensor | None,
    cfg: TsmConfig,
) -> torch.Tensor:
    scale = float(cfg.definition_position_feature_scale)
    if scale <= 0.0 or normalized_position is None or features.numel() == 0:
        return features
    count = min(features.shape[0], normalized_position.shape[0])
    if count == 0:
        return features
    position = normalized_position[:count].to(device=features.device, dtype=features.dtype)
    if features.shape[0] != count:
        features = features[:count]
    return torch.cat([features, scale * position], dim=-1)


def _paired_contrastive_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    detach_target: bool = True,
) -> torch.Tensor:
    if source.shape[0] < 2 or target.shape[0] < 2:
        return _zero_like_scalar(source)
    pair_count = min(source.shape[0], target.shape[0])
    source = F.normalize(source[:pair_count], dim=-1, eps=1e-6)
    target = target[:pair_count].to(device=source.device, dtype=source.dtype)
    if detach_target:
        target = target.detach()
    target = F.normalize(target, dim=-1, eps=1e-6)
    logits = torch.matmul(source, target.t()) / max(temperature, 1e-6)
    labels = torch.arange(pair_count, device=source.device)
    return F.cross_entropy(logits, labels)


def _bidirectional_paired_contrastive_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if source.shape[0] < 2 or target.shape[0] < 2:
        return _zero_like_scalar(source)
    return 0.5 * (
        _paired_contrastive_loss(source, target, temperature, detach_target=False)
        + _paired_contrastive_loss(target, source, temperature, detach_target=False)
    )


def _grouped_contrastive_query_loss(
    query: torch.Tensor,
    files: torch.Tensor,
    instance_labels: torch.Tensor,
    group_labels: torch.Tensor,
    temperature: float,
    detach_files: bool = False,
) -> torch.Tensor:
    if query.shape[0] < 2 or files.shape[0] < 2:
        return _zero_like_scalar(query)
    pair_count = min(query.shape[0], files.shape[0], instance_labels.shape[0], group_labels.shape[0])
    query = F.normalize(query[:pair_count], dim=-1, eps=1e-6)
    files = files[:pair_count].to(device=query.device, dtype=query.dtype)
    if detach_files:
        files = files.detach()
    files = F.normalize(files, dim=-1, eps=1e-6)
    instance_labels = instance_labels[:pair_count].to(device=query.device, dtype=torch.long)
    group_labels = group_labels[:pair_count].to(device=query.device, dtype=torch.long)
    valid = (instance_labels >= 0) & (group_labels >= 0)
    query = query[valid]
    files = files[valid]
    instance_labels = instance_labels[valid]
    group_labels = group_labels[valid]
    if query.shape[0] < 2:
        return _zero_like_scalar(query)
    losses: list[torch.Tensor] = []
    for row in range(query.shape[0]):
        same_group = group_labels == group_labels[row]
        same_instance = instance_labels == instance_labels[row]
        candidates = same_group
        targets = same_instance & candidates
        if int(candidates.to(torch.long).sum().item()) < 2 or not bool(targets.any().item()):
            continue
        logits = torch.matmul(query[row : row + 1], files[candidates].t()) / max(temperature, 1e-6)
        target_positions = torch.nonzero(targets[candidates], as_tuple=False).flatten()
        losses.append(F.cross_entropy(logits, target_positions[:1]))
    return torch.stack(losses).mean() if losses else _zero_like_scalar(query)


def _candidate_masked_query_loss(
    query: torch.Tensor,
    files: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if query.shape[0] < 2 or files.shape[0] < 2 or candidate_mask.numel() == 0:
        return _zero_like_scalar(query)
    pair_count = min(query.shape[0], files.shape[0], candidate_mask.shape[0], candidate_mask.shape[1])
    query = F.normalize(query[:pair_count], dim=-1, eps=1e-6)
    files = F.normalize(files[:pair_count].to(device=query.device, dtype=query.dtype), dim=-1, eps=1e-6)
    candidate_mask = candidate_mask[:pair_count, :pair_count].to(device=query.device, dtype=torch.bool)
    candidate_mask = candidate_mask | torch.eye(pair_count, dtype=torch.bool, device=query.device)
    losses: list[torch.Tensor] = []
    candidate_indices = torch.arange(pair_count, device=query.device)
    for row in range(pair_count):
        candidates = candidate_mask[row]
        if int(candidates.to(torch.long).sum().item()) < 2:
            continue
        logits = torch.matmul(query[row : row + 1], files[candidates].t()) / max(temperature, 1e-6)
        target_position = torch.nonzero(candidate_indices[candidates] == row, as_tuple=False).flatten()
        if bool(target_position.numel()):
            losses.append(F.cross_entropy(logits, target_position[:1]))
    return torch.stack(losses).mean() if losses else _zero_like_scalar(query)


def _active_file_gate_input_dim(cfg: TsmConfig) -> int:
    dim = cfg.definitions_per_context * 4 + 2
    if cfg.learned_active_file_gate_context_features:
        dim += cfg.contexts * 3
    if cfg.learned_active_file_gate_expectation_features:
        dim += cfg.definitions_per_context * 3
    return dim


def _active_file_expectation_input_dim(cfg: TsmConfig) -> int:
    dim = cfg.definitions_per_context + cfg.contexts + 2
    if cfg.active_file_expectation_trajectory_features:
        dim += 13 + 2 * cfg.active_file_expectation_phase_count
    return dim


def _active_file_trajectory_width(cfg: TsmConfig) -> int:
    return 13 + 2 * cfg.active_file_expectation_phase_count


def _active_file_dynamics_input_dim(cfg: TsmConfig) -> int:
    return _active_file_trajectory_width(cfg) + cfg.contexts + 2


def _active_file_trajectory_features(
    batch: dict[str, torch.Tensor],
    memory_read,
    mask: torch.Tensor,
    cfg: TsmConfig,
    dtype: torch.dtype,
    device: torch.device,
    projected_position: torch.Tensor | None = None,
) -> torch.Tensor:
    count = int(mask.to(torch.long).sum().item())
    width = _active_file_trajectory_width(cfg)
    if count == 0:
        return torch.zeros((0, width), dtype=dtype, device=device)
    position = memory_read.position[mask].to(device=device, dtype=dtype)
    velocity = memory_read.velocity[mask].to(device=device, dtype=dtype)
    position_valid = memory_read.position_valid[mask].to(device=device, dtype=dtype).unsqueeze(-1)
    velocity_valid = memory_read.velocity_valid[mask].to(device=device, dtype=dtype).unsqueeze(-1)
    scale = float(max(1, cfg.image_size))
    if projected_position is None:
        projected = position + velocity
        wrap_span = _active_file_wrap_span(cfg)
        if wrap_span is not None and wrap_span > 0:
            margin = float(max(4, cfg.image_size // 7))
            projected = ((projected - margin) % float(wrap_span)) + margin
    else:
        projected = projected_position[:count].to(device=device, dtype=dtype)
    velocity_mag = velocity.norm(dim=-1, keepdim=True) / scale
    base = [
        position / scale,
        velocity / scale,
        projected / scale,
        velocity_mag,
        position_valid,
        velocity_valid,
    ]
    phase_count = cfg.active_file_expectation_phase_count
    if "phase" in batch and phase_count > 0:
        phase = batch["phase"][mask].to(device=device, dtype=torch.long).clamp_min(0) % phase_count
        phase_onehot = F.one_hot(phase, num_classes=phase_count).to(dtype)
        next_phase = (phase + 1) % phase_count
        next_phase_onehot = F.one_hot(next_phase, num_classes=phase_count).to(dtype)
    else:
        phase_onehot = torch.zeros((count, phase_count), dtype=dtype, device=device)
        next_phase_onehot = torch.zeros((count, phase_count), dtype=dtype, device=device)
    flags = []
    for key in ("visible_t", "occluded_t", "visible_tp1", "occluded_tp1"):
        value = batch.get(key)
        if torch.is_tensor(value):
            flags.append(value[mask].to(device=device, dtype=dtype).unsqueeze(-1))
        else:
            flags.append(torch.zeros((count, 1), dtype=dtype, device=device))
    return torch.cat([*base, phase_onehot, next_phase_onehot, *flags], dim=-1)


def _active_file_projected_position(
    memory_read,
    mask: torch.Tensor,
    cfg: TsmConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(mask.to(torch.long).sum().item())
    if count == 0:
        return torch.zeros((0, 2), dtype=dtype, device=device), torch.zeros((0,), dtype=torch.bool, device=device)
    position = memory_read.position[mask].to(device=device, dtype=dtype)
    velocity = memory_read.velocity[mask].to(device=device, dtype=dtype)
    projected = position + velocity
    wrap_span = _active_file_wrap_span(cfg)
    if wrap_span is not None and wrap_span > 0:
        margin = float(max(4, cfg.image_size // 7))
        projected = ((projected - margin) % float(wrap_span)) + margin
    valid = memory_read.position_valid[mask].to(device=device) & memory_read.velocity_valid[mask].to(device=device)
    return projected, valid


def _active_file_ballistic_position(
    memory_read,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    cfg: TsmConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(mask.to(torch.long).sum().item())
    if count == 0:
        return torch.zeros((0, 2), dtype=dtype, device=device), torch.zeros((0,), dtype=torch.bool, device=device)

    position = memory_read.position[mask].to(device=device, dtype=dtype)
    velocity = memory_read.velocity[mask].to(device=device, dtype=dtype)
    valid = (
        memory_read.position_valid[mask].to(device=device)
        & memory_read.velocity_valid[mask].to(device=device)
        & memory_read.hit[mask].to(device=device)
    )

    if "phase" in batch and bool(memory_read.phase_valid[mask].any().item()):
        phase_count = max(1, int(cfg.active_file_expectation_phase_count))
        current_phase = batch["phase"].to(device=device, dtype=dtype)[mask].view(-1, 1)
        next_phase = torch.remainder(current_phase + 1.0, float(phase_count))
        last_phase = memory_read.phase[mask].to(device=device, dtype=dtype)
        elapsed = torch.remainder(next_phase - last_phase, float(phase_count))
        elapsed = torch.where(elapsed <= 0.0, torch.ones_like(elapsed), elapsed)
        valid = valid & memory_read.phase_valid[mask].to(device=device)
    else:
        elapsed = memory_read.age[mask].to(device=device, dtype=dtype).clamp_min(0.0) + 1.0

    projected = position + velocity * elapsed
    wrap_span = _active_file_wrap_span(cfg)
    if wrap_span is not None and wrap_span > 0:
        margin = float(max(4, cfg.image_size // 7))
        projected = ((projected - margin) % float(wrap_span)) + margin
    else:
        projected = projected.clamp(0.0, float(cfg.image_size - 1))
    return projected, valid


def _active_file_dynamics_features(
    batch: dict[str, torch.Tensor],
    memory_read,
    mask: torch.Tensor,
    cfg: TsmConfig,
    dtype: torch.dtype,
    device: torch.device,
    file_context: torch.Tensor,
    file_confidence: torch.Tensor,
    file_age: torch.Tensor,
) -> torch.Tensor:
    count = int(mask.to(torch.long).sum().item())
    if count == 0:
        return torch.zeros((0, _active_file_dynamics_input_dim(cfg)), dtype=dtype, device=device)
    trajectory = _active_file_trajectory_features(batch, memory_read, mask, cfg, dtype, device)
    count = min(count, trajectory.shape[0], file_context.shape[0], file_confidence.shape[0], file_age.shape[0])
    context_features = file_context[:count].to(device=device, dtype=dtype)
    confidence = file_confidence[:count].to(device=device, dtype=dtype)
    age = file_age[:count].to(device=device, dtype=dtype)
    age = torch.log1p(age.clamp_min(0.0)) / max(cfg.active_file_candidate_max_age, 1.0)
    return torch.cat([trajectory[:count], context_features, confidence, age], dim=-1)


def _active_file_dynamics_position(
    dynamics: nn.Module,
    dynamics_features: torch.Tensor,
    projected_position: torch.Tensor,
    cfg: TsmConfig,
) -> torch.Tensor:
    if dynamics_features.numel() == 0 or projected_position.numel() == 0:
        return torch.zeros((0, 2), dtype=dynamics_features.dtype, device=dynamics_features.device)
    count = min(dynamics_features.shape[0], projected_position.shape[0])
    scale = float(max(1, cfg.image_size))
    base_position = projected_position[:count].to(device=dynamics_features.device, dtype=dynamics_features.dtype) / scale
    delta = torch.tanh(dynamics(dynamics_features[:count])) * float(cfg.active_file_dynamics_delta_scale)
    return (base_position + delta).clamp(0.0, 1.0) * scale


def _active_file_expectation(
    expectation: nn.Module,
    files: torch.Tensor,
    file_context: torch.Tensor,
    file_confidence: torch.Tensor,
    file_age: torch.Tensor,
    age_scale: float,
    trajectory_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if files.numel() == 0:
        return torch.zeros((0, files.shape[-1]), dtype=files.dtype, device=files.device)
    count = min(files.shape[0], file_context.shape[0], file_confidence.shape[0], file_age.shape[0])
    files = files[:count]
    file_context = file_context[:count].to(device=files.device, dtype=files.dtype)
    confidence = file_confidence[:count].to(device=files.device, dtype=files.dtype)
    age = file_age[:count].to(device=files.device, dtype=files.dtype)
    age = torch.log1p(age.clamp_min(0.0)) / max(age_scale, 1.0)
    parts = [files, file_context, confidence, age]
    if trajectory_features is not None:
        parts.append(trajectory_features[:count].to(device=files.device, dtype=files.dtype))
    features = torch.cat(parts, dim=-1)
    expected_width = expectation[0].in_features if isinstance(expectation, nn.Sequential) else features.shape[-1]
    if features.shape[-1] < expected_width:
        padding = torch.zeros((features.shape[0], expected_width - features.shape[-1]), dtype=files.dtype, device=files.device)
        features = torch.cat([features, padding], dim=-1)
    return files + expectation(features)


def _active_file_gate_logits(
    gate: nn.Module,
    query: torch.Tensor,
    files: torch.Tensor,
    file_confidence: torch.Tensor,
    file_age: torch.Tensor,
    age_scale: float,
    query_context: torch.Tensor | None = None,
    file_context: torch.Tensor | None = None,
    expected_query: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.numel() == 0 or files.numel() == 0:
        return torch.zeros((0, 0), dtype=query.dtype, device=query.device)
    count = min(query.shape[0], files.shape[0], file_confidence.shape[0], file_age.shape[0])
    if query_context is not None and file_context is not None:
        count = min(count, query_context.shape[0], file_context.shape[0])
    if expected_query is not None:
        count = min(count, expected_query.shape[0])
    query = query[:count]
    files = files[:count].to(device=query.device, dtype=query.dtype)
    confidence = file_confidence[:count].view(1, count, 1).to(device=query.device, dtype=query.dtype)
    age = file_age[:count].view(1, count, 1).to(device=query.device, dtype=query.dtype)
    age = torch.log1p(age.clamp_min(0.0)) / max(age_scale, 1.0)
    query_pairs = query.unsqueeze(1).expand(count, count, -1)
    file_pairs = files.unsqueeze(0).expand(count, count, -1)
    confidence_pairs = confidence.expand(count, count, -1)
    age_pairs = age.expand(count, count, -1)
    feature_parts = [
        query_pairs,
        file_pairs,
        (query_pairs - file_pairs).abs(),
        query_pairs * file_pairs,
        confidence_pairs,
        age_pairs,
    ]
    if query_context is not None and file_context is not None:
        query_context = query_context[:count].to(device=query.device, dtype=query.dtype)
        file_context = file_context[:count].to(device=query.device, dtype=query.dtype)
        query_context_pairs = query_context.unsqueeze(1).expand(count, count, -1)
        file_context_pairs = file_context.unsqueeze(0).expand(count, count, -1)
        feature_parts.extend([
            query_context_pairs,
            file_context_pairs,
            (query_context_pairs - file_context_pairs).abs(),
        ])
    if expected_query is not None:
        expected_query = expected_query[:count].to(device=query.device, dtype=query.dtype)
        expected_pairs = expected_query.unsqueeze(0).expand(count, count, -1)
        feature_parts.extend([
            expected_pairs,
            (query_pairs - expected_pairs).abs(),
            query_pairs * expected_pairs,
        ])
    features = torch.cat(feature_parts, dim=-1)
    return gate(features).squeeze(-1)


def _learned_candidate_mask(
    logits: torch.Tensor,
    file_valid: torch.Tensor,
    topk: int,
    threshold: float,
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    count = min(logits.shape[0], logits.shape[1], file_valid.shape[0])
    logits = logits[:count, :count]
    file_valid = file_valid[:count].to(device=logits.device, dtype=torch.bool)
    probs = torch.sigmoid(logits).masked_fill(~file_valid.unsqueeze(0), -1.0)
    if topk > 0:
        k = min(topk, count)
        selected = torch.zeros((count, count), dtype=torch.bool, device=logits.device)
        indices = probs.topk(k, dim=1).indices
        selected.scatter_(1, indices, True)
        return selected & file_valid.unsqueeze(0)
    return (probs >= threshold) & file_valid.unsqueeze(0)


def _candidate_mask_agreement(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    dtype = predicted.dtype if predicted.is_floating_point() else torch.float32
    device = predicted.device
    if predicted.numel() == 0 or target.numel() == 0:
        zero = torch.zeros((), dtype=dtype, device=device)
        return {
            "precision": zero,
            "recall": zero,
            "f1": zero,
            "predicted_fraction": zero,
            "target_fraction": zero,
        }
    count = min(predicted.shape[0], predicted.shape[1], target.shape[0], target.shape[1])
    predicted = predicted[:count, :count].to(device=device, dtype=torch.bool)
    target = target[:count, :count].to(device=device, dtype=torch.bool)
    true_positive = (predicted & target).to(torch.float32).sum()
    predicted_positive = predicted.to(torch.float32).sum()
    target_positive = target.to(torch.float32).sum()
    precision = true_positive / predicted_positive.clamp_min(1.0)
    recall = true_positive / target_positive.clamp_min(1.0)
    f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1e-6)
    total = torch.tensor(float(count * count), dtype=torch.float32, device=device)
    return {
        "precision": precision.to(dtype),
        "recall": recall.to(dtype),
        "f1": f1.to(dtype),
        "predicted_fraction": (predicted_positive / total).to(dtype),
        "target_fraction": (target_positive / total).to(dtype),
    }


def _active_file_wrap_span(cfg: TsmConfig) -> float | None:
    if not cfg.active_file_candidate_wrap:
        return None
    margin = max(4, cfg.image_size // 7)
    return float(max(1, cfg.image_size - 2 * margin))


def _active_file_candidate_mask(
    file_positions: torch.Tensor,
    query_positions: torch.Tensor,
    file_position_valid: torch.Tensor,
    file_hit: torch.Tensor,
    file_age: torch.Tensor,
    radius: float,
    max_age: float,
    wrap_span: float | None = None,
) -> torch.Tensor:
    if file_positions.numel() == 0 or query_positions.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=file_positions.device)
    count = min(file_positions.shape[0], query_positions.shape[0])
    file_positions = file_positions[:count]
    query_positions = query_positions[:count].to(device=file_positions.device, dtype=file_positions.dtype)
    file_position_valid = file_position_valid[:count].to(device=file_positions.device, dtype=torch.bool)
    file_hit = file_hit[:count].to(device=file_positions.device, dtype=torch.bool)
    file_age = file_age[:count].view(-1).to(device=file_positions.device, dtype=file_positions.dtype)
    if wrap_span is not None and wrap_span > 0:
        diff = (query_positions.unsqueeze(1) - file_positions.unsqueeze(0)).abs()
        span = torch.tensor(wrap_span, dtype=file_positions.dtype, device=file_positions.device)
        diff = torch.minimum(diff, (span - diff).abs())
        distances = diff.square().sum(dim=-1).sqrt()
    else:
        distances = torch.cdist(query_positions, file_positions)
    candidates = distances <= radius
    candidates = candidates & file_position_valid.unsqueeze(0) & file_hit.unsqueeze(0)
    if max_age >= 0:
        candidates = candidates & (file_age <= max_age).unsqueeze(0)
    return candidates


def _active_file_feature_only_candidate_mask(
    file_valid: torch.Tensor,
    query_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    count = min(int(query_count), file_valid.shape[0])
    if count <= 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=device)
    valid = file_valid[:count].to(device=device, dtype=torch.bool)
    return valid.unsqueeze(0).expand(count, count)


def _local_reappearance_images(
    images: torch.Tensor,
    file_positions: torch.Tensor,
    cfg: TsmConfig,
) -> torch.Tensor:
    if images.numel() == 0 or file_positions.numel() == 0:
        return images.new_zeros((0, images.shape[1], images.shape[2], images.shape[3]))
    query_count = images.shape[0]
    file_count = file_positions.shape[0]
    height, width = images.shape[-2:]
    y = torch.arange(height, device=images.device, dtype=images.dtype)
    x = torch.arange(width, device=images.device, dtype=images.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    positions = file_positions.to(device=images.device, dtype=images.dtype)
    dx = xx.unsqueeze(0) - positions[:, 0].view(-1, 1, 1)
    dy = yy.unsqueeze(0) - positions[:, 1].view(-1, 1, 1)
    wrap_span = _active_file_wrap_span(cfg)
    if wrap_span is not None and wrap_span > 0:
        span = torch.tensor(wrap_span, device=images.device, dtype=images.dtype)
        dx = torch.minimum(dx.abs(), (span - dx.abs()).abs())
        dy = torch.minimum(dy.abs(), (span - dy.abs()).abs())
    sigma = float(max(1.5, cfg.image_size / 10.0))
    window = torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma * sigma)).clamp_min(1e-4)
    window = window.view(1, file_count, 1, height, width)
    expanded = images.unsqueeze(1).expand(query_count, file_count, -1, -1, -1)
    background = images.amin(dim=(-3, -2, -1), keepdim=True).view(query_count, 1, 1, 1, 1)
    local = background + (expanded - background) * window
    return local.reshape(query_count * file_count, images.shape[1], height, width)


def _state_prediction_error_matrix(actual_state: torch.Tensor, expected_state: torch.Tensor) -> torch.Tensor:
    if actual_state.numel() == 0 or expected_state.numel() == 0:
        return torch.zeros((0, 0), dtype=actual_state.dtype, device=actual_state.device)
    count = min(actual_state.shape[0], expected_state.shape[0])
    actual = actual_state[:count]
    expected = expected_state[:count].to(device=actual.device, dtype=actual.dtype)
    return (actual.unsqueeze(1) - expected.unsqueeze(0)).square().mean(dim=-1)


def _object_slot_position_metrics(
    slot_state: torch.Tensor,
    slot_position: torch.Tensor,
    slot_occupancy: torch.Tensor,
    slot_valid: torch.Tensor,
    target_position: torch.Tensor,
    distractor_position: torch.Tensor | None,
    cfg: TsmConfig,
) -> dict[str, torch.Tensor]:
    dtype = slot_state.dtype
    device = slot_state.device
    zero = torch.zeros((), dtype=dtype, device=device)
    if slot_state.numel() == 0 or slot_position.numel() == 0 or target_position.numel() == 0:
        metrics = {
            "count": zero,
            "valid_fraction": zero,
            "used_count": zero,
            "occupancy_entropy": zero,
            "separation": zero,
            "collapse_fraction": zero,
            "target_position_error": zero,
            "target_recall": zero,
            "distractor_position_error": zero,
            "distractor_recall": zero,
            "pair_position_error": zero,
            "assignment_object_file_id_usage": zero,
            "assignment_object_id_usage": zero,
        }
        metrics.update(position_recoverability_diagnostics(slot_state.new_zeros((0, 1)), slot_position.new_zeros((0, 2))))
        return metrics

    count = min(slot_state.shape[0], slot_position.shape[0], slot_occupancy.shape[0], slot_valid.shape[0], target_position.shape[0])
    slot_state = slot_state[:count]
    slot_position = slot_position[:count].to(device=device, dtype=dtype)
    slot_occupancy = slot_occupancy[:count].to(device=device, dtype=dtype)
    slot_valid = slot_valid[:count].to(device=device, dtype=torch.bool)
    target_position = target_position[:count].to(device=device, dtype=dtype)
    object_positions = [target_position.unsqueeze(1)]
    has_distractor = distractor_position is not None and distractor_position.numel() > 0
    if has_distractor:
        distractor_position = distractor_position[:count].to(device=device, dtype=dtype)
        object_positions.append(distractor_position.unsqueeze(1))
    objects = torch.cat(object_positions, dim=1)
    distances = torch.cdist(slot_position, objects) / float(max(1, cfg.image_size))
    distances = distances.masked_fill(~slot_valid.unsqueeze(-1), float("inf"))
    threshold = float(cfg.object_slot_match_radius) / float(max(1, cfg.image_size))

    target_min = distances[:, :, 0].amin(dim=1)
    target_finite = torch.isfinite(target_min)
    target_error = target_min[target_finite].mean() if bool(target_finite.any().item()) else zero
    target_recall = (target_min <= threshold).to(dtype).mean()
    if has_distractor:
        distractor_min = distances[:, :, 1].amin(dim=1)
        distractor_finite = torch.isfinite(distractor_min)
        distractor_error = distractor_min[distractor_finite].mean() if bool(distractor_finite.any().item()) else zero
        distractor_recall = (distractor_min <= threshold).to(dtype).mean()
    else:
        distractor_error = zero
        distractor_recall = zero

    matched_features: list[torch.Tensor] = []
    matched_positions: list[torch.Tensor] = []
    pair_errors: list[torch.Tensor] = []
    object_count = objects.shape[1]
    slot_count = slot_position.shape[1]
    for row in range(count):
        if object_count == 1:
            best_slot = distances[row, :, 0].argmin()
            if torch.isfinite(distances[row, best_slot, 0]):
                matched_features.append(slot_state[row, best_slot])
                matched_positions.append(objects[row, 0])
                pair_errors.append(distances[row, best_slot, 0])
            continue
        best_error = torch.tensor(float("inf"), dtype=dtype, device=device)
        best_indices: tuple[int, int] | None = None
        for target_slot in range(slot_count):
            if not bool(slot_valid[row, target_slot].item()):
                continue
            for distractor_slot in range(slot_count):
                if target_slot == distractor_slot or not bool(slot_valid[row, distractor_slot].item()):
                    continue
                error = 0.5 * (distances[row, target_slot, 0] + distances[row, distractor_slot, 1])
                if bool((error < best_error).item()):
                    best_error = error
                    best_indices = (target_slot, distractor_slot)
        if best_indices is not None and torch.isfinite(best_error):
            pair_errors.append(best_error)
            matched_features.append(slot_state[row, best_indices[0]])
            matched_positions.append(objects[row, 0])
            matched_features.append(slot_state[row, best_indices[1]])
            matched_positions.append(objects[row, 1])

    if matched_features:
        recover_features = torch.stack(matched_features)
        recover_positions = torch.stack(matched_positions)
        recoverability = position_recoverability_diagnostics(
            recover_features,
            recover_positions,
            scale=float(max(1, cfg.image_size)),
        )
    else:
        recoverability = position_recoverability_diagnostics(slot_state.new_zeros((0, 1)), slot_position.new_zeros((0, 2)))

    valid_fraction = slot_valid.to(dtype).mean()
    occupancy = slot_occupancy * slot_valid.to(dtype)
    occupancy_sum = occupancy.sum(dim=1, keepdim=True).clamp_min(1e-6)
    probs = occupancy / occupancy_sum
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    if slot_count > 1:
        entropy = entropy / torch.log(torch.tensor(float(slot_count), dtype=dtype, device=device)).clamp_min(1e-6)
        pairwise = torch.cdist(slot_position, slot_position) / float(max(1, cfg.image_size))
        pair_valid = slot_valid.unsqueeze(1) & slot_valid.unsqueeze(2)
        eye = torch.eye(slot_count, dtype=torch.bool, device=device).unsqueeze(0)
        pair_valid = pair_valid & ~eye
        if bool(pair_valid.any().item()):
            separation = pairwise[pair_valid].mean()
            min_pair = pairwise.masked_fill(~pair_valid, float("inf")).amin(dim=(1, 2))
            collapse = (min_pair <= threshold).to(dtype).mean()
        else:
            separation = zero
            collapse = zero
    else:
        separation = zero
        collapse = zero
    used_count = (occupancy > float(cfg.object_slot_salience_threshold)).to(dtype).sum(dim=1).mean()

    metrics = {
        "count": torch.tensor(float(slot_count), dtype=dtype, device=device),
        "valid_fraction": valid_fraction,
        "used_count": used_count,
        "occupancy_entropy": entropy.mean() if entropy.numel() else zero,
        "separation": separation,
        "collapse_fraction": collapse,
        "target_position_error": target_error,
        "target_recall": target_recall,
        "distractor_position_error": distractor_error,
        "distractor_recall": distractor_recall,
        "pair_position_error": torch.stack(pair_errors).mean() if pair_errors else zero,
        "assignment_object_file_id_usage": zero,
        "assignment_object_id_usage": zero,
    }
    metrics.update(recoverability)
    return metrics


def _slot_ternary_metrics(slot_ternary: torch.Tensor, slot_valid: torch.Tensor) -> dict[str, torch.Tensor]:
    dtype = slot_ternary.dtype
    device = slot_ternary.device
    zero = torch.zeros((), dtype=dtype, device=device)
    if slot_ternary.numel() == 0 or slot_valid.numel() == 0:
        return {
            "ternary_zero_fraction": zero,
            "ternary_nonzero_fraction": zero,
            "ternary_positive_fraction": zero,
            "ternary_negative_fraction": zero,
            "ternary_axis_usage_count": zero,
            "ternary_axis_usage_fraction": zero,
            "ternary_always_on_axis_fraction": zero,
        }
    valid = slot_valid.reshape(-1).to(device=device, dtype=torch.bool)
    ternary = slot_ternary.reshape(valid.shape[0], -1)[valid]
    if ternary.numel() == 0:
        return {
            "ternary_zero_fraction": zero,
            "ternary_nonzero_fraction": zero,
            "ternary_positive_fraction": zero,
            "ternary_negative_fraction": zero,
            "ternary_axis_usage_count": zero,
            "ternary_axis_usage_fraction": zero,
            "ternary_always_on_axis_fraction": zero,
        }
    signs = ternary.sign()
    axis_nonzero = (signs != 0).to(dtype).mean(dim=0)
    return {
        "ternary_zero_fraction": (signs == 0).to(dtype).mean(),
        "ternary_nonzero_fraction": (signs != 0).to(dtype).mean(),
        "ternary_positive_fraction": (signs > 0).to(dtype).mean(),
        "ternary_negative_fraction": (signs < 0).to(dtype).mean(),
        "ternary_axis_usage_count": axis_nonzero.gt(0).to(dtype).sum(),
        "ternary_axis_usage_fraction": axis_nonzero.gt(0).to(dtype).mean(),
        "ternary_always_on_axis_fraction": axis_nonzero.ge(0.95).to(dtype).mean(),
    }


def _file_slot_assignment_metrics(
    file_positions: torch.Tensor,
    file_valid: torch.Tensor,
    slot_positions: torch.Tensor,
    slot_valid: torch.Tensor,
    target_positions: torch.Tensor,
    file_instance_labels: torch.Tensor,
    target_instance_labels: torch.Tensor,
    group_labels: torch.Tensor,
    cfg: TsmConfig,
    distractor_positions: torch.Tensor | None = None,
    distractor_instance_labels: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    dtype = slot_positions.dtype
    device = slot_positions.device
    zero = torch.zeros((), dtype=dtype, device=device)
    if (
        file_positions.numel() == 0
        or slot_positions.numel() == 0
        or target_positions.numel() == 0
        or file_instance_labels.numel() == 0
        or target_instance_labels.numel() == 0
    ):
        return {
            "target_match_accuracy": zero,
            "target_hard_match_accuracy": zero,
            "distractor_match_accuracy": zero,
            "pair_match_accuracy": zero,
            "candidate_mean_count": zero,
            "row_coverage_fraction": zero,
            "target_file_recall_fraction": zero,
            "distractor_file_recall_fraction": zero,
            "assignment_position_error": zero,
            "target_assignment_position_error": zero,
            "distractor_assignment_position_error": zero,
            "assignment_object_file_id_usage": zero,
            "assignment_object_id_usage": zero,
            "assignment_sequence_id_usage": zero,
        }

    file_count = min(file_positions.shape[0], file_valid.shape[0], file_instance_labels.shape[0], group_labels.shape[0])
    query_count = min(slot_positions.shape[0], slot_valid.shape[0], target_positions.shape[0], target_instance_labels.shape[0])
    slot_count = slot_positions.shape[1]
    if file_count == 0 or query_count == 0 or slot_count == 0:
        return {
            "target_match_accuracy": zero,
            "target_hard_match_accuracy": zero,
            "distractor_match_accuracy": zero,
            "pair_match_accuracy": zero,
            "candidate_mean_count": zero,
            "row_coverage_fraction": zero,
            "target_file_recall_fraction": zero,
            "distractor_file_recall_fraction": zero,
            "assignment_position_error": zero,
            "target_assignment_position_error": zero,
            "distractor_assignment_position_error": zero,
            "assignment_object_file_id_usage": zero,
            "assignment_object_id_usage": zero,
            "assignment_sequence_id_usage": zero,
        }

    scale = float(max(1, cfg.image_size))
    file_positions = file_positions[:file_count].to(device=device, dtype=dtype)
    file_valid = file_valid[:file_count].to(device=device, dtype=torch.bool)
    file_instance_labels = file_instance_labels[:file_count].to(device=device, dtype=torch.long)
    group_labels = group_labels[:file_count].to(device=device, dtype=torch.long)
    slot_positions = slot_positions[:query_count].to(device=device, dtype=dtype)
    slot_valid = slot_valid[:query_count].to(device=device, dtype=torch.bool)
    target_positions = target_positions[:query_count].to(device=device, dtype=dtype)
    target_instance_labels = target_instance_labels[:query_count].to(device=device, dtype=torch.long)
    if distractor_positions is not None and distractor_positions.numel() > 0:
        distractor_positions = distractor_positions[:query_count].to(device=device, dtype=dtype)
    if distractor_instance_labels is not None and distractor_instance_labels.numel() > 0:
        distractor_instance_labels = distractor_instance_labels[:query_count].to(device=device, dtype=torch.long)
    if candidate_mask is not None and candidate_mask.numel() > 0:
        candidate_mask = candidate_mask[:query_count, :file_count].to(device=device, dtype=torch.bool)

    target_hits: list[torch.Tensor] = []
    hard_hits: list[torch.Tensor] = []
    distractor_hits: list[torch.Tensor] = []
    pair_hits: list[torch.Tensor] = []
    target_recalls: list[torch.Tensor] = []
    distractor_recalls: list[torch.Tensor] = []
    row_coverages: list[torch.Tensor] = []
    candidate_counts: list[torch.Tensor] = []
    assignment_errors: list[torch.Tensor] = []
    target_errors: list[torch.Tensor] = []
    distractor_errors: list[torch.Tensor] = []
    for row in range(query_count):
        row_slot_valid = slot_valid[row]
        valid_slot_indices = torch.nonzero(row_slot_valid, as_tuple=False).flatten()
        row_file_valid = file_valid
        if candidate_mask is not None:
            row_file_valid = row_file_valid & candidate_mask[row]
        valid_file_indices = torch.nonzero(row_file_valid, as_tuple=False).flatten()
        has_assignment = valid_file_indices.numel() > 0 and valid_slot_indices.numel() > 0
        row_coverages.append(torch.tensor(float(has_assignment), dtype=dtype, device=device))
        if not has_assignment:
            continue
        row_slots = slot_positions[row, valid_slot_indices]
        distances = torch.cdist(file_positions[valid_file_indices], row_slots) / scale
        file_scores = distances.min(dim=1).values
        k = min(valid_file_indices.numel(), max(1, valid_slot_indices.numel()))
        selected_local = file_scores.topk(k, largest=False).indices
        selected_files = valid_file_indices[selected_local]
        selected_distances = distances[selected_local]
        candidate_counts.append(torch.tensor(float(valid_file_indices.numel()), dtype=dtype, device=device))

        target_label = target_instance_labels[row]
        target_recalls.append((file_instance_labels[selected_files] == target_label).any().to(dtype))
        if distractor_instance_labels is not None:
            distractor_label = distractor_instance_labels[row]
            distractor_recalls.append((file_instance_labels[selected_files] == distractor_label).any().to(dtype))
        else:
            distractor_label = None

        assignment: dict[int, int] = {}
        best_error = torch.tensor(float("inf"), dtype=dtype, device=device)
        selected_count = int(selected_files.numel())
        slot_local_count = int(valid_slot_indices.numel())
        for file_order in permutations(range(selected_count), min(selected_count, slot_local_count)):
            total = zero
            for slot_local, file_local in enumerate(file_order):
                total = total + selected_distances[file_local, slot_local]
            if bool((total < best_error).item()):
                best_error = total
                assignment = {
                    int(valid_slot_indices[slot_local].item()): int(selected_files[file_local].item())
                    for slot_local, file_local in enumerate(file_order)
                }
        if not assignment:
            continue
        assignment_errors.append(best_error / max(1, len(assignment)))

        target_slot = torch.cdist(
            slot_positions[row].unsqueeze(0),
            target_positions[row].view(1, 1, 2),
        ).squeeze(0).squeeze(-1).masked_fill(~row_slot_valid, float("inf")).argmin()
        target_file = assignment.get(int(target_slot.item()))
        if target_file is not None:
            hit = (file_instance_labels[target_file] == target_label).to(dtype)
            target_hits.append(hit)
            target_error = (file_positions[target_file] - slot_positions[row, target_slot]).norm() / scale
            target_errors.append(target_error)
            same_group_other = (group_labels == group_labels[target_file]) & (file_instance_labels != target_label)
            if bool(same_group_other.any().item()):
                hard_hits.append(hit)

        if distractor_positions is not None and distractor_label is not None:
            distractor_slot = torch.cdist(
                slot_positions[row].unsqueeze(0),
                distractor_positions[row].view(1, 1, 2),
            ).squeeze(0).squeeze(-1).masked_fill(~row_slot_valid, float("inf")).argmin()
            distractor_file = assignment.get(int(distractor_slot.item()))
            if distractor_file is not None:
                distractor_hit = (file_instance_labels[distractor_file] == distractor_label).to(dtype)
                distractor_hits.append(distractor_hit)
                distractor_error = (file_positions[distractor_file] - slot_positions[row, distractor_slot]).norm() / scale
                distractor_errors.append(distractor_error)
                if target_file is not None:
                    pair_hits.append((hit.bool() & distractor_hit.bool()).to(dtype))

    return {
        "target_match_accuracy": torch.stack(target_hits).mean() if target_hits else zero,
        "target_hard_match_accuracy": torch.stack(hard_hits).mean() if hard_hits else zero,
        "distractor_match_accuracy": torch.stack(distractor_hits).mean() if distractor_hits else zero,
        "pair_match_accuracy": torch.stack(pair_hits).mean() if pair_hits else zero,
        "candidate_mean_count": torch.stack(candidate_counts).mean() if candidate_counts else zero,
        "row_coverage_fraction": torch.stack(row_coverages).mean() if row_coverages else zero,
        "target_file_recall_fraction": torch.stack(target_recalls).mean() if target_recalls else zero,
        "distractor_file_recall_fraction": torch.stack(distractor_recalls).mean() if distractor_recalls else zero,
        "assignment_position_error": torch.stack(assignment_errors).mean() if assignment_errors else zero,
        "target_assignment_position_error": torch.stack(target_errors).mean() if target_errors else zero,
        "distractor_assignment_position_error": torch.stack(distractor_errors).mean() if distractor_errors else zero,
        "assignment_object_file_id_usage": zero,
        "assignment_object_id_usage": zero,
        "assignment_sequence_id_usage": zero,
    }


def _oracle_pair_file_slot_ceiling_metrics(
    slot_positions: torch.Tensor,
    slot_valid: torch.Tensor,
    target_positions: torch.Tensor,
    target_instance_labels: torch.Tensor,
    group_labels: torch.Tensor,
    cfg: TsmConfig,
    distractor_positions: torch.Tensor | None,
    distractor_instance_labels: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    dtype = slot_positions.dtype
    device = slot_positions.device
    zero = torch.zeros((), dtype=dtype, device=device)
    if (
        distractor_positions is None
        or distractor_instance_labels is None
        or distractor_positions.numel() == 0
        or distractor_instance_labels.numel() == 0
    ):
        return {
            "target_match_accuracy": zero,
            "target_hard_match_accuracy": zero,
            "distractor_match_accuracy": zero,
            "pair_match_accuracy": zero,
            "candidate_mean_count": zero,
            "row_coverage_fraction": zero,
            "target_file_recall_fraction": zero,
            "distractor_file_recall_fraction": zero,
            "assignment_position_error": zero,
            "target_assignment_position_error": zero,
            "distractor_assignment_position_error": zero,
            "assignment_object_file_id_usage": zero,
            "assignment_object_id_usage": zero,
            "assignment_sequence_id_usage": zero,
        }

    query_count = min(
        slot_positions.shape[0],
        slot_valid.shape[0],
        target_positions.shape[0],
        target_instance_labels.shape[0],
        distractor_positions.shape[0],
        distractor_instance_labels.shape[0],
        group_labels.shape[0],
    )
    if query_count == 0:
        return {
            "target_match_accuracy": zero,
            "target_hard_match_accuracy": zero,
            "distractor_match_accuracy": zero,
            "pair_match_accuracy": zero,
            "candidate_mean_count": zero,
            "row_coverage_fraction": zero,
            "target_file_recall_fraction": zero,
            "distractor_file_recall_fraction": zero,
            "assignment_position_error": zero,
            "target_assignment_position_error": zero,
            "distractor_assignment_position_error": zero,
            "assignment_object_file_id_usage": zero,
            "assignment_object_id_usage": zero,
            "assignment_sequence_id_usage": zero,
        }

    collected: dict[str, list[torch.Tensor]] = {}
    for row in range(query_count):
        row_file_positions = torch.stack((target_positions[row], distractor_positions[row]), dim=0)
        row_file_labels = torch.stack((target_instance_labels[row], distractor_instance_labels[row]), dim=0)
        row_group_labels = group_labels[row].view(1).expand(2)
        row_metrics = _file_slot_assignment_metrics(
            row_file_positions,
            torch.ones(2, dtype=torch.bool, device=device),
            slot_positions[row:row + 1],
            slot_valid[row:row + 1],
            target_positions[row:row + 1],
            row_file_labels,
            target_instance_labels[row:row + 1],
            row_group_labels,
            cfg,
            distractor_positions=distractor_positions[row:row + 1],
            distractor_instance_labels=distractor_instance_labels[row:row + 1],
        )
        for key, value in row_metrics.items():
            collected.setdefault(key, []).append(value)

    return {
        key: torch.stack(values).mean() if values else zero
        for key, values in collected.items()
    }


def _oracle_pair_file_slot_noise_sweep_metrics(
    slot_positions: torch.Tensor,
    slot_valid: torch.Tensor,
    target_positions: torch.Tensor,
    target_instance_labels: torch.Tensor,
    group_labels: torch.Tensor,
    cfg: TsmConfig,
    distractor_positions: torch.Tensor | None,
    distractor_instance_labels: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    dtype = slot_positions.dtype
    device = slot_positions.device
    zero_base = _oracle_pair_file_slot_ceiling_metrics(
        slot_positions,
        slot_valid,
        target_positions,
        target_instance_labels,
        group_labels,
        cfg,
        distractor_positions,
        distractor_instance_labels,
    )
    if (
        distractor_positions is None
        or distractor_instance_labels is None
        or distractor_positions.numel() == 0
        or distractor_instance_labels.numel() == 0
    ):
        out: dict[str, torch.Tensor] = {}
        for noise_px in ORACLE_POSITION_NOISE_SWEEP_PX:
            label = f"{int(noise_px)}px"
            for key, value in zero_base.items():
                out[f"noise_{label}_{key}"] = value
            out[f"noise_{label}_position_noise_px"] = torch.tensor(noise_px, dtype=dtype, device=device)
            out[f"noise_{label}_position_noise_normalized"] = torch.tensor(
                noise_px / float(max(1, cfg.image_size)),
                dtype=dtype,
                device=device,
            )
        return out

    query_count = min(
        slot_positions.shape[0],
        slot_valid.shape[0],
        target_positions.shape[0],
        target_instance_labels.shape[0],
        distractor_positions.shape[0],
        distractor_instance_labels.shape[0],
        group_labels.shape[0],
    )
    if query_count == 0:
        out = {}
        for noise_px in ORACLE_POSITION_NOISE_SWEEP_PX:
            label = f"{int(noise_px)}px"
            for key, value in zero_base.items():
                out[f"noise_{label}_{key}"] = value
            out[f"noise_{label}_position_noise_px"] = torch.tensor(noise_px, dtype=dtype, device=device)
            out[f"noise_{label}_position_noise_normalized"] = torch.tensor(
                noise_px / float(max(1, cfg.image_size)),
                dtype=dtype,
                device=device,
            )
        return out

    target_positions = target_positions[:query_count].to(device=device, dtype=dtype)
    distractor_positions = distractor_positions[:query_count].to(device=device, dtype=dtype)
    target_instance_labels = target_instance_labels[:query_count].to(device=device, dtype=torch.long)
    distractor_instance_labels = distractor_instance_labels[:query_count].to(device=device, dtype=torch.long)
    group_labels = group_labels[:query_count].to(device=device, dtype=torch.long)
    slot_positions = slot_positions[:query_count].to(device=device, dtype=dtype)
    slot_valid = slot_valid[:query_count].to(device=device, dtype=torch.bool)

    rows = torch.arange(query_count, device=device, dtype=dtype)

    out: dict[str, torch.Tensor] = {}
    for noise_px in ORACLE_POSITION_NOISE_SWEEP_PX:
        label = f"{int(noise_px)}px"
        collected: dict[str, list[torch.Tensor]] = {}
        adversarial_delta = distractor_positions - target_positions
        adversarial_direction = adversarial_delta / adversarial_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for trial in range(ORACLE_POSITION_NOISE_SWEEP_TRIALS):
            if trial == 0:
                target_direction = adversarial_direction
                distractor_direction = -adversarial_direction
            else:
                target_angle = rows * 1.61803398875 + 0.37 + float(trial) * 0.78539816339
                distractor_angle = rows * 1.61803398875 + 2.11 + float(trial) * 1.17809724510
                target_direction = torch.stack((torch.cos(target_angle), torch.sin(target_angle)), dim=-1)
                distractor_direction = torch.stack((torch.cos(distractor_angle), torch.sin(distractor_angle)), dim=-1)
            noisy_target = target_positions + target_direction * float(noise_px)
            noisy_distractor = distractor_positions + distractor_direction * float(noise_px)
            for row in range(query_count):
                row_file_positions = torch.stack((noisy_target[row], noisy_distractor[row]), dim=0)
                row_file_labels = torch.stack((target_instance_labels[row], distractor_instance_labels[row]), dim=0)
                row_group_labels = group_labels[row].view(1).expand(2)
                row_metrics = _file_slot_assignment_metrics(
                    row_file_positions,
                    torch.ones(2, dtype=torch.bool, device=device),
                    slot_positions[row:row + 1],
                    slot_valid[row:row + 1],
                    target_positions[row:row + 1],
                    row_file_labels,
                    target_instance_labels[row:row + 1],
                    row_group_labels,
                    cfg,
                    distractor_positions=distractor_positions[row:row + 1],
                    distractor_instance_labels=distractor_instance_labels[row:row + 1],
                )
                for key, value in row_metrics.items():
                    collected.setdefault(key, []).append(value)
        for key, values in collected.items():
            out[f"noise_{label}_{key}"] = torch.stack(values).mean() if values else zero_base[key]
        out[f"noise_{label}_position_noise_px"] = torch.tensor(noise_px, dtype=dtype, device=device)
        out[f"noise_{label}_position_noise_normalized"] = torch.tensor(
            noise_px / float(max(1, cfg.image_size)),
            dtype=dtype,
            device=device,
        )
        out[f"noise_{label}_trial_count"] = torch.tensor(
            float(ORACLE_POSITION_NOISE_SWEEP_TRIALS),
            dtype=dtype,
            device=device,
        )
    return out


def _candidate_path_context_metrics(
    diagnostics: dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_scalar(reference)
    return {
        "occluded_bridge_delta": diagnostics.get("occluded_memory_definition_object_probe_delta", zero),
        "ternary_nonzero_fraction": diagnostics.get("ternary_nonzero_fraction", zero),
        "dynamics_position_error": diagnostics.get("reappeared_dynamics_position_error", zero),
        "dynamics_valid_fraction": diagnostics.get("reappeared_dynamics_valid_fraction", zero),
    }


def _object_cycle_loss(
    hidden_scores: torch.Tensor,
    reappeared_scores: torch.Tensor,
    hidden_file_scores: torch.Tensor,
    reappeared_file_scores: torch.Tensor,
    temperature: float,
    pair_weight: float,
    file_weight: float,
) -> torch.Tensor:
    if (
        hidden_scores.shape[0] < 2
        or reappeared_scores.shape[0] < 2
        or hidden_file_scores.shape[0] < 2
        or reappeared_file_scores.shape[0] < 2
    ):
        return _zero_like_scalar(hidden_scores)
    hidden_to_file = _bidirectional_paired_contrastive_loss(hidden_scores, hidden_file_scores, temperature)
    reappeared_to_file = _bidirectional_paired_contrastive_loss(reappeared_scores, reappeared_file_scores, temperature)
    file_context_cycle = _bidirectional_paired_contrastive_loss(hidden_file_scores, reappeared_file_scores, temperature)
    hidden_to_reappeared = _bidirectional_paired_contrastive_loss(hidden_scores, reappeared_scores, temperature)
    return hidden_to_file + reappeared_to_file + file_weight * file_context_cycle + pair_weight * hidden_to_reappeared


class Self(nn.Module):
    def __init__(self, cfg: TsmConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TsmConfig()
        self.perception = PerceptionSurface(self.cfg)
        self.reality = Reality(self.cfg)
        self.mind = Mind(self.cfg)
        self.sae = SAE(self.cfg)
        self.contexts = ContextRouter(self.cfg)
        self.defs = DefinitionBank(self.cfg)
        self.object_slots = ObjectSlotReadout(self.cfg)
        gate_input = _active_file_gate_input_dim(self.cfg)
        gate_hidden = max(16, self.cfg.definitions_per_context * 2)
        self.active_file_gate = nn.Sequential(
            nn.Linear(gate_input, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        expectation_input = _active_file_expectation_input_dim(self.cfg)
        expectation_hidden = max(16, self.cfg.definitions_per_context * 2)
        self.active_file_expectation = nn.Sequential(
            nn.Linear(expectation_input, expectation_hidden),
            nn.GELU(),
            nn.Linear(expectation_hidden, self.cfg.definitions_per_context),
        )
        nn.init.zeros_(self.active_file_expectation[-1].weight)
        nn.init.zeros_(self.active_file_expectation[-1].bias)
        dynamics_input = _active_file_dynamics_input_dim(self.cfg)
        dynamics_hidden = max(16, dynamics_input)
        self.active_file_dynamics = nn.Sequential(
            nn.Linear(dynamics_input, dynamics_hidden),
            nn.GELU(),
            nn.Linear(dynamics_hidden, 2),
        )
        nn.init.zeros_(self.active_file_dynamics[-1].weight)
        nn.init.zeros_(self.active_file_dynamics[-1].bias)
        self.drive_dynamics = DriveDynamics()
        self.memory = Memory()
        self.evidence = EvidenceAccumulator()
        self.gate = MutationGate()
        self.trauma = TraumaMonitor()
        self.stage = DevelopmentalScheduler()

    @torch.no_grad()
    def _diagnostic_ternary_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        perception = self.perception(image, dataset_id=dataset_id)
        batch_size = image.shape[0]
        initial_q = self.mind.initial(batch_size, image.device)
        context = self.contexts(initial_q, perception.latents)
        expected0 = self.reality.predict_latents(initial_q, context.embedding)
        drives = DriveState.zeros(batch_size, image.device)
        attach_power = torch.zeros_like(perception.latents)
        sae0 = self.sae(
            perception.latents,
            expected0,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        q, _delta_q = self.mind.infer(
            perception.latents,
            sae0.eps,
            context.embedding,
            steps=self.cfg.inference_steps,
        )
        expected = self.reality.predict_latents(q, context.embedding)
        sae = self.sae(
            perception.latents,
            expected,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        return self.defs.project(sae.eps, context.probs, token_position=perception.meta.position)

    def _definition_state_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
        include_binding_position: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        perception = self.perception(image, dataset_id=dataset_id)
        batch_size = image.shape[0]
        initial_q = self.mind.initial(batch_size, image.device)
        context = self.contexts(initial_q, perception.latents)
        expected0 = self.reality.predict_latents(initial_q, context.embedding)
        drives = DriveState.zeros(batch_size, image.device)
        attach_power = torch.zeros_like(perception.latents)
        sae0 = self.sae(
            perception.latents,
            expected0,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        q, _delta_q = self.mind.infer(
            perception.latents,
            sae0.eps,
            context.embedding,
            steps=self.cfg.inference_steps,
        )
        expected = self.reality.predict_latents(q, context.embedding)
        sae = self.sae(
            perception.latents,
            expected,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        scores = self.defs.raw_scores(sae.eps, context.probs, token_position=perception.meta.position)
        if include_binding_position:
            return scores, context.probs, perception.meta.binding_position
        return scores, context.probs

    def _definition_scores_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores, _context_probs = self._definition_state_for_image(image, dataset_id)
        return scores

    @torch.no_grad()
    def _diagnostic_definition_scores_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._definition_scores_for_image(image, dataset_id)

    @torch.no_grad()
    def _diagnostic_definition_state_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
        include_binding_position: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self._definition_state_for_image(image, dataset_id, include_binding_position)

    def forward_train(self, batch: dict[str, torch.Tensor], include_label_diagnostics: bool = True) -> TrainOutput:
        image_t = batch["image_t"]
        image_tp1 = batch["image_tp1"]
        dataset_id = batch.get("dataset_id")
        mode = batch.get("mode", batch.get("label"))
        perception = self.perception(image_t, dataset_id=dataset_id)
        batch_size = image_t.shape[0]
        initial_q = self.mind.initial(batch_size, image_t.device)
        context = self.contexts(initial_q, perception.latents)
        expected0 = self.reality.predict_latents(initial_q, context.embedding)
        drives = DriveState.zeros(batch_size, image_t.device)
        attach_power = torch.zeros_like(perception.latents)
        sae0 = self.sae(
            perception.latents,
            expected0,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        q, delta_q = self.mind.infer(
            perception.latents,
            sae0.eps,
            context.embedding,
            steps=self.cfg.inference_steps,
        )
        expected = self.reality.predict_latents(q, context.embedding)
        sae = self.sae(
            perception.latents,
            expected,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        memory_read = self.memory.read_write_object_files(
            batch,
            perception.latents.mean(dim=1),
            step=len(self.memory.records),
        )
        memory_feature = memory_read.feature
        memory_confidence = memory_read.confidence
        memory_prediction_confidence = memory_confidence
        if "occluded_t" in batch:
            occluded_gate = batch["occluded_t"].to(device=image_t.device, dtype=image_t.dtype).unsqueeze(-1)
            memory_prediction_confidence = memory_prediction_confidence * occluded_gate
        if not self.cfg.use_memory_conditioning:
            memory_prediction_confidence = torch.zeros_like(memory_prediction_confidence)
        memory_definition_confidence = memory_prediction_confidence if self.cfg.use_ternary_conditioning else torch.zeros_like(memory_prediction_confidence)
        ternary_base = self.defs.project(sae.eps, context.probs, token_position=perception.meta.position)
        ternary = self.defs.project(
            sae.eps,
            context.probs,
            memory_feature,
            memory_definition_confidence,
            token_position=perception.meta.position,
        )
        ternary_condition = ternary if self.cfg.use_ternary_conditioning else None
        ternary_base_condition = ternary_base if self.cfg.use_ternary_conditioning else None
        recon = self.reality.reconstruct_image(q, ternary_condition)
        next_image = self.reality.predict_next_image(
            q,
            context.embedding,
            ternary_condition,
            memory_feature,
            memory_prediction_confidence,
        )
        reappearance_alignment = _zero_like_scalar(image_t)
        object_cycle_consistency = _zero_like_scalar(image_t)
        reappearance_file_query = _zero_like_scalar(image_t)
        active_file_query = _zero_like_scalar(image_t)
        active_file_expectation = _zero_like_scalar(image_t)
        active_file_expectation_pair = _zero_like_scalar(image_t)
        active_file_expectation_hard = _zero_like_scalar(image_t)
        active_file_dynamics = _zero_like_scalar(image_t)
        learned_active_file_gate = _zero_like_scalar(image_t)
        needs_reappearance_target = (
            self.cfg.reappearance_alignment_weight > 0.0
            or self.cfg.object_cycle_weight > 0.0
            or self.cfg.reappearance_file_query_weight > 0.0
            or self.cfg.active_file_query_weight > 0.0
            or self.cfg.active_file_expectation_weight > 0.0
            or self.cfg.active_file_dynamics_weight > 0.0
            or self.cfg.learned_active_file_gate_weight > 0.0
        )
        if needs_reappearance_target and "visible_tp1" in batch and "occluded_t" in batch:
            visible_tp1_for_alignment = batch["visible_tp1"].to(device=image_t.device, dtype=image_t.dtype)
            occluded_t_for_alignment = batch["occluded_t"].to(device=image_t.device, dtype=image_t.dtype)
            reappeared_for_alignment = (occluded_t_for_alignment > 0.5) & (visible_tp1_for_alignment > 0.5)
            if bool(reappeared_for_alignment.any().item()):
                target_dataset_id = dataset_id[reappeared_for_alignment] if dataset_id is not None else None
                source_scores = self.defs.raw_scores(
                    sae.eps,
                    context.probs,
                    memory_feature,
                    memory_definition_confidence,
                    token_position=perception.meta.position,
                )
                target_scores, target_context_probs, target_binding_position = self._definition_state_for_image(
                    image_tp1[reappeared_for_alignment],
                    dataset_id=target_dataset_id,
                    include_binding_position=True,
                )
                source_reappeared_scores = source_scores[reappeared_for_alignment]
                if self.cfg.reappearance_alignment_weight > 0.0:
                    reappearance_alignment = _paired_contrastive_loss(
                        source_reappeared_scores,
                        target_scores,
                        self.cfg.reappearance_alignment_temperature,
                    )
                needs_file_anchor = (
                    self.cfg.object_cycle_weight > 0.0
                    or self.cfg.reappearance_file_query_weight > 0.0
                    or self.cfg.active_file_query_weight > 0.0
                    or self.cfg.active_file_expectation_weight > 0.0
                    or self.cfg.learned_active_file_gate_expectation_features
                    or self.cfg.learned_active_file_gate_weight > 0.0
                )
                if needs_file_anchor:
                    source_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_for_alignment],
                        context.probs[reappeared_for_alignment],
                        memory_definition_confidence[reappeared_for_alignment],
                    )
                    target_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_for_alignment],
                        target_context_probs,
                        memory_definition_confidence[reappeared_for_alignment],
                    )
                    target_query_scores = self.defs.file_query_scores(target_scores)
                expected_state_scores = None
                expected_query_scores = None
                dynamics_position = None
                projected_position = None
                dynamics_valid = None
                needs_dynamics_position = (
                    self.cfg.active_file_dynamics_weight > 0.0
                    or self.cfg.active_file_expectation_dynamics_features
                )
                if needs_dynamics_position and "object_position_tp1" in batch:
                    projected_position, _projected_valid = _active_file_projected_position(
                        memory_read,
                        reappeared_for_alignment,
                        self.cfg,
                        image_t.dtype,
                        image_t.device,
                    )
                    dynamics_features = _active_file_dynamics_features(
                        batch,
                        memory_read,
                        reappeared_for_alignment,
                        self.cfg,
                        image_t.dtype,
                        image_t.device,
                        context.probs[reappeared_for_alignment],
                        memory_definition_confidence[reappeared_for_alignment],
                        memory_read.age[reappeared_for_alignment],
                    )
                    dynamics_valid = (
                        memory_read.position_valid[reappeared_for_alignment]
                        & memory_read.hit[reappeared_for_alignment]
                        & (
                            memory_read.age[reappeared_for_alignment].view(-1)
                            <= self.cfg.active_file_candidate_max_age
                        )
                    )
                    if self.cfg.active_file_dynamics_detach_inputs:
                        dynamics_features = dynamics_features.detach()
                        projected_position = projected_position.detach()
                    dynamics_position = _active_file_dynamics_position(
                        self.active_file_dynamics,
                        dynamics_features,
                        projected_position,
                        self.cfg,
                    )
                    if self.cfg.active_file_dynamics_weight > 0.0 and bool(dynamics_valid.any().item()):
                        target_position = batch["object_position_tp1"].to(device=image_t.device, dtype=image_t.dtype)[
                            reappeared_for_alignment
                        ]
                        scale = float(max(1, self.cfg.image_size))
                        active_file_dynamics = F.smooth_l1_loss(
                            dynamics_position[dynamics_valid] / scale,
                            target_position[dynamics_valid] / scale,
                        )
                needs_file_expectation = (
                    self.cfg.active_file_expectation_weight > 0.0
                    or self.cfg.learned_active_file_gate_expectation_features
                )
                if needs_file_expectation:
                    expectation_file_scores = source_file_scores
                    expectation_file_context = context.probs[reappeared_for_alignment]
                    expectation_confidence = memory_definition_confidence[reappeared_for_alignment]
                    expectation_age = memory_read.age[reappeared_for_alignment]
                    expectation_trajectory_features = (
                        _active_file_trajectory_features(
                            batch,
                            memory_read,
                            reappeared_for_alignment,
                            self.cfg,
                            image_t.dtype,
                            image_t.device,
                            dynamics_position
                            if self.cfg.active_file_expectation_dynamics_features
                            else None,
                        )
                        if self.cfg.active_file_expectation_trajectory_features
                        else None
                    )
                    if self.cfg.active_file_expectation_detach_inputs:
                        expectation_file_scores = expectation_file_scores.detach()
                        expectation_file_context = expectation_file_context.detach()
                        expectation_confidence = expectation_confidence.detach()
                        expectation_age = expectation_age.detach()
                        if expectation_trajectory_features is not None:
                            expectation_trajectory_features = expectation_trajectory_features.detach()
                    expected_state_scores = _active_file_expectation(
                        self.active_file_expectation,
                        expectation_file_scores,
                        expectation_file_context,
                        expectation_confidence,
                        expectation_age,
                        self.cfg.active_file_candidate_max_age,
                        expectation_trajectory_features,
                    )
                    expected_query_scores = self.defs.file_query_scores(expected_state_scores)
                if self.cfg.object_cycle_weight > 0.0:
                    object_cycle_consistency = _object_cycle_loss(
                        source_reappeared_scores,
                        target_scores,
                        source_file_scores,
                        target_file_scores,
                        self.cfg.object_cycle_temperature,
                        self.cfg.object_cycle_pair_weight,
                        self.cfg.object_cycle_file_weight,
                    )
                if self.cfg.reappearance_file_query_weight > 0.0:
                    file_query_pair = _bidirectional_paired_contrastive_loss(
                        target_query_scores,
                        target_file_scores,
                        self.cfg.reappearance_file_query_temperature,
                    )
                    file_query_hard = _zero_like_scalar(image_t)
                    instance_ids = _object_instance_ids(batch, image_t.device)
                    if instance_ids is not None and "object_id" in batch:
                        object_id = batch["object_id"].to(device=image_t.device, dtype=torch.long)
                        file_query_hard = _grouped_contrastive_query_loss(
                            target_query_scores,
                            target_file_scores,
                            instance_ids[reappeared_for_alignment],
                            object_id[reappeared_for_alignment],
                            self.cfg.reappearance_file_query_temperature,
                        )
                    reappearance_file_query = (
                        file_query_pair + self.cfg.reappearance_file_query_hard_weight * file_query_hard
                    )
                if self.cfg.active_file_expectation_weight > 0.0 and expected_state_scores is not None:
                    active_file_expectation_pair = _paired_contrastive_loss(
                        expected_state_scores,
                        target_scores,
                        self.cfg.active_file_expectation_temperature,
                    )
                    if (
                        self.cfg.active_file_expectation_hard_weight > 0.0
                        and "object_id" in batch
                    ):
                        instance_ids = _object_instance_ids(batch, image_t.device)
                        object_id = batch["object_id"].to(device=image_t.device, dtype=torch.long)
                        if instance_ids is not None:
                            active_file_expectation_hard = _grouped_contrastive_query_loss(
                                expected_state_scores,
                                target_scores,
                                instance_ids[reappeared_for_alignment],
                                object_id[reappeared_for_alignment],
                                self.cfg.active_file_expectation_temperature,
                                detach_files=True,
                            )
                    active_file_expectation = (
                        active_file_expectation_pair
                        + self.cfg.active_file_expectation_hard_weight * active_file_expectation_hard
                    )
                if (
                    (self.cfg.active_file_query_weight > 0.0 or self.cfg.learned_active_file_gate_weight > 0.0)
                    and "object_position_tp1" in batch
                ):
                    file_valid = (
                        memory_read.position_valid[reappeared_for_alignment]
                        & memory_read.hit[reappeared_for_alignment]
                        & (
                            memory_read.age[reappeared_for_alignment].view(-1)
                            <= self.cfg.active_file_candidate_max_age
                        )
                    )
                    feature_only_candidates = _active_file_feature_only_candidate_mask(
                        file_valid,
                        int(reappeared_for_alignment.to(torch.long).sum().item()),
                        image_t.dtype,
                        image_t.device,
                    )
                    predicted_position_candidates = None
                    if dynamics_position is not None:
                        predicted_position_candidates = _active_file_candidate_mask(
                            memory_read.position[reappeared_for_alignment],
                            dynamics_position.detach(),
                            memory_read.position_valid[reappeared_for_alignment],
                            memory_read.hit[reappeared_for_alignment],
                            memory_read.age[reappeared_for_alignment],
                            self.cfg.active_file_candidate_radius,
                            self.cfg.active_file_candidate_max_age,
                            _active_file_wrap_span(self.cfg),
                        )
                    active_candidates = (
                        predicted_position_candidates
                        if predicted_position_candidates is not None
                        else feature_only_candidates
                    )
                    if self.cfg.active_file_query_weight > 0.0:
                        active_file_query = _candidate_masked_query_loss(
                            target_query_scores,
                            target_file_scores,
                            active_candidates,
                            self.cfg.active_file_query_temperature,
                        )
                    if self.cfg.learned_active_file_gate_weight > 0.0:
                        gate_query_scores = target_query_scores
                        gate_file_scores = target_file_scores
                        gate_query_context = target_context_probs
                        gate_file_context = context.probs[reappeared_for_alignment]
                        if self.cfg.learned_active_file_gate_detach_inputs:
                            gate_query_scores = gate_query_scores.detach()
                            gate_file_scores = gate_file_scores.detach()
                            gate_query_context = gate_query_context.detach()
                            gate_file_context = gate_file_context.detach()
                        if not self.cfg.learned_active_file_gate_context_features:
                            gate_query_context = None
                            gate_file_context = None
                        gate_expected_query = (
                            expected_query_scores
                            if self.cfg.learned_active_file_gate_expectation_features
                            else None
                        )
                        if self.cfg.learned_active_file_gate_detach_inputs and gate_expected_query is not None:
                            gate_expected_query = gate_expected_query.detach()
                        learned_logits = _active_file_gate_logits(
                            self.active_file_gate,
                            gate_query_scores,
                            gate_file_scores,
                            memory_definition_confidence[reappeared_for_alignment],
                            memory_read.age[reappeared_for_alignment],
                            self.cfg.active_file_candidate_max_age,
                            gate_query_context,
                            gate_file_context,
                            gate_expected_query,
                        )
                        if learned_logits.numel() > 0:
                            count = min(learned_logits.shape[0], learned_logits.shape[1], active_candidates.shape[0])
                            valid_pairs = file_valid[:count].to(device=image_t.device, dtype=torch.bool).unsqueeze(0)
                            valid_pairs = valid_pairs.expand(count, count)
                            if bool(valid_pairs.any().item()):
                                learned_active_file_gate = F.binary_cross_entropy_with_logits(
                                    learned_logits[:count, :count][valid_pairs],
                                    active_candidates[:count, :count].to(dtype=image_t.dtype)[valid_pairs],
                                )
        prior = self.reality.context_prior(context.probs)
        free_energy = variational_free_energy(sae.raw, sae.precision, q, prior)
        ternary_nonzero = ternary.ne(0)
        ternary_base_nonzero = ternary_base.ne(0)
        memory_definition_flip = ternary.ne(ternary_base)
        context_hard = context.probs.argmax(dim=-1)
        context_used = torch.bincount(context_hard, minlength=self.cfg.contexts).gt(0).float().sum()
        context_entropy_value = context_entropy(context.probs)
        context_balance_value = context_balance_loss(context.probs)
        prediction_error_per_sample = (next_image - image_tp1).square().flatten(1).mean(dim=1)
        reconstruction_error_per_sample = (recon - image_t).square().flatten(1).mean(dim=1)
        severity_per_sample = sae.severity.mean(dim=1)
        coherence_per_sample = sae.coherence.mean(dim=1)
        losses = {
            "reconstruction": F.mse_loss(recon, image_t),
            "prediction": F.mse_loss(next_image, image_tp1),
            "free_energy": free_energy,
            "complexity": F.mse_loss(q.mean(dim=1), prior),
            "context_entropy": context_entropy_value,
            "context_balance": context_balance_value,
            "ternary_activation_l1": ternary.abs().mean(),
            "bit_cost": self.defs.bit_cost(),
            "reappearance_alignment": reappearance_alignment,
            "object_cycle_consistency": object_cycle_consistency,
            "reappearance_file_query": reappearance_file_query,
            "active_file_query": active_file_query,
            "active_file_expectation": active_file_expectation,
            "active_file_expectation_pair": active_file_expectation_pair,
            "active_file_expectation_hard": active_file_expectation_hard,
            "active_file_dynamics": active_file_dynamics,
            "learned_active_file_gate": learned_active_file_gate,
        }
        diagnostics = {
            "context_max_probability": context.probs.max(dim=-1).values.mean(),
            "context_effective_count": torch.exp(context_entropy_value.detach()),
            "context_used_count": context_used,
            "ternary_zero_fraction": ternary.eq(0).float().mean(),
            "ternary_nonzero_fraction": ternary_nonzero.float().mean(),
            "ternary_positive_fraction": ternary.gt(0).float().mean(),
            "ternary_negative_fraction": ternary.lt(0).float().mean(),
            "memory_definition_flip_fraction": memory_definition_flip.to(image_t.dtype).mean(),
            "memory_definition_activation_delta": (
                ternary_nonzero.to(image_t.dtype).mean() - ternary_base_nonzero.to(image_t.dtype).mean()
            ),
            "ternary_condition_norm": (
                self.reality.condition_latents(q, ternary_condition) - q
            ).detach().norm(dim=-1).mean(),
            "memory_definition_condition_norm": (
                self.reality.condition_latents(q, ternary_condition)
                - self.reality.condition_latents(q, ternary_base_condition)
            ).detach().norm(dim=-1).mean(),
            "memory_condition_norm": (
                self.reality.condition_latents(q, ternary_condition, memory_feature, memory_prediction_confidence)
                - self.reality.condition_latents(q, ternary_condition)
            ).detach().norm(dim=-1).mean(),
            "memory_object_file_count": torch.tensor(
                len(self.memory.object_files),
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_object_read_count": torch.tensor(
                self.memory.object_read_count,
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_object_write_count": torch.tensor(
                self.memory.object_write_count,
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_object_hit_fraction": memory_read.hit.to(image_t.dtype).mean(),
            "memory_object_write_fraction": memory_read.write.to(image_t.dtype).mean(),
            "memory_object_confidence_mean": memory_confidence.mean(),
            "memory_object_prediction_confidence_mean": memory_prediction_confidence.mean(),
            "memory_object_age_mean": memory_read.age.mean(),
            "sae_severity_mean": sae.severity.mean(),
            "sae_coherence_mean": sae.coherence.mean(),
            "gate_accept_count": torch.tensor(
                sum(decision.accepted for decision in self.gate.decisions),
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "gate_reject_count": torch.tensor(
                sum(not decision.accepted for decision in self.gate.decisions),
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_records": torch.tensor(len(self.memory.records), dtype=image_t.dtype, device=image_t.device),
        }
        if mode is not None:
            mode = mode.to(device=image_t.device, dtype=torch.long)
            diagnostics.update(_mode_context_stats(
                mode,
                context_hard,
                self.cfg.contexts,
                image_t.dtype,
            ))
            diagnostics["mode_count"] = torch.tensor(
                mode[mode >= 0].unique().numel(),
                dtype=image_t.dtype,
                device=image_t.device,
            )
            if include_label_diagnostics:
                diagnostics.update(ternary_label_diagnostics(ternary, mode, context_hard))
        if "phase" in batch:
            phase = batch["phase"].to(device=image_t.device, dtype=torch.long)
            diagnostics.update(_prefix_metrics(
                "phase_",
                _mode_context_stats(
                    phase,
                    context_hard,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
            ))
            diagnostics["phase_count"] = torch.tensor(
                phase[phase >= 0].unique().numel(),
                dtype=image_t.dtype,
                device=image_t.device,
            )
            if include_label_diagnostics:
                diagnostics.update(_prefix_metrics(
                    "phase_",
                    ternary_label_diagnostics(ternary, phase, context_hard),
                ))
        if "object_id" in batch:
            object_id = batch["object_id"].to(device=image_t.device, dtype=torch.long)
            diagnostics.update(_prefix_metrics(
                "object_",
                _mode_context_stats(
                    object_id,
                    context_hard,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
            ))
            diagnostics["object_count"] = torch.tensor(
                object_id[object_id >= 0].unique().numel(),
                dtype=image_t.dtype,
                device=image_t.device,
            )
            if include_label_diagnostics:
                diagnostics.update(_prefix_metrics(
                    "object_",
                    ternary_label_diagnostics(ternary, object_id, context_hard),
                ))
                active_memory = memory_read.hit.to(device=image_t.device)
                if bool(active_memory.any().item()):
                    diagnostics.update(_prefix_metrics(
                        "memory_object_",
                        feature_label_diagnostics(memory_feature[active_memory], object_id[active_memory]),
                    ))
        if "visible_t" in batch and "occluded_t" in batch:
            visible_t = batch["visible_t"].to(device=image_t.device, dtype=image_t.dtype)
            occluded_t = batch["occluded_t"].to(device=image_t.device, dtype=image_t.dtype)
            visible_tp1 = batch.get("visible_tp1", torch.zeros_like(visible_t)).to(device=image_t.device, dtype=image_t.dtype)
            occluded_tp1 = batch.get("occluded_tp1", torch.zeros_like(occluded_t)).to(device=image_t.device, dtype=image_t.dtype)
            moved = batch.get("moved", torch.zeros_like(visible_t)).to(device=image_t.device, dtype=image_t.dtype)
            identity_preserved = batch.get("identity_preserved", torch.zeros_like(visible_t)).to(
                device=image_t.device,
                dtype=image_t.dtype,
            )
            unexpected_disappearance = batch.get("unexpected_disappearance", torch.zeros_like(visible_t)).to(
                device=image_t.device,
                dtype=image_t.dtype,
            )
            reappeared = (occluded_t > 0.5) & (visible_tp1 > 0.5)
            visible_mask = visible_t > 0.5
            occluded_mask = occluded_t > 0.5
            moved_mask = moved > 0.5
            disappearance_mask = unexpected_disappearance > 0.5
            diagnostics.update({
                "temporal_visible_fraction": visible_t.mean(),
                "temporal_occluded_fraction": occluded_t.mean(),
                "temporal_moved_fraction": moved.mean(),
                "temporal_identity_preserved_fraction": identity_preserved.mean(),
                "temporal_unexpected_disappearance_fraction": unexpected_disappearance.mean(),
                "temporal_reappeared_fraction": reappeared.to(image_t.dtype).mean(),
                "temporal_visible_tp1_fraction": visible_tp1.mean(),
                "temporal_occluded_tp1_fraction": occluded_tp1.mean(),
                "temporal_context_visible_used_count": _masked_context_used(
                    context_hard,
                    visible_mask,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
                "temporal_context_occluded_used_count": _masked_context_used(
                    context_hard,
                    occluded_mask,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
                "temporal_memory_visible_hit_fraction": _masked_mean(
                    memory_read.hit.to(image_t.dtype),
                    visible_mask,
                    image_t.dtype,
                ),
                "temporal_memory_occluded_hit_fraction": _masked_mean(
                    memory_read.hit.to(image_t.dtype),
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_memory_occluded_confidence_mean": _masked_mean(
                    memory_prediction_confidence.squeeze(-1),
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_memory_definition_occluded_flip_fraction": _masked_mean(
                    memory_definition_flip.to(image_t.dtype).mean(dim=1),
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_memory_definition_occluded_activation_delta": _masked_mean(
                    ternary_nonzero.to(image_t.dtype).mean(dim=1)
                    - ternary_base_nonzero.to(image_t.dtype).mean(dim=1),
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_sae_visible_mean": _masked_mean(severity_per_sample, visible_mask, image_t.dtype),
                "temporal_sae_occluded_mean": _masked_mean(severity_per_sample, occluded_mask, image_t.dtype),
                "temporal_sae_moved_mean": _masked_mean(severity_per_sample, moved_mask, image_t.dtype),
                "temporal_sae_disappearance_mean": _masked_mean(
                    severity_per_sample,
                    disappearance_mask,
                    image_t.dtype,
                ),
                "temporal_sae_reappeared_mean": _masked_mean(severity_per_sample, reappeared, image_t.dtype),
                "temporal_coherence_visible_mean": _masked_mean(coherence_per_sample, visible_mask, image_t.dtype),
                "temporal_coherence_occluded_mean": _masked_mean(coherence_per_sample, occluded_mask, image_t.dtype),
                "temporal_prediction_visible_mean": _masked_mean(
                    prediction_error_per_sample,
                    visible_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_occluded_mean": _masked_mean(
                    prediction_error_per_sample,
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_moved_mean": _masked_mean(
                    prediction_error_per_sample,
                    moved_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_disappearance_mean": _masked_mean(
                    prediction_error_per_sample,
                    disappearance_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_reappeared_mean": _masked_mean(
                    prediction_error_per_sample,
                    reappeared,
                    image_t.dtype,
                ),
                "temporal_reconstruction_visible_mean": _masked_mean(
                    reconstruction_error_per_sample,
                    visible_mask,
                    image_t.dtype,
                ),
                "temporal_reconstruction_occluded_mean": _masked_mean(
                    reconstruction_error_per_sample,
                    occluded_mask,
                    image_t.dtype,
                ),
            })
            if "same_class_contested" in batch:
                contested = batch["same_class_contested"].to(device=image_t.device, dtype=image_t.dtype)
                diagnostics["temporal_same_class_contested_fraction"] = contested.mean()
            if "object_file_id" in batch:
                diagnostics["object_file_id_storage_key_present"] = torch.ones((), dtype=image_t.dtype, device=image_t.device)
                diagnostics["object_file_id_bind_time_candidate_filter_usage"] = _zero_like_scalar(image_t)
                diagnostics["object_file_id_bind_time_leakage_audit_pass"] = torch.ones(
                    (),
                    dtype=image_t.dtype,
                    device=image_t.device,
                )
                auxiliary_label_used = float(
                    self.cfg.reappearance_file_query_weight > 0.0
                    or (
                        self.cfg.active_file_expectation_weight > 0.0
                        and self.cfg.active_file_expectation_hard_weight > 0.0
                    )
                )
                diagnostics["object_file_id_auxiliary_label_usage"] = torch.tensor(
                    auxiliary_label_used,
                    dtype=image_t.dtype,
                    device=image_t.device,
                )
            diagnostics["temporal_sae_occlusion_delta"] = (
                diagnostics["temporal_sae_occluded_mean"] - diagnostics["temporal_sae_visible_mean"]
            )
            diagnostics["temporal_prediction_occlusion_delta"] = (
                diagnostics["temporal_prediction_occluded_mean"] - diagnostics["temporal_prediction_visible_mean"]
            )
            with torch.no_grad():
                next_without_memory = self.reality.predict_next_image(q, context.embedding, ternary_condition)
                next_without_any_memory = self.reality.predict_next_image(q, context.embedding, ternary_base_condition)
                no_memory_error = (next_without_memory - image_tp1).square().flatten(1).mean(dim=1)
                no_any_memory_error = (next_without_any_memory - image_tp1).square().flatten(1).mean(dim=1)
                memory_impact = no_memory_error - prediction_error_per_sample.detach()
                memory_total_impact = no_any_memory_error - prediction_error_per_sample.detach()
                memory_definition_impact = no_any_memory_error - no_memory_error
            diagnostics.update({
                "memory_prediction_impact_mean": memory_impact.mean(),
                "memory_prediction_occluded_impact_mean": _masked_mean(memory_impact, occluded_mask, image_t.dtype),
                "memory_prediction_reappeared_impact_mean": _masked_mean(memory_impact, reappeared, image_t.dtype),
                "memory_prediction_disappearance_impact_mean": _masked_mean(
                    memory_impact,
                    disappearance_mask,
                    image_t.dtype,
                ),
                "memory_total_prediction_impact_mean": memory_total_impact.mean(),
                "memory_total_prediction_occluded_impact_mean": _masked_mean(
                    memory_total_impact,
                    occluded_mask,
                    image_t.dtype,
                ),
                "memory_total_prediction_reappeared_impact_mean": _masked_mean(
                    memory_total_impact,
                    reappeared,
                    image_t.dtype,
                ),
                "memory_definition_prediction_impact_mean": memory_definition_impact.mean(),
                "memory_definition_prediction_occluded_impact_mean": _masked_mean(
                    memory_definition_impact,
                    occluded_mask,
                    image_t.dtype,
                ),
                "memory_definition_prediction_reappeared_impact_mean": _masked_mean(
                    memory_definition_impact,
                    reappeared,
                    image_t.dtype,
                ),
            })
            if include_label_diagnostics and "object_id" in batch and bool(occluded_mask.any().item()):
                object_id = batch["object_id"].to(device=image_t.device, dtype=torch.long)
                diagnostics.update(_prefix_metrics(
                    "occluded_object_",
                    ternary_label_diagnostics(
                        ternary[occluded_mask],
                        object_id[occluded_mask],
                        context_hard[occluded_mask],
                    ),
                ))
                diagnostics.update(_prefix_metrics(
                    "occluded_base_",
                    ternary_label_diagnostics(
                        ternary_base[occluded_mask],
                        object_id[occluded_mask],
                        context_hard[occluded_mask],
                    ),
                ))
                diagnostics["occluded_memory_definition_object_probe_delta"] = (
                    diagnostics["occluded_object_ternary_mode_probe_accuracy"]
                    - diagnostics["occluded_base_ternary_mode_probe_accuracy"]
                )
                occluded_active = occluded_mask & memory_read.hit.to(device=image_t.device)
                if bool(occluded_active.any().item()):
                    diagnostics.update(_prefix_metrics(
                        "occluded_memory_object_",
                        feature_label_diagnostics(memory_feature[occluded_active], object_id[occluded_active]),
                    ))
                reappeared_active = reappeared & (object_id >= 0)
                if bool(reappeared_active.any().item()):
                    target_dataset_id = dataset_id[reappeared_active] if dataset_id is not None else None
                    target_ternary = self._diagnostic_ternary_for_image(
                        image_tp1[reappeared_active],
                        dataset_id=target_dataset_id,
                    )
                    (
                        target_definition_scores,
                        target_context_probs,
                        target_binding_position,
                    ) = self._diagnostic_definition_state_for_image(
                        image_tp1[reappeared_active],
                        dataset_id=target_dataset_id,
                        include_binding_position=True,
                    )
                    source_definition_scores = self.defs.raw_scores(
                        sae.eps,
                        context.probs,
                        memory_feature,
                        memory_definition_confidence,
                        token_position=perception.meta.position,
                    )[reappeared_active]
                    object_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_active],
                        context.probs[reappeared_active],
                        memory_definition_confidence[reappeared_active],
                    )
                    target_object_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_active],
                        target_context_probs,
                        memory_definition_confidence[reappeared_active],
                    )
                    target_query_scores = self.defs.file_query_scores(target_definition_scores)
                    expected_state_scores = None
                    expected_query_scores = None
                    dynamics_position = None
                    dynamics_valid = None
                    ballistic_position = None
                    ballistic_valid = None
                    projected_position = None
                    needs_dynamics_position = (
                        self.cfg.active_file_dynamics_weight > 0.0
                        or self.cfg.active_file_expectation_dynamics_features
                    )
                    if needs_dynamics_position and "object_position_tp1" in batch:
                        projected_position, _projected_valid = _active_file_projected_position(
                            memory_read,
                            reappeared_active,
                            self.cfg,
                            image_t.dtype,
                            image_t.device,
                        )
                        dynamics_features = _active_file_dynamics_features(
                            batch,
                            memory_read,
                            reappeared_active,
                            self.cfg,
                            image_t.dtype,
                            image_t.device,
                            context.probs[reappeared_active],
                            memory_definition_confidence[reappeared_active],
                            memory_read.age[reappeared_active],
                        )
                        dynamics_valid = (
                            memory_read.position_valid[reappeared_active]
                            & memory_read.hit[reappeared_active]
                            & (
                                memory_read.age[reappeared_active].view(-1)
                                <= self.cfg.active_file_candidate_max_age
                            )
                        )
                        dynamics_features = dynamics_features.detach()
                        projected_position = projected_position.detach()
                        dynamics_position = _active_file_dynamics_position(
                            self.active_file_dynamics,
                            dynamics_features,
                            projected_position,
                            self.cfg,
                        )
                    if (
                        self.cfg.active_file_expectation_weight > 0.0
                        or self.cfg.learned_active_file_gate_expectation_features
                    ):
                        expectation_trajectory_features = (
                            _active_file_trajectory_features(
                                batch,
                                memory_read,
                                reappeared_active,
                                self.cfg,
                                image_t.dtype,
                                image_t.device,
                                dynamics_position
                                if self.cfg.active_file_expectation_dynamics_features
                                else None,
                            )
                            if self.cfg.active_file_expectation_trajectory_features
                            else None
                        )
                        expected_state_scores = _active_file_expectation(
                            self.active_file_expectation,
                            object_file_scores,
                            context.probs[reappeared_active],
                            memory_definition_confidence[reappeared_active],
                            memory_read.age[reappeared_active],
                            self.cfg.active_file_candidate_max_age,
                            expectation_trajectory_features,
                        )
                        expected_query_scores = self.defs.file_query_scores(expected_state_scores)
                    source_labels = object_id[reappeared_active]
                    diagnostics.update(_prefix_metrics(
                        "reappeared_",
                        feature_match_diagnostics(
                            ternary[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                            source_labels,
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_base_",
                        feature_match_diagnostics(
                            ternary_base[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                            source_labels,
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_",
                        paired_feature_match_diagnostics(
                            ternary[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_base_",
                        paired_feature_match_diagnostics(
                            ternary_base[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_file_",
                        paired_feature_match_diagnostics(
                            source_definition_scores.to(image_t.dtype),
                            object_file_scores.to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_target_file_",
                        paired_feature_match_diagnostics(
                            target_definition_scores.to(image_t.dtype),
                            target_object_file_scores.to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_query_file_",
                        paired_feature_match_diagnostics(
                            target_query_scores.to(image_t.dtype),
                            target_object_file_scores.to(image_t.dtype),
                        ),
                    ))
                    if expected_query_scores is not None:
                        diagnostics.update(_prefix_metrics(
                            "reappeared_expected_file_",
                            paired_feature_match_diagnostics(
                                expected_query_scores.to(image_t.dtype),
                                target_query_scores.to(image_t.dtype),
                            ),
                        ))
                    if expected_state_scores is not None:
                        diagnostics.update(_prefix_metrics(
                            "reappeared_expected_state_",
                            paired_feature_match_diagnostics(
                                expected_state_scores.to(image_t.dtype),
                                target_definition_scores.to(image_t.dtype),
                            ),
                        ))
                    if (
                        (
                            self.cfg.active_file_expectation_trajectory_features
                            or self.cfg.active_file_dynamics_weight > 0.0
                            or self.cfg.active_file_expectation_dynamics_features
                        )
                        and "object_position_tp1" in batch
                    ):
                        projected_position, projected_valid = _active_file_projected_position(
                            memory_read,
                            reappeared_active,
                            self.cfg,
                            image_t.dtype,
                            image_t.device,
                        )
                        target_position = batch["object_position_tp1"].to(device=image_t.device, dtype=image_t.dtype)[
                            reappeared_active
                        ]
                        if bool(projected_valid.any().item()):
                            position_error = (
                                projected_position[projected_valid] - target_position[projected_valid]
                            ).norm(dim=-1).mean() / float(max(1, self.cfg.image_size))
                        else:
                            position_error = _zero_like_scalar(image_t)
                        diagnostics["reappeared_trajectory_position_error"] = position_error
                        diagnostics["reappeared_trajectory_valid_fraction"] = projected_valid.to(image_t.dtype).mean()
                        ballistic_position, ballistic_valid = _active_file_ballistic_position(
                            memory_read,
                            batch,
                            reappeared_active,
                            self.cfg,
                            image_t.dtype,
                            image_t.device,
                        )
                        if bool(ballistic_valid.any().item()):
                            ballistic_error = (
                                ballistic_position[ballistic_valid] - target_position[ballistic_valid]
                            ).norm(dim=-1).mean() / float(max(1, self.cfg.image_size))
                        else:
                            ballistic_error = _zero_like_scalar(image_t)
                        diagnostics["reappeared_ballistic_position_error"] = ballistic_error
                        diagnostics["reappeared_ballistic_valid_fraction"] = ballistic_valid.to(image_t.dtype).mean()
                        if dynamics_position is not None and dynamics_valid is not None:
                            shared_valid = projected_valid & dynamics_valid
                            if bool(dynamics_valid.any().item()):
                                dynamics_error = (
                                    dynamics_position[dynamics_valid] - target_position[dynamics_valid]
                                ).norm(dim=-1).mean() / float(max(1, self.cfg.image_size))
                            else:
                                dynamics_error = _zero_like_scalar(image_t)
                            if bool(shared_valid.any().item()):
                                shared_projected_error = (
                                    projected_position[shared_valid] - target_position[shared_valid]
                                ).norm(dim=-1).mean() / float(max(1, self.cfg.image_size))
                                shared_dynamics_error = (
                                    dynamics_position[shared_valid] - target_position[shared_valid]
                                ).norm(dim=-1).mean() / float(max(1, self.cfg.image_size))
                                dynamics_improvement = shared_projected_error - shared_dynamics_error
                            else:
                                dynamics_improvement = _zero_like_scalar(image_t)
                            diagnostics["reappeared_dynamics_position_error"] = dynamics_error
                            diagnostics["reappeared_dynamics_position_improvement"] = dynamics_improvement
                            diagnostics["reappeared_dynamics_valid_fraction"] = dynamics_valid.to(image_t.dtype).mean()
                            shared_ballistic_valid = ballistic_valid & dynamics_valid
                            if bool(shared_ballistic_valid.any().item()):
                                shared_ballistic_error = (
                                    ballistic_position[shared_ballistic_valid]
                                    - target_position[shared_ballistic_valid]
                                ).norm(dim=-1).mean() / float(max(1, self.cfg.image_size))
                                shared_dynamics_for_ballistic_error = (
                                    dynamics_position[shared_ballistic_valid]
                                    - target_position[shared_ballistic_valid]
                                ).norm(dim=-1).mean() / float(max(1, self.cfg.image_size))
                                diagnostics["reappeared_dynamics_over_ballistic_position_improvement"] = (
                                    shared_ballistic_error - shared_dynamics_for_ballistic_error
                                )
                            else:
                                diagnostics["reappeared_dynamics_over_ballistic_position_improvement"] = (
                                    _zero_like_scalar(image_t)
                                )
                    file_binding_position = (
                        dynamics_position.detach()
                        if dynamics_position is not None
                        else memory_read.position[reappeared_active]
                    )
                    target_definition_binding_scores = _binding_position_features(
                        target_definition_scores,
                        target_binding_position,
                        self.cfg,
                    )
                    target_query_binding_scores = _binding_position_features(
                        target_query_scores,
                        target_binding_position,
                        self.cfg,
                    )
                    source_definition_binding_scores = _binding_position_features(
                        source_definition_scores,
                        _normalized_pixel_position(file_binding_position, self.cfg),
                        self.cfg,
                    )
                    target_object_file_binding_scores = _binding_position_features(
                        target_object_file_scores,
                        _normalized_pixel_position(file_binding_position, self.cfg),
                        self.cfg,
                    )
                    expected_query_binding_scores = _binding_position_features(
                        expected_query_scores if expected_query_scores is not None else target_object_file_scores,
                        _normalized_pixel_position(file_binding_position, self.cfg),
                        self.cfg,
                    )
                    state_prediction_error = _state_prediction_error_matrix(
                        target_query_binding_scores.to(image_t.dtype),
                        expected_query_binding_scores.to(image_t.dtype),
                    )
                    if state_prediction_error.numel() > 0:
                        diagnostics["reappeared_state_prediction_error_mean"] = state_prediction_error.mean()
                    target_position_for_reappeared = None
                    slot_output = None
                    slot_distractor_position = None
                    if "object_position_tp1" in batch:
                        target_position_for_reappeared = batch["object_position_tp1"].to(
                            device=image_t.device,
                            dtype=image_t.dtype,
                        )[reappeared_active]
                        diagnostics.update(_prefix_metrics(
                            "reappeared_definition_",
                            position_recoverability_diagnostics(
                                target_definition_binding_scores.to(image_t.dtype),
                                target_position_for_reappeared,
                                scale=float(max(1, self.cfg.image_size)),
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_definition_position_ablated_",
                            position_recoverability_diagnostics(
                                target_definition_scores.to(image_t.dtype),
                                target_position_for_reappeared,
                                scale=float(max(1, self.cfg.image_size)),
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_file_query_position_ablated_",
                            position_recoverability_diagnostics(
                                target_query_scores.to(image_t.dtype),
                                target_position_for_reappeared,
                                scale=float(max(1, self.cfg.image_size)),
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_memory_definition_position_ablated_",
                            position_recoverability_diagnostics(
                                source_definition_scores.to(image_t.dtype),
                                target_position_for_reappeared,
                                scale=float(max(1, self.cfg.image_size)),
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_file_query_",
                            position_recoverability_diagnostics(
                                target_query_binding_scores.to(image_t.dtype),
                                target_position_for_reappeared,
                                scale=float(max(1, self.cfg.image_size)),
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_memory_definition_",
                            position_recoverability_diagnostics(
                                source_definition_binding_scores.to(image_t.dtype),
                                target_position_for_reappeared,
                                scale=float(max(1, self.cfg.image_size)),
                            ),
                        ))
                        if self.cfg.object_slot_count > 0:
                            slot_output = self.object_slots(image_tp1[reappeared_active])
                            slot_distractor_position = (
                                batch["distractor_position_tp1"].to(device=image_t.device, dtype=image_t.dtype)[
                                    reappeared_active
                                ]
                                if "distractor_position_tp1" in batch
                                else None
                            )
                            diagnostics.update(_prefix_metrics(
                                "reappeared_object_slot_",
                                _object_slot_position_metrics(
                                    slot_output.state.to(image_t.dtype),
                                    slot_output.position.to(image_t.dtype),
                                    slot_output.occupancy.to(image_t.dtype),
                                    slot_output.valid,
                                    target_position_for_reappeared,
                                    slot_distractor_position,
                                    self.cfg,
                                ),
                            ))
                            if self.cfg.object_slot_ternary_diagnostics and slot_output.local_images.numel() > 0:
                                slot_count = slot_output.local_images.shape[1]
                                flat_slots = slot_output.local_images.reshape(
                                    -1,
                                    image_tp1.shape[1],
                                    image_tp1.shape[2],
                                    image_tp1.shape[3],
                                )
                                slot_dataset_id = (
                                    target_dataset_id.repeat_interleave(slot_count)
                                    if target_dataset_id is not None
                                    else None
                                )
                                slot_ternary = self._diagnostic_ternary_for_image(
                                    flat_slots,
                                    dataset_id=slot_dataset_id,
                                ).view(slot_output.local_images.shape[0], slot_count, -1)
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_object_slot_",
                                    _slot_ternary_metrics(slot_ternary.to(image_t.dtype), slot_output.valid),
                                ))
                    oracle_position_candidates = None
                    predicted_position_candidates = None
                    feature_only_candidates = None
                    active_candidates = None
                    if target_position_for_reappeared is not None:
                        file_valid = (
                            memory_read.position_valid[reappeared_active]
                            & memory_read.hit[reappeared_active]
                            & (
                                memory_read.age[reappeared_active].view(-1)
                                <= self.cfg.active_file_candidate_max_age
                            )
                        )
                        oracle_position_candidates = _active_file_candidate_mask(
                            memory_read.position[reappeared_active],
                            target_position_for_reappeared,
                            memory_read.position_valid[reappeared_active],
                            memory_read.hit[reappeared_active],
                            memory_read.age[reappeared_active],
                            self.cfg.active_file_candidate_radius,
                            self.cfg.active_file_candidate_max_age,
                            _active_file_wrap_span(self.cfg),
                        )
                        if dynamics_position is not None:
                            predicted_position_candidates = _active_file_candidate_mask(
                                memory_read.position[reappeared_active],
                                dynamics_position.detach(),
                                memory_read.position_valid[reappeared_active],
                                memory_read.hit[reappeared_active],
                                memory_read.age[reappeared_active],
                                self.cfg.active_file_candidate_radius,
                                self.cfg.active_file_candidate_max_age,
                                _active_file_wrap_span(self.cfg),
                            )
                        feature_only_candidates = _active_file_feature_only_candidate_mask(
                            file_valid,
                            int(reappeared_active.to(torch.long).sum().item()),
                            image_t.dtype,
                            image_t.device,
                        )
                        active_candidates = (
                            predicted_position_candidates
                            if predicted_position_candidates is not None
                            else feature_only_candidates
                        )
                    local_prediction_error = None
                    if feature_only_candidates is not None and file_binding_position.numel() > 0:
                        query_count = int(reappeared_active.to(torch.long).sum().item())
                        file_count = min(query_count, file_binding_position.shape[0])
                        if file_count > 0:
                            local_images = _local_reappearance_images(
                                image_tp1[reappeared_active][:query_count],
                                file_binding_position.detach()[:file_count],
                                self.cfg,
                            )
                            if local_images.numel() > 0:
                                local_dataset_id = (
                                    target_dataset_id[:query_count].repeat_interleave(file_count)
                                    if target_dataset_id is not None
                                    else None
                                )
                                local_definition_scores, _local_context, _local_binding_position = (
                                    self._diagnostic_definition_state_for_image(
                                        local_images,
                                        dataset_id=local_dataset_id,
                                        include_binding_position=True,
                                    )
                                )
                                local_query_scores = self.defs.file_query_scores(local_definition_scores)
                                repeated_position = file_binding_position.detach()[:file_count].unsqueeze(0).expand(
                                    query_count,
                                    file_count,
                                    2,
                                ).reshape(query_count * file_count, 2)
                                local_query_binding_scores = _binding_position_features(
                                    local_query_scores.to(image_t.dtype),
                                    _normalized_pixel_position(repeated_position, self.cfg),
                                    self.cfg,
                                )
                                expected_local = expected_query_binding_scores[:file_count].unsqueeze(0).expand(
                                    query_count,
                                    file_count,
                                    -1,
                                ).reshape(query_count * file_count, -1)
                                local_prediction_error = (
                                    local_query_binding_scores - expected_local.to(local_query_binding_scores.dtype)
                                ).square().mean(dim=-1).view(query_count, file_count)
                                diagnostics["reappeared_local_prediction_error_mean"] = local_prediction_error.mean()
                    instance_ids = _object_instance_ids(batch, image_t.device)
                    if instance_ids is not None:
                        source_sequences = instance_ids[reappeared_active]
                        if (
                            slot_output is not None
                            and target_position_for_reappeared is not None
                            and dynamics_position is not None
                            and dynamics_valid is not None
                        ):
                            file_slot_valid = (
                                memory_read.position_valid[reappeared_active]
                                & memory_read.hit[reappeared_active]
                                & dynamics_valid.to(device=image_t.device, dtype=torch.bool)
                                & (
                                    memory_read.age[reappeared_active].view(-1)
                                    <= self.cfg.active_file_candidate_max_age
                                )
                            )
                            distractor_sequences = None
                            if "track_id" in batch:
                                reappeared_tracks = batch["track_id"].to(device=image_t.device, dtype=torch.long)[
                                    reappeared_active
                                ]
                                source_sequences_long = source_sequences.to(device=image_t.device, dtype=torch.long)
                                distractor_sequences = torch.where(
                                    reappeared_tracks == 0,
                                    source_sequences_long + 1,
                                    source_sequences_long - 1,
                                )
                            diagnostics.update(_prefix_metrics(
                                "reappeared_file_slot_",
                                _file_slot_assignment_metrics(
                                    dynamics_position.detach().to(image_t.dtype),
                                    file_slot_valid,
                                    slot_output.position.to(image_t.dtype),
                                    slot_output.valid,
                                    target_position_for_reappeared,
                                    source_sequences,
                                    source_sequences,
                                    source_labels,
                                    self.cfg,
                                    distractor_positions=slot_distractor_position,
                                    distractor_instance_labels=distractor_sequences,
                                ),
                            ))
                            diagnostics.update(_prefix_metrics(
                                "reappeared_file_slot_",
                                _candidate_path_context_metrics(diagnostics, image_t),
                            ))
                            diagnostics.update(_prefix_metrics(
                                "reappeared_oracle_position_global_file_slot_",
                                _file_slot_assignment_metrics(
                                    target_position_for_reappeared.detach().to(image_t.dtype),
                                    file_slot_valid,
                                    slot_output.position.to(image_t.dtype),
                                    slot_output.valid,
                                    target_position_for_reappeared,
                                    source_sequences,
                                    source_sequences,
                                    source_labels,
                                    self.cfg,
                                    distractor_positions=slot_distractor_position,
                                    distractor_instance_labels=distractor_sequences,
                                ),
                            ))
                            diagnostics.update(_prefix_metrics(
                                "reappeared_oracle_position_global_file_slot_",
                                _candidate_path_context_metrics(diagnostics, image_t),
                            ))
                            diagnostics.update(_prefix_metrics(
                                "reappeared_oracle_position_ceiling_file_slot_",
                                _oracle_pair_file_slot_ceiling_metrics(
                                    slot_output.position.to(image_t.dtype),
                                    slot_output.valid,
                                    target_position_for_reappeared,
                                    source_sequences,
                                    source_labels,
                                    self.cfg,
                                    slot_distractor_position,
                                    distractor_sequences,
                                ),
                            ))
                            diagnostics.update(_prefix_metrics(
                                "reappeared_oracle_position_ceiling_file_slot_",
                                _candidate_path_context_metrics(diagnostics, image_t),
                            ))
                            diagnostics.update(_prefix_metrics(
                                "reappeared_oracle_noise_file_slot_",
                                _oracle_pair_file_slot_noise_sweep_metrics(
                                    slot_output.position.to(image_t.dtype),
                                    slot_output.valid,
                                    target_position_for_reappeared,
                                    source_sequences,
                                    source_labels,
                                    self.cfg,
                                    slot_distractor_position,
                                    distractor_sequences,
                                ),
                            ))
                            if ballistic_position is not None and ballistic_valid is not None:
                                ballistic_file_slot_valid = (
                                    memory_read.position_valid[reappeared_active]
                                    & memory_read.hit[reappeared_active]
                                    & ballistic_valid.to(device=image_t.device, dtype=torch.bool)
                                    & (
                                        memory_read.age[reappeared_active].view(-1)
                                        <= self.cfg.active_file_candidate_max_age
                                    )
                                )
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_ballistic_file_slot_",
                                    _file_slot_assignment_metrics(
                                        ballistic_position.detach().to(image_t.dtype),
                                        ballistic_file_slot_valid,
                                        slot_output.position.to(image_t.dtype),
                                        slot_output.valid,
                                        target_position_for_reappeared,
                                        source_sequences,
                                        source_sequences,
                                        source_labels,
                                        self.cfg,
                                        distractor_positions=slot_distractor_position,
                                        distractor_instance_labels=distractor_sequences,
                                    ),
                                ))
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_ballistic_file_slot_",
                                    _candidate_path_context_metrics(diagnostics, image_t),
                                ))
                            file_slot_candidate_paths: list[tuple[str, torch.Tensor, torch.Tensor]] = []
                            if active_candidates is not None:
                                file_slot_candidate_paths.append((
                                    "reappeared_active_file_slot_",
                                    dynamics_position.detach().to(image_t.dtype),
                                    active_candidates,
                                ))
                            if oracle_position_candidates is not None:
                                file_slot_candidate_paths.append((
                                    "reappeared_oracle_position_file_slot_",
                                    target_position_for_reappeared.detach().to(image_t.dtype),
                                    oracle_position_candidates,
                                ))
                            if predicted_position_candidates is not None:
                                file_slot_candidate_paths.append((
                                    "reappeared_predicted_position_file_slot_",
                                    dynamics_position.detach().to(image_t.dtype),
                                    predicted_position_candidates,
                                ))
                            if feature_only_candidates is not None:
                                file_slot_candidate_paths.append((
                                    "reappeared_feature_only_file_slot_",
                                    dynamics_position.detach().to(image_t.dtype),
                                    feature_only_candidates,
                                ))
                            for file_slot_prefix, file_slot_positions, file_slot_candidates in file_slot_candidate_paths:
                                diagnostics.update(_prefix_metrics(
                                    file_slot_prefix,
                                    _file_slot_assignment_metrics(
                                        file_slot_positions,
                                        file_slot_valid,
                                        slot_output.position.to(image_t.dtype),
                                        slot_output.valid,
                                        target_position_for_reappeared,
                                        source_sequences,
                                        source_sequences,
                                        source_labels,
                                        self.cfg,
                                        distractor_positions=slot_distractor_position,
                                        distractor_instance_labels=distractor_sequences,
                                        candidate_mask=file_slot_candidates,
                                    ),
                                ))
                                diagnostics.update(_prefix_metrics(
                                    file_slot_prefix,
                                    _candidate_path_context_metrics(diagnostics, image_t),
                                ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_file_",
                            grouped_instance_match_diagnostics(
                                source_definition_scores.to(image_t.dtype),
                                object_file_scores.to(image_t.dtype),
                                source_sequences,
                                source_sequences,
                                source_labels,
                                source_labels,
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_target_file_",
                            grouped_instance_match_diagnostics(
                                target_definition_scores.to(image_t.dtype),
                                target_object_file_scores.to(image_t.dtype),
                                source_sequences,
                                source_sequences,
                                source_labels,
                                source_labels,
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_query_file_",
                            grouped_instance_match_diagnostics(
                                target_query_scores.to(image_t.dtype),
                                target_object_file_scores.to(image_t.dtype),
                                source_sequences,
                                source_sequences,
                                source_labels,
                                source_labels,
                            ),
                        ))
                        if expected_query_scores is not None:
                            diagnostics.update(_prefix_metrics(
                                "reappeared_expected_file_",
                                grouped_instance_match_diagnostics(
                                    expected_query_scores.to(image_t.dtype),
                                    target_query_scores.to(image_t.dtype),
                                    source_sequences,
                                    source_sequences,
                                    source_labels,
                                    source_labels,
                                ),
                            ))
                        if expected_state_scores is not None:
                            diagnostics.update(_prefix_metrics(
                                "reappeared_expected_state_",
                                grouped_instance_match_diagnostics(
                                    expected_state_scores.to(image_t.dtype),
                                    target_definition_scores.to(image_t.dtype),
                                    source_sequences,
                                    source_sequences,
                                    source_labels,
                                    source_labels,
                                ),
                            ))
                        if active_candidates is not None:
                            candidate_paths = [
                                ("reappeared_active_query_file_", active_candidates),
                            ]
                            if oracle_position_candidates is not None:
                                candidate_paths.append((
                                    "reappeared_oracle_position_query_file_",
                                    oracle_position_candidates,
                                ))
                            if predicted_position_candidates is not None:
                                candidate_paths.append((
                                    "reappeared_predicted_position_query_file_",
                                    predicted_position_candidates,
                                ))
                            if feature_only_candidates is not None:
                                candidate_paths.append((
                                    "reappeared_feature_only_query_file_",
                                    feature_only_candidates,
                                ))
                            for prefix, candidates in candidate_paths:
                                diagnostics.update(_prefix_metrics(
                                    prefix,
                                    candidate_instance_match_diagnostics(
                                        target_query_binding_scores.to(image_t.dtype),
                                        target_object_file_binding_scores.to(image_t.dtype),
                                        source_sequences,
                                        source_sequences,
                                        source_labels,
                                        source_labels,
                                        candidates,
                                    ),
                                ))
                                diagnostics.update(_prefix_metrics(
                                    prefix,
                                    _candidate_path_context_metrics(diagnostics, image_t),
                                ))
                                if state_prediction_error.numel() > 0:
                                    diagnostics.update(_prefix_metrics(
                                        prefix.replace("query_file_", "state_prediction_error_query_file_"),
                                        candidate_error_match_diagnostics(
                                            state_prediction_error.to(image_t.dtype),
                                            source_sequences,
                                            source_sequences,
                                            source_labels,
                                            source_labels,
                                            candidates,
                                        ),
                                    ))
                                    diagnostics.update(_prefix_metrics(
                                        prefix.replace("query_file_", "state_prediction_error_query_file_"),
                                        _candidate_path_context_metrics(diagnostics, image_t),
                                    ))
                                if local_prediction_error is not None:
                                    diagnostics.update(_prefix_metrics(
                                        prefix.replace("query_file_", "local_prediction_error_query_file_"),
                                        candidate_error_match_diagnostics(
                                            local_prediction_error.to(image_t.dtype),
                                            source_sequences,
                                            source_sequences,
                                            source_labels,
                                            source_labels,
                                            candidates,
                                        ),
                                    ))
                                    diagnostics.update(_prefix_metrics(
                                        prefix.replace("query_file_", "local_prediction_error_query_file_"),
                                        _candidate_path_context_metrics(diagnostics, image_t),
                                    ))
                            if feature_only_candidates is not None:
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_feature_only_position_ablated_query_file_",
                                    candidate_instance_match_diagnostics(
                                        target_query_scores.to(image_t.dtype),
                                        target_object_file_scores.to(image_t.dtype),
                                        source_sequences,
                                        source_sequences,
                                        source_labels,
                                        source_labels,
                                        feature_only_candidates,
                                    ),
                                ))
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_feature_only_position_ablated_query_file_",
                                    _candidate_path_context_metrics(diagnostics, image_t),
                                ))
                            with torch.no_grad():
                                learned_logits = _active_file_gate_logits(
                                    self.active_file_gate,
                                    target_query_scores,
                                    target_object_file_scores,
                                    memory_definition_confidence[reappeared_active],
                                    memory_read.age[reappeared_active],
                                    self.cfg.active_file_candidate_max_age,
                                    target_context_probs
                                    if self.cfg.learned_active_file_gate_context_features
                                    else None,
                                    context.probs[reappeared_active]
                                    if self.cfg.learned_active_file_gate_context_features
                                    else None,
                                    expected_query_scores
                                    if self.cfg.learned_active_file_gate_expectation_features
                                    else None,
                                )
                                file_valid = (
                                    memory_read.position_valid[reappeared_active]
                                    & memory_read.hit[reappeared_active]
                                    & (
                                        memory_read.age[reappeared_active].view(-1)
                                        <= self.cfg.active_file_candidate_max_age
                                    )
                                )
                                learned_candidates = _learned_candidate_mask(
                                    learned_logits,
                                    file_valid,
                                    self.cfg.learned_active_file_gate_topk,
                                    self.cfg.learned_active_file_gate_threshold,
                                )
                            diagnostics.update(_prefix_metrics(
                                "reappeared_learned_active_query_file_",
                                candidate_instance_match_diagnostics(
                                    target_query_scores.to(image_t.dtype),
                                    target_object_file_scores.to(image_t.dtype),
                                    source_sequences,
                                    source_sequences,
                                    source_labels,
                                    source_labels,
                                    learned_candidates,
                                ),
                            ))
                            if (
                                slot_output is not None
                                and target_position_for_reappeared is not None
                                and dynamics_position is not None
                                and dynamics_valid is not None
                            ):
                                learned_file_slot_valid = (
                                    memory_read.position_valid[reappeared_active]
                                    & memory_read.hit[reappeared_active]
                                    & dynamics_valid.to(device=image_t.device, dtype=torch.bool)
                                    & (
                                        memory_read.age[reappeared_active].view(-1)
                                        <= self.cfg.active_file_candidate_max_age
                                    )
                                )
                                learned_distractor_sequences = None
                                if "track_id" in batch:
                                    reappeared_tracks = batch["track_id"].to(device=image_t.device, dtype=torch.long)[
                                        reappeared_active
                                    ]
                                    source_sequences_long = source_sequences.to(device=image_t.device, dtype=torch.long)
                                    learned_distractor_sequences = torch.where(
                                        reappeared_tracks == 0,
                                        source_sequences_long + 1,
                                        source_sequences_long - 1,
                                    )
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_learned_active_file_slot_",
                                    _file_slot_assignment_metrics(
                                        dynamics_position.detach().to(image_t.dtype),
                                        learned_file_slot_valid,
                                        slot_output.position.to(image_t.dtype),
                                        slot_output.valid,
                                        target_position_for_reappeared,
                                        source_sequences,
                                        source_sequences,
                                        source_labels,
                                        self.cfg,
                                        distractor_positions=slot_distractor_position,
                                        distractor_instance_labels=learned_distractor_sequences,
                                        candidate_mask=learned_candidates,
                                    ),
                                ))
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_learned_active_file_slot_",
                                    _candidate_path_context_metrics(diagnostics, image_t),
                                ))
                            if state_prediction_error.numel() > 0:
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_learned_active_state_prediction_error_query_file_",
                                    candidate_error_match_diagnostics(
                                        state_prediction_error.to(image_t.dtype),
                                        source_sequences,
                                        source_sequences,
                                        source_labels,
                                        source_labels,
                                        learned_candidates,
                                    ),
                                ))
                            if local_prediction_error is not None:
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_learned_active_local_prediction_error_query_file_",
                                    candidate_error_match_diagnostics(
                                        local_prediction_error.to(image_t.dtype),
                                        source_sequences,
                                        source_sequences,
                                        source_labels,
                                        source_labels,
                                        learned_candidates,
                                    ),
                                ))
                            diagnostics.update(_prefix_metrics(
                                "reappeared_learned_active_file_gate_active_",
                                _candidate_mask_agreement(learned_candidates, active_candidates),
                            ))
                            if oracle_position_candidates is not None:
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_learned_active_file_gate_oracle_position_",
                                    _candidate_mask_agreement(learned_candidates, oracle_position_candidates),
                                ))
                            if predicted_position_candidates is not None:
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_learned_active_file_gate_predicted_position_",
                                    _candidate_mask_agreement(learned_candidates, predicted_position_candidates),
                                ))
                            if feature_only_candidates is not None:
                                diagnostics.update(_prefix_metrics(
                                    "reappeared_learned_active_file_gate_feature_only_",
                                    _candidate_mask_agreement(learned_candidates, feature_only_candidates),
                                ))
                    diagnostics["reappeared_memory_definition_match_delta"] = (
                        diagnostics["reappeared_feature_match_accuracy"]
                        - diagnostics["reappeared_base_feature_match_accuracy"]
                    )
                    diagnostics["reappeared_paired_memory_definition_match_delta"] = (
                        diagnostics["reappeared_paired_feature_match_accuracy"]
                        - diagnostics["reappeared_base_paired_feature_match_accuracy"]
                    )
                elif include_label_diagnostics:
                    diagnostics.update({
                        "reappeared_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_file_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_file_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_expected_file_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_expected_file_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_expected_file_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_file_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_file_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_expected_state_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_expected_state_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_expected_state_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_state_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_expected_state_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_trajectory_position_error": _zero_like_scalar(image_t),
                        "reappeared_trajectory_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_dynamics_position_error": _zero_like_scalar(image_t),
                        "reappeared_dynamics_position_improvement": _zero_like_scalar(image_t),
                        "reappeared_dynamics_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_mean_count": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_target_present_fraction": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_row_coverage_fraction": _zero_like_scalar(image_t),
                        "reappeared_active_query_file_candidate_target_recall_fraction": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_mean_count": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_target_present_fraction": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_row_coverage_fraction": _zero_like_scalar(image_t),
                        "reappeared_learned_active_query_file_candidate_target_recall_fraction": _zero_like_scalar(image_t),
                        "reappeared_base_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_base_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_base_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_base_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_memory_definition_match_delta": _zero_like_scalar(image_t),
                        "reappeared_paired_memory_definition_match_delta": _zero_like_scalar(image_t),
                    })
        total = (
            self.cfg.recon_weight * losses["reconstruction"]
            + self.cfg.pred_weight * losses["prediction"]
            + self.cfg.free_energy_weight * losses["free_energy"]
            + self.cfg.complexity_weight * losses["complexity"]
            + self.cfg.context_entropy_weight * losses["context_entropy"]
            + self.cfg.context_balance_weight * losses["context_balance"]
            + self.cfg.ternary_activation_weight * losses["ternary_activation_l1"]
            + self.cfg.bit_cost_weight * losses["bit_cost"]
            + self.cfg.reappearance_alignment_weight * losses["reappearance_alignment"]
            + self.cfg.object_cycle_weight * losses["object_cycle_consistency"]
            + self.cfg.reappearance_file_query_weight * losses["reappearance_file_query"]
            + self.cfg.active_file_query_weight * losses["active_file_query"]
            + self.cfg.active_file_expectation_weight * losses["active_file_expectation"]
            + self.cfg.active_file_dynamics_weight * losses["active_file_dynamics"]
            + self.cfg.learned_active_file_gate_weight * losses["learned_active_file_gate"]
        )
        self.memory.write(sae.eps, sae.coherence)
        for candidate in self.evidence.accumulate(delta_q, sae.coherence):
            self.gate.consider(candidate)
        self.trauma.scan(sae.eps, self.gate.rejections)
        return TrainOutput(
            total_loss=total,
            losses=losses,
            diagnostics=diagnostics,
            recon_image=recon,
            next_image=next_image,
            context=context,
            sae=sae,
            ternary=ternary,
            latent_state=q,
            memory_feature=memory_feature,
            memory_confidence=memory_prediction_confidence,
        )

    @torch.no_grad()
    def ternary_prediction_impacts(self, output: TrainOutput, image_tp1: torch.Tensor) -> torch.Tensor:
        axis_count = output.ternary.shape[-1]
        if not self.cfg.use_ternary_conditioning:
            return torch.zeros(axis_count, dtype=output.ternary.dtype, device=output.ternary.device)
        base_error = (output.next_image.detach() - image_tp1).square().flatten(1).mean(dim=1)
        impacts: list[torch.Tensor] = []
        latent = output.latent_state.detach()
        context_embedding = output.context.embedding.detach()
        ternary = output.ternary.detach()
        for axis in range(axis_count):
            ablated = ternary.clone()
            ablated[:, axis] = 0
            prediction = self.reality.predict_next_image(
                latent,
                context_embedding,
                ablated,
                output.memory_feature,
                output.memory_confidence,
            )
            ablated_error = (prediction - image_tp1).square().flatten(1).mean(dim=1)
            impacts.append((ablated_error - base_error).clamp_min(0.0).mean())
        return torch.stack(impacts) if impacts else torch.zeros(0, dtype=output.ternary.dtype, device=output.ternary.device)

    @torch.no_grad()
    def tick(self, raw_inputs: torch.Tensor | dict[str, torch.Tensor]) -> TickOutput:
        image = raw_inputs["image_t"] if isinstance(raw_inputs, dict) else raw_inputs
        batch = {"image_t": image, "image_tp1": image}
        out = self.forward_train(batch)
        return TickOutput(
            reconstruction=out.recon_image,
            next_prediction=out.next_image,
            context_probs=out.context.probs,
            sae=out.sae,
            ternary=out.ternary,
            action=None,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> TrainOutput:
        return self.forward_train(batch)
