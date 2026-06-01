from __future__ import annotations

from typing import Any

import torch


def _zero(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=dtype, device=device)


def _discrete_mutual_information(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = x.device
    if x.numel() == 0 or y.numel() == 0:
        return _zero(dtype, device)
    x = x.to(torch.long)
    y = y.to(torch.long)
    n = torch.tensor(float(x.numel()), dtype=dtype, device=device)
    mi = _zero(dtype, device)
    for x_value in x.unique():
        x_mask = x == x_value
        px = x_mask.to(dtype).sum() / n
        for y_value in y.unique():
            y_mask = y == y_value
            py = y_mask.to(dtype).sum() / n
            pxy = (x_mask & y_mask).to(dtype).sum() / n
            if bool((pxy > 0).item()):
                mi = mi + pxy * torch.log(pxy / (px * py).clamp_min(1e-8))
    return mi


def _axis_mutual_information(states: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if states.numel() == 0 or labels.numel() == 0:
        zero = _zero(dtype, states.device)
        return zero, zero
    values = [_discrete_mutual_information(states[:, axis], labels, dtype) for axis in range(states.shape[1])]
    stacked = torch.stack(values) if values else _zero(dtype, states.device).view(1)
    return stacked.mean(), stacked.max()


def _nearest_centroid_probe(features: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = features.device
    unique_labels = labels.unique(sorted=True)
    if features.numel() == 0 or unique_labels.numel() < 2:
        return _zero(dtype, device)
    centroids = torch.stack([features[labels == label].mean(dim=0) for label in unique_labels])
    distances = (features.unsqueeze(1) - centroids.unsqueeze(0)).square().mean(dim=-1)
    pred = unique_labels[distances.argmin(dim=1)]
    return (pred == labels).to(dtype).mean()


def _per_mode_nonzero(nonzero: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    device = nonzero.device
    values = [nonzero[labels == label].to(dtype).mean() for label in labels.unique(sorted=True)]
    if not values:
        zero = _zero(dtype, device)
        return {
            "ternary_per_mode_nonzero_fraction": zero,
            "ternary_per_mode_nonzero_std": zero,
            "ternary_per_mode_nonzero_min": zero,
            "ternary_per_mode_nonzero_max": zero,
        }
    stacked = torch.stack(values)
    std = stacked.std(unbiased=False) if stacked.numel() > 1 else _zero(dtype, device)
    return {
        "ternary_per_mode_nonzero_fraction": stacked.mean(),
        "ternary_per_mode_nonzero_std": std,
        "ternary_per_mode_nonzero_min": stacked.min(),
        "ternary_per_mode_nonzero_max": stacked.max(),
    }


def _axis_stability(signs: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = signs.device
    values: list[torch.Tensor] = []
    for axis in range(signs.shape[1]):
        axis_sign = signs[:, axis]
        for label in labels.unique(sorted=True):
            selected = axis_sign[labels == label]
            active = selected != 0
            if bool(active.any().item()):
                active_sign = selected[active]
                pos = (active_sign > 0).to(dtype).sum()
                neg = (active_sign < 0).to(dtype).sum()
                values.append(torch.maximum(pos, neg) / active.to(dtype).sum())
    if not values:
        return _zero(dtype, device)
    return torch.stack(values).mean()


def ternary_label_diagnostics(
    ternary: torch.Tensor,
    labels: torch.Tensor,
    context_hard: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dtype = ternary.dtype
    device = ternary.device
    labels = labels.to(device=device, dtype=torch.long)
    context_hard = context_hard.to(device=device, dtype=torch.long)
    valid = labels >= 0
    if not bool(valid.any().item()):
        zero = _zero(dtype, device)
        return {
            "ternary_mode_mutual_information": zero,
            "ternary_mode_max_mutual_information": zero,
            "ternary_context_mutual_information": zero,
            "ternary_context_max_mutual_information": zero,
            "ternary_mode_probe_accuracy": zero,
            "ternary_per_mode_nonzero_fraction": zero,
            "ternary_per_mode_nonzero_std": zero,
            "ternary_per_mode_nonzero_min": zero,
            "ternary_per_mode_nonzero_max": zero,
            "ternary_axis_usage_count": zero,
            "ternary_axis_usage_fraction": zero,
            "ternary_always_on_axis_fraction": zero,
            "ternary_axis_stability": zero,
        }

    ternary = ternary.detach()[valid]
    labels = labels[valid]
    context_hard = context_hard[valid]
    signs = ternary.sign()
    states = torch.where(signs > 0, torch.full_like(signs, 2), torch.where(signs < 0, torch.zeros_like(signs), torch.ones_like(signs)))
    states = states.to(torch.long)
    nonzero = signs != 0
    axis_nonzero_fraction = nonzero.to(dtype).mean(dim=0)
    mode_mi, mode_max_mi = _axis_mutual_information(states, labels, dtype)
    ctx_mi, ctx_max_mi = _axis_mutual_information(states, context_hard, dtype)
    metrics = {
        "ternary_mode_mutual_information": mode_mi,
        "ternary_mode_max_mutual_information": mode_max_mi,
        "ternary_context_mutual_information": ctx_mi,
        "ternary_context_max_mutual_information": ctx_max_mi,
        "ternary_mode_probe_accuracy": _nearest_centroid_probe(signs.to(dtype), labels, dtype),
        "ternary_axis_usage_count": axis_nonzero_fraction.gt(0).to(dtype).sum(),
        "ternary_axis_usage_fraction": axis_nonzero_fraction.gt(0).to(dtype).mean(),
        "ternary_always_on_axis_fraction": axis_nonzero_fraction.ge(0.95).to(dtype).mean(),
        "ternary_axis_stability": _axis_stability(signs, labels, dtype),
    }
    metrics.update(_per_mode_nonzero(nonzero, labels, dtype))
    return metrics


def feature_label_diagnostics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    dtype = features.dtype
    device = features.device
    labels = labels.to(device=device, dtype=torch.long)
    valid = labels >= 0
    if features.numel() == 0 or not bool(valid.any().item()):
        zero = _zero(dtype, device)
        return {
            "feature_probe_accuracy": zero,
            "feature_centroid_separation": zero,
            "feature_label_count": zero,
        }
    features = features.detach()[valid]
    labels = labels[valid]
    unique_labels = labels.unique(sorted=True)
    if unique_labels.numel() < 2:
        zero = _zero(dtype, device)
        return {
            "feature_probe_accuracy": zero,
            "feature_centroid_separation": zero,
            "feature_label_count": torch.tensor(float(unique_labels.numel()), dtype=dtype, device=device),
        }
    centroids = torch.stack([features[labels == label].mean(dim=0) for label in unique_labels])
    distances = torch.pdist(centroids, p=2)
    return {
        "feature_probe_accuracy": _nearest_centroid_probe(features.to(dtype), labels, dtype),
        "feature_centroid_separation": distances.mean() if distances.numel() else _zero(dtype, device),
        "feature_label_count": torch.tensor(float(unique_labels.numel()), dtype=dtype, device=device),
    }


def feature_match_diagnostics(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    dtype = source_features.dtype
    device = source_features.device
    target_labels = source_labels if target_labels is None else target_labels
    source_labels = source_labels.to(device=device, dtype=torch.long)
    target_labels = target_labels.to(device=device, dtype=torch.long)
    source_valid = source_labels >= 0
    target_valid = target_labels >= 0
    if (
        source_features.numel() == 0
        or target_features.numel() == 0
        or not bool(source_valid.any().item())
        or not bool(target_valid.any().item())
    ):
        zero = _zero(dtype, device)
        return {
            "feature_match_accuracy": zero,
            "feature_match_margin": zero,
            "feature_same_distance": zero,
            "feature_nearest_other_distance": zero,
        }

    source = source_features.detach()[source_valid].to(dtype)
    source_labels = source_labels[source_valid]
    target = target_features.detach()[target_valid].to(device=device, dtype=dtype)
    target_labels = target_labels[target_valid]
    distances = (source.unsqueeze(1) - target.unsqueeze(0)).square().mean(dim=-1)
    nearest = distances.argmin(dim=1)
    pred_labels = target_labels[nearest]
    accuracy = (pred_labels == source_labels).to(dtype).mean()

    same_values: list[torch.Tensor] = []
    other_values: list[torch.Tensor] = []
    for row in range(source.shape[0]):
        same = target_labels == source_labels[row]
        other = ~same
        same_values.append(distances[row, same].min() if bool(same.any().item()) else _zero(dtype, device))
        other_values.append(distances[row, other].min() if bool(other.any().item()) else _zero(dtype, device))
    same_distance = torch.stack(same_values).mean() if same_values else _zero(dtype, device)
    other_distance = torch.stack(other_values).mean() if other_values else _zero(dtype, device)
    return {
        "feature_match_accuracy": accuracy,
        "feature_match_margin": other_distance - same_distance,
        "feature_same_distance": same_distance,
        "feature_nearest_other_distance": other_distance,
    }


def paired_feature_match_diagnostics(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dtype = source_features.dtype
    device = source_features.device
    if source_features.numel() == 0 or target_features.numel() == 0:
        zero = _zero(dtype, device)
        return {
            "paired_feature_match_accuracy": zero,
            "paired_feature_match_margin": zero,
            "paired_feature_same_distance": zero,
            "paired_feature_nearest_other_distance": zero,
        }
    pair_count = min(source_features.shape[0], target_features.shape[0])
    if pair_count < 2:
        zero = _zero(dtype, device)
        return {
            "paired_feature_match_accuracy": zero,
            "paired_feature_match_margin": zero,
            "paired_feature_same_distance": zero,
            "paired_feature_nearest_other_distance": zero,
        }
    source = source_features.detach()[:pair_count].to(dtype)
    target = target_features.detach()[:pair_count].to(device=device, dtype=dtype)
    distances = (source.unsqueeze(1) - target.unsqueeze(0)).square().mean(dim=-1)
    nearest = distances.argmin(dim=1)
    labels = torch.arange(pair_count, device=device)
    accuracy = (nearest == labels).to(dtype).mean()
    same_distance = distances[labels, labels].mean()
    other_mask = ~torch.eye(pair_count, dtype=torch.bool, device=device)
    other_distance = distances.masked_fill(~other_mask, float("inf")).min(dim=1).values.mean()
    return {
        "paired_feature_match_accuracy": accuracy,
        "paired_feature_match_margin": other_distance - same_distance,
        "paired_feature_same_distance": same_distance,
        "paired_feature_nearest_other_distance": other_distance,
    }


def grouped_instance_match_diagnostics(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_instance_labels: torch.Tensor,
    target_instance_labels: torch.Tensor,
    source_group_labels: torch.Tensor,
    target_group_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dtype = source_features.dtype
    device = source_features.device
    source_instance_labels = source_instance_labels.to(device=device, dtype=torch.long)
    target_instance_labels = target_instance_labels.to(device=device, dtype=torch.long)
    source_group_labels = source_group_labels.to(device=device, dtype=torch.long)
    target_group_labels = target_group_labels.to(device=device, dtype=torch.long)
    source_valid = (source_instance_labels >= 0) & (source_group_labels >= 0)
    target_valid = (target_instance_labels >= 0) & (target_group_labels >= 0)
    if (
        source_features.numel() == 0
        or target_features.numel() == 0
        or not bool(source_valid.any().item())
        or not bool(target_valid.any().item())
    ):
        zero = _zero(dtype, device)
        return {
            "instance_match_accuracy": zero,
            "instance_match_margin": zero,
            "instance_same_distance": zero,
            "instance_nearest_other_distance": zero,
            "instance_hard_match_accuracy": zero,
            "instance_hard_margin": zero,
            "instance_hard_same_distance": zero,
            "instance_nearest_same_group_other_distance": zero,
            "instance_hard_valid_fraction": zero,
        }

    source = source_features.detach()[source_valid].to(dtype)
    source_instances = source_instance_labels[source_valid]
    source_groups = source_group_labels[source_valid]
    target = target_features.detach()[target_valid].to(device=device, dtype=dtype)
    target_instances = target_instance_labels[target_valid]
    target_groups = target_group_labels[target_valid]
    distances = (source.unsqueeze(1) - target.unsqueeze(0)).square().mean(dim=-1)
    nearest = distances.argmin(dim=1)
    overall_accuracy = (target_instances[nearest] == source_instances).to(dtype).mean()

    same_values: list[torch.Tensor] = []
    other_values: list[torch.Tensor] = []
    hard_same_values: list[torch.Tensor] = []
    hard_other_values: list[torch.Tensor] = []
    hard_hits: list[torch.Tensor] = []
    for row in range(source.shape[0]):
        same = target_instances == source_instances[row]
        other = ~same
        if bool(same.any().item()):
            same_distance = distances[row, same].min()
        else:
            same_distance = _zero(dtype, device)
        same_values.append(same_distance)
        other_values.append(distances[row, other].min() if bool(other.any().item()) else _zero(dtype, device))

        same_group = target_groups == source_groups[row]
        hard_other = same_group & other
        hard_pool = same_group & (same | hard_other)
        if bool(same.any().item()) and bool(hard_other.any().item()) and bool(hard_pool.any().item()):
            hard_nearest = distances[row].masked_fill(~hard_pool, float("inf")).argmin()
            hard_hits.append((target_instances[hard_nearest] == source_instances[row]).to(dtype))
            hard_same_values.append(same_distance)
            hard_other_values.append(distances[row, hard_other].min())

    same_distance = torch.stack(same_values).mean() if same_values else _zero(dtype, device)
    other_distance = torch.stack(other_values).mean() if other_values else _zero(dtype, device)
    if hard_hits:
        hard_accuracy = torch.stack(hard_hits).mean()
        hard_same_distance = torch.stack(hard_same_values).mean()
        hard_other_distance = torch.stack(hard_other_values).mean()
        hard_valid_fraction = torch.tensor(
            len(hard_hits) / max(1, source.shape[0]),
            dtype=dtype,
            device=device,
        )
    else:
        hard_accuracy = _zero(dtype, device)
        hard_same_distance = _zero(dtype, device)
        hard_other_distance = _zero(dtype, device)
        hard_valid_fraction = _zero(dtype, device)

    return {
        "instance_match_accuracy": overall_accuracy,
        "instance_match_margin": other_distance - same_distance,
        "instance_same_distance": same_distance,
        "instance_nearest_other_distance": other_distance,
        "instance_hard_match_accuracy": hard_accuracy,
        "instance_hard_margin": hard_other_distance - hard_same_distance,
        "instance_hard_same_distance": hard_same_distance,
        "instance_nearest_same_group_other_distance": hard_other_distance,
        "instance_hard_valid_fraction": hard_valid_fraction,
    }


def ternary_axis_specialization(
    ternary: torch.Tensor,
    labels: torch.Tensor,
    min_usage: float = 0.0,
) -> list[dict[str, Any]]:
    labels = labels.detach().cpu().to(torch.long)
    valid = labels >= 0
    if ternary.numel() == 0 or not bool(valid.any().item()):
        return []

    signs = ternary.detach().cpu().sign()[valid]
    labels = labels[valid]
    modes = [int(label.item()) for label in labels.unique(sorted=True)]
    axes: list[dict[str, Any]] = []
    for axis in range(signs.shape[1]):
        axis_sign = signs[:, axis]
        usage_fraction = float((axis_sign != 0).to(torch.float32).mean().item())
        if usage_fraction < min_usage:
            continue
        positive_modes: list[int] = []
        negative_modes: list[int] = []
        neutral_modes: list[int] = []
        mode_fractions: dict[str, dict[str, float]] = {}
        stability_values: list[float] = []
        for mode in modes:
            selected = axis_sign[labels == mode]
            pos = float((selected > 0).to(torch.float32).mean().item())
            neg = float((selected < 0).to(torch.float32).mean().item())
            neutral = float((selected == 0).to(torch.float32).mean().item())
            mode_fractions[str(mode)] = {
                "positive": pos,
                "negative": neg,
                "neutral": neutral,
                "nonzero": pos + neg,
            }
            dominant_value = max((pos, "positive"), (neg, "negative"), (neutral, "neutral"))[1]
            if dominant_value == "positive":
                positive_modes.append(mode)
            elif dominant_value == "negative":
                negative_modes.append(mode)
            else:
                neutral_modes.append(mode)
            active = pos + neg
            if active > 0:
                stability_values.append(max(pos, neg) / active)
        axes.append(
            {
                "axis_id": axis,
                "usage_fraction": usage_fraction,
                "stability": sum(stability_values) / len(stability_values) if stability_values else 0.0,
                "positive_modes": positive_modes,
                "negative_modes": negative_modes,
                "neutral_modes": neutral_modes,
                "mode_fractions": mode_fractions,
            }
        )
    return axes
