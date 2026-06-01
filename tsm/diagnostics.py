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
