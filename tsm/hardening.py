from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .config import DefinitionHardeningConfig
from .diagnostics import ternary_axis_specialization


@dataclass
class AxisWindowEvidence:
    step: int
    axis_id: int
    usage_fraction: float
    stability: float
    mode_mutual_information: float
    prediction_impact: float
    positive_modes: list[int]
    negative_modes: list[int]
    neutral_modes: list[int]
    sparse: bool
    stable: bool
    associated: bool
    useful: bool

    @property
    def candidate_ready(self) -> bool:
        return self.sparse and self.stable and self.associated and self.useful


@dataclass
class HardenedDefinition:
    axis_id: int
    state: str
    first_step: int
    hardened_step: int | None
    evidence_windows: int
    mean_usage_fraction: float
    mean_stability: float
    mean_mode_mutual_information: float
    mean_prediction_impact: float
    positive_modes: list[int]
    negative_modes: list[int]
    neutral_modes: list[int]


def _discrete_mi(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    x = x.to(torch.long)
    y = y.to(torch.long)
    n = float(x.numel())
    mi = 0.0
    for x_value in x.unique():
        x_mask = x == x_value
        px = float(x_mask.to(torch.float32).sum().item()) / n
        for y_value in y.unique():
            y_mask = y == y_value
            py = float(y_mask.to(torch.float32).sum().item()) / n
            pxy = float((x_mask & y_mask).to(torch.float32).sum().item()) / n
            if pxy > 0.0 and px > 0.0 and py > 0.0:
                mi += pxy * torch.log(torch.tensor(pxy / (px * py))).item()
    return float(mi)


def _axis_state(signs: torch.Tensor) -> torch.Tensor:
    return torch.where(signs > 0, torch.full_like(signs, 2), torch.where(signs < 0, torch.zeros_like(signs), torch.ones_like(signs)))


def _consensus_modes(windows: list[AxisWindowEvidence], attr: str) -> list[int]:
    counts: dict[int, int] = defaultdict(int)
    for window in windows:
        for mode in getattr(window, attr):
            counts[int(mode)] += 1
    threshold = max(1, len(windows) // 2 + len(windows) % 2)
    return sorted(mode for mode, count in counts.items() if count >= threshold)


def _mean(windows: list[AxisWindowEvidence], attr: str) -> float:
    if not windows:
        return 0.0
    return sum(float(getattr(window, attr)) for window in windows) / len(windows)


class DefinitionEvidenceTracker:
    """Promotes within-run ternary axes from impressions to candidate definitions."""

    def __init__(self, cfg: DefinitionHardeningConfig, axis_count: int) -> None:
        self.cfg = cfg
        self.axis_count = axis_count
        self.history: dict[int, deque[AxisWindowEvidence]] = {
            axis: deque(maxlen=cfg.recent_window_limit) for axis in range(axis_count)
        }
        self.candidates: dict[int, HardenedDefinition] = {}
        self.hardened: dict[int, HardenedDefinition] = {}
        self.windows: list[AxisWindowEvidence] = []
        self.updates = 0

    def update(
        self,
        step: int,
        ternary: torch.Tensor,
        labels: torch.Tensor,
        prediction_impacts: torch.Tensor,
    ) -> dict[str, float]:
        self.updates += 1
        labels = labels.detach().cpu().to(torch.long)
        valid = labels >= 0
        new_candidates = 0
        new_hardened = 0
        if ternary.numel() > 0 and bool(valid.any().item()):
            signs = ternary.detach().cpu().sign()[valid]
            labels = labels[valid]
            prediction_impacts = prediction_impacts.detach().cpu().to(torch.float32)
            specializations = {
                int(axis["axis_id"]): axis
                for axis in ternary_axis_specialization(signs, labels, min_usage=0.0)
            }
            states = _axis_state(signs)
            for axis in range(min(self.axis_count, signs.shape[1])):
                spec = specializations.get(axis)
                if spec is None:
                    continue
                usage = float(spec["usage_fraction"])
                mode_mi = _discrete_mi(states[:, axis], labels)
                impact = float(prediction_impacts[axis].item()) if axis < prediction_impacts.numel() else 0.0
                window = AxisWindowEvidence(
                    step=step,
                    axis_id=axis,
                    usage_fraction=usage,
                    stability=float(spec["stability"]),
                    mode_mutual_information=mode_mi,
                    prediction_impact=impact,
                    positive_modes=list(spec["positive_modes"]),
                    negative_modes=list(spec["negative_modes"]),
                    neutral_modes=list(spec["neutral_modes"]),
                    sparse=self.cfg.min_usage <= usage <= self.cfg.max_usage,
                    stable=float(spec["stability"]) >= self.cfg.min_stability,
                    associated=mode_mi >= self.cfg.min_mode_mutual_information,
                    useful=impact > self.cfg.min_prediction_impact,
                )
                self.history[axis].append(window)
                self.windows.append(window)
                if window.candidate_ready and axis not in self.candidates:
                    self.candidates[axis] = self._build_definition(axis, "candidate", None)
                    new_candidates += 1
                recent_ready = [item for item in self.history[axis] if item.candidate_ready]
                if len(recent_ready) >= self.cfg.harden_after_windows and axis not in self.hardened:
                    hardened = self._build_definition(axis, "hardened", step)
                    self.candidates[axis] = hardened
                    self.hardened[axis] = hardened
                    new_hardened += 1
                elif axis in self.candidates and axis not in self.hardened:
                    self.candidates[axis] = self._build_definition(axis, "candidate", None)

        candidate_only = len(set(self.candidates) - set(self.hardened))
        latest_ready = [
            window
            for axis_history in self.history.values()
            for window in axis_history
            if window.candidate_ready
        ]
        return {
            "definition_update_count": float(self.updates),
            "definition_candidate_count": float(candidate_only),
            "definition_hardened_count": float(len(self.hardened)),
            "definition_new_candidate_count": float(new_candidates),
            "definition_new_hardened_count": float(new_hardened),
            "definition_ready_window_count": float(len(latest_ready)),
            "definition_mean_ready_stability": _mean(latest_ready, "stability"),
            "definition_mean_ready_mode_mi": _mean(latest_ready, "mode_mutual_information"),
            "definition_mean_ready_prediction_impact": _mean(latest_ready, "prediction_impact"),
        }

    def _build_definition(self, axis: int, state: str, hardened_step: int | None) -> HardenedDefinition:
        ready = [window for window in self.history[axis] if window.candidate_ready]
        source = ready or list(self.history[axis])
        return HardenedDefinition(
            axis_id=axis,
            state=state,
            first_step=min((window.step for window in source), default=0),
            hardened_step=hardened_step,
            evidence_windows=len(ready),
            mean_usage_fraction=_mean(source, "usage_fraction"),
            mean_stability=_mean(source, "stability"),
            mean_mode_mutual_information=_mean(source, "mode_mutual_information"),
            mean_prediction_impact=_mean(source, "prediction_impact"),
            positive_modes=_consensus_modes(source, "positive_modes"),
            negative_modes=_consensus_modes(source, "negative_modes"),
            neutral_modes=_consensus_modes(source, "neutral_modes"),
        )

    def to_dict(self) -> dict[str, Any]:
        candidates = [
            item
            for axis, item in self.candidates.items()
            if axis not in self.hardened
        ]
        definitions = list(self.candidates.values())
        return {
            "config": asdict(self.cfg),
            "axis_count": self.axis_count,
            "updates": self.updates,
            "candidates": [asdict(item) for item in sorted(candidates, key=lambda item: item.axis_id)],
            "hardened": [asdict(item) for item in sorted(self.hardened.values(), key=lambda item: item.axis_id)],
            "definitions": [asdict(item) for item in sorted(definitions, key=lambda item: item.axis_id)],
            "windows": [
                asdict(window)
                for window in self.windows
            ],
            "recent_windows": [
                asdict(window)
                for axis in sorted(self.history)
                for window in self.history[axis]
            ],
        }
