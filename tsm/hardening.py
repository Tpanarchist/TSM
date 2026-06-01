from __future__ import annotations

import math
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
    valid_modes: int
    min_valid_modes: int
    quarantined: bool = False
    counter_evidence: bool = False
    counter_reasons: list[str] | None = None

    @property
    def candidate_ready(self) -> bool:
        return (
            not self.quarantined
            and self.valid_modes >= self.min_valid_modes
            and self.sparse
            and self.stable
            and self.associated
            and self.useful
        )


@dataclass
class HardenedDefinition:
    axis_id: int
    state: str
    first_step: int
    hardened_step: int | None
    softened_step: int | None
    rejected_step: int | None
    evidence_windows: int
    counter_evidence_windows: int
    mean_usage_fraction: float
    mean_stability: float
    mean_mode_mutual_information: float
    mean_prediction_impact: float
    positive_modes: list[int]
    negative_modes: list[int]
    neutral_modes: list[int]
    soften_reason: str | None = None


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


def _counter_reasons(window: AxisWindowEvidence) -> list[str]:
    reasons: list[str] = []
    if window.valid_modes < window.min_valid_modes:
        reasons.append("insufficient_modes")
    if not window.sparse:
        reasons.append("not_sparse")
    if not window.stable:
        reasons.append("unstable")
    if not window.associated:
        reasons.append("not_associated")
    if not window.useful:
        reasons.append("not_useful")
    return reasons


def _dominant_reason(windows: list[AxisWindowEvidence]) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for window in windows:
        for reason in window.counter_reasons or []:
            counts[reason] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


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
        self.softened: dict[int, HardenedDefinition] = {}
        self.rejected: dict[int, HardenedDefinition] = {}
        self.windows: list[AxisWindowEvidence] = []
        self.events: list[dict[str, Any]] = []
        self.loss_count = 0
        self.loss_mean = 0.0
        self.loss_m2 = 0.0
        self.quarantined_updates = 0
        self.updates = 0

    def update(
        self,
        step: int,
        ternary: torch.Tensor,
        labels: torch.Tensor,
        prediction_impacts: torch.Tensor,
        prediction_loss: float | torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.updates += 1
        labels = labels.detach().cpu().to(torch.long)
        valid = labels >= 0
        valid_mode_count = int(labels[valid].unique().numel()) if bool(valid.any().item()) else 0
        quarantined = self._is_quarantined(prediction_loss)
        new_candidates = 0
        new_hardened = 0
        new_softened = 0
        new_rejected = 0
        new_rehardened = 0
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
                sparse = self.cfg.min_usage <= usage <= self.cfg.max_usage
                stable = float(spec["stability"]) >= self.cfg.min_stability
                associated = mode_mi >= self.cfg.min_mode_mutual_information
                useful = impact > self.cfg.min_prediction_impact
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
                    sparse=sparse,
                    stable=stable,
                    associated=associated,
                    useful=useful,
                    valid_modes=valid_mode_count,
                    min_valid_modes=self.cfg.min_valid_modes,
                    quarantined=quarantined,
                )
                reasons = _counter_reasons(window)
                window.counter_reasons = ["quarantined_update"] if quarantined else reasons
                window.counter_evidence = (not quarantined) and bool(reasons)
                self.history[axis].append(window)
                self.windows.append(window)
                if quarantined:
                    continue
                if axis in self.rejected and window.candidate_ready:
                    del self.rejected[axis]
                    self.candidates[axis] = self._build_definition(axis, "candidate", None)
                    new_candidates += 1
                    self._event(step, "revived", axis, "ready_after_rejection")
                if window.candidate_ready and axis not in self.candidates:
                    self.candidates[axis] = self._build_definition(axis, "candidate", None)
                    new_candidates += 1
                    self._event(step, "candidate", axis, None)
                recent_ready = [item for item in self.history[axis] if item.candidate_ready]
                recent_counter = [item for item in self.history[axis] if item.counter_evidence]
                consecutive_counter = self._consecutive_counter_count(axis)
                if window.candidate_ready and len(recent_ready) >= self.cfg.harden_after_windows and axis not in self.hardened:
                    hardened = self._build_definition(axis, "hardened", step)
                    self.candidates[axis] = hardened
                    self.hardened[axis] = hardened
                    if axis in self.softened:
                        del self.softened[axis]
                        new_rehardened += 1
                        self._event(step, "rehardened", axis, None)
                    else:
                        new_hardened += 1
                        self._event(step, "hardened", axis, None)
                if window.counter_evidence and axis in self.hardened and consecutive_counter >= self.cfg.soften_after_windows:
                    softened = self._build_definition(axis, "softened", self.hardened[axis].hardened_step)
                    self.candidates[axis] = softened
                    self.softened[axis] = softened
                    del self.hardened[axis]
                    new_softened += 1
                    self._event(step, "softened", axis, softened.soften_reason)
                if window.counter_evidence and axis in self.softened and consecutive_counter >= self.cfg.reject_after_windows:
                    rejected = self._build_definition(axis, "rejected", self.softened[axis].hardened_step)
                    self.rejected[axis] = rejected
                    self.candidates.pop(axis, None)
                    self.softened.pop(axis, None)
                    new_rejected += 1
                    self._event(step, "rejected", axis, rejected.soften_reason)
                if axis in self.hardened:
                    self.candidates[axis] = self._build_definition(axis, "hardened", self.hardened[axis].hardened_step)
                elif axis in self.softened:
                    self.candidates[axis] = self._build_definition(axis, "softened", self.softened[axis].hardened_step)
                elif axis in self.candidates and axis not in self.rejected:
                    self.candidates[axis] = self._build_definition(axis, "candidate", None)

        candidate_only = sum(1 for item in self.candidates.values() if item.state == "candidate")
        latest_ready = [
            window
            for axis_history in self.history.values()
            for window in axis_history
            if window.candidate_ready
        ]
        latest_counter = [
            window
            for axis_history in self.history.values()
            for window in axis_history
            if window.counter_evidence
        ]
        return {
            "definition_update_count": float(self.updates),
            "definition_candidate_count": float(candidate_only),
            "definition_hardened_count": float(len(self.hardened)),
            "definition_softened_count": float(len(self.softened)),
            "definition_rejected_count": float(len(self.rejected)),
            "definition_new_candidate_count": float(new_candidates),
            "definition_new_hardened_count": float(new_hardened),
            "definition_new_rehardened_count": float(new_rehardened),
            "definition_new_softened_count": float(new_softened),
            "definition_new_rejected_count": float(new_rejected),
            "definition_ready_window_count": float(len(latest_ready)),
            "definition_counter_window_count": float(len(latest_counter)),
            "definition_quarantined_update_count": float(self.quarantined_updates),
            "definition_mean_ready_stability": _mean(latest_ready, "stability"),
            "definition_mean_ready_mode_mi": _mean(latest_ready, "mode_mutual_information"),
            "definition_mean_ready_prediction_impact": _mean(latest_ready, "prediction_impact"),
        }

    def _build_definition(self, axis: int, state: str, hardened_step: int | None) -> HardenedDefinition:
        ready = [window for window in self.history[axis] if window.candidate_ready]
        counter = [window for window in self.history[axis] if window.counter_evidence]
        source = ready or list(self.history[axis])
        return HardenedDefinition(
            axis_id=axis,
            state=state,
            first_step=min((window.step for window in source), default=0),
            hardened_step=hardened_step,
            softened_step=max((window.step for window in counter), default=None) if state in {"softened", "rejected"} else None,
            rejected_step=max((window.step for window in counter), default=None) if state == "rejected" else None,
            evidence_windows=len(ready),
            counter_evidence_windows=len(counter),
            mean_usage_fraction=_mean(source, "usage_fraction"),
            mean_stability=_mean(source, "stability"),
            mean_mode_mutual_information=_mean(source, "mode_mutual_information"),
            mean_prediction_impact=_mean(source, "prediction_impact"),
            positive_modes=_consensus_modes(source, "positive_modes"),
            negative_modes=_consensus_modes(source, "negative_modes"),
            neutral_modes=_consensus_modes(source, "neutral_modes"),
            soften_reason=_dominant_reason(counter) if state in {"softened", "rejected"} else None,
        )

    def _event(self, step: int, event: str, axis: int, reason: str | None) -> None:
        self.events.append({"step": step, "event": event, "axis_id": axis, "reason": reason})

    def _consecutive_counter_count(self, axis: int) -> int:
        count = 0
        for window in reversed(self.history[axis]):
            if not window.counter_evidence:
                break
            count += 1
        return count

    def _is_quarantined(self, prediction_loss: float | torch.Tensor | None) -> bool:
        if prediction_loss is None:
            return False
        loss = float(prediction_loss.detach().cpu()) if torch.is_tensor(prediction_loss) else float(prediction_loss)
        if self.loss_count < self.cfg.min_loss_history:
            self._update_loss_stats(loss)
            return False
        variance = self.loss_m2 / max(1, self.loss_count - 1)
        std = math.sqrt(max(0.0, variance))
        fallback = max(abs(self.loss_mean) * 0.1, 1e-8)
        scale = max(std, fallback)
        quarantined = loss > self.loss_mean + self.cfg.quarantine_loss_z * scale
        if quarantined:
            self.quarantined_updates += 1
            return True
        self._update_loss_stats(loss)
        return False

    def _update_loss_stats(self, loss: float) -> None:
        self.loss_count += 1
        delta = loss - self.loss_mean
        self.loss_mean += delta / self.loss_count
        self.loss_m2 += delta * (loss - self.loss_mean)

    def to_dict(self) -> dict[str, Any]:
        candidates = [
            item
            for axis, item in self.candidates.items()
            if item.state == "candidate"
        ]
        definitions = list(self.candidates.values())
        return {
            "config": asdict(self.cfg),
            "axis_count": self.axis_count,
            "updates": self.updates,
            "quarantined_updates": self.quarantined_updates,
            "candidates": [asdict(item) for item in sorted(candidates, key=lambda item: item.axis_id)],
            "hardened": [asdict(item) for item in sorted(self.hardened.values(), key=lambda item: item.axis_id)],
            "softened": [asdict(item) for item in sorted(self.softened.values(), key=lambda item: item.axis_id)],
            "rejected": [asdict(item) for item in sorted(self.rejected.values(), key=lambda item: item.axis_id)],
            "definitions": [asdict(item) for item in sorted(definitions, key=lambda item: item.axis_id)],
            "events": self.events,
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
