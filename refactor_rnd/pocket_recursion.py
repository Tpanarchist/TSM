from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import log
from typing import Any

import numpy as np


class TritVote(IntEnum):
    """One primitive ternary cognitive vote."""

    DENY = -1
    ABSTAIN = 0
    APPROVE = 1


@dataclass(frozen=True)
class Trit:
    """A local ternary unit: one vote over one lower-level unit."""

    unit_id: int
    vote: TritVote
    time: int
    frame: int = 0


@dataclass(frozen=True)
class Tryte:
    """A variable-size relation packet made from TRITs.

    A TRYTE is intentionally not "three TRITs." It is the pocket dimension:
    lower units plus their directed relation structure inside a frame.
    """

    trits: tuple[Trit, ...]
    frame: int
    level: int = 0
    relation_matrix: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0)))

    @property
    def size(self) -> int:
        return len(self.trits)

    @property
    def unit_ids(self) -> tuple[int, ...]:
        return tuple(trit.unit_id for trit in self.trits)


@dataclass
class Trion:
    """A stable classified ternary object.

    A TRION is the TSM-native name for what a generic cognition vocabulary
    might call a Cognit.
    """

    id: int
    prototype: np.ndarray
    support: int = 1
    label_counts: dict[int, int] = field(default_factory=dict)

    def update(self, vector: np.ndarray, label: int | None = None) -> None:
        self.support += 1
        rate = 1.0 / float(self.support)
        self.prototype = (1.0 - rate) * self.prototype + rate * vector
        if label is not None:
            self.label_counts[label] = self.label_counts.get(label, 0) + 1


@dataclass(frozen=True)
class ProbeMetrics:
    accuracy: float
    nmi: float
    trion_count: int
    abstain_rate: float


def one_hot_symbols(sequence: list[int] | tuple[int, ...], vocab_size: int) -> tuple[Trit, ...]:
    return tuple(
        Trit(unit_id=int(unit_id), vote=TritVote.APPROVE, time=idx, frame=0)
        for idx, unit_id in enumerate(sequence)
        if 0 <= int(unit_id) < vocab_size
    )


def build_tryte(sequence: list[int] | tuple[int, ...], vocab_size: int, frame: int = 0, level: int = 0) -> Tryte:
    trits = one_hot_symbols(sequence, vocab_size)
    relation_matrix = directed_transition_matrix(sequence, vocab_size)
    return Tryte(trits=trits, frame=frame, level=level, relation_matrix=relation_matrix)


def directed_transition_matrix(sequence: list[int] | tuple[int, ...], vocab_size: int) -> np.ndarray:
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    if len(sequence) < 2:
        return matrix
    for idx, source in enumerate(sequence):
        target = sequence[(idx + 1) % len(sequence)]
        if 0 <= int(source) < vocab_size and 0 <= int(target) < vocab_size:
            matrix[int(source), int(target)] += 1.0
    total = matrix.sum()
    return matrix / total if total > 0.0 else matrix


def histogram_encoder(sequence: list[int] | tuple[int, ...], vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size,), dtype=np.float64)
    for unit_id in sequence:
        if 0 <= int(unit_id) < vocab_size:
            counts[int(unit_id)] += 1.0
    total = counts.sum()
    return counts / total if total > 0.0 else counts


def tryte_relation_space_encoder(sequence: list[int] | tuple[int, ...], vocab_size: int) -> np.ndarray:
    tryte = build_tryte(sequence, vocab_size)
    return tryte.relation_matrix.reshape(-1)


class AbstainGatedTrionizer:
    """Online prototype learner with explicit Abstain before TRION creation."""

    def __init__(self, approve_radius: float = 0.05, min_evidence: int = 2) -> None:
        self.approve_radius = float(approve_radius)
        self.min_evidence = int(max(1, min_evidence))
        self.trions: list[Trion] = []
        self._pending: list[tuple[np.ndarray, int | None]] = []
        self.votes: list[TritVote] = []

    def fit_predict(self, features: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        predictions: list[int] = []
        for idx, vector in enumerate(np.asarray(features, dtype=np.float64)):
            label = int(labels[idx]) if labels is not None else None
            predictions.append(self.observe(vector, label=label))
        return np.asarray(predictions, dtype=np.int64)

    def observe(self, vector: np.ndarray, label: int | None = None) -> int:
        vector = np.asarray(vector, dtype=np.float64)
        if self.trions:
            distances = np.asarray([np.linalg.norm(vector - trion.prototype) for trion in self.trions])
            nearest = int(distances.argmin())
            if float(distances[nearest]) <= self.approve_radius:
                self.trions[nearest].update(vector, label)
                self.votes.append(TritVote.APPROVE)
                return nearest

        self.votes.append(TritVote.ABSTAIN)
        self._pending.append((vector.copy(), label))
        if len(self._pending) >= self.min_evidence:
            pending_vectors = np.stack([item[0] for item in self._pending])
            prototype = pending_vectors.mean(axis=0)
            label_counts: dict[int, int] = {}
            for _, pending_label in self._pending:
                if pending_label is not None:
                    label_counts[int(pending_label)] = label_counts.get(int(pending_label), 0) + 1
            trion_id = len(self.trions)
            self.trions.append(Trion(id=trion_id, prototype=prototype, support=len(self._pending), label_counts=label_counts))
            self._pending.clear()
            return trion_id
        return -1

    @property
    def abstain_rate(self) -> float:
        if not self.votes:
            return 0.0
        return sum(1 for vote in self.votes if vote == TritVote.ABSTAIN) / float(len(self.votes))


def same_marginal_order_sequences(samples_per_regime: int = 48) -> tuple[list[list[int]], np.ndarray, int]:
    patterns = (
        [0, 1, 2, 3, 0, 1, 2, 3],
        [0, 3, 2, 1, 0, 3, 2, 1],
    )
    return _repeat_patterns(patterns, samples_per_regime=samples_per_regime, vocab_size=4)


def shared_anchor_sequences(samples_per_regime: int = 48) -> tuple[list[list[int]], np.ndarray, int]:
    patterns = (
        [0, 1, 0, 1, 2, 2],
        [0, 2, 0, 2, 1, 1],
    )
    return _repeat_patterns(patterns, samples_per_regime=samples_per_regime, vocab_size=3)


def _repeat_patterns(
    patterns: tuple[list[int], ...],
    samples_per_regime: int,
    vocab_size: int,
) -> tuple[list[list[int]], np.ndarray, int]:
    sequences: list[list[int]] = []
    labels: list[int] = []
    for label, pattern in enumerate(patterns):
        for _ in range(samples_per_regime):
            sequences.append(list(pattern))
            labels.append(label)
    return sequences, np.asarray(labels, dtype=np.int64), vocab_size


def encode_sequences(
    sequences: list[list[int]],
    vocab_size: int,
    encoder: str,
) -> np.ndarray:
    if encoder == "histogram":
        rows = [histogram_encoder(sequence, vocab_size) for sequence in sequences]
    elif encoder == "tryte_relation_space":
        rows = [tryte_relation_space_encoder(sequence, vocab_size) for sequence in sequences]
    else:
        raise ValueError(f"unknown encoder: {encoder}")
    return np.stack(rows).astype(np.float64)


def run_probe(
    sequences: list[list[int]],
    labels: np.ndarray,
    vocab_size: int,
    encoder: str,
    approve_radius: float = 0.02,
    min_evidence: int = 2,
) -> ProbeMetrics:
    features = encode_sequences(sequences, vocab_size, encoder)
    trionizer = AbstainGatedTrionizer(approve_radius=approve_radius, min_evidence=min_evidence)
    predictions = trionizer.fit_predict(features, labels)
    return ProbeMetrics(
        accuracy=nearest_centroid_accuracy(features, labels),
        nmi=normalized_mutual_information(labels, predictions),
        trion_count=len(trionizer.trions),
        abstain_rate=trionizer.abstain_rate,
    )


def novelty_stream_metrics() -> dict[str, float]:
    patterns = [
        [0, 1, 2, 3, 0, 1, 2, 3],
        [0, 3, 2, 1, 0, 3, 2, 1],
        [0, 2, 1, 3, 0, 2, 1, 3],
    ]
    labels = [0] * 20 + [1] * 20 + [2] * 20
    sequences = [patterns[label] for label in labels]
    features = encode_sequences(sequences, vocab_size=4, encoder="tryte_relation_space")
    trionizer = AbstainGatedTrionizer(approve_radius=0.02, min_evidence=3)
    predictions = trionizer.fit_predict(features, np.asarray(labels, dtype=np.int64))
    votes = np.asarray([1 if vote == TritVote.ABSTAIN else 0 for vote in trionizer.votes], dtype=np.float64)
    phase_rates = [float(votes[start : start + 20].mean()) for start in (0, 20, 40)]
    early_rates = [float(votes[start : start + 5].mean()) for start in (0, 20, 40)]
    late_rates = [float(votes[start + 10 : start + 20].mean()) for start in (0, 20, 40)]
    return {
        "phase0_abstain_rate": phase_rates[0],
        "phase1_abstain_rate": phase_rates[1],
        "phase2_abstain_rate": phase_rates[2],
        "novelty_spike_1": early_rates[1] - late_rates[0],
        "novelty_spike_2": early_rates[2] - late_rates[1],
        "phase1_settle_delta": early_rates[1] - late_rates[1],
        "phase2_settle_delta": early_rates[2] - late_rates[2],
        "trion_count": float(len(trionizer.trions)),
        "nmi": normalized_mutual_information(np.asarray(labels, dtype=np.int64), predictions),
    }


def depth_probe_metrics() -> dict[str, Any]:
    level0_patterns = (
        [0, 1, 2, 3, 0, 1, 2, 3],
        [0, 3, 2, 1, 0, 3, 2, 1],
        [0, 2, 1, 3, 0, 2, 1, 3],
    )
    level1_sequences, level1_labels, level0_vocab = _repeat_patterns(level0_patterns, samples_per_regime=16, vocab_size=4)
    level1_features = encode_sequences(level1_sequences, level0_vocab, "tryte_relation_space")
    level1_trionizer = AbstainGatedTrionizer(approve_radius=0.02, min_evidence=2)
    level1_ids = level1_trionizer.fit_predict(level1_features, level1_labels)

    level2_sequences: list[list[int]] = []
    level2_labels: list[int] = []
    for label, template in enumerate(([0, 1, 0, 1, 2, 2], [0, 2, 0, 2, 1, 1])):
        for _ in range(12):
            level2_sequences.append([_first_trion_for_label(level1_ids, level1_labels, unit) for unit in template])
            level2_labels.append(label)
    level2_features = encode_sequences(level2_sequences, vocab_size=max(1, len(level1_trionizer.trions)), encoder="tryte_relation_space")
    level2_trionizer = AbstainGatedTrionizer(approve_radius=0.02, min_evidence=2)
    level2_ids = level2_trionizer.fit_predict(level2_features, np.asarray(level2_labels, dtype=np.int64))

    level3_sequences: list[list[int]] = []
    level3_labels: list[int] = []
    for label, template in enumerate(([0, 1, 0, 1], [1, 0, 1, 0])):
        for _ in range(8):
            level3_sequences.append([_first_trion_for_label(level2_ids, np.asarray(level2_labels), unit) for unit in template])
            level3_labels.append(label)
    level3_features = encode_sequences(level3_sequences, vocab_size=max(1, len(level2_trionizer.trions)), encoder="tryte_relation_space")
    level3_trionizer = AbstainGatedTrionizer(approve_radius=0.02, min_evidence=2)
    level3_ids = level3_trionizer.fit_predict(level3_features, np.asarray(level3_labels, dtype=np.int64))

    return {
        "level1": {
            "nmi": normalized_mutual_information(level1_labels, level1_ids),
            "accuracy": nearest_centroid_accuracy(level1_features, level1_labels),
            "trion_count": len(level1_trionizer.trions),
            "abstain_rate": level1_trionizer.abstain_rate,
        },
        "level2": {
            "nmi": normalized_mutual_information(np.asarray(level2_labels), level2_ids),
            "accuracy": nearest_centroid_accuracy(level2_features, np.asarray(level2_labels)),
            "trion_count": len(level2_trionizer.trions),
            "abstain_rate": level2_trionizer.abstain_rate,
        },
        "level3": {
            "nmi": normalized_mutual_information(np.asarray(level3_labels), level3_ids),
            "accuracy": nearest_centroid_accuracy(level3_features, np.asarray(level3_labels)),
            "trion_count": len(level3_trionizer.trions),
            "abstain_rate": level3_trionizer.abstain_rate,
        },
    }


def _first_trion_for_label(predictions: np.ndarray, labels: np.ndarray, label: int) -> int:
    matches = predictions[(labels == label) & (predictions >= 0)]
    if matches.size == 0:
        return 0
    values, counts = np.unique(matches, return_counts=True)
    return int(values[int(counts.argmax())])


def run_all(seed: int = 31) -> dict[str, Any]:
    # The current worlds are deterministic; seed is kept in the API so future
    # noisy variants can be compared without changing result schema.
    np.random.default_rng(seed)
    same_sequences, same_labels, same_vocab = same_marginal_order_sequences()
    anchor_sequences, anchor_labels, anchor_vocab = shared_anchor_sequences()
    same_hist = run_probe(same_sequences, same_labels, same_vocab, "histogram")
    same_relation = run_probe(same_sequences, same_labels, same_vocab, "tryte_relation_space")
    anchor_hist = run_probe(anchor_sequences, anchor_labels, anchor_vocab, "histogram")
    anchor_relation = run_probe(anchor_sequences, anchor_labels, anchor_vocab, "tryte_relation_space")
    return {
        "seed": seed,
        "same_marginal": {
            "histogram": same_hist.__dict__,
            "tryte_relation_space": same_relation.__dict__,
            "relation_nmi_delta": same_relation.nmi - same_hist.nmi,
            "acceptance_pass": (
                (same_hist.nmi <= 0.25 or same_hist.accuracy <= 0.60)
                and same_relation.nmi >= 0.70
                and same_relation.accuracy >= 0.85
            ),
        },
        "shared_anchor": {
            "histogram": anchor_hist.__dict__,
            "tryte_relation_space": anchor_relation.__dict__,
            "relation_nmi_delta": anchor_relation.nmi - anchor_hist.nmi,
            "acceptance_pass": (anchor_relation.nmi - anchor_hist.nmi) >= 0.40,
        },
        "novelty": novelty_stream_metrics(),
        "depth_probe": depth_probe_metrics(),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(path: tuple[str, ...]) -> list[float]:
        values = []
        for result in results:
            current: Any = result
            for key in path:
                current = current[key]
            values.append(float(current))
        return values

    metric_paths = {
        "same_histogram_nmi": ("same_marginal", "histogram", "nmi"),
        "same_relation_nmi": ("same_marginal", "tryte_relation_space", "nmi"),
        "same_relation_accuracy": ("same_marginal", "tryte_relation_space", "accuracy"),
        "shared_histogram_nmi": ("shared_anchor", "histogram", "nmi"),
        "shared_relation_nmi": ("shared_anchor", "tryte_relation_space", "nmi"),
        "shared_relation_nmi_delta": ("shared_anchor", "relation_nmi_delta"),
        "novelty_spike_1": ("novelty", "novelty_spike_1"),
        "novelty_spike_2": ("novelty", "novelty_spike_2"),
    }
    aggregate: dict[str, Any] = {"seed_count": len(results)}
    for name, path in metric_paths.items():
        values = collect(path)
        aggregate[name] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    aggregate["acceptance"] = {
        "same_marginal_pass_fraction": float(np.mean([result["same_marginal"]["acceptance_pass"] for result in results])),
        "shared_anchor_pass_fraction": float(np.mean([result["shared_anchor"]["acceptance_pass"] for result in results])),
        "novelty_pass_fraction": float(np.mean([
            result["novelty"]["novelty_spike_1"] > 0.0
            and result["novelty"]["novelty_spike_2"] > 0.0
            and result["novelty"]["phase1_settle_delta"] > 0.0
            and result["novelty"]["phase2_settle_delta"] > 0.0
            for result in results
        ])),
    }
    return aggregate


def nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    centroids = np.stack([features[labels == label].mean(axis=0) for label in unique_labels])
    distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=-1)
    predictions = unique_labels[distances.argmin(axis=1)]
    return float((predictions == labels).mean())


def normalized_mutual_information(labels: np.ndarray | list[int], predictions: np.ndarray | list[int]) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    count = min(labels.size, predictions.size)
    if count == 0:
        return 0.0
    labels = labels[:count]
    predictions = predictions[:count]
    valid = predictions >= 0
    if not bool(valid.any()):
        return 0.0
    labels = labels[valid]
    predictions = predictions[valid]
    total = float(labels.size)
    if total <= 0.0:
        return 0.0
    label_values, label_counts = np.unique(labels, return_counts=True)
    pred_values, pred_counts = np.unique(predictions, return_counts=True)
    label_probs = label_counts.astype(np.float64) / total
    pred_probs = pred_counts.astype(np.float64) / total
    label_entropy = _entropy(label_probs)
    pred_entropy = _entropy(pred_probs)
    if label_entropy <= 0.0 or pred_entropy <= 0.0:
        return 0.0
    mutual_info = 0.0
    for label, label_count in zip(label_values, label_counts):
        for pred, pred_count in zip(pred_values, pred_counts):
            joint = float(((labels == label) & (predictions == pred)).sum()) / total
            if joint <= 0.0:
                continue
            label_prob = float(label_count) / total
            pred_prob = float(pred_count) / total
            mutual_info += joint * log(joint / (label_prob * pred_prob))
    return float(mutual_info / np.sqrt(label_entropy * pred_entropy))


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
