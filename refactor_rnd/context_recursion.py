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
class Context:
    """A bounded relation field made from TRITs.

    A CONTEXT is intentionally not "three TRITs." It preserves lower units
    plus their transition, lag, order, frame, and ranked ordered implication
    structure inside a bounded field.
    """

    trits: tuple[Trit, ...]
    frame: int = 0
    level: int = 0
    relation_matrix: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0)))

    @property
    def size(self) -> int:
        return len(self.trits)

    @property
    def unit_ids(self) -> tuple[int, ...]:
        return tuple(trit.unit_id for trit in self.trits)


@dataclass
class Abstraction:
    """A reusable classified structure.

    An ABSTRACTION is the product that remains after stable Knowledge is
    classified into a reusable unit.
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
    abstraction_count: int
    abstain_rate: float


def one_hot_symbols(sequence: list[int] | tuple[int, ...], vocab_size: int) -> tuple[Trit, ...]:
    return tuple(
        Trit(unit_id=int(unit_id), vote=TritVote.APPROVE, time=idx, frame=0)
        for idx, unit_id in enumerate(sequence)
        if 0 <= int(unit_id) < vocab_size
    )


def build_context(sequence: list[int] | tuple[int, ...], vocab_size: int, frame: int = 0, level: int = 0) -> Context:
    trits = one_hot_symbols(sequence, vocab_size)
    relation_matrix = directed_transition_matrix(sequence, vocab_size)
    return Context(trits=trits, frame=frame, level=level, relation_matrix=relation_matrix)


def directed_transition_matrix(sequence: list[int] | tuple[int, ...], vocab_size: int, lag: int = 1) -> np.ndarray:
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    if len(sequence) < 2:
        return matrix
    lag = max(1, int(lag))
    for idx, source in enumerate(sequence):
        target = sequence[(idx + lag) % len(sequence)]
        if 0 <= int(source) < vocab_size and 0 <= int(target) < vocab_size:
            matrix[int(source), int(target)] += 1.0
    total = matrix.sum()
    return matrix / total if total > 0.0 else matrix


def ordered_transition_onehots(
    sequence: list[int] | tuple[int, ...],
    vocab_size: int,
    slots: int | None = None,
) -> np.ndarray:
    slot_count = len(sequence) if slots is None else max(0, int(slots))
    if slot_count <= 0:
        return np.zeros((0, vocab_size * vocab_size), dtype=np.float64)
    rows = np.zeros((slot_count, vocab_size * vocab_size), dtype=np.float64)
    if not sequence:
        return rows
    for idx, source in enumerate(sequence[:slot_count]):
        target = sequence[(idx + 1) % len(sequence)]
        if 0 <= int(source) < vocab_size and 0 <= int(target) < vocab_size:
            rows[idx, int(source) * vocab_size + int(target)] = 1.0
    return rows


def histogram_encoder(sequence: list[int] | tuple[int, ...], vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size,), dtype=np.float64)
    for unit_id in sequence:
        if 0 <= int(unit_id) < vocab_size:
            counts[int(unit_id)] += 1.0
    total = counts.sum()
    return counts / total if total > 0.0 else counts


def cooccurrence_encoder(sequence: list[int] | tuple[int, ...], vocab_size: int) -> np.ndarray:
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    valid = [int(unit_id) for unit_id in sequence if 0 <= int(unit_id) < vocab_size]
    for left_idx, left in enumerate(valid):
        for right in valid[left_idx + 1 :]:
            matrix[left, right] += 1.0
            matrix[right, left] += 1.0
    total = matrix.sum()
    flat = matrix.reshape(-1)
    return flat / total if total > 0.0 else flat


def transition_encoder(sequence: list[int] | tuple[int, ...], vocab_size: int) -> np.ndarray:
    return directed_transition_matrix(sequence, vocab_size, lag=1).reshape(-1)


def transition_lag_order_encoder(
    sequence: list[int] | tuple[int, ...],
    vocab_size: int,
    ordered_slots: int | None = None,
) -> np.ndarray:
    lag1 = directed_transition_matrix(sequence, vocab_size, lag=1).reshape(-1)
    lag2 = directed_transition_matrix(sequence, vocab_size, lag=2).reshape(-1)
    ordered = ordered_transition_onehots(sequence, vocab_size, slots=ordered_slots).reshape(-1)
    return np.concatenate([lag1, lag2, ordered]).astype(np.float64)


def degraded_transition_lag_order_encoder(
    sequence: list[int] | tuple[int, ...],
    vocab_size: int,
    loss: float,
    ordered_slots: int | None = None,
) -> np.ndarray:
    loss = float(np.clip(loss, 0.0, 1.0))
    full = transition_lag_order_encoder(sequence, vocab_size, ordered_slots=ordered_slots)
    lag_dim = vocab_size * vocab_size
    degraded = np.zeros_like(full)
    degraded[:lag_dim] = full[:lag_dim]
    degraded[lag_dim : 2 * lag_dim] = max(0.0, 1.0 - 2.0 * loss) * full[lag_dim : 2 * lag_dim]
    degraded[2 * lag_dim :] = max(0.0, 1.0 - 4.0 * loss) * full[2 * lag_dim :]
    return degraded.astype(np.float64)


def context_relation_space_encoder(sequence: list[int] | tuple[int, ...], vocab_size: int) -> np.ndarray:
    return transition_encoder(sequence, vocab_size)


def vector_context_sequence_encoder(sequence: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    vectors = [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in sequence]
    if not vectors:
        return np.zeros((0,), dtype=np.float64)
    deltas = [vectors[(idx + 1) % len(vectors)] - vector for idx, vector in enumerate(vectors)]
    return np.concatenate([*vectors, *deltas]).astype(np.float64)


def canonical_encoder_name(encoder: str) -> str:
    if encoder == "context_relation_space":
        return "transition"
    return encoder


def available_encoder_names() -> tuple[str, ...]:
    return ("histogram", "cooccurrence", "transition", "transition_lag_order")


def available_stress_encoder_names() -> tuple[str, ...]:
    return ("histogram", "transition", "transition_lag_order", "degraded_transition_lag_order")


def available_stress_case_names() -> tuple[str, ...]:
    return (
        "variable_lag",
        "pure_phase_offset",
        "noisy_order",
        "missing_events",
        "repeated_symbols",
        "ambiguous_shared_anchors",
        "frame_shift",
    )


def stress_degradation_levels() -> tuple[float, ...]:
    return (0.0, 0.25, 0.50, 0.75, 1.0)


def _stress_positions(length: int, degradation: float, label: int, sample_index: int, max_count: int | None = None) -> list[int]:
    if length <= 0 or degradation <= 0.0:
        return []
    cap = max_count if max_count is not None else max(1, length // 2)
    count = min(cap, max(1, int(np.ceil(float(degradation) * cap))))
    positions: list[int] = []
    cursor = (int(label) + int(sample_index)) % length
    step = 2 if length > 2 else 1
    while len(positions) < count and len(positions) < length:
        pos = cursor % length
        if pos not in positions:
            positions.append(pos)
        cursor += step
    return positions


def apply_stress_case(
    sequence: list[int] | tuple[int, ...],
    stress_case: str,
    degradation: float,
    label: int,
    sample_index: int,
    vocab_size: int,
) -> list[int]:
    result = [int(unit_id) for unit_id in sequence]
    if stress_case == "baseline" or degradation <= 0.0:
        return result
    length = len(result)
    if length == 0:
        return result
    anchor = 0 if vocab_size > 0 else 0
    positions = _stress_positions(length, degradation, label, sample_index, max_count=max(1, length // 2))

    if stress_case == "variable_lag":
        base = list(result)
        offset = 0
        for pos in sorted(positions):
            insert_at = min(pos + offset, len(result))
            filler = base[pos - 1] if pos > 0 else base[0]
            result.insert(insert_at, filler)
            offset += 1
        return result

    if stress_case == "pure_phase_offset":
        max_shift = max(1, int(np.ceil(float(degradation) * max(1, length // 2))))
        shift = (int(label) + int(sample_index)) % (max_shift + 1)
        if shift <= 0:
            return result
        translated = ([-1] * shift) + result[: max(0, length - shift)]
        return translated[:length]

    if stress_case == "noisy_order":
        used: set[int] = set()
        for pos in positions:
            next_pos = (pos + 1) % length
            if pos in used or next_pos in used:
                continue
            result[pos], result[next_pos] = result[next_pos], result[pos]
            used.add(pos)
            used.add(next_pos)
        return result

    if stress_case == "missing_events":
        max_drop = max(0, length - 2)
        drop_positions = set(sorted(positions)[:max_drop])
        if not drop_positions:
            return result
        return [unit_id for idx, unit_id in enumerate(result) if idx not in drop_positions]

    if stress_case == "repeated_symbols":
        for pos in positions:
            next_pos = (pos + 1) % length
            result[next_pos] = result[pos]
        return result

    if stress_case == "ambiguous_shared_anchors":
        for pos in positions:
            result[pos] = anchor
        return result

    if stress_case == "frame_shift":
        shift = max(1, int(np.ceil(float(degradation) * max(1, length - 1))))
        shift = (shift + int(label) + int(sample_index)) % length
        if shift:
            return result[shift:] + result[:shift]
        return result

    raise ValueError(f"unknown stress case: {stress_case}")


def stress_sequences(
    sequences: list[list[int]],
    labels: np.ndarray,
    stress_case: str,
    degradation: float,
    vocab_size: int,
) -> list[list[int]]:
    return [
        apply_stress_case(sequence, stress_case, degradation, int(labels[idx]), idx, vocab_size)
        for idx, sequence in enumerate(sequences)
    ]


class AbstainGatedAbstractor:
    """Mechanism that mints abstractions only after Abstain-gated evidence accumulates."""

    def __init__(self, approve_radius: float = 0.05, min_evidence: int = 2) -> None:
        self.approve_radius = float(approve_radius)
        self.min_evidence = int(max(1, min_evidence))
        self.abstractions: list[Abstraction] = []
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
        if self.abstractions:
            distances = np.asarray([np.linalg.norm(vector - abstraction.prototype) for abstraction in self.abstractions])
            nearest = int(distances.argmin())
            if float(distances[nearest]) <= self.approve_radius:
                self.abstractions[nearest].update(vector, label)
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
            abstraction_id = len(self.abstractions)
            self.abstractions.append(Abstraction(id=abstraction_id, prototype=prototype, support=len(self._pending), label_counts=label_counts))
            self._pending.clear()
            return abstraction_id
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
    degradation_loss: float = 0.0,
) -> np.ndarray:
    encoder = canonical_encoder_name(encoder)
    ordered_slots = max((len(sequence) for sequence in sequences), default=0)
    if encoder == "histogram":
        rows = [histogram_encoder(sequence, vocab_size) for sequence in sequences]
    elif encoder == "cooccurrence":
        rows = [cooccurrence_encoder(sequence, vocab_size) for sequence in sequences]
    elif encoder == "transition":
        rows = [transition_encoder(sequence, vocab_size) for sequence in sequences]
    elif encoder == "transition_lag_order":
        rows = [transition_lag_order_encoder(sequence, vocab_size, ordered_slots=ordered_slots) for sequence in sequences]
    elif encoder == "degraded_transition_lag_order":
        rows = [
            degraded_transition_lag_order_encoder(
                sequence,
                vocab_size,
                degradation_loss,
                ordered_slots=ordered_slots,
            )
            for sequence in sequences
        ]
    elif encoder == "context_relation_space":
        rows = [context_relation_space_encoder(sequence, vocab_size) for sequence in sequences]
    else:
        raise ValueError(f"unknown encoder: {encoder}")
    return np.stack(rows).astype(np.float64)


def encode_vector_sequences(sequences: list[list[np.ndarray]]) -> np.ndarray:
    rows = [vector_context_sequence_encoder(sequence) for sequence in sequences]
    return np.stack(rows).astype(np.float64)


def run_probe(
    sequences: list[list[int]],
    labels: np.ndarray,
    vocab_size: int,
    encoder: str,
    approve_radius: float = 0.02,
    min_evidence: int = 2,
    degradation_loss: float = 0.0,
) -> ProbeMetrics:
    features = encode_sequences(sequences, vocab_size, encoder, degradation_loss=degradation_loss)
    abstractor = AbstainGatedAbstractor(approve_radius=approve_radius, min_evidence=min_evidence)
    predictions = abstractor.fit_predict(features, labels)
    return ProbeMetrics(
        accuracy=nearest_centroid_accuracy(features, labels),
        nmi=normalized_mutual_information(labels, predictions),
        abstraction_count=len(abstractor.abstractions),
        abstain_rate=abstractor.abstain_rate,
    )


def supervised_probe_predictions(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    predictions = np.zeros_like(labels)
    for idx, vector in enumerate(features):
        best_label = int(unique_labels[0])
        best_distance = float("inf")
        for label in unique_labels:
            mask = labels == label
            if labels[idx] == label:
                mask = mask.copy()
                mask[idx] = False
            if bool(mask.any()):
                centroid = features[mask].mean(axis=0)
            else:
                centroid = features[labels == label].mean(axis=0)
            distance = float(((vector - centroid) ** 2).mean())
            if distance < best_distance:
                best_distance = distance
                best_label = int(label)
        predictions[idx] = best_label
    return predictions


def supervised_probe_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = supervised_probe_predictions(features, labels)
    return {
        "accuracy": float((predictions == np.asarray(labels)).mean()),
        "nmi": normalized_mutual_information(labels, predictions),
    }


def level_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    abstraction_count: int,
    abstain_rate: float,
) -> dict[str, float]:
    probe = supervised_probe_metrics(features, labels)
    abstraction_nmi = normalized_mutual_information(labels, predictions)
    return {
        "nmi": abstraction_nmi,
        "accuracy": probe["accuracy"],
        "abstraction_nmi": abstraction_nmi,
        "abstraction_ari": adjusted_rand_index(labels, predictions),
        "supervised_probe_accuracy": probe["accuracy"],
        "supervised_probe_nmi": probe["nmi"],
        "abstraction_count": int(abstraction_count),
        "abstain_rate": float(abstain_rate),
    }


def novelty_stream_metrics(
    encoder: str = "context_relation_space",
    stress_case: str = "baseline",
    degradation: float = 0.0,
    degradation_loss: float = 0.0,
) -> dict[str, float]:
    patterns = [
        [0, 1, 2, 3, 0, 1, 2, 3],
        [0, 3, 2, 1, 0, 3, 2, 1],
        [0, 2, 1, 3, 0, 2, 1, 3],
    ]
    labels = [0] * 20 + [1] * 20 + [2] * 20
    sequences = [patterns[label] for label in labels]
    label_array = np.asarray(labels, dtype=np.int64)
    sequences = stress_sequences(sequences, label_array, stress_case, degradation, vocab_size=4)
    features = encode_sequences(sequences, vocab_size=4, encoder=encoder, degradation_loss=degradation_loss)
    abstractor = AbstainGatedAbstractor(approve_radius=0.02, min_evidence=3)
    predictions = abstractor.fit_predict(features, label_array)
    votes = np.asarray([1 if vote == TritVote.ABSTAIN else 0 for vote in abstractor.votes], dtype=np.float64)
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
        "abstain_spike_rate": 0.5 * ((early_rates[1] - late_rates[0]) + (early_rates[2] - late_rates[1])),
        "abstraction_count": float(len(abstractor.abstractions)),
        "nmi": normalized_mutual_information(label_array, predictions),
    }


def depth_level_specs() -> tuple[
    tuple[tuple[list[int], ...], int],
    tuple[tuple[list[int], ...], int],
    tuple[tuple[list[int], ...], int],
]:
    return (
        ((
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 3, 2, 1, 0, 3, 2, 1],
            [0, 2, 1, 3, 0, 2, 1, 3],
        ), 16),
        ((
            [0, 1, 0, 1, 2, 2],
            [0, 2, 0, 2, 1, 1],
        ), 12),
        ((
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ), 8),
    )


def depth_base_world() -> tuple[list[list[int]], np.ndarray, int]:
    level0_patterns, samples_per_regime = depth_level_specs()[0]
    return _repeat_patterns(level0_patterns, samples_per_regime=samples_per_regime, vocab_size=4)


def label_mean_vectors(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(label): np.asarray(features[labels == label], dtype=np.float64).mean(axis=0)
        for label in np.unique(labels)
    }


def abstraction_vectors_by_label(
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    abstractions: list[Abstraction],
) -> dict[int, np.ndarray]:
    by_label = label_mean_vectors(features, labels)
    if not abstractions:
        return by_label
    for label in np.unique(labels):
        matches = predictions[(labels == label) & (predictions >= 0)]
        if matches.size == 0:
            continue
        values, counts = np.unique(matches, return_counts=True)
        abstraction_id = int(values[int(counts.argmax())])
        by_label[int(label)] = abstractions[abstraction_id].prototype.copy()
    return by_label


def build_vector_level_sequences(
    source_vectors: dict[int, np.ndarray],
    templates: tuple[list[int], ...],
    samples_per_regime: int,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    sequences: list[list[np.ndarray]] = []
    labels: list[int] = []
    for label, template in enumerate(templates):
        for _ in range(samples_per_regime):
            sequences.append([source_vectors[int(unit)].copy() for unit in template])
            labels.append(label)
    return sequences, np.asarray(labels, dtype=np.int64)


def _pack_depth_level_state(
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    abstraction_count: int,
    abstain_rate: float,
    vocab_size: int,
) -> dict[str, Any]:
    return {
        "features": np.asarray(features, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "abstraction_count": int(abstraction_count),
        "abstain_rate": float(abstain_rate),
        "vocab_size": int(vocab_size),
    }


def _depth_encoder_state(
    encoder: str,
    stress_case: str = "baseline",
    degradation: float = 0.0,
    degradation_loss: float = 0.0,
) -> dict[str, dict[str, Any]]:
    encoder = canonical_encoder_name(encoder)
    lower_level_stress_case = "baseline" if stress_case == "pure_phase_offset" else stress_case
    level_specs = depth_level_specs()
    level1_sequences, level1_labels, level0_vocab = depth_base_world()
    level1_labels = np.asarray(level1_labels, dtype=np.int64)
    level1_sequences = stress_sequences(level1_sequences, level1_labels, lower_level_stress_case, degradation, level0_vocab)
    level1_features = encode_sequences(level1_sequences, level0_vocab, encoder, degradation_loss=degradation_loss)
    level1_abstractor = AbstainGatedAbstractor(approve_radius=0.02, min_evidence=2)
    level1_ids = level1_abstractor.fit_predict(level1_features, level1_labels)

    level2_templates, level2_samples = level_specs[1]
    level2_sequences: list[list[int]] = []
    level2_labels: list[int] = []
    for label, template in enumerate(level2_templates):
        for _ in range(level2_samples):
            level2_sequences.append([_first_abstraction_for_label(level1_ids, level1_labels, unit) for unit in template])
            level2_labels.append(label)
    level2_label_array = np.asarray(level2_labels, dtype=np.int64)
    level2_vocab = max(1, len(level1_abstractor.abstractions))
    level2_sequences = stress_sequences(level2_sequences, level2_label_array, lower_level_stress_case, degradation, level2_vocab)
    level2_features = encode_sequences(level2_sequences, vocab_size=level2_vocab, encoder=encoder, degradation_loss=degradation_loss)
    level2_abstractor = AbstainGatedAbstractor(approve_radius=0.02, min_evidence=2)
    level2_ids = level2_abstractor.fit_predict(level2_features, level2_label_array)

    level3_templates, level3_samples = level_specs[2]
    level3_sequences: list[list[int]] = []
    level3_labels: list[int] = []
    for label, template in enumerate(level3_templates):
        for _ in range(level3_samples):
            level3_sequences.append([_first_abstraction_for_label(level2_ids, level2_label_array, unit) for unit in template])
            level3_labels.append(label)
    level3_label_array = np.asarray(level3_labels, dtype=np.int64)
    level3_vocab = max(1, len(level2_abstractor.abstractions))
    level3_sequences = stress_sequences(level3_sequences, level3_label_array, stress_case, degradation, level3_vocab)
    level3_features = encode_sequences(level3_sequences, vocab_size=level3_vocab, encoder=encoder, degradation_loss=degradation_loss)
    level3_abstractor = AbstainGatedAbstractor(approve_radius=0.02, min_evidence=2)
    level3_ids = level3_abstractor.fit_predict(level3_features, level3_label_array)

    return {
        "level1": _pack_depth_level_state(
            level1_features,
            level1_labels,
            level1_ids,
            len(level1_abstractor.abstractions),
            level1_abstractor.abstain_rate,
            level0_vocab,
        ),
        "level2": _pack_depth_level_state(
            level2_features,
            level2_label_array,
            level2_ids,
            len(level2_abstractor.abstractions),
            level2_abstractor.abstain_rate,
            level2_vocab,
        ),
        "level3": _pack_depth_level_state(
            level3_features,
            level3_label_array,
            level3_ids,
            len(level3_abstractor.abstractions),
            level3_abstractor.abstain_rate,
            level3_vocab,
        ),
    }


def _level_metrics_from_state(state: dict[str, Any]) -> dict[str, float]:
    return level_metrics(
        state["features"],
        state["labels"],
        state["predictions"],
        state["abstraction_count"],
        state["abstain_rate"],
    )


def run_depth_encoder_diagnostics(
    encoder: str,
    stress_case: str = "baseline",
    degradation: float = 0.0,
    degradation_loss: float = 0.0,
) -> dict[str, dict[str, float]]:
    state = _depth_encoder_state(
        encoder,
        stress_case=stress_case,
        degradation=degradation,
        degradation_loss=degradation_loss,
    )
    return {level_name: _level_metrics_from_state(level_state) for level_name, level_state in state.items()}


def run_classify_before_recursion_ablation(recursion_mode: str) -> dict[str, dict[str, float]]:
    if recursion_mode not in {"raw_context", "classified_abstraction"}:
        raise ValueError(f"unknown recursion mode: {recursion_mode}")

    level_specs = depth_level_specs()
    level1_sequences, level1_labels, level0_vocab = depth_base_world()
    level1_labels = np.asarray(level1_labels, dtype=np.int64)
    level1_features = encode_sequences(level1_sequences, level0_vocab, "transition")
    level1_abstractor = AbstainGatedAbstractor(approve_radius=0.02, min_evidence=2)
    level1_ids = level1_abstractor.fit_predict(level1_features, level1_labels)

    if recursion_mode == "classified_abstraction":
        level1_units = abstraction_vectors_by_label(level1_features, level1_labels, level1_ids, level1_abstractor.abstractions)
    else:
        level1_units = label_mean_vectors(level1_features, level1_labels)

    level2_templates, level2_samples = level_specs[1]
    level2_sequences, level2_labels = build_vector_level_sequences(level1_units, level2_templates, level2_samples)
    level2_features = encode_vector_sequences(level2_sequences)
    level2_abstractor = AbstainGatedAbstractor(approve_radius=0.02, min_evidence=2)
    level2_ids = level2_abstractor.fit_predict(level2_features, level2_labels)

    if recursion_mode == "classified_abstraction":
        level2_units = abstraction_vectors_by_label(level2_features, level2_labels, level2_ids, level2_abstractor.abstractions)
    else:
        level2_units = label_mean_vectors(level2_features, level2_labels)

    level3_templates, level3_samples = level_specs[2]
    level3_sequences, level3_labels = build_vector_level_sequences(level2_units, level3_templates, level3_samples)
    level3_features = encode_vector_sequences(level3_sequences)
    level3_abstractor = AbstainGatedAbstractor(approve_radius=0.02, min_evidence=2)
    level3_ids = level3_abstractor.fit_predict(level3_features, level3_labels)

    return {
        "level1": level_metrics(level1_features, level1_labels, level1_ids, len(level1_abstractor.abstractions), level1_abstractor.abstain_rate),
        "level2": level_metrics(level2_features, level2_labels, level2_ids, len(level2_abstractor.abstractions), level2_abstractor.abstain_rate),
        "level3": level_metrics(level3_features, level3_labels, level3_ids, len(level3_abstractor.abstractions), level3_abstractor.abstain_rate),
    }


def diagnose_level3_collapse(
    encoder_results: dict[str, dict[str, dict[str, float]]],
    recursion_results: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    baseline = encoder_results["transition"]["level3"]
    order_rescue = encoder_results["transition_lag_order"]["level3"]
    classified = recursion_results["classified_abstraction"]["level3"]
    raw_context = recursion_results["raw_context"]["level3"]

    if baseline["supervised_probe_accuracy"] >= 0.75 and baseline["abstraction_nmi"] <= 0.25:
        primary = "classifier_weakness"
        summary = (
            "Level-3 raw context vectors still support a simple supervised probe, but the abstain-gated abstractor does not separate them."
        )
    elif baseline["supervised_probe_accuracy"] <= 0.60 and order_rescue["supervised_probe_accuracy"] >= 0.75:
        primary = "information_loss"
        summary = (
            "The transition-only context encoder destroys the level-3 label signal; adding lag/order restores it, so the collapse is representational rather than a hard depth wall."
        )
    elif baseline["supervised_probe_accuracy"] <= 0.60 and raw_context["supervised_probe_accuracy"] <= 0.60:
        primary = "true_recursion_depth_failure"
        summary = (
            "Neither the transition encoder nor raw-context recursion retains a supervised level-3 signal, which points to a genuine toy depth limit."
        )
    else:
        primary = "mixed"
        summary = (
            "Level-3 collapse is mixed: some signal survives under richer encoders or alternate recursion paths, but the current transition-only path still fails."
        )

    if classified["abstraction_nmi"] > raw_context["abstraction_nmi"] + 0.10:
        recursion_takeaway = "classification_before_recursion_helps"
    elif raw_context["abstraction_nmi"] > classified["abstraction_nmi"] + 0.10:
        recursion_takeaway = "raw_context_recursion_helps"
    else:
        recursion_takeaway = "no_clear_recursion_mode_winner"

    return {
        "primary_cause": primary,
        "summary": summary,
        "recursion_takeaway": recursion_takeaway,
        "baseline_level3_supervised_probe_accuracy": baseline["supervised_probe_accuracy"],
        "baseline_level3_abstraction_nmi": baseline["abstraction_nmi"],
        "order_rescue_level3_supervised_probe_accuracy": order_rescue["supervised_probe_accuracy"],
        "order_rescue_level3_abstraction_nmi": order_rescue["abstraction_nmi"],
        "classified_abstraction_level3_nmi": classified["abstraction_nmi"],
        "raw_context_level3_nmi": raw_context["abstraction_nmi"],
    }


def collapse_level(levels: dict[str, dict[str, float]], nmi_threshold: float = 0.70, probe_threshold: float = 0.75) -> str:
    for level_name in ("level1", "level2", "level3"):
        metrics = levels[level_name]
        if metrics["abstraction_nmi"] < nmi_threshold or metrics["supervised_probe_accuracy"] < probe_threshold:
            return level_name
    return "none"


def _threshold_score(value: float | None) -> float:
    return 2.0 if value is None else float(value)


def _level3_damage_components(full_payload: dict[str, Any], degraded_payload: dict[str, Any]) -> dict[str, float]:
    full_level3 = full_payload["levels"]["level3"]
    degraded_level3 = degraded_payload["levels"]["level3"]
    level3_nmi_damage = max(0.0, float(full_level3["abstraction_nmi"]) - float(degraded_level3["abstraction_nmi"]))
    level3_probe_damage = max(
        0.0,
        float(full_level3["supervised_probe_accuracy"]) - float(degraded_level3["supervised_probe_accuracy"]),
    )
    return {
        "level3_nmi_damage": level3_nmi_damage,
        "level3_probe_damage": level3_probe_damage,
        "level3_damage_score": level3_nmi_damage + level3_probe_damage,
    }


def _has_level3_damage(full_payload: dict[str, Any], degraded_payload: dict[str, Any], tolerance: float = 1e-9) -> bool:
    return _level3_damage_components(full_payload, degraded_payload)["level3_damage_score"] > tolerance


def diagnose_probe_d(
    case_results: dict[str, Any],
    thresholds: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    better_cases = [
        case_name
        for case_name, encoder_thresholds in thresholds.items()
        if _threshold_score(encoder_thresholds["transition_lag_order"]) > _threshold_score(encoder_thresholds["transition"])
    ]
    degraded_cases = [
        case_name
        for case_name, encoder_results in case_results.items()
        if any(
            float(degradation_key) > 0.0
            and _has_level3_damage(
                encoder_results["transition_lag_order"][degradation_key],
                encoder_results["degraded_transition_lag_order"][degradation_key],
            )
            for degradation_key in encoder_results["transition_lag_order"]
        )
    ]
    damage_ranking: list[dict[str, Any]] = []
    for case_name, encoder_results in case_results.items():
        damage_components = [
            _level3_damage_components(
                encoder_results["transition_lag_order"][degradation_key],
                encoder_results["degraded_transition_lag_order"][degradation_key],
            )
            for degradation_key in encoder_results["transition_lag_order"]
            if float(degradation_key) > 0.0
        ]
        if damage_components:
            level3_nmi_damage = float(np.mean([item["level3_nmi_damage"] for item in damage_components]))
            level3_probe_damage = float(np.mean([item["level3_probe_damage"] for item in damage_components]))
            level3_damage_score = float(np.mean([item["level3_damage_score"] for item in damage_components]))
        else:
            level3_nmi_damage = 0.0
            level3_probe_damage = 0.0
            level3_damage_score = 0.0
        damage_ranking.append({
            "stress_case": case_name,
            "level3_damage_score": level3_damage_score,
            "level3_nmi_damage": level3_nmi_damage,
            "level3_probe_damage": level3_probe_damage,
        })
    damage_ranking.sort(
        key=lambda row: (
            -row["level3_damage_score"],
            -row["level3_nmi_damage"],
            -row["level3_probe_damage"],
            row["stress_case"],
        )
    )
    most_damaging_case = damage_ranking[0]["stress_case"] if damage_ranking else "none"
    least_damaging_case = damage_ranking[-1]["stress_case"] if damage_ranking else "none"
    robustness = "robust" if len(better_cases) >= max(1, len(thresholds) // 2) else "brittle"
    summary = (
        f"transition_lag_order survives degradation better than transition in {len(better_cases)}/{len(thresholds)} stress cases; "
        f"controlled loss hurts its level-3 signal in {len(degraded_cases)}/{len(thresholds)} cases; "
        f"mean level-3 damage is highest under {most_damaging_case}, so Context is currently {robustness} rather than free."
    )
    return {
        "transition_lag_order_beats_transition_case_count": len(better_cases),
        "degraded_transition_lag_order_weaker_case_count": len(degraded_cases),
        "degraded_transition_lag_order_weaker_cases": degraded_cases,
        "most_damaging_case": most_damaging_case,
        "least_damaging_case": least_damaging_case,
        "level3_damage_ranking": damage_ranking,
        "summary": summary,
    }


def probe_d_stress_bench() -> dict[str, Any]:
    case_results: dict[str, Any] = {}
    thresholds: dict[str, dict[str, float | None]] = {}
    table: list[dict[str, Any]] = []
    for stress_case in available_stress_case_names():
        case_results[stress_case] = {}
        thresholds[stress_case] = {}
        case_rows_by_encoder: dict[str, list[dict[str, Any]]] = {}
        for encoder in available_stress_encoder_names():
            case_results[stress_case][encoder] = {}
            case_rows_by_encoder[encoder] = []
            threshold: float | None = None
            for degradation in stress_degradation_levels():
                degradation_loss = degradation if encoder == "degraded_transition_lag_order" else 0.0
                levels = run_depth_encoder_diagnostics(
                    encoder,
                    stress_case=stress_case,
                    degradation=degradation,
                    degradation_loss=degradation_loss,
                )
                novelty = novelty_stream_metrics(
                    encoder=encoder,
                    stress_case=stress_case,
                    degradation=degradation,
                    degradation_loss=degradation_loss,
                )
                current_collapse_level = collapse_level(levels)
                if threshold is None and current_collapse_level != "none":
                    threshold = float(degradation)
                degradation_key = f"{degradation:.2f}"
                case_results[stress_case][encoder][degradation_key] = {
                    "levels": levels,
                    "novelty": novelty,
                    "collapse_level": current_collapse_level,
                }
                row = {
                    "stress_case": stress_case,
                    "degradation": float(degradation),
                    "encoder": encoder,
                    "level1_abstraction_nmi": levels["level1"]["abstraction_nmi"],
                    "level1_supervised_probe_accuracy": levels["level1"]["supervised_probe_accuracy"],
                    "level1_abstraction_count": levels["level1"]["abstraction_count"],
                    "level2_abstraction_nmi": levels["level2"]["abstraction_nmi"],
                    "level2_supervised_probe_accuracy": levels["level2"]["supervised_probe_accuracy"],
                    "level2_abstraction_count": levels["level2"]["abstraction_count"],
                    "level3_abstraction_nmi": levels["level3"]["abstraction_nmi"],
                    "level3_supervised_probe_accuracy": levels["level3"]["supervised_probe_accuracy"],
                    "level3_abstraction_count": levels["level3"]["abstraction_count"],
                    "abstain_spike_rate": novelty["abstain_spike_rate"],
                    "collapse_level": current_collapse_level,
                    "degradation_threshold": None,
                }
                case_rows_by_encoder[encoder].append(row)
                level3_metrics = levels["level3"]
                level3_collapsed = (
                    level3_metrics["abstraction_nmi"] < 0.50
                    or level3_metrics["supervised_probe_accuracy"] < 0.75
                )
                if threshold is None and degradation > 0.0 and level3_collapsed:
                    threshold = float(degradation)
            thresholds[stress_case][encoder] = threshold
            for row in case_rows_by_encoder[encoder]:
                row["degradation_threshold"] = threshold
                table.append(row)

    return {
        "results": case_results,
        "thresholds": thresholds,
        "table": table,
        "diagnosis": diagnose_probe_d(case_results, thresholds),
    }


def _ordered_feature_block(features: np.ndarray, vocab_size: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    lag_dim = int(vocab_size) * int(vocab_size)
    ordered = features[:, 2 * lag_dim :]
    if ordered.shape[1] == 0:
        return np.zeros((features.shape[0], 1), dtype=np.float64)
    return ordered


def _ordered_feature_tensor(features: np.ndarray, vocab_size: int) -> np.ndarray:
    ordered = _ordered_feature_block(features, vocab_size)
    pair_dim = max(1, int(vocab_size) * int(vocab_size))
    slot_count = max(1, ordered.shape[1] // pair_dim)
    return np.asarray(ordered, dtype=np.float64).reshape(ordered.shape[0], slot_count, pair_dim)


def _shift_ordered_slots(tensor: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(tensor)
    if shift == 0:
        return tensor
    if shift > 0:
        shifted[shift:] = tensor[:-shift]
        return shifted
    shifted[:shift] = tensor[-shift:]
    return shifted


def _order_alignment_errors(features: np.ndarray, labels: np.ndarray, vocab_size: int) -> tuple[float, float]:
    ordered = _ordered_feature_tensor(features, vocab_size)
    labels = np.asarray(labels, dtype=np.int64)
    slot_count = int(ordered.shape[1])
    strict_errors: list[float] = []
    phase_tolerant_errors: list[float] = []
    for idx, tensor in enumerate(ordered):
        mask = labels == labels[idx]
        if mask.sum() > 1:
            mask = mask.copy()
            mask[idx] = False
        centroid = ordered[mask].mean(axis=0)
        strict_error = float(((tensor - centroid) ** 2).mean())
        phase_tolerant_error = min(
            float(((tensor - _shift_ordered_slots(centroid, shift=shift)) ** 2).mean())
            for shift in range(-(max(1, slot_count) - 1), max(1, slot_count))
        )
        strict_errors.append(strict_error)
        phase_tolerant_errors.append(phase_tolerant_error)
    return float(np.mean(strict_errors)), float(np.mean(phase_tolerant_errors))


def _strict_order_recovery_accuracy(features: np.ndarray, labels: np.ndarray, vocab_size: int) -> float:
    return supervised_probe_metrics(_ordered_feature_block(features, vocab_size), labels)["accuracy"]


def _phase_tolerant_order_recovery_accuracy(features: np.ndarray, labels: np.ndarray, vocab_size: int) -> float:
    ordered = _ordered_feature_tensor(features, vocab_size)
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    predictions = np.zeros_like(labels)
    slot_count = int(ordered.shape[1])

    for idx, tensor in enumerate(ordered):
        best_label = int(unique_labels[0])
        best_distance = float("inf")
        for label in unique_labels:
            mask = labels == label
            if labels[idx] == label:
                mask = mask.copy()
                mask[idx] = False
            if bool(mask.any()):
                centroid = ordered[mask].mean(axis=0)
            else:
                centroid = ordered[labels == label].mean(axis=0)
            distance = min(
                float(((tensor - _shift_ordered_slots(centroid, shift=shift)) ** 2).mean())
                for shift in range(-(max(1, slot_count) - 1), max(1, slot_count))
            )
            if distance < best_distance:
                best_distance = distance
                best_label = int(label)
        predictions[idx] = best_label
    return float((predictions == labels).mean())


def _probe_e_failure_mode(
    semantic_confusion_score: float,
    order_loss_score: float,
    high_threshold: float = 0.10,
    margin: float = 0.05,
) -> str:
    if semantic_confusion_score >= high_threshold and order_loss_score >= high_threshold:
        return "context_to_abstraction_bridge_failure"
    if order_loss_score > semantic_confusion_score + margin:
        return "context_ordering_problem"
    if semantic_confusion_score > order_loss_score + margin:
        return "abstraction_classification_problem"
    if semantic_confusion_score > 0.0 or order_loss_score > 0.0:
        return "mixed_low_damage"
    return "low_damage"


def _probe_f_failure_mode(
    phase_error_score: float,
    rank_instability_score: float,
    high_threshold: float = 0.10,
    margin: float = 0.05,
) -> str:
    if phase_error_score >= high_threshold and rank_instability_score >= high_threshold:
        return "mixed_order_damage"
    if phase_error_score > rank_instability_score + margin:
        return "phase_error_problem"
    if rank_instability_score > phase_error_score + margin:
        return "rank_instability_problem"
    if phase_error_score > 0.0 or rank_instability_score > 0.0:
        return "mixed_low_order_damage"
    return "low_order_damage"


def probe_e_damage_decomposition() -> dict[str, Any]:
    case_results: dict[str, dict[str, dict[str, Any]]] = {}
    table: list[dict[str, Any]] = []
    for stress_case in available_stress_case_names():
        case_results[stress_case] = {}
        degradation_rows: list[dict[str, Any]] = []
        for degradation in stress_degradation_levels():
            if degradation <= 0.0:
                continue
            full_state = _depth_encoder_state(
                "transition_lag_order",
                stress_case=stress_case,
                degradation=degradation,
            )
            degraded_state = _depth_encoder_state(
                "degraded_transition_lag_order",
                stress_case=stress_case,
                degradation=degradation,
                degradation_loss=degradation,
            )
            full_level3 = _level_metrics_from_state(full_state["level3"])
            degraded_level3 = _level_metrics_from_state(degraded_state["level3"])
            full_order_recovery_accuracy = supervised_probe_metrics(
                _ordered_feature_block(full_state["level3"]["features"], full_state["level3"]["vocab_size"]),
                full_state["level3"]["labels"],
            )["accuracy"]
            degraded_order_recovery_accuracy = supervised_probe_metrics(
                _ordered_feature_block(degraded_state["level3"]["features"], degraded_state["level3"]["vocab_size"]),
                degraded_state["level3"]["labels"],
            )["accuracy"]
            semantic_confusion_score = max(
                0.0,
                float(full_level3["abstraction_nmi"]) - float(degraded_level3["abstraction_nmi"]),
            )
            order_loss_score = max(0.0, float(full_order_recovery_accuracy) - float(degraded_order_recovery_accuracy))
            total_damage_score = semantic_confusion_score + order_loss_score
            row = {
                "stress_case": stress_case,
                "degradation": float(degradation),
                "semantic_confusion_score": semantic_confusion_score,
                "order_loss_score": order_loss_score,
                "total_damage_score": total_damage_score,
                "supervised_probe_accuracy": float(degraded_level3["supervised_probe_accuracy"]),
                "abstraction_nmi": float(degraded_level3["abstraction_nmi"]),
                "order_recovery_accuracy": float(degraded_order_recovery_accuracy),
                "failure_mode": _probe_e_failure_mode(semantic_confusion_score, order_loss_score),
            }
            degradation_key = f"{degradation:.2f}"
            case_results[stress_case][degradation_key] = row
            degradation_rows.append(row)
        semantic_confusion_score = float(np.mean([row["semantic_confusion_score"] for row in degradation_rows]))
        order_loss_score = float(np.mean([row["order_loss_score"] for row in degradation_rows]))
        total_damage_score = float(np.mean([row["total_damage_score"] for row in degradation_rows]))
        supervised_probe_accuracy = float(np.mean([row["supervised_probe_accuracy"] for row in degradation_rows]))
        abstraction_nmi = float(np.mean([row["abstraction_nmi"] for row in degradation_rows]))
        order_recovery_accuracy = float(np.mean([row["order_recovery_accuracy"] for row in degradation_rows]))
        table.append({
            "stress_case": stress_case,
            "semantic_confusion_score": semantic_confusion_score,
            "order_loss_score": order_loss_score,
            "total_damage_score": total_damage_score,
            "supervised_probe_accuracy": supervised_probe_accuracy,
            "abstraction_nmi": abstraction_nmi,
            "order_recovery_accuracy": order_recovery_accuracy,
            "failure_mode": _probe_e_failure_mode(semantic_confusion_score, order_loss_score),
        })

    table.sort(
        key=lambda row: (
            -row["total_damage_score"],
            -row["order_loss_score"],
            -row["semantic_confusion_score"],
            row["stress_case"],
        )
    )
    most_total_damage_case = table[0]["stress_case"] if table else "none"
    least_total_damage_case = table[-1]["stress_case"] if table else "none"
    most_order_loss_case = max(table, key=lambda row: row["order_loss_score"])["stress_case"] if table else "none"
    most_semantic_confusion_case = max(table, key=lambda row: row["semantic_confusion_score"])["stress_case"] if table else "none"
    summary = (
        f"level-3 damage splits into semantic confusion and order loss; total damage is highest under {most_total_damage_case}, "
        f"order loss is highest under {most_order_loss_case}, and semantic confusion is highest under {most_semantic_confusion_case}."
    )
    return {
        "results": case_results,
        "table": table,
        "diagnosis": {
            "most_total_damage_case": most_total_damage_case,
            "least_total_damage_case": least_total_damage_case,
            "most_order_loss_case": most_order_loss_case,
            "most_semantic_confusion_case": most_semantic_confusion_case,
            "summary": summary,
        },
    }


def probe_f_order_loss_split() -> dict[str, Any]:
    case_results: dict[str, dict[str, dict[str, Any]]] = {}
    table: list[dict[str, Any]] = []
    for stress_case in available_stress_case_names():
        case_results[stress_case] = {}
        degradation_rows: list[dict[str, Any]] = []
        for degradation in stress_degradation_levels():
            if degradation <= 0.0:
                continue
            full_state = _depth_encoder_state(
                "transition_lag_order",
                stress_case=stress_case,
                degradation=degradation,
            )
            degraded_state = _depth_encoder_state(
                "degraded_transition_lag_order",
                stress_case=stress_case,
                degradation=degradation,
                degradation_loss=degradation,
            )
            full_level3 = full_state["level3"]
            degraded_level3 = degraded_state["level3"]
            full_strict_order_recovery_accuracy = _strict_order_recovery_accuracy(
                full_level3["features"],
                full_level3["labels"],
                full_level3["vocab_size"],
            )
            degraded_strict_order_recovery_accuracy = _strict_order_recovery_accuracy(
                degraded_level3["features"],
                degraded_level3["labels"],
                degraded_level3["vocab_size"],
            )
            full_phase_tolerant_order_recovery_accuracy = _phase_tolerant_order_recovery_accuracy(
                full_level3["features"],
                full_level3["labels"],
                full_level3["vocab_size"],
            )
            degraded_phase_tolerant_order_recovery_accuracy = _phase_tolerant_order_recovery_accuracy(
                degraded_level3["features"],
                degraded_level3["labels"],
                degraded_level3["vocab_size"],
            )
            total_order_loss_score = max(
                0.0,
                float(full_strict_order_recovery_accuracy) - float(degraded_strict_order_recovery_accuracy),
            )
            full_strict_order_alignment_error, full_phase_tolerant_order_alignment_error = _order_alignment_errors(
                full_level3["features"],
                full_level3["labels"],
                full_level3["vocab_size"],
            )
            degraded_strict_order_alignment_error, degraded_phase_tolerant_order_alignment_error = _order_alignment_errors(
                degraded_level3["features"],
                degraded_level3["labels"],
                degraded_level3["vocab_size"],
            )
            raw_phase_error = max(0.0, full_strict_order_alignment_error - full_phase_tolerant_order_alignment_error)
            raw_rank_instability = max(0.0, full_phase_tolerant_order_alignment_error)
            raw_total = raw_phase_error + raw_rank_instability
            if total_order_loss_score <= 0.0:
                phase_error_score = 0.0
                rank_instability_score = 0.0
            elif raw_total > 0.0:
                phase_error_score = float(total_order_loss_score) * (float(raw_phase_error) / float(raw_total))
                rank_instability_score = max(0.0, float(total_order_loss_score) - float(phase_error_score))
            else:
                phase_error_score = 0.0
                rank_instability_score = float(total_order_loss_score)
            row = {
                "stress_case": stress_case,
                "degradation": float(degradation),
                "phase_error_score": phase_error_score,
                "rank_instability_score": rank_instability_score,
                "ordered_implication_damage_score": total_order_loss_score,
                "total_order_loss_score": total_order_loss_score,
                "strict_order_alignment_error": float(full_strict_order_alignment_error),
                "phase_tolerant_order_alignment_error": float(full_phase_tolerant_order_alignment_error),
                "order_failure_mode": _probe_f_failure_mode(phase_error_score, rank_instability_score),
            }
            degradation_key = f"{degradation:.2f}"
            case_results[stress_case][degradation_key] = row
            degradation_rows.append(row)
        phase_error_score = float(np.mean([row["phase_error_score"] for row in degradation_rows]))
        rank_instability_score = float(np.mean([row["rank_instability_score"] for row in degradation_rows]))
        total_order_loss_score = float(np.mean([row["total_order_loss_score"] for row in degradation_rows]))
        strict_order_alignment_error = float(np.mean([row["strict_order_alignment_error"] for row in degradation_rows]))
        phase_tolerant_order_alignment_error = float(
            np.mean([row["phase_tolerant_order_alignment_error"] for row in degradation_rows])
        )
        table.append({
            "stress_case": stress_case,
            "phase_error_score": phase_error_score,
            "rank_instability_score": rank_instability_score,
            "ordered_implication_damage_score": total_order_loss_score,
            "total_order_loss_score": total_order_loss_score,
            "strict_order_alignment_error": strict_order_alignment_error,
            "phase_tolerant_order_alignment_error": phase_tolerant_order_alignment_error,
            "order_failure_mode": _probe_f_failure_mode(phase_error_score, rank_instability_score),
        })

    table.sort(
        key=lambda row: (
            -row["total_order_loss_score"],
            -row["rank_instability_score"],
            -row["phase_error_score"],
            row["stress_case"],
        )
    )
    natural_rows = [row for row in table if row["stress_case"] != "pure_phase_offset"]
    most_total_order_loss_case = table[0]["stress_case"] if table else "none"
    least_total_order_loss_case = table[-1]["stress_case"] if table else "none"
    max_phase_error_score = max((row["phase_error_score"] for row in table), default=0.0)
    max_rank_instability_score = max((row["rank_instability_score"] for row in table), default=0.0)
    max_natural_phase_error_score = max((row["phase_error_score"] for row in natural_rows), default=0.0)
    max_natural_rank_instability_score = max((row["rank_instability_score"] for row in natural_rows), default=0.0)
    overall_most_phase_error_case = "none"
    if max_phase_error_score > 0.0:
        overall_most_phase_error_case = max(
            table,
            key=lambda row: (row["phase_error_score"], row["total_order_loss_score"], row["stress_case"]),
        )["stress_case"]
    overall_most_rank_instability_case = "none"
    if max_rank_instability_score > 0.0:
        overall_most_rank_instability_case = max(
            table,
            key=lambda row: (row["rank_instability_score"], row["total_order_loss_score"], row["stress_case"]),
        )["stress_case"]
    most_phase_error_case = "none"
    if max_natural_phase_error_score > 0.0:
        most_phase_error_case = max(
            natural_rows,
            key=lambda row: (row["phase_error_score"], row["total_order_loss_score"], row["stress_case"]),
        )["stress_case"]
    most_rank_instability_case = "none"
    if max_natural_rank_instability_score > 0.0:
        most_rank_instability_case = max(
            natural_rows,
            key=lambda row: (row["rank_instability_score"], row["total_order_loss_score"], row["stress_case"]),
        )["stress_case"]
    phase_summary = "no natural stressor shows measurable phase error"
    if most_phase_error_case != "none":
        phase_summary = f"the most phase-like natural stressor is {most_phase_error_case}"
    rank_summary = "no natural stressor shows measurable rank instability"
    if most_rank_instability_case != "none":
        rank_summary = f"the most rank-unstable natural stressor is {most_rank_instability_case}"
    variable_lag_row = next((row for row in table if row["stress_case"] == "variable_lag"), None)
    variable_lag_interpretation = "variable_lag is absent from the current Probe F table"
    if variable_lag_row is not None:
        if (
            variable_lag_row["phase_error_score"] > 0.0
            and variable_lag_row["phase_error_score"] > variable_lag_row["rank_instability_score"]
        ):
            variable_lag_interpretation = (
                "variable_lag behaves like phase drift: the same ranked implication is preserved but shifted in time"
            )
        elif variable_lag_row["rank_instability_score"] > 0.0:
            variable_lag_interpretation = (
                "variable_lag remains rank instability dominated: candidate continuation order is unstable rather than merely delayed"
            )
        else:
            variable_lag_interpretation = "variable_lag shows no measurable ordered implication damage in the current toy"
    pure_phase_offset_row = next((row for row in table if row["stress_case"] == "pure_phase_offset"), None)
    pure_phase_offset_interpretation = "pure_phase_offset is absent from the current Probe F table"
    phase_control_validated = False
    if pure_phase_offset_row is not None:
        phase_control_validated = pure_phase_offset_row["phase_error_score"] > 0.0
        if pure_phase_offset_row["phase_error_score"] > pure_phase_offset_row["rank_instability_score"]:
            pure_phase_offset_interpretation = (
                "pure_phase_offset behaves like phase drift: the same ranked implication is preserved but shifted in time"
            )
        elif pure_phase_offset_row["phase_error_score"] > 0.0:
            pure_phase_offset_interpretation = (
                "pure_phase_offset validates the phase side: measurable phase error appears without variable_lag-scale rank instability"
            )
        else:
            pure_phase_offset_interpretation = "pure_phase_offset does not trigger measurable phase error: the phase side remains blind"
    synthetic_phase_control_case = "pure_phase_offset" if pure_phase_offset_row is not None else "none"
    synthetic_phase_control_summary = "synthetic phase control is absent"
    if synthetic_phase_control_case != "none":
        synthetic_phase_control_summary = f"synthetic phase control {synthetic_phase_control_case} validated: {phase_control_validated}"
    ordered_implication_interpretation = (
        "ranked ordered implications are damaged when the model no longer preserves which continuation lawfully follows"
    )
    phase_pov_interpretation = "phase drift = same thread, shifted in time"
    rank_pov_interpretation = "rank instability = candidate threads lost lawful ordering"
    summary = (
        f"Probe F sharpens ordered implication damage into phase drift and rank instability; ordered implication damage is highest under {most_total_order_loss_case}; "
        f"{synthetic_phase_control_summary}; {phase_summary}; {rank_summary}."
    )
    return {
        "results": case_results,
        "table": table,
        "diagnosis": {
            "most_total_order_loss_case": most_total_order_loss_case,
            "most_ordered_implication_damage_case": most_total_order_loss_case,
            "least_total_order_loss_case": least_total_order_loss_case,
            "synthetic_phase_control_case": synthetic_phase_control_case,
            "overall_most_phase_error_case": overall_most_phase_error_case,
            "overall_most_rank_instability_case": overall_most_rank_instability_case,
            "most_natural_phase_error_case": most_phase_error_case,
            "most_natural_rank_instability_case": most_rank_instability_case,
            "most_phase_error_case": most_phase_error_case,
            "most_rank_instability_case": most_rank_instability_case,
            "ordered_implication_interpretation": ordered_implication_interpretation,
            "phase_pov_interpretation": phase_pov_interpretation,
            "rank_instability_pov_interpretation": rank_pov_interpretation,
            "variable_lag_interpretation": variable_lag_interpretation,
            "pure_phase_offset_interpretation": pure_phase_offset_interpretation,
            "phase_control_validated": phase_control_validated,
            "summary": summary,
        },
    }


def level_collapse_diagnostics() -> dict[str, Any]:
    encoder_results = {encoder: run_depth_encoder_diagnostics(encoder) for encoder in available_encoder_names()}
    encoder_level_table: list[dict[str, Any]] = []
    for encoder, levels in encoder_results.items():
        for level_name, metrics in levels.items():
            encoder_level_table.append({
                "encoder": encoder,
                "level": level_name,
                "abstraction_nmi": metrics["abstraction_nmi"],
                "abstraction_ari": metrics["abstraction_ari"],
                "supervised_probe_accuracy": metrics["supervised_probe_accuracy"],
                "supervised_probe_nmi": metrics["supervised_probe_nmi"],
                "abstraction_count": metrics["abstraction_count"],
            })

    recursion_results = {
        "classified_abstraction": run_classify_before_recursion_ablation("classified_abstraction"),
        "raw_context": run_classify_before_recursion_ablation("raw_context"),
    }
    recursion_table: list[dict[str, Any]] = []
    for mode, levels in recursion_results.items():
        for level_name, metrics in levels.items():
            recursion_table.append({
                "mode": mode,
                "level": level_name,
                "abstraction_nmi": metrics["abstraction_nmi"],
                "abstraction_ari": metrics["abstraction_ari"],
                "supervised_probe_accuracy": metrics["supervised_probe_accuracy"],
                "supervised_probe_nmi": metrics["supervised_probe_nmi"],
                "abstraction_count": metrics["abstraction_count"],
            })

    return {
        "encoder_results": encoder_results,
        "encoder_level_table": encoder_level_table,
        "classify_before_recursion": {
            "results": recursion_results,
            "table": recursion_table,
        },
        "diagnosis": diagnose_level3_collapse(encoder_results, recursion_results),
    }


def depth_probe_metrics() -> dict[str, Any]:
    return level_collapse_diagnostics()["encoder_results"]["transition"]


def _first_abstraction_for_label(predictions: np.ndarray, labels: np.ndarray, label: int) -> int:
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
    same_relation = run_probe(same_sequences, same_labels, same_vocab, "context_relation_space")
    anchor_hist = run_probe(anchor_sequences, anchor_labels, anchor_vocab, "histogram")
    anchor_relation = run_probe(anchor_sequences, anchor_labels, anchor_vocab, "context_relation_space")
    collapse_diagnostics = level_collapse_diagnostics()
    probe_d_stress = probe_d_stress_bench()
    probe_e_decomposition = probe_e_damage_decomposition()
    probe_f_order_loss = probe_f_order_loss_split()
    return {
        "seed": seed,
        "same_marginal": {
            "histogram": same_hist.__dict__,
            "context_relation_space": same_relation.__dict__,
            "relation_nmi_delta": same_relation.nmi - same_hist.nmi,
            "acceptance_pass": (
                (same_hist.nmi <= 0.25 or same_hist.accuracy <= 0.60)
                and same_relation.nmi >= 0.70
                and same_relation.accuracy >= 0.85
            ),
        },
        "shared_anchor": {
            "histogram": anchor_hist.__dict__,
            "context_relation_space": anchor_relation.__dict__,
            "relation_nmi_delta": anchor_relation.nmi - anchor_hist.nmi,
            "acceptance_pass": (anchor_relation.nmi - anchor_hist.nmi) >= 0.40,
        },
        "novelty": novelty_stream_metrics(),
        "depth_probe": collapse_diagnostics["encoder_results"]["transition"],
        "collapse_diagnostics": collapse_diagnostics,
        "probe_d_stress": probe_d_stress,
        "probe_e_decomposition": probe_e_decomposition,
        "probe_f_order_loss_split": probe_f_order_loss,
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
        "same_relation_nmi": ("same_marginal", "context_relation_space", "nmi"),
        "same_relation_accuracy": ("same_marginal", "context_relation_space", "accuracy"),
        "shared_histogram_nmi": ("shared_anchor", "histogram", "nmi"),
        "shared_relation_nmi": ("shared_anchor", "context_relation_space", "nmi"),
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


def adjusted_rand_index(labels: np.ndarray | list[int], predictions: np.ndarray | list[int]) -> float:
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
    total = int(labels.size)
    if total < 2:
        return 0.0
    label_values, label_inverse = np.unique(labels, return_inverse=True)
    pred_values, pred_inverse = np.unique(predictions, return_inverse=True)
    contingency = np.zeros((label_values.size, pred_values.size), dtype=np.int64)
    for left, right in zip(label_inverse, pred_inverse):
        contingency[int(left), int(right)] += 1

    sum_comb = float(np.sum(contingency * (contingency - 1) // 2))
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    row_comb = float(np.sum(row_sums * (row_sums - 1) // 2))
    col_comb = float(np.sum(col_sums * (col_sums - 1) // 2))
    total_comb = float(total * (total - 1) // 2)
    if total_comb <= 0.0:
        return 0.0
    expected = (row_comb * col_comb) / total_comb
    maximum = 0.5 * (row_comb + col_comb)
    denominator = maximum - expected
    if abs(denominator) <= 1e-12:
        return 0.0
    return float((sum_comb - expected) / denominator)


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
