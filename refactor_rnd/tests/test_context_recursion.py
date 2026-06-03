from __future__ import annotations

import numpy as np

from refactor_rnd.context_recursion import (
    Abstraction,
    AbstainGatedAbstractor,
    Context,
    Trit,
    TritVote,
    build_context,
    context_relation_space_encoder,
    depth_probe_metrics,
    histogram_encoder,
    novelty_stream_metrics,
    run_all,
    run_probe,
    same_marginal_order_sequences,
    shared_anchor_sequences,
)


def test_context_is_variable_size_relation_context():
    trits = tuple(Trit(unit_id=i % 3, vote=TritVote.APPROVE, time=i) for i in range(5))
    context = Context(trits=trits, frame=7, level=1, relation_matrix=np.zeros((3, 3)))

    assert context.size == 5
    assert context.unit_ids == (0, 1, 2, 0, 1)
    assert context.frame == 7
    assert context.level == 1


def test_context_relation_space_changes_when_marginals_match():
    left = [0, 1, 0, 1, 2, 2]
    right = [0, 2, 0, 2, 1, 1]

    assert np.allclose(histogram_encoder(left, 3), histogram_encoder(right, 3))
    assert not np.allclose(context_relation_space_encoder(left, 3), context_relation_space_encoder(right, 3))


def test_build_context_preserves_directed_relations():
    context = build_context([0, 1, 0, 1], vocab_size=2)

    assert context.size == 4
    assert context.relation_matrix.shape == (2, 2)
    assert context.relation_matrix[0, 1] > 0.0
    assert context.relation_matrix[1, 0] > 0.0


def test_same_marginal_histogram_fails_relation_space_passes():
    sequences, labels, vocab_size = same_marginal_order_sequences(samples_per_regime=12)

    histogram = run_probe(sequences, labels, vocab_size, "histogram")
    relation = run_probe(sequences, labels, vocab_size, "context_relation_space")

    assert histogram.nmi <= 0.25 or histogram.accuracy <= 0.60
    assert relation.nmi >= 0.70
    assert relation.accuracy >= 0.85


def test_shared_anchor_relation_space_beats_histogram():
    sequences, labels, vocab_size = shared_anchor_sequences(samples_per_regime=12)

    histogram = run_probe(sequences, labels, vocab_size, "histogram")
    relation = run_probe(sequences, labels, vocab_size, "context_relation_space")

    assert relation.nmi - histogram.nmi >= 0.40


def test_novelty_stream_re_spikes_abstain_and_settles():
    metrics = novelty_stream_metrics()

    assert metrics["novelty_spike_1"] > 0.0
    assert metrics["novelty_spike_2"] > 0.0
    assert metrics["phase1_settle_delta"] > 0.0
    assert metrics["phase2_settle_delta"] > 0.0
    assert metrics["abstraction_count"] == 3.0


def test_depth_probe_reports_levels_without_pass_claim():
    metrics = depth_probe_metrics()

    assert set(metrics) == {"level1", "level2", "level3"}
    for level in metrics.values():
        assert set(level) == {"nmi", "accuracy", "abstraction_count", "abstain_rate"}
        assert 0.0 <= level["nmi"] <= 1.0
        assert 0.0 <= level["accuracy"] <= 1.0
        assert level["abstraction_count"] >= 1


def test_abstractor_creates_abstractions_after_evidence_threshold():
    abstractor = AbstainGatedAbstractor(approve_radius=0.0, min_evidence=2)
    first = np.array([1.0, 0.0])
    second = np.array([1.0, 0.0])

    assert abstractor.observe(first, label=0) == -1
    created_id = abstractor.observe(second, label=0)

    assert created_id == 0
    assert len(abstractor.abstractions) == 1
    assert isinstance(abstractor.abstractions[0], Abstraction)


def test_run_all_reports_acceptance():
    metrics = run_all(seed=31)

    assert metrics["same_marginal"]["acceptance_pass"]
    assert metrics["shared_anchor"]["acceptance_pass"]
    assert metrics["novelty"]["novelty_spike_1"] > 0.0
