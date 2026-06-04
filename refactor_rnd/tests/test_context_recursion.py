from __future__ import annotations

import numpy as np

from refactor_rnd.context_recursion import (
    Abstraction,
    AbstainGatedAbstractor,
    aggregate_results,
    apply_stress_case,
    Context,
    Trit,
    TritVote,
    available_stress_case_names,
    available_stress_encoder_names,
    build_context,
    canonical_encoder_name,
    cooccurrence_encoder,
    context_relation_space_encoder,
    encode_sequences,
    depth_probe_metrics,
    degraded_transition_lag_order_encoder,
    histogram_encoder,
    level_collapse_diagnostics,
    novelty_stream_metrics,
    probe_d_stress_bench,
    probe_e_damage_decomposition,
    probe_f_order_loss_split,
    run_all,
    run_classify_before_recursion_ablation,
    run_depth_encoder_diagnostics,
    run_probe,
    same_marginal_order_sequences,
    shared_anchor_sequences,
    stress_degradation_levels,
    transition_lag_order_encoder,
)
from refactor_rnd.run_context_recursion import _summary_markdown


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
    assert np.allclose(cooccurrence_encoder(left, 3), cooccurrence_encoder(right, 3))
    assert not np.allclose(context_relation_space_encoder(left, 3), context_relation_space_encoder(right, 3))
    assert not np.allclose(transition_lag_order_encoder(left, 3), transition_lag_order_encoder(right, 3))
    assert not np.allclose(degraded_transition_lag_order_encoder(left, 3, 0.75), degraded_transition_lag_order_encoder(right, 3, 0.75))


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
        assert {"nmi", "accuracy", "abstraction_nmi", "abstraction_ari", "supervised_probe_accuracy", "supervised_probe_nmi", "abstraction_count", "abstain_rate"}.issubset(level)
        assert 0.0 <= level["abstraction_nmi"] <= 1.0
        assert -1.0 <= level["abstraction_ari"] <= 1.0
        assert 0.0 <= level["supervised_probe_accuracy"] <= 1.0
        assert 0.0 <= level["supervised_probe_nmi"] <= 1.0
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
    assert "collapse_diagnostics" in metrics


def test_transition_encoder_level3_drops_signal_but_order_encoder_recovers_it():
    transition = run_depth_encoder_diagnostics("transition")
    ordered = run_depth_encoder_diagnostics("transition_lag_order")

    assert transition["level3"]["supervised_probe_accuracy"] <= 0.60
    assert transition["level3"]["abstraction_nmi"] <= 0.25
    assert ordered["level3"]["supervised_probe_accuracy"] >= transition["level3"]["supervised_probe_accuracy"]


def test_classify_before_recursion_ablation_reports_modes():
    classified = run_classify_before_recursion_ablation("classified_abstraction")
    raw_context = run_classify_before_recursion_ablation("raw_context")

    assert set(classified) == {"level1", "level2", "level3"}
    assert set(raw_context) == {"level1", "level2", "level3"}
    assert classified["level3"]["abstraction_count"] >= 1
    assert raw_context["level3"]["abstraction_count"] >= 1


def test_level_collapse_diagnostics_reports_tables_and_diagnosis():
    diagnostics = level_collapse_diagnostics()

    assert set(diagnostics["encoder_results"]) == {"histogram", "cooccurrence", "transition", "transition_lag_order"}
    assert diagnostics["diagnosis"]["primary_cause"] in {"classifier_weakness", "information_loss", "true_recursion_depth_failure", "mixed"}
    assert diagnostics["diagnosis"]["recursion_takeaway"] in {
        "classification_before_recursion_helps",
        "raw_context_recursion_helps",
        "no_clear_recursion_mode_winner",
    }
    assert canonical_encoder_name("context_relation_space") == "transition"


def test_variable_lag_inserts_delay_steps_literally():
    stressed = apply_stress_case([0, 1, 2, 3], "variable_lag", degradation=0.50, label=0, sample_index=0, vocab_size=4)

    assert stressed == [0, 0, 1, 2, 3]


def test_pure_phase_offset_translates_sequence_with_blank_padding():
    stressed = apply_stress_case([0, 1, 2, 3], "pure_phase_offset", degradation=1.00, label=0, sample_index=1, vocab_size=4)

    assert stressed == [-1, 0, 1, 2]


def test_missing_events_removes_steps_literally():
    stressed = apply_stress_case([0, 1, 2, 3, 4, 5], "missing_events", degradation=0.75, label=0, sample_index=0, vocab_size=6)

    assert stressed == [1, 3, 5]
    assert -1 not in stressed


def test_probe_d_stress_bench_reports_cases_and_thresholds():
    stress = probe_d_stress_bench()

    assert set(stress["results"]) == set(available_stress_case_names())
    assert stress["diagnosis"]["transition_lag_order_beats_transition_case_count"] >= 1
    assert stress["diagnosis"]["degraded_transition_lag_order_weaker_case_count"] >= 1
    assert stress["diagnosis"]["degraded_transition_lag_order_weaker_cases"]
    assert len(stress["diagnosis"]["level3_damage_ranking"]) == len(available_stress_case_names())
    for stress_case in available_stress_case_names():
        assert set(stress["results"][stress_case]) == set(available_stress_encoder_names())
    assert len(stress["table"]) == (
        len(available_stress_case_names())
        * len(available_stress_encoder_names())
        * len(stress_degradation_levels())
    )


def test_probe_e_damage_decomposition_reports_case_scores():
    decomposition = probe_e_damage_decomposition()

    assert set(decomposition["results"]) == set(available_stress_case_names())
    assert len(decomposition["table"]) == len(available_stress_case_names())
    assert decomposition["diagnosis"]["most_total_damage_case"] in set(available_stress_case_names())
    assert decomposition["diagnosis"]["most_order_loss_case"] in set(available_stress_case_names())
    assert decomposition["diagnosis"]["most_semantic_confusion_case"] in set(available_stress_case_names())


def test_probe_e_table_matches_mean_degradation_rows():
    decomposition = probe_e_damage_decomposition()
    table_by_case = {row["stress_case"]: row for row in decomposition["table"]}

    for stress_case, rows in decomposition["results"].items():
        expected_semantic = float(np.mean([row["semantic_confusion_score"] for row in rows.values()]))
        expected_order = float(np.mean([row["order_loss_score"] for row in rows.values()]))
        expected_total = float(np.mean([row["total_damage_score"] for row in rows.values()]))
        expected_probe = float(np.mean([row["supervised_probe_accuracy"] for row in rows.values()]))
        expected_nmi = float(np.mean([row["abstraction_nmi"] for row in rows.values()]))
        expected_order_recovery = float(np.mean([row["order_recovery_accuracy"] for row in rows.values()]))
        row = table_by_case[stress_case]

        assert np.isclose(row["semantic_confusion_score"], expected_semantic)
        assert np.isclose(row["order_loss_score"], expected_order)
        assert np.isclose(row["total_damage_score"], expected_total)
        assert np.isclose(row["supervised_probe_accuracy"], expected_probe)
        assert np.isclose(row["abstraction_nmi"], expected_nmi)
        assert np.isclose(row["order_recovery_accuracy"], expected_order_recovery)
        assert row["failure_mode"] in {
            "context_ordering_problem",
            "abstraction_classification_problem",
            "context_to_abstraction_bridge_failure",
            "mixed_low_damage",
            "low_damage",
        }


def test_probe_f_order_loss_split_reports_case_scores():
    decomposition = probe_f_order_loss_split()
    diagnosis = decomposition["diagnosis"]

    assert set(decomposition["results"]) == set(available_stress_case_names())
    assert len(decomposition["table"]) == len(available_stress_case_names())
    assert diagnosis["most_total_order_loss_case"] in set(available_stress_case_names())
    assert diagnosis["most_ordered_implication_damage_case"] == diagnosis["most_total_order_loss_case"]
    assert diagnosis["synthetic_phase_control_case"] == "pure_phase_offset"
    assert diagnosis["overall_most_phase_error_case"] in set(available_stress_case_names()) | {"none"}
    assert diagnosis["overall_most_rank_instability_case"] in set(available_stress_case_names()) | {"none"}
    assert diagnosis["most_natural_phase_error_case"] in (set(available_stress_case_names()) - {"pure_phase_offset"}) | {"none"}
    assert diagnosis["most_natural_rank_instability_case"] in (set(available_stress_case_names()) - {"pure_phase_offset"}) | {"none"}
    assert diagnosis["most_phase_error_case"] in set(available_stress_case_names()) | {"none"}
    assert diagnosis["most_rank_instability_case"] in set(available_stress_case_names()) | {"none"}
    assert "ranked ordered implications" in diagnosis["ordered_implication_interpretation"]
    assert diagnosis["phase_pov_interpretation"] == "phase drift = same thread, shifted in time"
    assert diagnosis["rank_instability_pov_interpretation"] == "rank instability = candidate threads lost lawful ordering"
    assert diagnosis["phase_control_validated"] is True
    assert "phase" in diagnosis["pure_phase_offset_interpretation"]


def test_probe_f_table_matches_mean_degradation_rows_and_ranking():
    decomposition = probe_f_order_loss_split()
    table_by_case = {row["stress_case"]: row for row in decomposition["table"]}

    for stress_case, rows in decomposition["results"].items():
        expected_phase = float(np.mean([row["phase_error_score"] for row in rows.values()]))
        expected_rank = float(np.mean([row["rank_instability_score"] for row in rows.values()]))
        expected_ordered_implication_damage = float(
            np.mean([row["ordered_implication_damage_score"] for row in rows.values()])
        )
        expected_total = float(np.mean([row["total_order_loss_score"] for row in rows.values()]))
        expected_strict = float(np.mean([row["strict_order_alignment_error"] for row in rows.values()]))
        expected_phase_tolerant = float(np.mean([row["phase_tolerant_order_alignment_error"] for row in rows.values()]))
        row = table_by_case[stress_case]

        assert np.isclose(row["phase_error_score"], expected_phase)
        assert np.isclose(row["rank_instability_score"], expected_rank)
        assert np.isclose(row["ordered_implication_damage_score"], expected_ordered_implication_damage)
        assert np.isclose(row["total_order_loss_score"], expected_total)
        assert np.isclose(row["ordered_implication_damage_score"], row["total_order_loss_score"])
        assert np.isclose(row["strict_order_alignment_error"], expected_strict)
        assert np.isclose(row["phase_tolerant_order_alignment_error"], expected_phase_tolerant)
        assert row["order_failure_mode"] in {
            "phase_error_problem",
            "rank_instability_problem",
            "mixed_order_damage",
            "mixed_low_order_damage",
            "low_order_damage",
        }

    expected_order = [
        row["stress_case"]
        for row in sorted(
            decomposition["table"],
            key=lambda row: (
                -row["total_order_loss_score"],
                -row["rank_instability_score"],
                -row["phase_error_score"],
                row["stress_case"],
            ),
        )
    ]
    actual_order = [row["stress_case"] for row in decomposition["table"]]
    assert actual_order == expected_order
    assert decomposition["diagnosis"]["most_total_order_loss_case"] == expected_order[0]
    pure_phase_offset = table_by_case["pure_phase_offset"]
    variable_lag = table_by_case["variable_lag"]
    noisy_order = table_by_case["noisy_order"]
    assert decomposition["diagnosis"]["phase_control_validated"] is True
    assert decomposition["diagnosis"]["most_natural_rank_instability_case"] == "variable_lag"
    assert decomposition["diagnosis"]["most_natural_phase_error_case"] != "pure_phase_offset"
    assert pure_phase_offset["phase_error_score"] > 0.0
    assert pure_phase_offset["rank_instability_score"] < variable_lag["rank_instability_score"]
    assert pure_phase_offset["rank_instability_score"] < noisy_order["rank_instability_score"]
    if np.isclose(max(row["phase_error_score"] for row in decomposition["table"]), 0.0):
        assert decomposition["diagnosis"]["most_phase_error_case"] == "none"
        assert "phase drift" not in decomposition["diagnosis"]["variable_lag_interpretation"]
    assert "rank instability" in decomposition["diagnosis"]["variable_lag_interpretation"] or "phase drift" in decomposition["diagnosis"]["variable_lag_interpretation"]


def test_probe_f_summary_uses_ordered_implication_language():
    metrics = run_all(seed=31)
    payload = {
        "aggregate": aggregate_results([metrics]),
        "per_seed": [metrics],
    }

    summary = _summary_markdown(payload)

    assert "## Probe F - Ordered Implication Split" in summary
    assert "ordered_implication_damage_score" in summary
    assert "synthetic_phase_control_case: pure_phase_offset" in summary
    assert "phase_control_validated: True" in summary
    assert "natural_stressor_most_phase_like_case:" in summary
    assert "natural_stressor_most_rank_instability_case: variable_lag" in summary
    assert "pure_phase_offset_interpretation:" in summary
    assert "variable_lag_interpretation:" in summary
    assert "rank instability" in summary


def test_probe_d_damage_ranking_matches_level3_deltas():
    stress = probe_d_stress_bench()
    weaker_cases: set[str] = set()
    expected_ranking: dict[str, dict[str, float]] = {}
    for stress_case, encoder_results in stress["results"].items():
        case_scores: list[float] = []
        case_nmi_damage: list[float] = []
        case_probe_damage: list[float] = []
        for degradation_key, full_payload in encoder_results["transition_lag_order"].items():
            if float(degradation_key) <= 0.0:
                continue
            degraded_payload = encoder_results["degraded_transition_lag_order"][degradation_key]
            full_level3 = full_payload["levels"]["level3"]
            degraded_level3 = degraded_payload["levels"]["level3"]
            nmi_damage = max(0.0, full_level3["abstraction_nmi"] - degraded_level3["abstraction_nmi"])
            probe_damage = max(0.0, full_level3["supervised_probe_accuracy"] - degraded_level3["supervised_probe_accuracy"])
            case_nmi_damage.append(nmi_damage)
            case_probe_damage.append(probe_damage)
            case_scores.append(nmi_damage + probe_damage)
            if (
                nmi_damage > 0.0
                or probe_damage > 0.0
            ):
                weaker_cases.add(stress_case)
        expected_ranking[stress_case] = {
            "level3_damage_score": float(np.mean(case_scores)),
            "level3_nmi_damage": float(np.mean(case_nmi_damage)),
            "level3_probe_damage": float(np.mean(case_probe_damage)),
        }

    expected_order = [
        name
        for name, _ in sorted(
            expected_ranking.items(),
            key=lambda item: (
                -item[1]["level3_damage_score"],
                -item[1]["level3_nmi_damage"],
                -item[1]["level3_probe_damage"],
                item[0],
            ),
        )
    ]
    actual_ranking = stress["diagnosis"]["level3_damage_ranking"]

    assert stress["diagnosis"]["degraded_transition_lag_order_weaker_case_count"] == len(weaker_cases)
    assert set(stress["diagnosis"]["degraded_transition_lag_order_weaker_cases"]) == weaker_cases
    assert stress["diagnosis"]["degraded_transition_lag_order_weaker_case_count"] >= 1
    assert [row["stress_case"] for row in actual_ranking] == expected_order
    assert stress["diagnosis"]["most_damaging_case"] == expected_order[0]
    assert stress["diagnosis"]["least_damaging_case"] == expected_order[-1]
    for row in actual_ranking:
        expected = expected_ranking[row["stress_case"]]
        assert np.isclose(row["level3_damage_score"], expected["level3_damage_score"])
        assert np.isclose(row["level3_nmi_damage"], expected["level3_nmi_damage"])
        assert np.isclose(row["level3_probe_damage"], expected["level3_probe_damage"])


def test_order_aware_encoder_handles_variable_length_stress_batches():
    features = encode_sequences(
        [[0, 1, 2, 3], [0, 1, 1, 2, 3], [0, 2, 3]],
        vocab_size=4,
        encoder="transition_lag_order",
    )

    assert features.shape == (3, (2 + 5) * 16)


def test_transition_lag_order_survives_ambiguous_shared_anchors_better_than_transition():
    transition = run_depth_encoder_diagnostics("transition", stress_case="ambiguous_shared_anchors", degradation=0.50)
    ordered = run_depth_encoder_diagnostics("transition_lag_order", stress_case="ambiguous_shared_anchors", degradation=0.50)

    assert ordered["level3"]["supervised_probe_accuracy"] >= transition["level3"]["supervised_probe_accuracy"]


def test_degraded_transition_lag_order_reports_milder_level3_signal_than_full_encoder():
    full = run_depth_encoder_diagnostics("transition_lag_order", stress_case="repeated_symbols", degradation=0.75)
    degraded = run_depth_encoder_diagnostics(
        "degraded_transition_lag_order",
        stress_case="repeated_symbols",
        degradation=0.75,
        degradation_loss=0.75,
    )

    assert degraded["level3"]["supervised_probe_accuracy"] <= full["level3"]["supervised_probe_accuracy"]


def test_run_all_reports_probe_d_stress():
    metrics = run_all(seed=31)

    assert "probe_d_stress" in metrics
    assert "probe_e_decomposition" in metrics
    assert "probe_f_order_loss_split" in metrics
