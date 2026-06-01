import torch

from tsm.diagnostics import (
    feature_label_diagnostics,
    feature_match_diagnostics,
    grouped_instance_match_diagnostics,
    paired_feature_match_diagnostics,
    ternary_axis_specialization,
    ternary_label_diagnostics,
)


def test_ternary_label_diagnostics_detect_mode_structure():
    ternary = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    contexts = torch.tensor([0, 0, 1, 1])

    metrics = ternary_label_diagnostics(ternary, labels, contexts)

    assert metrics["ternary_mode_probe_accuracy"].item() == 1.0
    assert metrics["ternary_axis_usage_count"].item() == 2.0
    assert metrics["ternary_mode_mutual_information"].item() > 0.0
    assert metrics["ternary_context_mutual_information"].item() > 0.0
    assert metrics["ternary_axis_stability"].item() == 1.0


def test_ternary_axis_specialization_reports_mode_boundaries():
    ternary = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    axes = ternary_axis_specialization(ternary, labels, min_usage=0.1)

    assert axes[0]["axis_id"] == 0
    assert axes[0]["positive_modes"] == [0]
    assert axes[1]["axis_id"] == 1
    assert axes[1]["negative_modes"] == [1]
    assert axes[0]["stability"] == 1.0


def test_feature_label_diagnostics_probe_recovers_cluster_labels():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    metrics = feature_label_diagnostics(features, labels)

    assert metrics["feature_probe_accuracy"].item() == 1.0
    assert metrics["feature_centroid_separation"].item() > 0.0
    assert metrics["feature_label_count"].item() == 2.0


def test_feature_match_diagnostics_scores_source_target_identity():
    source = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    target = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    source_labels = torch.tensor([0, 1])
    target_labels = torch.tensor([1, 0])

    metrics = feature_match_diagnostics(source, target, source_labels, target_labels)

    assert metrics["feature_match_accuracy"].item() == 1.0
    assert metrics["feature_match_margin"].item() > 0.0


def test_paired_feature_match_diagnostics_scores_exact_pairs():
    source = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    metrics = paired_feature_match_diagnostics(source, target)

    assert metrics["paired_feature_match_accuracy"].item() == 1.0
    assert metrics["paired_feature_match_margin"].item() > 0.0


def test_grouped_instance_match_diagnostics_uses_same_group_hard_negatives():
    source = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]
    )
    instances = torch.tensor([10, 11, 12])
    groups = torch.tensor([0, 1, 0])

    metrics = grouped_instance_match_diagnostics(source, target, instances, instances, groups, groups)

    assert metrics["instance_match_accuracy"].item() == 1.0
    assert metrics["instance_hard_match_accuracy"].item() == 1.0
    assert metrics["instance_hard_valid_fraction"].item() > 0.0
