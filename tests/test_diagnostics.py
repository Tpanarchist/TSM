import torch

from tsm.diagnostics import ternary_axis_specialization, ternary_label_diagnostics


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
