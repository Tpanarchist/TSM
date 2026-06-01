import torch

from tsm.config import DefinitionHardeningConfig
from tsm.hardening import DefinitionEvidenceTracker


def test_definition_evidence_hardens_repeated_stable_useful_axis():
    cfg = DefinitionHardeningConfig(
        min_usage=0.1,
        max_usage=0.9,
        min_stability=0.9,
        min_mode_mutual_information=0.01,
        min_prediction_impact=0.001,
        harden_after_windows=2,
    )
    tracker = DefinitionEvidenceTracker(cfg, axis_count=3)
    ternary = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    impacts = torch.tensor([0.01, 0.02, 0.0])

    first = tracker.update(10, ternary, labels, impacts)
    second = tracker.update(20, ternary, labels, impacts)
    report = tracker.to_dict()

    assert first["definition_new_candidate_count"] == 2.0
    assert first["definition_hardened_count"] == 0.0
    assert second["definition_new_hardened_count"] == 2.0
    assert second["definition_hardened_count"] == 2.0
    assert {item["axis_id"] for item in report["hardened"]} == {0, 1}


def test_definition_evidence_does_not_harden_without_prediction_usefulness():
    cfg = DefinitionHardeningConfig(
        min_usage=0.1,
        max_usage=0.9,
        min_stability=0.9,
        min_mode_mutual_information=0.01,
        min_prediction_impact=0.001,
        harden_after_windows=2,
    )
    tracker = DefinitionEvidenceTracker(cfg, axis_count=2)
    ternary = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, -1.0]])
    labels = torch.tensor([0, 0, 1, 1])

    tracker.update(10, ternary, labels, torch.zeros(2))
    metrics = tracker.update(20, ternary, labels, torch.zeros(2))

    assert metrics["definition_candidate_count"] == 0.0
    assert metrics["definition_hardened_count"] == 0.0


def test_hardened_definition_softens_after_repeated_counter_evidence():
    cfg = DefinitionHardeningConfig(
        min_usage=0.1,
        max_usage=0.9,
        min_stability=0.9,
        min_mode_mutual_information=0.01,
        min_prediction_impact=0.001,
        harden_after_windows=2,
        soften_after_windows=2,
        reject_after_windows=4,
    )
    tracker = DefinitionEvidenceTracker(cfg, axis_count=2)
    labels = torch.tensor([0, 0, 1, 1])
    useful = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, -1.0]])
    collapsed = torch.zeros_like(useful)

    tracker.update(10, useful, labels, torch.tensor([0.01, 0.02]))
    tracker.update(20, useful, labels, torch.tensor([0.01, 0.02]))
    first_counter = tracker.update(30, collapsed, labels, torch.zeros(2))
    second_counter = tracker.update(40, collapsed, labels, torch.zeros(2))

    assert first_counter["definition_softened_count"] == 0.0
    assert second_counter["definition_new_softened_count"] == 2.0
    assert second_counter["definition_hardened_count"] == 0.0
    assert second_counter["definition_softened_count"] == 2.0


def test_softened_definition_rejects_after_continued_counter_evidence():
    cfg = DefinitionHardeningConfig(
        min_usage=0.1,
        max_usage=0.9,
        min_stability=0.9,
        min_mode_mutual_information=0.01,
        min_prediction_impact=0.001,
        harden_after_windows=2,
        soften_after_windows=2,
        reject_after_windows=4,
    )
    tracker = DefinitionEvidenceTracker(cfg, axis_count=1)
    labels = torch.tensor([0, 0, 1, 1])
    useful = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    collapsed = torch.zeros_like(useful)

    tracker.update(10, useful, labels, torch.tensor([0.01]))
    tracker.update(20, useful, labels, torch.tensor([0.01]))
    tracker.update(30, collapsed, labels, torch.zeros(1))
    tracker.update(40, collapsed, labels, torch.zeros(1))
    tracker.update(50, collapsed, labels, torch.zeros(1))
    metrics = tracker.update(60, collapsed, labels, torch.zeros(1))
    report = tracker.to_dict()

    assert metrics["definition_new_rejected_count"] == 1.0
    assert metrics["definition_rejected_count"] == 1.0
    assert report["rejected"][0]["soften_reason"] == "not_sparse"


def test_outlier_quarantine_does_not_soften_from_one_bad_update():
    cfg = DefinitionHardeningConfig(
        min_usage=0.1,
        max_usage=0.9,
        min_stability=0.9,
        min_mode_mutual_information=0.01,
        min_prediction_impact=0.001,
        harden_after_windows=2,
        soften_after_windows=1,
        quarantine_loss_z=1.0,
        min_loss_history=1,
    )
    tracker = DefinitionEvidenceTracker(cfg, axis_count=1)
    labels = torch.tensor([0, 0, 1, 1])
    useful = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    collapsed = torch.zeros_like(useful)

    tracker.update(10, useful, labels, torch.tensor([0.01]), prediction_loss=0.1)
    tracker.update(20, useful, labels, torch.tensor([0.01]), prediction_loss=0.1)
    metrics = tracker.update(30, collapsed, labels, torch.zeros(1), prediction_loss=1.0)

    assert metrics["definition_quarantined_update_count"] == 1.0
    assert metrics["definition_hardened_count"] == 1.0
    assert metrics["definition_softened_count"] == 0.0
