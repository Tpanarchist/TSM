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
