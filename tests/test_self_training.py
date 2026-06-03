import torch

from tsm.config import TsmConfig
from tsm.memory import Memory
from tsm.self_field import (
    Self,
    _active_file_candidate_mask,
    _active_file_ballistic_position,
    _active_file_calibration_features,
    _active_file_calibration_input_dim,
    _active_file_calibration_uncertainty,
    _active_file_dynamics_features,
    _active_file_dynamics_input_dim,
    _active_file_dynamics_position,
    _active_file_expectation,
    _active_file_feature_only_candidate_mask,
    _active_file_gate_input_dim,
    _active_file_gate_logits,
    _all_track_endpoint_slot_cleanliness_metrics,
    _all_track_endpoint_spacing_metrics,
    _all_track_file_slot_assignment_metrics,
    _all_track_neutral_file_slot_metrics,
    _all_track_predicted_file_slot_metrics,
    _all_track_runtime_confidence_metrics,
    _file_slot_assignment_metrics,
    _local_reappearance_images,
    _oracle_error_shape_file_slot_metrics,
    _oracle_pair_file_slot_ceiling_metrics,
    _oracle_pair_file_slot_noise_sweep_metrics,
    _paired_endpoint_error_structure_metrics,
    _paired_predicted_file_slot_metrics,
    _state_prediction_error_matrix,
)


def test_forward_train_returns_loss_dict_and_images():
    cfg = TsmConfig(
        d_model=32,
        workspace_latents=8,
        contexts=3,
        definitions_per_context=4,
        image_size=16,
        image_channels=1,
        patch_size=4,
        attention_heads=4,
        inference_steps=1,
    )
    model = Self(cfg)
    batch = {
        "image_t": torch.rand(2, 1, 16, 16),
        "image_tp1": torch.rand(2, 1, 16, 16),
        "dataset_id": torch.zeros(2, dtype=torch.long),
        "mode": torch.tensor([0, 1], dtype=torch.long),
    }

    out = model.forward_train(batch)

    assert out.total_loss.ndim == 0
    assert set(out.losses) >= {
        "reconstruction",
        "prediction",
        "free_energy",
        "complexity",
        "context_entropy",
        "context_balance",
        "ternary_activation_l1",
        "bit_cost",
        "reappearance_alignment",
        "object_cycle_consistency",
        "reappearance_file_query",
        "active_file_query",
        "active_file_expectation",
        "active_file_expectation_pair",
        "active_file_expectation_hard",
        "active_file_dynamics",
        "active_file_calibration",
        "learned_active_file_gate",
    }
    assert set(out.diagnostics) >= {
        "context_effective_count",
        "context_used_count",
        "ternary_zero_fraction",
        "ternary_nonzero_fraction",
        "ternary_positive_fraction",
        "ternary_negative_fraction",
        "ternary_condition_norm",
        "sae_severity_mean",
        "sae_coherence_mean",
        "mode_context_consistency",
        "context_mode_purity",
        "mode_context_separation",
        "mode_context_used_count",
        "mode_count",
        "ternary_mode_mutual_information",
        "ternary_context_mutual_information",
        "ternary_mode_probe_accuracy",
        "ternary_per_mode_nonzero_fraction",
        "ternary_axis_usage_count",
        "ternary_axis_stability",
    }
    ternary_total = (
        out.diagnostics["ternary_zero_fraction"]
        + out.diagnostics["ternary_positive_fraction"]
        + out.diagnostics["ternary_negative_fraction"]
    )
    assert torch.allclose(ternary_total, torch.tensor(1.0), atol=1e-6)
    assert out.recon_image.shape == (2, 1, 16, 16)
    assert out.next_image.shape == (2, 1, 16, 16)
    assert out.latent_state.shape == (2, 8, 32)
    impacts = model.ternary_prediction_impacts(out, batch["image_tp1"])
    assert impacts.shape == (4,)
    assert torch.all(impacts >= 0)


def test_tick_has_no_embodiment_action():
    cfg = TsmConfig(
        d_model=32,
        workspace_latents=8,
        contexts=3,
        definitions_per_context=4,
        image_size=16,
        image_channels=1,
        patch_size=4,
        attention_heads=4,
        inference_steps=1,
    )
    model = Self(cfg)
    out = model.tick(torch.rand(1, 1, 16, 16))

    assert out.action is None
    assert out.context_probs.shape == (1, 3)


def test_active_file_candidate_mask_supports_wrapped_positions():
    file_positions = torch.tensor([[16.0, 5.0]])
    query_positions = torch.tensor([[4.0, 5.0]])
    valid = torch.tensor([True])
    hit = torch.tensor([True])
    age = torch.zeros(1, 1)

    direct = _active_file_candidate_mask(file_positions, query_positions, valid, hit, age, 8.1, 8.0)
    wrapped = _active_file_candidate_mask(file_positions, query_positions, valid, hit, age, 8.1, 8.0, 20.0)

    assert not bool(direct[0, 0].item())
    assert bool(wrapped[0, 0].item())


def test_active_file_feature_only_candidate_mask_uses_valid_files():
    valid = torch.tensor([True, False, True])

    mask = _active_file_feature_only_candidate_mask(valid, 3, torch.float32, torch.device("cpu"))

    expected = torch.tensor(
        [
            [True, False, True],
            [True, False, True],
            [True, False, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_local_reappearance_images_focuses_candidate_regions():
    cfg = TsmConfig(image_size=8, active_file_candidate_wrap=False)
    image = torch.full((1, 1, 8, 8), 0.02)
    image[0, 0, 1, 1] = 1.0
    image[0, 0, 6, 6] = 0.8
    positions = torch.tensor([[1.0, 1.0], [6.0, 6.0]])

    local = _local_reappearance_images(image, positions, cfg)

    assert local.shape == (2, 1, 8, 8)
    assert local[0, 0, 1, 1] > local[0, 0, 6, 6]
    assert local[1, 0, 6, 6] > local[1, 0, 1, 1]


def test_state_prediction_error_matrix_scores_expected_states():
    actual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected = torch.tensor([[1.0, 0.1], [0.1, 1.0]])

    errors = _state_prediction_error_matrix(actual, expected)

    assert errors.shape == (2, 2)
    assert errors[0, 0] < errors[0, 1]
    assert errors[1, 1] < errors[1, 0]


def test_active_file_gate_accepts_context_features():
    cfg = TsmConfig(
        d_model=32,
        workspace_latents=8,
        contexts=3,
        definitions_per_context=4,
        image_size=16,
        image_channels=1,
        patch_size=4,
        attention_heads=4,
        inference_steps=1,
        learned_active_file_gate_context_features=True,
        learned_active_file_gate_expectation_features=True,
        object_slot_count=2,
        object_slot_sigma=1.8,
        object_slot_nms_radius=4.0,
        object_slot_match_radius=4.0,
    )
    model = Self(cfg)
    query = torch.rand(2, cfg.definitions_per_context)
    files = torch.rand(2, cfg.definitions_per_context)
    confidence = torch.ones(2, 1)
    age = torch.zeros(2, 1)
    query_context = torch.softmax(torch.rand(2, cfg.contexts), dim=-1)
    file_context = torch.softmax(torch.rand(2, cfg.contexts), dim=-1)
    expected_query = torch.rand(2, cfg.definitions_per_context)

    assert model.active_file_gate[0].in_features == _active_file_gate_input_dim(cfg)
    logits = _active_file_gate_logits(
        model.active_file_gate,
        query,
        files,
        confidence,
        age,
        8.0,
        query_context,
        file_context,
        expected_query,
    )

    assert logits.shape == (2, 2)


def test_active_file_expectation_predicts_query_shape():
    cfg = TsmConfig(
        d_model=32,
        workspace_latents=8,
        contexts=3,
        definitions_per_context=4,
        image_size=16,
        image_channels=1,
        patch_size=4,
        attention_heads=4,
        inference_steps=1,
        active_file_expectation_trajectory_features=True,
    )
    model = Self(cfg)
    files = torch.rand(2, cfg.definitions_per_context)
    file_context = torch.softmax(torch.rand(2, cfg.contexts), dim=-1)
    confidence = torch.ones(2, 1)
    age = torch.zeros(2, 1)
    trajectory = torch.rand(2, 13 + 2 * cfg.active_file_expectation_phase_count)

    expected_query = _active_file_expectation(
        model.active_file_expectation,
        files,
        file_context,
        confidence,
        age,
        8.0,
        trajectory,
    )

    assert expected_query.shape == (2, cfg.definitions_per_context)


def test_active_file_dynamics_predicts_position_shape():
    cfg = TsmConfig(
        d_model=32,
        workspace_latents=8,
        contexts=3,
        definitions_per_context=4,
        image_size=16,
        image_channels=1,
        patch_size=4,
        attention_heads=4,
        inference_steps=1,
        active_file_expectation_trajectory_features=True,
    )
    model = Self(cfg)
    features = torch.rand(2, _active_file_dynamics_input_dim(cfg))
    projected = torch.tensor([[3.0, 4.0], [5.0, 6.0]])

    position = _active_file_dynamics_position(model.active_file_dynamics, features, projected, cfg)

    assert position.shape == (2, 2)
    assert torch.allclose(position, projected)


def test_active_file_calibration_predicts_uncertainty_shape():
    cfg = TsmConfig(
        d_model=32,
        workspace_latents=8,
        contexts=3,
        definitions_per_context=4,
        image_size=16,
        image_channels=1,
        patch_size=4,
        attention_heads=4,
        inference_steps=1,
        active_file_expectation_trajectory_features=True,
        object_slot_count=2,
    )
    model = Self(cfg)
    dynamics_features = torch.rand(2, _active_file_dynamics_input_dim(cfg))
    features = _active_file_calibration_features(
        dynamics_features,
        predicted_positions=torch.tensor([[3.0, 4.0], [12.0, 6.0]]),
        slot_positions=torch.tensor([[[3.0, 4.0], [12.0, 6.0]], [[12.0, 6.0], [3.0, 4.0]]]),
        slot_valid=torch.tensor([[True, True], [True, True]]),
        slot_occupancy=torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        cfg=cfg,
        reference_positions=torch.tensor([[2.5, 4.0], [11.0, 6.0]]),
        reference_valid=torch.tensor([True, True]),
    )
    uncertainty = _active_file_calibration_uncertainty(model.active_file_calibration, features)

    assert features.shape == (2, _active_file_calibration_input_dim(cfg))
    assert uncertainty.shape == (2,)
    assert torch.all((uncertainty >= 0.0) & (uncertainty <= 1.0))


def test_object_memory_tracks_velocity_and_phase():
    memory = Memory()
    batch = {
        "sequence_id": torch.zeros(3, dtype=torch.long),
        "visible_t": torch.ones(3),
        "phase": torch.tensor([0, 1, 2], dtype=torch.long),
        "object_position_t": torch.tensor([[1.0, 1.0], [3.0, 2.0], [6.0, 4.0]]),
    }
    features = torch.arange(12, dtype=torch.float32).view(3, 4)

    read = memory.read_write_object_files(batch, features, step=0)

    assert bool(read.velocity_valid[2].item())
    assert torch.allclose(read.velocity[2], torch.tensor([2.0, 1.0]))
    assert bool(read.phase_valid[2].item())
    assert torch.allclose(read.phase[2], torch.tensor([1.0]))


def test_file_slot_assignment_matches_swapped_slots_without_identity_labels():
    cfg = TsmConfig(image_size=16, object_slot_count=2)
    metrics = _file_slot_assignment_metrics(
        file_positions=torch.tensor([[2.0, 4.0], [12.0, 4.0]]),
        file_valid=torch.tensor([True, True]),
        slot_positions=torch.tensor([[[12.1, 4.0], [2.1, 4.0]]]),
        slot_valid=torch.tensor([[True, True]]),
        target_positions=torch.tensor([[2.0, 4.0]]),
        file_instance_labels=torch.tensor([10, 11], dtype=torch.long),
        target_instance_labels=torch.tensor([10], dtype=torch.long),
        group_labels=torch.tensor([1, 1], dtype=torch.long),
        cfg=cfg,
        distractor_positions=torch.tensor([[12.0, 4.0]]),
        distractor_instance_labels=torch.tensor([11], dtype=torch.long),
    )

    assert metrics["target_match_accuracy"].item() == 1.0
    assert metrics["target_hard_match_accuracy"].item() == 1.0
    assert metrics["distractor_match_accuracy"].item() == 1.0
    assert metrics["pair_match_accuracy"].item() == 1.0
    assert metrics["candidate_mean_count"].item() == 2.0
    assert metrics["row_coverage_fraction"].item() == 1.0
    assert metrics["target_file_recall_fraction"].item() == 1.0
    assert metrics["distractor_file_recall_fraction"].item() == 1.0
    assert metrics["assignment_object_file_id_usage"].item() == 0.0
    assert metrics["assignment_object_id_usage"].item() == 0.0
    assert metrics["assignment_sequence_id_usage"].item() == 0.0


def test_oracle_pair_file_slot_ceiling_uses_positions_without_identity_assignment():
    cfg = TsmConfig(image_size=16, object_slot_count=2)
    metrics = _oracle_pair_file_slot_ceiling_metrics(
        slot_positions=torch.tensor([[[12.1, 4.0], [2.1, 4.0]]]),
        slot_valid=torch.tensor([[True, True]]),
        target_positions=torch.tensor([[2.0, 4.0]]),
        target_instance_labels=torch.tensor([10], dtype=torch.long),
        group_labels=torch.tensor([1], dtype=torch.long),
        cfg=cfg,
        distractor_positions=torch.tensor([[12.0, 4.0]]),
        distractor_instance_labels=torch.tensor([11], dtype=torch.long),
    )

    assert metrics["target_match_accuracy"].item() == 1.0
    assert metrics["target_hard_match_accuracy"].item() == 1.0
    assert metrics["distractor_match_accuracy"].item() == 1.0
    assert metrics["pair_match_accuracy"].item() == 1.0
    assert metrics["candidate_mean_count"].item() == 2.0
    assert metrics["assignment_object_file_id_usage"].item() == 0.0
    assert metrics["assignment_object_id_usage"].item() == 0.0
    assert metrics["assignment_sequence_id_usage"].item() == 0.0


def test_oracle_pair_file_slot_noise_sweep_reports_error_budget_curve():
    cfg = TsmConfig(image_size=16, object_slot_count=2)
    metrics = _oracle_pair_file_slot_noise_sweep_metrics(
        slot_positions=torch.tensor([[[12.1, 4.0], [2.1, 4.0]]]),
        slot_valid=torch.tensor([[True, True]]),
        target_positions=torch.tensor([[2.0, 4.0]]),
        target_instance_labels=torch.tensor([10], dtype=torch.long),
        group_labels=torch.tensor([1], dtype=torch.long),
        cfg=cfg,
        distractor_positions=torch.tensor([[12.0, 4.0]]),
        distractor_instance_labels=torch.tensor([11], dtype=torch.long),
    )

    assert metrics["noise_0px_target_match_accuracy"].item() == 1.0
    assert metrics["noise_0px_pair_match_accuracy"].item() == 1.0
    assert metrics["noise_1px_position_noise_px"].item() == 1.0
    assert metrics["noise_6px_position_noise_normalized"].item() == 6.0 / 16.0
    assert metrics["noise_6px_trial_count"].item() == 8.0
    assert metrics["noise_8px_position_noise_px"].item() == 8.0
    assert metrics["noise_6px_assignment_object_file_id_usage"].item() == 0.0


def test_paired_endpoint_error_structure_reports_compression_and_tail():
    cfg = TsmConfig(image_size=16)
    metrics = _paired_endpoint_error_structure_metrics(
        predicted_positions=torch.tensor([[5.0, 4.0], [9.0, 4.0]]),
        predicted_valid=torch.tensor([True, True]),
        target_positions=torch.tensor([[2.0, 4.0]]),
        file_instance_labels=torch.tensor([10, 11], dtype=torch.long),
        target_instance_labels=torch.tensor([10], dtype=torch.long),
        cfg=cfg,
        distractor_positions=torch.tensor([[12.0, 4.0]]),
        distractor_instance_labels=torch.tensor([11], dtype=torch.long),
    )

    assert torch.allclose(metrics["true_pair_distance"], torch.tensor(10.0 / 16.0))
    assert torch.allclose(metrics["predicted_pair_distance"], torch.tensor(4.0 / 16.0))
    assert torch.allclose(metrics["pair_distance_compression"], torch.tensor(6.0 / 16.0))
    assert metrics["midpoint_pull"].item() > 0.0
    assert metrics["error_p95"].item() >= metrics["error_median"].item()


def test_oracle_error_shape_file_slot_reports_failure_modes():
    cfg = TsmConfig(image_size=16, object_slot_count=2)
    metrics = _oracle_error_shape_file_slot_metrics(
        slot_positions=torch.tensor([[[12.1, 4.0], [2.1, 4.0]]]),
        slot_valid=torch.tensor([[True, True]]),
        target_positions=torch.tensor([[2.0, 4.0]]),
        target_instance_labels=torch.tensor([10], dtype=torch.long),
        group_labels=torch.tensor([1], dtype=torch.long),
        cfg=cfg,
        distractor_positions=torch.tensor([[12.0, 4.0]]),
        distractor_instance_labels=torch.tensor([11], dtype=torch.long),
    )

    assert "center_bias_target_match_accuracy" in metrics
    assert "correlated_target_match_accuracy" in metrics
    assert "heavy_tail_target_match_accuracy" in metrics
    assert metrics["center_bias_position_noise_px"].item() == 6.5
    assert metrics["center_bias_pair_distance_ratio"].item() < 1.0
    assert metrics["correlated_pair_distance_ratio"].item() > 0.9


def test_paired_predicted_file_slot_metrics_uses_local_file_pair():
    cfg = TsmConfig(image_size=16, object_slot_count=2)
    metrics = _paired_predicted_file_slot_metrics(
        predicted_positions=torch.tensor([[2.1, 4.0], [12.1, 4.0], [7.0, 7.0]]),
        predicted_valid=torch.tensor([True, True, True]),
        slot_positions=torch.tensor([[[12.0, 4.0], [2.0, 4.0]]]),
        slot_valid=torch.tensor([[True, True]]),
        target_positions=torch.tensor([[2.0, 4.0]]),
        file_instance_labels=torch.tensor([10, 11, 12], dtype=torch.long),
        target_instance_labels=torch.tensor([10], dtype=torch.long),
        group_labels=torch.tensor([1], dtype=torch.long),
        cfg=cfg,
        distractor_positions=torch.tensor([[12.0, 4.0]]),
        distractor_instance_labels=torch.tensor([11], dtype=torch.long),
    )

    assert metrics["target_match_accuracy"].item() == 1.0
    assert metrics["distractor_match_accuracy"].item() == 1.0
    assert metrics["pair_match_accuracy"].item() == 1.0
    assert metrics["candidate_mean_count"].item() == 2.0
    assert metrics["valid_pair_fraction"].item() == 1.0


def test_all_track_file_slot_assignment_scores_full_local_set():
    cfg = TsmConfig(image_size=16, object_slot_count=3)
    all_positions = torch.tensor([[[2.0, 4.0], [8.0, 4.0], [12.0, 10.0]]])
    metrics = _all_track_file_slot_assignment_metrics(
        file_positions=all_positions + 0.1,
        file_valid=torch.tensor([[True, True, True]]),
        slot_positions=torch.tensor([[[12.0, 10.0], [2.0, 4.0], [8.0, 4.0]]]),
        slot_valid=torch.tensor([[True, True, True]]),
        all_positions=all_positions,
        all_instance_labels=torch.tensor([[10, 11, 12]], dtype=torch.long),
        target_instance_labels=torch.tensor([11], dtype=torch.long),
        cfg=cfg,
    )

    assert metrics["object_count"].item() == 3.0
    assert metrics["object_match_accuracy"].item() == 1.0
    assert metrics["set_match_accuracy"].item() == 1.0
    assert metrics["target_match_accuracy"].item() == 1.0
    assert metrics["assignment_object_file_id_usage"].item() == 0.0


def test_all_track_predicted_file_slot_metrics_gathers_scene_files():
    cfg = TsmConfig(image_size=16, object_slot_count=3)
    metrics = _all_track_predicted_file_slot_metrics(
        predicted_positions=torch.tensor([[2.0, 4.0], [12.0, 10.0], [8.0, 4.0], [1.0, 1.0]]),
        predicted_valid=torch.tensor([True, True, True, True]),
        slot_positions=torch.tensor([[[12.0, 10.0], [2.0, 4.0], [8.0, 4.0]]]),
        slot_valid=torch.tensor([[True, True, True]]),
        file_instance_labels=torch.tensor([10, 12, 11, 99], dtype=torch.long),
        target_instance_labels=torch.tensor([10], dtype=torch.long),
        all_positions=torch.tensor([[[2.0, 4.0], [8.0, 4.0], [12.0, 10.0]]]),
        all_instance_labels=torch.tensor([[10, 11, 12]], dtype=torch.long),
        cfg=cfg,
    )

    assert metrics["object_match_accuracy"].item() == 1.0
    assert metrics["set_match_accuracy"].item() == 1.0
    assert metrics["candidate_mean_count"].item() == 3.0


def test_all_track_neutral_file_slot_metrics_buckets_deadband_decisions():
    cfg = TsmConfig(image_size=16, object_slot_count=3)
    metrics = _all_track_neutral_file_slot_metrics(
        predicted_positions=torch.tensor([[0.1, 0.0], [5.9, 0.0], [5.9, 0.0]]),
        predicted_valid=torch.tensor([True, True, True]),
        file_instance_labels=torch.tensor([10, 11, 12], dtype=torch.long),
        slot_positions=torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]]),
        slot_valid=torch.tensor([[True, True, True]]),
        all_positions=torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]]),
        all_instance_labels=torch.tensor([[10, 11, 12]], dtype=torch.long),
        cfg=cfg,
    )

    assert metrics["object_count"].item() == 3.0
    assert metrics["decision_coverage_fraction"].item() == 1.0
    assert torch.isclose(metrics["forced_correct_fraction"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(metrics["forced_wrong_fraction"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(metrics["neutral_decline_fraction"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(metrics["confident_correct_bind_fraction"], torch.tensor(1.0 / 3.0))
    assert metrics["confident_wrong_bind_fraction"].item() == 0.0
    assert torch.isclose(metrics["correct_decline_fraction"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(metrics["wrong_decline_fraction"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(metrics["decline_precision"], torch.tensor(0.5))
    assert metrics["assignment_object_file_id_usage"].item() == 0.0


def test_all_track_runtime_confidence_reports_error_calibration_without_label_leakage():
    cfg = TsmConfig(image_size=16, object_slot_count=3)
    metrics = _all_track_runtime_confidence_metrics(
        predicted_positions=torch.tensor([[0.1, 0.0], [4.5, 0.0], [5.9, 0.0]]),
        predicted_valid=torch.tensor([True, True, True]),
        file_instance_labels=torch.tensor([10, 11, 12], dtype=torch.long),
        slot_positions=torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]]),
        slot_valid=torch.tensor([[True, True, True]]),
        slot_occupancy=torch.tensor([[0.95, 0.70, 0.35]]),
        file_confidence=torch.tensor([[1.0], [0.7], [0.2]]),
        file_age=torch.tensor([[0.0], [2.0], [7.0]]),
        all_positions=torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]]),
        all_instance_labels=torch.tensor([[10, 11, 12]], dtype=torch.long),
        cfg=cfg,
        reference_positions=torch.tensor([[0.0, 0.0], [4.1, 0.0], [8.0, 0.0]]),
        reference_valid=torch.tensor([True, True, True]),
        calibrated_uncertainty=torch.tensor([0.05, 0.25, 0.70]),
    )

    assert metrics["object_count"].item() == 3.0
    assert metrics["decision_coverage_fraction"].item() == 1.0
    assert metrics["runtime_uncertainty_error_pearson"].item() > 0.5
    assert metrics["runtime_uncertainty_error_spearman"].item() > 0.5
    assert metrics["calibrated_uncertainty_error_pearson"].item() > 0.5
    assert metrics["calibrated_uncertainty_high_error_lift"].item() > 0.0
    assert metrics["runtime_confidence_drop_on_correct_declines"].item() > 0.0
    assert metrics["confidence_true_position_usage"].item() == 0.0
    assert metrics["confidence_endpoint_error_usage"].item() == 0.0
    assert metrics["confidence_object_file_id_usage"].item() == 0.0
    assert metrics["confidence_object_id_usage"].item() == 0.0
    assert metrics["confidence_sequence_id_usage"].item() == 0.0


def test_all_track_runtime_confidence_reports_tail_danger_detection():
    cfg = TsmConfig(image_size=16, object_slot_count=3)
    metrics = _all_track_runtime_confidence_metrics(
        predicted_positions=torch.tensor([[0.1, 0.0], [4.5, 0.0], [3.0, 0.0]]),
        predicted_valid=torch.tensor([True, True, True]),
        file_instance_labels=torch.tensor([10, 11, 12], dtype=torch.long),
        slot_positions=torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]]),
        slot_valid=torch.tensor([[True, True, True]]),
        slot_occupancy=torch.tensor([[0.95, 0.70, 0.35]]),
        file_confidence=torch.tensor([[1.0], [0.7], [0.2]]),
        file_age=torch.tensor([[0.0], [2.0], [7.0]]),
        all_positions=torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]]),
        all_instance_labels=torch.tensor([[10, 11, 12]], dtype=torch.long),
        cfg=cfg,
        calibrated_uncertainty=torch.tensor([0.05, 0.25, 0.90]),
    )

    assert metrics["unsafe_endpoint_error_ratio_threshold"].item() == 0.5
    assert torch.isclose(metrics["unsafe_endpoint_error_fraction"], torch.tensor(1.0 / 3.0))
    assert metrics["slot_unsafe_fraction"].item() == 0.0
    assert torch.isclose(metrics["pair_unsafe_fraction"], torch.tensor(1.0 / 3.0))
    assert metrics["pair_unsafe_within_scene_valid_fraction"].item() == 1.0
    assert metrics["endpoint_error_to_spacing_ratio_p90"].item() > 1.0
    assert metrics["calibrated_uncertainty_unsafe_auroc"].item() == 1.0
    assert metrics["calibrated_uncertainty_unsafe_auprc"].item() == 1.0
    assert metrics["calibrated_uncertainty_pair_unsafe_within_scene_auroc"].item() == 1.0
    assert metrics["calibrated_uncertainty_pair_unsafe_within_scene_auprc"].item() == 1.0
    assert metrics["calibrated_uncertainty_scene_adjusted_pair_unsafe_auroc"].item() == 1.0
    assert metrics["calibrated_uncertainty_within_scene_variance_mean"].item() > 0.0
    assert metrics["calibrated_uncertainty_within_scene_pair_unsafe_gap"].item() > 0.0
    assert metrics["calibrated_uncertainty_unsafe_lift"].item() > 0.0
    assert metrics["calibrated_uncertainty_error_high_bucket_mean"].item() > (
        metrics["calibrated_uncertainty_error_low_bucket_mean"].item()
    )
    assert metrics["candidate_margin_uncertainty_unsafe_auroc"].item() >= 0.0
    assert metrics["confidence_endpoint_error_usage"].item() == 0.0
    assert metrics["confidence_object_file_id_usage"].item() == 0.0


def test_all_track_endpoint_spacing_metrics_reports_error_budget():
    cfg = TsmConfig(image_size=16, object_slot_count=3)
    metrics = _all_track_endpoint_spacing_metrics(
        predicted_positions=torch.tensor([[2.0, 4.0], [12.0, 10.0], [8.0, 4.0], [1.0, 1.0]]),
        predicted_valid=torch.tensor([True, True, True, True]),
        file_instance_labels=torch.tensor([10, 12, 11, 99], dtype=torch.long),
        all_positions=torch.tensor([[[2.0, 4.0], [8.0, 4.0], [12.0, 10.0]]]),
        all_instance_labels=torch.tensor([[10, 11, 12]], dtype=torch.long),
        cfg=cfg,
    )

    assert metrics["object_count"].item() == 3.0
    assert metrics["valid_row_fraction"].item() == 1.0
    assert metrics["min_interobject_spacing_px"].item() == 6.0
    assert metrics["endpoint_error_mean"].item() == 0.0
    assert metrics["endpoint_error_to_spacing_ratio"].item() == 0.0
    assert metrics["shared_track_endpoint_error_mean"].item() == 0.0
    assert metrics["extra_track_endpoint_error_mean"].item() == 0.0
    assert metrics["track0_endpoint_error_mean"].item() == 0.0


def test_all_track_endpoint_slot_cleanliness_metrics_splits_endpoint_errors():
    cfg = TsmConfig(image_size=16, object_slot_count=3, object_slot_match_radius=2.0)
    metrics = _all_track_endpoint_slot_cleanliness_metrics(
        predicted_positions=torch.tensor([[2.0, 4.0], [3.0, 10.0], [10.0, 4.0]]),
        predicted_valid=torch.tensor([True, True, True]),
        file_instance_labels=torch.tensor([10, 12, 11], dtype=torch.long),
        slot_positions=torch.tensor([[[2.5, 4.0], [7.5, 4.0], [1.0, 1.0]]]),
        slot_valid=torch.tensor([[True, True, True]]),
        all_positions=torch.tensor([[[2.0, 4.0], [8.0, 4.0], [12.0, 10.0]]]),
        all_instance_labels=torch.tensor([[10, 11, 12]], dtype=torch.long),
        cfg=cfg,
    )

    assert metrics["object_count"].item() == 3.0
    assert metrics["valid_object_fraction"].item() == 1.0
    assert torch.isclose(metrics["slot_clean_object_fraction"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(metrics["slot_dirty_object_fraction"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(metrics["clean_endpoint_error_mean"], torch.tensor(0.0625))
    assert torch.isclose(metrics["dirty_endpoint_error_mean"], torch.tensor(9.0 / 16.0))
    assert metrics["high_error_object_fraction"].item() > 0.0
    assert metrics["high_error_clean_fraction"].item() == 0.0


def test_active_file_ballistic_position_uses_phase_elapsed_with_wrap():
    cfg = TsmConfig(image_size=28, active_file_expectation_phase_count=5, active_file_candidate_wrap=True)
    memory = Memory()
    batch = {
        "object_file_id": torch.zeros(3, dtype=torch.long),
        "visible_t": torch.ones(3),
        "phase": torch.tensor([0, 1, 2], dtype=torch.long),
        "object_position_t": torch.tensor([[4.0, 4.0], [7.0, 4.0], [10.0, 5.0]]),
    }
    memory.read_write_object_files(batch, torch.rand(3, 4), step=0)
    reappeared_batch = {
        "object_file_id": torch.zeros(1, dtype=torch.long),
        "visible_t": torch.zeros(1),
        "phase": torch.tensor([4], dtype=torch.long),
    }
    read = memory.read_write_object_files(reappeared_batch, torch.rand(1, 4), step=1)

    position, valid = _active_file_ballistic_position(
        read,
        reappeared_batch,
        torch.tensor([True]),
        cfg,
        torch.float32,
        torch.device("cpu"),
    )

    assert bool(valid.item())
    assert torch.allclose(position, torch.tensor([[19.0, 8.0]]))


def test_forward_train_reports_temporal_object_diagnostics():
    cfg = TsmConfig(
        d_model=32,
        workspace_latents=8,
        contexts=4,
        definitions_per_context=4,
        image_size=16,
        image_channels=1,
        patch_size=4,
        attention_heads=4,
        inference_steps=1,
        active_file_expectation_weight=0.003,
        active_file_expectation_hard_weight=1.0,
        active_file_expectation_trajectory_features=True,
        active_file_expectation_dynamics_features=True,
        active_file_dynamics_weight=0.01,
        learned_active_file_gate_context_features=True,
        learned_active_file_gate_expectation_features=True,
        object_slot_count=2,
        object_slot_sigma=1.8,
        object_slot_nms_radius=4.0,
        object_slot_match_radius=4.0,
    )
    model = Self(cfg)
    batch = {
        "image_t": torch.rand(5, 1, 16, 16),
        "image_tp1": torch.rand(5, 1, 16, 16),
        "dataset_id": torch.zeros(5, dtype=torch.long),
        "sequence_id": torch.zeros(5, dtype=torch.long),
        "object_file_id": torch.tensor([0, 1, 2, 3, 0], dtype=torch.long),
        "mode": torch.tensor([0, 1, 2, 2, 3], dtype=torch.long),
        "phase": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "object_id": torch.tensor([0, 1, 2, 3, 0], dtype=torch.long),
        "track_id": torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
        "visible_t": torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32),
        "visible_tp1": torch.tensor([1, 1, 0, 0, 1], dtype=torch.float32),
        "occluded_t": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32),
        "occluded_tp1": torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32),
        "moved": torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32),
        "identity_preserved": torch.ones(5),
        "unexpected_disappearance": torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32),
        "object_position_t": torch.zeros(5, 2),
        "object_position_tp1": torch.zeros(5, 2),
        "distractor_position_tp1": torch.ones(5, 2) * 8.0,
        "all_object_positions_tp1": torch.stack((
            torch.zeros(5, 2),
            torch.ones(5, 2) * 8.0,
        ), dim=1),
        "all_object_file_ids": torch.tensor([
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [0, 1],
        ], dtype=torch.long),
        "contested_track_count": torch.full((5,), 2, dtype=torch.long),
        "nonlinear_contested_motion": torch.ones(5),
    }

    out = model.forward_train(batch)

    assert set(out.diagnostics) >= {
        "phase_ternary_mode_probe_accuracy",
        "object_ternary_mode_probe_accuracy",
        "occluded_object_ternary_mode_probe_accuracy",
        "temporal_visible_fraction",
        "object_file_id_storage_key_present",
        "object_file_id_bind_time_candidate_filter_usage",
        "object_file_id_bind_time_leakage_audit_pass",
        "object_file_id_auxiliary_label_usage",
        "temporal_occluded_fraction",
        "temporal_reappeared_fraction",
        "temporal_nonlinear_contested_motion_fraction",
        "temporal_sae_occlusion_delta",
        "temporal_prediction_occluded_mean",
        "temporal_prediction_visible_mean",
        "temporal_context_occluded_used_count",
        "temporal_context_visible_used_count",
        "temporal_memory_occluded_hit_fraction",
        "memory_condition_norm",
        "memory_definition_condition_norm",
        "memory_definition_flip_fraction",
        "memory_prediction_occluded_impact_mean",
        "memory_total_prediction_occluded_impact_mean",
        "memory_definition_prediction_occluded_impact_mean",
        "temporal_memory_definition_occluded_flip_fraction",
        "memory_object_feature_probe_accuracy",
        "occluded_memory_object_feature_probe_accuracy",
        "occluded_base_ternary_mode_probe_accuracy",
        "occluded_memory_definition_object_probe_delta",
        "reappeared_feature_match_accuracy",
        "reappeared_base_feature_match_accuracy",
        "reappeared_memory_definition_match_delta",
        "reappeared_paired_feature_match_accuracy",
        "reappeared_base_paired_feature_match_accuracy",
        "reappeared_paired_memory_definition_match_delta",
        "reappeared_file_paired_feature_match_accuracy",
        "reappeared_file_instance_match_accuracy",
        "reappeared_file_instance_hard_match_accuracy",
        "reappeared_target_file_paired_feature_match_accuracy",
        "reappeared_target_file_instance_match_accuracy",
        "reappeared_target_file_instance_hard_match_accuracy",
        "reappeared_query_file_paired_feature_match_accuracy",
        "reappeared_query_file_instance_match_accuracy",
        "reappeared_query_file_instance_hard_match_accuracy",
        "reappeared_expected_file_paired_feature_match_accuracy",
        "reappeared_expected_file_instance_match_accuracy",
        "reappeared_expected_file_instance_hard_match_accuracy",
        "reappeared_expected_state_paired_feature_match_accuracy",
        "reappeared_expected_state_instance_match_accuracy",
        "reappeared_expected_state_instance_hard_match_accuracy",
        "reappeared_trajectory_position_error",
        "reappeared_trajectory_valid_fraction",
        "reappeared_ballistic_position_error",
        "reappeared_ballistic_valid_fraction",
        "reappeared_dynamics_position_error",
        "reappeared_dynamics_position_improvement",
        "reappeared_dynamics_over_ballistic_position_improvement",
        "reappeared_dynamics_valid_fraction",
        "reappeared_definition_position_linear_error",
        "reappeared_definition_position_linear_improvement",
        "reappeared_file_query_position_linear_error",
        "reappeared_file_query_position_linear_improvement",
        "reappeared_memory_definition_position_linear_error",
        "reappeared_memory_definition_position_linear_improvement",
        "reappeared_definition_position_ablated_position_linear_error",
        "reappeared_definition_position_ablated_position_linear_improvement",
        "reappeared_file_query_position_ablated_position_linear_error",
        "reappeared_file_query_position_ablated_position_linear_improvement",
        "reappeared_memory_definition_position_ablated_position_linear_error",
        "reappeared_memory_definition_position_ablated_position_linear_improvement",
        "reappeared_object_slot_count",
        "reappeared_object_slot_valid_fraction",
        "reappeared_object_slot_used_count",
        "reappeared_object_slot_occupancy_entropy",
        "reappeared_object_slot_separation",
        "reappeared_object_slot_collapse_fraction",
        "reappeared_object_slot_target_position_error",
        "reappeared_object_slot_target_recall",
        "reappeared_object_slot_distractor_position_error",
        "reappeared_object_slot_distractor_recall",
        "reappeared_object_slot_pair_position_error",
        "reappeared_object_slot_assignment_object_file_id_usage",
        "reappeared_object_slot_assignment_object_id_usage",
        "reappeared_object_slot_position_linear_error",
        "reappeared_object_slot_position_linear_improvement",
        "reappeared_object_slot_position_linear_r2",
        "reappeared_object_slot_ternary_zero_fraction",
        "reappeared_object_slot_ternary_nonzero_fraction",
        "reappeared_object_slot_ternary_axis_usage_count",
        "reappeared_file_slot_target_match_accuracy",
        "reappeared_file_slot_target_hard_match_accuracy",
        "reappeared_file_slot_distractor_match_accuracy",
        "reappeared_file_slot_pair_match_accuracy",
        "reappeared_file_slot_candidate_mean_count",
        "reappeared_file_slot_row_coverage_fraction",
        "reappeared_file_slot_target_file_recall_fraction",
        "reappeared_file_slot_distractor_file_recall_fraction",
        "reappeared_file_slot_assignment_position_error",
        "reappeared_file_slot_target_assignment_position_error",
        "reappeared_file_slot_distractor_assignment_position_error",
        "reappeared_file_slot_assignment_object_file_id_usage",
        "reappeared_file_slot_assignment_object_id_usage",
        "reappeared_file_slot_assignment_sequence_id_usage",
        "reappeared_file_slot_occluded_bridge_delta",
        "reappeared_file_slot_ternary_nonzero_fraction",
        "reappeared_file_slot_dynamics_position_error",
        "reappeared_file_slot_dynamics_valid_fraction",
        "reappeared_oracle_position_global_file_slot_target_match_accuracy",
        "reappeared_oracle_position_global_file_slot_target_hard_match_accuracy",
        "reappeared_oracle_position_global_file_slot_distractor_match_accuracy",
        "reappeared_oracle_position_global_file_slot_pair_match_accuracy",
        "reappeared_oracle_position_global_file_slot_candidate_mean_count",
        "reappeared_oracle_position_global_file_slot_assignment_object_file_id_usage",
        "reappeared_oracle_position_global_file_slot_assignment_object_id_usage",
        "reappeared_oracle_position_global_file_slot_assignment_sequence_id_usage",
        "reappeared_oracle_position_ceiling_file_slot_target_match_accuracy",
        "reappeared_oracle_position_ceiling_file_slot_target_hard_match_accuracy",
        "reappeared_oracle_position_ceiling_file_slot_distractor_match_accuracy",
        "reappeared_oracle_position_ceiling_file_slot_pair_match_accuracy",
        "reappeared_oracle_position_ceiling_file_slot_candidate_mean_count",
        "reappeared_oracle_position_ceiling_file_slot_assignment_position_error",
        "reappeared_oracle_position_ceiling_file_slot_assignment_object_file_id_usage",
        "reappeared_oracle_position_ceiling_file_slot_assignment_object_id_usage",
        "reappeared_oracle_position_ceiling_file_slot_assignment_sequence_id_usage",
        "reappeared_oracle_noise_file_slot_noise_0px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_1px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_2px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_3px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_4px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_6px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_7px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_8px_target_match_accuracy",
        "reappeared_oracle_noise_file_slot_noise_6px_position_noise_normalized",
        "reappeared_ballistic_file_slot_target_match_accuracy",
        "reappeared_ballistic_file_slot_target_hard_match_accuracy",
        "reappeared_ballistic_file_slot_distractor_match_accuracy",
        "reappeared_ballistic_file_slot_pair_match_accuracy",
        "reappeared_ballistic_file_slot_assignment_position_error",
        "reappeared_ballistic_file_slot_assignment_object_file_id_usage",
        "reappeared_ballistic_file_slot_assignment_object_id_usage",
        "reappeared_ballistic_file_slot_assignment_sequence_id_usage",
        "reappeared_dynamics_endpoint_valid_pair_fraction",
        "reappeared_dynamics_endpoint_true_pair_distance",
        "reappeared_dynamics_endpoint_predicted_pair_distance",
        "reappeared_dynamics_endpoint_pair_distance_ratio",
        "reappeared_dynamics_endpoint_pair_distance_compression",
        "reappeared_dynamics_endpoint_midpoint_error",
        "reappeared_dynamics_endpoint_midpoint_pull",
        "reappeared_dynamics_endpoint_error_mean",
        "reappeared_dynamics_endpoint_error_median",
        "reappeared_dynamics_endpoint_error_p75",
        "reappeared_dynamics_endpoint_error_p90",
        "reappeared_dynamics_endpoint_error_p95",
        "reappeared_dynamics_endpoint_error_max",
        "reappeared_dynamics_endpoint_target_bias_x",
        "reappeared_dynamics_endpoint_target_bias_y",
        "reappeared_dynamics_endpoint_distractor_bias_x",
        "reappeared_dynamics_endpoint_distractor_bias_y",
        "reappeared_dynamics_endpoint_paired_error_cosine",
        "reappeared_dynamics_endpoint_paired_error_x_correlation",
        "reappeared_dynamics_endpoint_paired_error_y_correlation",
        "reappeared_dynamics_local_file_slot_target_match_accuracy",
        "reappeared_dynamics_local_file_slot_target_hard_match_accuracy",
        "reappeared_dynamics_local_file_slot_distractor_match_accuracy",
        "reappeared_dynamics_local_file_slot_pair_match_accuracy",
        "reappeared_dynamics_local_file_slot_valid_pair_fraction",
        "reappeared_dynamics_all_file_slot_object_count",
        "reappeared_dynamics_all_file_slot_object_match_accuracy",
        "reappeared_dynamics_all_file_slot_target_match_accuracy",
        "reappeared_dynamics_all_file_slot_set_match_accuracy",
        "reappeared_dynamics_all_file_slot_row_coverage_fraction",
        "reappeared_dynamics_all_endpoint_object_count",
        "reappeared_dynamics_all_endpoint_valid_row_fraction",
        "reappeared_dynamics_all_endpoint_min_interobject_spacing",
        "reappeared_dynamics_all_endpoint_min_interobject_spacing_px",
        "reappeared_dynamics_all_endpoint_endpoint_error_mean",
        "reappeared_dynamics_all_endpoint_endpoint_error_median",
        "reappeared_dynamics_all_endpoint_endpoint_error_p90",
        "reappeared_dynamics_all_endpoint_endpoint_error_to_spacing_ratio",
        "reappeared_dynamics_all_endpoint_shared_track_endpoint_error_mean",
        "reappeared_dynamics_all_endpoint_extra_track_endpoint_error_mean",
        "reappeared_dynamics_slot_clean_endpoint_slot_clean_object_fraction",
        "reappeared_dynamics_slot_clean_endpoint_clean_endpoint_error_p90",
        "reappeared_dynamics_slot_clean_endpoint_dirty_endpoint_error_p90",
        "reappeared_dynamics_slot_clean_endpoint_high_error_clean_fraction",
        "reappeared_dynamics_neutral_all_file_slot_forced_correct_fraction",
        "reappeared_dynamics_neutral_all_file_slot_neutral_decline_fraction",
        "reappeared_dynamics_neutral_all_file_slot_confident_correct_bind_fraction",
        "reappeared_dynamics_neutral_all_file_slot_confident_wrong_bind_fraction",
        "reappeared_dynamics_neutral_all_file_slot_correct_decline_fraction",
        "reappeared_dynamics_neutral_all_file_slot_wrong_decline_fraction",
        "reappeared_dynamics_runtime_confidence_object_count",
        "reappeared_dynamics_runtime_confidence_decision_coverage_fraction",
        "reappeared_dynamics_runtime_confidence_actual_endpoint_error_mean",
        "reappeared_dynamics_runtime_confidence_actual_endpoint_error_p90",
        "reappeared_dynamics_runtime_confidence_endpoint_error_to_spacing_ratio_p90",
        "reappeared_dynamics_runtime_confidence_unsafe_endpoint_error_fraction",
        "reappeared_dynamics_runtime_confidence_slot_unsafe_fraction",
        "reappeared_dynamics_runtime_confidence_pair_unsafe_fraction",
        "reappeared_dynamics_runtime_confidence_pair_unsafe_within_scene_valid_fraction",
        "reappeared_dynamics_runtime_confidence_runtime_uncertainty_mean",
        "reappeared_dynamics_runtime_confidence_runtime_confidence_mean",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_mean",
        "reappeared_dynamics_runtime_confidence_calibrated_confidence_mean",
        "reappeared_dynamics_runtime_confidence_candidate_margin_uncertainty_mean",
        "reappeared_dynamics_runtime_confidence_runtime_uncertainty_error_pearson",
        "reappeared_dynamics_runtime_confidence_runtime_uncertainty_error_spearman",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_error_pearson",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_error_spearman",
        "reappeared_dynamics_runtime_confidence_naive_margin_uncertainty_error_pearson",
        "reappeared_dynamics_runtime_confidence_candidate_margin_uncertainty_error_pearson",
        "reappeared_dynamics_runtime_confidence_runtime_uncertainty_unsafe_auroc",
        "reappeared_dynamics_runtime_confidence_runtime_uncertainty_unsafe_auprc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_unsafe_auroc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_unsafe_auprc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_pair_unsafe_auroc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_pair_unsafe_auprc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_pair_unsafe_within_scene_auroc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_pair_unsafe_within_scene_auprc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_scene_adjusted_pair_unsafe_auroc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_scene_adjusted_pair_unsafe_auprc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_within_scene_variance_mean",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_within_scene_pair_unsafe_gap",
        "reappeared_dynamics_runtime_confidence_naive_margin_uncertainty_unsafe_auroc",
        "reappeared_dynamics_runtime_confidence_naive_margin_uncertainty_unsafe_auprc",
        "reappeared_dynamics_runtime_confidence_candidate_margin_uncertainty_unsafe_auroc",
        "reappeared_dynamics_runtime_confidence_candidate_margin_uncertainty_unsafe_auprc",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_high_error_lift",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_unsafe_lift",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_error_low_bucket_mean",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_error_mid_bucket_mean",
        "reappeared_dynamics_runtime_confidence_calibrated_uncertainty_error_high_bucket_mean",
        "reappeared_dynamics_runtime_confidence_runtime_confidence_correct_decline_mean",
        "reappeared_dynamics_runtime_confidence_calibrated_confidence_correct_decline_mean",
        "reappeared_dynamics_runtime_confidence_runtime_confidence_drop_on_correct_declines",
        "reappeared_dynamics_runtime_confidence_calibrated_confidence_drop_on_correct_declines",
        "reappeared_dynamics_runtime_confidence_confidence_true_position_usage",
        "reappeared_dynamics_runtime_confidence_confidence_endpoint_error_usage",
        "reappeared_dynamics_runtime_confidence_confidence_object_file_id_usage",
        "reappeared_ballistic_endpoint_pair_distance_ratio",
        "reappeared_ballistic_endpoint_error_p95",
        "reappeared_ballistic_local_file_slot_target_match_accuracy",
        "reappeared_ballistic_local_file_slot_pair_match_accuracy",
        "reappeared_ballistic_all_file_slot_object_match_accuracy",
        "reappeared_ballistic_all_file_slot_set_match_accuracy",
        "reappeared_ballistic_all_endpoint_endpoint_error_mean",
        "reappeared_ballistic_all_endpoint_endpoint_error_p90",
        "reappeared_ballistic_all_endpoint_endpoint_error_to_spacing_ratio",
        "reappeared_ballistic_all_endpoint_shared_track_endpoint_error_mean",
        "reappeared_ballistic_slot_clean_endpoint_slot_clean_object_fraction",
        "reappeared_ballistic_slot_clean_endpoint_clean_endpoint_error_p90",
        "reappeared_ballistic_neutral_all_file_slot_neutral_decline_fraction",
        "reappeared_oracle_all_file_slot_object_match_accuracy",
        "reappeared_oracle_all_file_slot_set_match_accuracy",
        "reappeared_oracle_error_shape_file_slot_center_bias_target_match_accuracy",
        "reappeared_oracle_error_shape_file_slot_center_bias_pair_match_accuracy",
        "reappeared_oracle_error_shape_file_slot_center_bias_pair_distance_ratio",
        "reappeared_oracle_error_shape_file_slot_center_bias_injected_error",
        "reappeared_oracle_error_shape_file_slot_correlated_target_match_accuracy",
        "reappeared_oracle_error_shape_file_slot_correlated_pair_match_accuracy",
        "reappeared_oracle_error_shape_file_slot_correlated_pair_distance_ratio",
        "reappeared_oracle_error_shape_file_slot_heavy_tail_target_match_accuracy",
        "reappeared_oracle_error_shape_file_slot_heavy_tail_pair_match_accuracy",
        "reappeared_oracle_error_shape_file_slot_heavy_tail_injected_error",
        "reappeared_active_query_file_candidate_instance_match_accuracy",
        "reappeared_active_query_file_candidate_instance_hard_match_accuracy",
        "reappeared_active_query_file_candidate_mean_count",
        "reappeared_active_query_file_candidate_row_coverage_fraction",
        "reappeared_active_query_file_candidate_target_recall_fraction",
        "reappeared_active_state_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_active_state_prediction_error_query_file_candidate_error_hard_match_accuracy",
        "reappeared_active_state_prediction_error_query_file_candidate_row_coverage_fraction",
        "reappeared_active_state_prediction_error_query_file_candidate_target_recall_fraction",
        "reappeared_active_local_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_active_local_prediction_error_query_file_candidate_error_hard_match_accuracy",
        "reappeared_active_local_prediction_error_query_file_candidate_row_coverage_fraction",
        "reappeared_active_local_prediction_error_query_file_candidate_target_recall_fraction",
        "reappeared_oracle_position_query_file_candidate_instance_match_accuracy",
        "reappeared_oracle_position_query_file_candidate_instance_hard_match_accuracy",
        "reappeared_oracle_position_query_file_candidate_row_coverage_fraction",
        "reappeared_oracle_position_query_file_candidate_target_recall_fraction",
        "reappeared_predicted_position_query_file_candidate_instance_match_accuracy",
        "reappeared_predicted_position_query_file_candidate_instance_hard_match_accuracy",
        "reappeared_predicted_position_query_file_candidate_row_coverage_fraction",
        "reappeared_predicted_position_query_file_candidate_target_recall_fraction",
        "reappeared_predicted_position_state_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_predicted_position_local_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_feature_only_query_file_candidate_instance_match_accuracy",
        "reappeared_feature_only_query_file_candidate_instance_hard_match_accuracy",
        "reappeared_feature_only_query_file_candidate_row_coverage_fraction",
        "reappeared_feature_only_query_file_candidate_target_recall_fraction",
        "reappeared_feature_only_state_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_feature_only_local_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_feature_only_position_ablated_query_file_candidate_instance_match_accuracy",
        "reappeared_feature_only_position_ablated_query_file_candidate_instance_hard_match_accuracy",
        "reappeared_feature_only_position_ablated_query_file_candidate_row_coverage_fraction",
        "reappeared_feature_only_position_ablated_query_file_candidate_target_recall_fraction",
        "reappeared_learned_active_query_file_candidate_instance_match_accuracy",
        "reappeared_learned_active_query_file_candidate_instance_hard_match_accuracy",
        "reappeared_learned_active_query_file_candidate_mean_count",
        "reappeared_learned_active_query_file_candidate_row_coverage_fraction",
        "reappeared_learned_active_query_file_candidate_target_recall_fraction",
        "reappeared_learned_active_state_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_learned_active_state_prediction_error_query_file_candidate_error_hard_match_accuracy",
        "reappeared_learned_active_local_prediction_error_query_file_candidate_error_match_accuracy",
        "reappeared_learned_active_local_prediction_error_query_file_candidate_error_hard_match_accuracy",
        "reappeared_learned_active_file_gate_active_recall",
    }
    assert torch.allclose(out.diagnostics["temporal_visible_fraction"], torch.tensor(0.6), atol=1e-6)
    assert torch.allclose(out.diagnostics["temporal_occluded_fraction"], torch.tensor(0.4), atol=1e-6)
    assert out.diagnostics["object_file_id_bind_time_leakage_audit_pass"].item() == 1.0
    assert out.diagnostics["object_file_id_bind_time_candidate_filter_usage"].item() == 0.0
    assert out.diagnostics["object_file_id_auxiliary_label_usage"].item() == 1.0
    assert out.diagnostics["reappeared_object_slot_count"].item() == 2.0
    assert out.diagnostics["reappeared_object_slot_assignment_object_file_id_usage"].item() == 0.0
    assert out.diagnostics["reappeared_object_slot_assignment_object_id_usage"].item() == 0.0
    assert out.diagnostics["reappeared_file_slot_assignment_object_file_id_usage"].item() == 0.0
    assert out.diagnostics["reappeared_file_slot_assignment_object_id_usage"].item() == 0.0
    assert out.diagnostics["reappeared_file_slot_assignment_sequence_id_usage"].item() == 0.0
    assert out.diagnostics["reappeared_oracle_position_ceiling_file_slot_assignment_object_file_id_usage"].item() == 0.0
    assert out.diagnostics["reappeared_oracle_position_ceiling_file_slot_assignment_object_id_usage"].item() == 0.0
    assert out.diagnostics["reappeared_oracle_position_ceiling_file_slot_assignment_sequence_id_usage"].item() == 0.0

    dynamics_features = _active_file_dynamics_features(
        batch,
        model.memory.read_write_object_files(batch, torch.rand(5, cfg.d_model), step=1),
        torch.ones(5, dtype=torch.bool),
        cfg,
        torch.float32,
        torch.device("cpu"),
        torch.softmax(torch.rand(5, cfg.contexts), dim=-1),
        torch.ones(5, 1),
        torch.zeros(5, 1),
    )
    assert dynamics_features.shape == (5, _active_file_dynamics_input_dim(cfg))
