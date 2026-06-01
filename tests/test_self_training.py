import torch

from tsm.config import TsmConfig
from tsm.self_field import Self


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
    )
    model = Self(cfg)
    batch = {
        "image_t": torch.rand(5, 1, 16, 16),
        "image_tp1": torch.rand(5, 1, 16, 16),
        "dataset_id": torch.zeros(5, dtype=torch.long),
        "sequence_id": torch.zeros(5, dtype=torch.long),
        "mode": torch.tensor([0, 1, 2, 2, 3], dtype=torch.long),
        "phase": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "object_id": torch.tensor([0, 1, 2, 3, 0], dtype=torch.long),
        "visible_t": torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32),
        "visible_tp1": torch.tensor([1, 1, 0, 0, 1], dtype=torch.float32),
        "occluded_t": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32),
        "occluded_tp1": torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32),
        "moved": torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32),
        "identity_preserved": torch.ones(5),
        "unexpected_disappearance": torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32),
        "object_position_t": torch.zeros(5, 2),
    }

    out = model.forward_train(batch)

    assert set(out.diagnostics) >= {
        "phase_ternary_mode_probe_accuracy",
        "object_ternary_mode_probe_accuracy",
        "occluded_object_ternary_mode_probe_accuracy",
        "temporal_visible_fraction",
        "temporal_occluded_fraction",
        "temporal_reappeared_fraction",
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
    }
    assert torch.allclose(out.diagnostics["temporal_visible_fraction"], torch.tensor(0.6), atol=1e-6)
    assert torch.allclose(out.diagnostics["temporal_occluded_fraction"], torch.tensor(0.4), atol=1e-6)
