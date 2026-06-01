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
