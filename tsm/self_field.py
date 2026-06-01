from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import TsmConfig
from .context import ContextRouter, context_balance_loss, context_entropy
from .diagnostics import ternary_label_diagnostics
from .definitions import DefinitionBank
from .develop import DevelopmentalScheduler
from .drives import DriveDynamics
from .evidence import EvidenceAccumulator
from .free_energy import variational_free_energy
from .gate import MutationGate
from .memory import Memory
from .mind import Mind
from .perception import PerceptionSurface
from .reality import Reality
from .sae import SAE
from .trauma import TraumaMonitor
from .types import DriveState, TickOutput, TrainOutput


def _mode_context_stats(
    mode: torch.Tensor,
    context_hard: torch.Tensor,
    context_count: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    valid = mode >= 0
    if not bool(valid.any().item()):
        zero = torch.zeros((), dtype=dtype, device=context_hard.device)
        return {
            "mode_context_consistency": zero,
            "context_mode_purity": zero,
            "mode_context_separation": zero,
            "mode_context_used_count": zero,
        }
    mode = mode[valid]
    context_hard = context_hard[valid]
    total = mode.numel()
    mode_majority = torch.zeros((), dtype=dtype, device=context_hard.device)
    for mode_id in mode.unique():
        routed = context_hard[mode == mode_id]
        counts = torch.bincount(routed, minlength=context_count).to(dtype)
        mode_majority = mode_majority + counts.max()

    context_majority = torch.zeros((), dtype=dtype, device=context_hard.device)
    for ctx_id in context_hard.unique():
        routed_modes = mode[context_hard == ctx_id]
        counts = torch.bincount(routed_modes).to(dtype)
        context_majority = context_majority + counts.max()

    mode_count = mode.unique().numel()
    used_count = context_hard.unique().numel()
    possible = max(1, min(int(mode_count), context_count))
    return {
        "mode_context_consistency": mode_majority / total,
        "context_mode_purity": context_majority / total,
        "mode_context_separation": torch.tensor(used_count / possible, dtype=dtype, device=context_hard.device),
        "mode_context_used_count": torch.tensor(used_count, dtype=dtype, device=context_hard.device),
    }


def _prefix_metrics(prefix: str, metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = mask.to(device=values.device).bool()
    if not bool(mask.any().item()):
        return torch.zeros((), dtype=dtype, device=values.device)
    return values[mask].mean()


def _masked_context_used(
    context_hard: torch.Tensor,
    mask: torch.Tensor,
    context_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = mask.to(device=context_hard.device).bool()
    if not bool(mask.any().item()):
        return torch.zeros((), dtype=dtype, device=context_hard.device)
    return torch.bincount(context_hard[mask], minlength=context_count).gt(0).to(dtype).sum()


class Self(nn.Module):
    def __init__(self, cfg: TsmConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TsmConfig()
        self.perception = PerceptionSurface(self.cfg)
        self.reality = Reality(self.cfg)
        self.mind = Mind(self.cfg)
        self.sae = SAE(self.cfg)
        self.contexts = ContextRouter(self.cfg)
        self.defs = DefinitionBank(self.cfg)
        self.drive_dynamics = DriveDynamics()
        self.memory = Memory()
        self.evidence = EvidenceAccumulator()
        self.gate = MutationGate()
        self.trauma = TraumaMonitor()
        self.stage = DevelopmentalScheduler()

    def forward_train(self, batch: dict[str, torch.Tensor], include_label_diagnostics: bool = True) -> TrainOutput:
        image_t = batch["image_t"]
        image_tp1 = batch["image_tp1"]
        dataset_id = batch.get("dataset_id")
        mode = batch.get("mode", batch.get("label"))
        perception = self.perception(image_t, dataset_id=dataset_id)
        batch_size = image_t.shape[0]
        initial_q = self.mind.initial(batch_size, image_t.device)
        context = self.contexts(initial_q, perception.latents)
        expected0 = self.reality.predict_latents(initial_q, context.embedding)
        drives = DriveState.zeros(batch_size, image_t.device)
        attach_power = torch.zeros_like(perception.latents)
        sae0 = self.sae(
            perception.latents,
            expected0,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        q, delta_q = self.mind.infer(
            perception.latents,
            sae0.eps,
            context.embedding,
            steps=self.cfg.inference_steps,
        )
        expected = self.reality.predict_latents(q, context.embedding)
        sae = self.sae(
            perception.latents,
            expected,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        ternary = self.defs.project(sae.eps, context.probs)
        ternary_condition = ternary if self.cfg.use_ternary_conditioning else None
        recon = self.reality.reconstruct_image(q, ternary_condition)
        next_image = self.reality.predict_next_image(q, context.embedding, ternary_condition)
        prior = self.reality.context_prior(context.probs)
        free_energy = variational_free_energy(sae.raw, sae.precision, q, prior)
        ternary_nonzero = ternary.ne(0)
        context_hard = context.probs.argmax(dim=-1)
        context_used = torch.bincount(context_hard, minlength=self.cfg.contexts).gt(0).float().sum()
        context_entropy_value = context_entropy(context.probs)
        context_balance_value = context_balance_loss(context.probs)
        prediction_error_per_sample = (next_image - image_tp1).square().flatten(1).mean(dim=1)
        reconstruction_error_per_sample = (recon - image_t).square().flatten(1).mean(dim=1)
        severity_per_sample = sae.severity.mean(dim=1)
        coherence_per_sample = sae.coherence.mean(dim=1)
        losses = {
            "reconstruction": F.mse_loss(recon, image_t),
            "prediction": F.mse_loss(next_image, image_tp1),
            "free_energy": free_energy,
            "complexity": F.mse_loss(q.mean(dim=1), prior),
            "context_entropy": context_entropy_value,
            "context_balance": context_balance_value,
            "ternary_activation_l1": ternary.abs().mean(),
            "bit_cost": self.defs.bit_cost(),
        }
        diagnostics = {
            "context_max_probability": context.probs.max(dim=-1).values.mean(),
            "context_effective_count": torch.exp(context_entropy_value.detach()),
            "context_used_count": context_used,
            "ternary_zero_fraction": ternary.eq(0).float().mean(),
            "ternary_nonzero_fraction": ternary_nonzero.float().mean(),
            "ternary_positive_fraction": ternary.gt(0).float().mean(),
            "ternary_negative_fraction": ternary.lt(0).float().mean(),
            "ternary_condition_norm": (
                self.reality.condition_latents(q, ternary_condition) - q
            ).detach().norm(dim=-1).mean(),
            "sae_severity_mean": sae.severity.mean(),
            "sae_coherence_mean": sae.coherence.mean(),
            "gate_accept_count": torch.tensor(
                sum(decision.accepted for decision in self.gate.decisions),
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "gate_reject_count": torch.tensor(
                sum(not decision.accepted for decision in self.gate.decisions),
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_records": torch.tensor(len(self.memory.records), dtype=image_t.dtype, device=image_t.device),
        }
        if mode is not None:
            mode = mode.to(device=image_t.device, dtype=torch.long)
            diagnostics.update(_mode_context_stats(
                mode,
                context_hard,
                self.cfg.contexts,
                image_t.dtype,
            ))
            diagnostics["mode_count"] = torch.tensor(
                mode[mode >= 0].unique().numel(),
                dtype=image_t.dtype,
                device=image_t.device,
            )
            if include_label_diagnostics:
                diagnostics.update(ternary_label_diagnostics(ternary, mode, context_hard))
        if "phase" in batch:
            phase = batch["phase"].to(device=image_t.device, dtype=torch.long)
            diagnostics.update(_prefix_metrics(
                "phase_",
                _mode_context_stats(
                    phase,
                    context_hard,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
            ))
            diagnostics["phase_count"] = torch.tensor(
                phase[phase >= 0].unique().numel(),
                dtype=image_t.dtype,
                device=image_t.device,
            )
            if include_label_diagnostics:
                diagnostics.update(_prefix_metrics(
                    "phase_",
                    ternary_label_diagnostics(ternary, phase, context_hard),
                ))
        if "object_id" in batch:
            object_id = batch["object_id"].to(device=image_t.device, dtype=torch.long)
            diagnostics.update(_prefix_metrics(
                "object_",
                _mode_context_stats(
                    object_id,
                    context_hard,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
            ))
            diagnostics["object_count"] = torch.tensor(
                object_id[object_id >= 0].unique().numel(),
                dtype=image_t.dtype,
                device=image_t.device,
            )
            if include_label_diagnostics:
                diagnostics.update(_prefix_metrics(
                    "object_",
                    ternary_label_diagnostics(ternary, object_id, context_hard),
                ))
        if "visible_t" in batch and "occluded_t" in batch:
            visible_t = batch["visible_t"].to(device=image_t.device, dtype=image_t.dtype)
            occluded_t = batch["occluded_t"].to(device=image_t.device, dtype=image_t.dtype)
            visible_tp1 = batch.get("visible_tp1", torch.zeros_like(visible_t)).to(device=image_t.device, dtype=image_t.dtype)
            occluded_tp1 = batch.get("occluded_tp1", torch.zeros_like(occluded_t)).to(device=image_t.device, dtype=image_t.dtype)
            moved = batch.get("moved", torch.zeros_like(visible_t)).to(device=image_t.device, dtype=image_t.dtype)
            identity_preserved = batch.get("identity_preserved", torch.zeros_like(visible_t)).to(
                device=image_t.device,
                dtype=image_t.dtype,
            )
            unexpected_disappearance = batch.get("unexpected_disappearance", torch.zeros_like(visible_t)).to(
                device=image_t.device,
                dtype=image_t.dtype,
            )
            reappeared = (occluded_t > 0.5) & (visible_tp1 > 0.5)
            visible_mask = visible_t > 0.5
            occluded_mask = occluded_t > 0.5
            moved_mask = moved > 0.5
            disappearance_mask = unexpected_disappearance > 0.5
            diagnostics.update({
                "temporal_visible_fraction": visible_t.mean(),
                "temporal_occluded_fraction": occluded_t.mean(),
                "temporal_moved_fraction": moved.mean(),
                "temporal_identity_preserved_fraction": identity_preserved.mean(),
                "temporal_unexpected_disappearance_fraction": unexpected_disappearance.mean(),
                "temporal_reappeared_fraction": reappeared.to(image_t.dtype).mean(),
                "temporal_visible_tp1_fraction": visible_tp1.mean(),
                "temporal_occluded_tp1_fraction": occluded_tp1.mean(),
                "temporal_context_visible_used_count": _masked_context_used(
                    context_hard,
                    visible_mask,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
                "temporal_context_occluded_used_count": _masked_context_used(
                    context_hard,
                    occluded_mask,
                    self.cfg.contexts,
                    image_t.dtype,
                ),
                "temporal_sae_visible_mean": _masked_mean(severity_per_sample, visible_mask, image_t.dtype),
                "temporal_sae_occluded_mean": _masked_mean(severity_per_sample, occluded_mask, image_t.dtype),
                "temporal_sae_moved_mean": _masked_mean(severity_per_sample, moved_mask, image_t.dtype),
                "temporal_sae_disappearance_mean": _masked_mean(
                    severity_per_sample,
                    disappearance_mask,
                    image_t.dtype,
                ),
                "temporal_sae_reappeared_mean": _masked_mean(severity_per_sample, reappeared, image_t.dtype),
                "temporal_coherence_visible_mean": _masked_mean(coherence_per_sample, visible_mask, image_t.dtype),
                "temporal_coherence_occluded_mean": _masked_mean(coherence_per_sample, occluded_mask, image_t.dtype),
                "temporal_prediction_visible_mean": _masked_mean(
                    prediction_error_per_sample,
                    visible_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_occluded_mean": _masked_mean(
                    prediction_error_per_sample,
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_moved_mean": _masked_mean(
                    prediction_error_per_sample,
                    moved_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_disappearance_mean": _masked_mean(
                    prediction_error_per_sample,
                    disappearance_mask,
                    image_t.dtype,
                ),
                "temporal_prediction_reappeared_mean": _masked_mean(
                    prediction_error_per_sample,
                    reappeared,
                    image_t.dtype,
                ),
                "temporal_reconstruction_visible_mean": _masked_mean(
                    reconstruction_error_per_sample,
                    visible_mask,
                    image_t.dtype,
                ),
                "temporal_reconstruction_occluded_mean": _masked_mean(
                    reconstruction_error_per_sample,
                    occluded_mask,
                    image_t.dtype,
                ),
            })
            diagnostics["temporal_sae_occlusion_delta"] = (
                diagnostics["temporal_sae_occluded_mean"] - diagnostics["temporal_sae_visible_mean"]
            )
            diagnostics["temporal_prediction_occlusion_delta"] = (
                diagnostics["temporal_prediction_occluded_mean"] - diagnostics["temporal_prediction_visible_mean"]
            )
            if include_label_diagnostics and "object_id" in batch and bool(occluded_mask.any().item()):
                object_id = batch["object_id"].to(device=image_t.device, dtype=torch.long)
                diagnostics.update(_prefix_metrics(
                    "occluded_object_",
                    ternary_label_diagnostics(
                        ternary[occluded_mask],
                        object_id[occluded_mask],
                        context_hard[occluded_mask],
                    ),
                ))
        total = (
            self.cfg.recon_weight * losses["reconstruction"]
            + self.cfg.pred_weight * losses["prediction"]
            + self.cfg.free_energy_weight * losses["free_energy"]
            + self.cfg.complexity_weight * losses["complexity"]
            + self.cfg.context_entropy_weight * losses["context_entropy"]
            + self.cfg.context_balance_weight * losses["context_balance"]
            + self.cfg.ternary_activation_weight * losses["ternary_activation_l1"]
            + self.cfg.bit_cost_weight * losses["bit_cost"]
        )
        self.memory.write(sae.eps, sae.coherence)
        for candidate in self.evidence.accumulate(delta_q, sae.coherence):
            self.gate.consider(candidate)
        self.trauma.scan(sae.eps, self.gate.rejections)
        return TrainOutput(
            total_loss=total,
            losses=losses,
            diagnostics=diagnostics,
            recon_image=recon,
            next_image=next_image,
            context=context,
            sae=sae,
            ternary=ternary,
            latent_state=q,
        )

    @torch.no_grad()
    def ternary_prediction_impacts(self, output: TrainOutput, image_tp1: torch.Tensor) -> torch.Tensor:
        axis_count = output.ternary.shape[-1]
        if not self.cfg.use_ternary_conditioning:
            return torch.zeros(axis_count, dtype=output.ternary.dtype, device=output.ternary.device)
        base_error = (output.next_image.detach() - image_tp1).square().flatten(1).mean(dim=1)
        impacts: list[torch.Tensor] = []
        latent = output.latent_state.detach()
        context_embedding = output.context.embedding.detach()
        ternary = output.ternary.detach()
        for axis in range(axis_count):
            ablated = ternary.clone()
            ablated[:, axis] = 0
            prediction = self.reality.predict_next_image(latent, context_embedding, ablated)
            ablated_error = (prediction - image_tp1).square().flatten(1).mean(dim=1)
            impacts.append((ablated_error - base_error).clamp_min(0.0).mean())
        return torch.stack(impacts) if impacts else torch.zeros(0, dtype=output.ternary.dtype, device=output.ternary.device)

    @torch.no_grad()
    def tick(self, raw_inputs: torch.Tensor | dict[str, torch.Tensor]) -> TickOutput:
        image = raw_inputs["image_t"] if isinstance(raw_inputs, dict) else raw_inputs
        batch = {"image_t": image, "image_tp1": image}
        out = self.forward_train(batch)
        return TickOutput(
            reconstruction=out.recon_image,
            next_prediction=out.next_image,
            context_probs=out.context.probs,
            sae=out.sae,
            ternary=out.ternary,
            action=None,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> TrainOutput:
        return self.forward_train(batch)
