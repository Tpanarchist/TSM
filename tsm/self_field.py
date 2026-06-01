from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import TsmConfig
from .context import ContextRouter, context_balance_loss, context_entropy
from .diagnostics import (
    feature_label_diagnostics,
    feature_match_diagnostics,
    grouped_instance_match_diagnostics,
    paired_feature_match_diagnostics,
    ternary_label_diagnostics,
)
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


def _zero_like_scalar(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=reference.dtype, device=reference.device)


def _paired_contrastive_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    detach_target: bool = True,
) -> torch.Tensor:
    if source.shape[0] < 2 or target.shape[0] < 2:
        return _zero_like_scalar(source)
    pair_count = min(source.shape[0], target.shape[0])
    source = F.normalize(source[:pair_count], dim=-1, eps=1e-6)
    target = target[:pair_count].to(device=source.device, dtype=source.dtype)
    if detach_target:
        target = target.detach()
    target = F.normalize(target, dim=-1, eps=1e-6)
    logits = torch.matmul(source, target.t()) / max(temperature, 1e-6)
    labels = torch.arange(pair_count, device=source.device)
    return F.cross_entropy(logits, labels)


def _bidirectional_paired_contrastive_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if source.shape[0] < 2 or target.shape[0] < 2:
        return _zero_like_scalar(source)
    return 0.5 * (
        _paired_contrastive_loss(source, target, temperature, detach_target=False)
        + _paired_contrastive_loss(target, source, temperature, detach_target=False)
    )


def _grouped_contrastive_query_loss(
    query: torch.Tensor,
    files: torch.Tensor,
    instance_labels: torch.Tensor,
    group_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if query.shape[0] < 2 or files.shape[0] < 2:
        return _zero_like_scalar(query)
    pair_count = min(query.shape[0], files.shape[0], instance_labels.shape[0], group_labels.shape[0])
    query = F.normalize(query[:pair_count], dim=-1, eps=1e-6)
    files = F.normalize(files[:pair_count].to(device=query.device, dtype=query.dtype), dim=-1, eps=1e-6)
    instance_labels = instance_labels[:pair_count].to(device=query.device, dtype=torch.long)
    group_labels = group_labels[:pair_count].to(device=query.device, dtype=torch.long)
    valid = (instance_labels >= 0) & (group_labels >= 0)
    query = query[valid]
    files = files[valid]
    instance_labels = instance_labels[valid]
    group_labels = group_labels[valid]
    if query.shape[0] < 2:
        return _zero_like_scalar(query)
    losses: list[torch.Tensor] = []
    for row in range(query.shape[0]):
        same_group = group_labels == group_labels[row]
        same_instance = instance_labels == instance_labels[row]
        candidates = same_group
        targets = same_instance & candidates
        if int(candidates.to(torch.long).sum().item()) < 2 or not bool(targets.any().item()):
            continue
        logits = torch.matmul(query[row : row + 1], files[candidates].t()) / max(temperature, 1e-6)
        target_positions = torch.nonzero(targets[candidates], as_tuple=False).flatten()
        losses.append(F.cross_entropy(logits, target_positions[:1]))
    return torch.stack(losses).mean() if losses else _zero_like_scalar(query)


def _object_cycle_loss(
    hidden_scores: torch.Tensor,
    reappeared_scores: torch.Tensor,
    hidden_file_scores: torch.Tensor,
    reappeared_file_scores: torch.Tensor,
    temperature: float,
    pair_weight: float,
    file_weight: float,
) -> torch.Tensor:
    if (
        hidden_scores.shape[0] < 2
        or reappeared_scores.shape[0] < 2
        or hidden_file_scores.shape[0] < 2
        or reappeared_file_scores.shape[0] < 2
    ):
        return _zero_like_scalar(hidden_scores)
    hidden_to_file = _bidirectional_paired_contrastive_loss(hidden_scores, hidden_file_scores, temperature)
    reappeared_to_file = _bidirectional_paired_contrastive_loss(reappeared_scores, reappeared_file_scores, temperature)
    file_context_cycle = _bidirectional_paired_contrastive_loss(hidden_file_scores, reappeared_file_scores, temperature)
    hidden_to_reappeared = _bidirectional_paired_contrastive_loss(hidden_scores, reappeared_scores, temperature)
    return hidden_to_file + reappeared_to_file + file_weight * file_context_cycle + pair_weight * hidden_to_reappeared


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

    @torch.no_grad()
    def _diagnostic_ternary_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        perception = self.perception(image, dataset_id=dataset_id)
        batch_size = image.shape[0]
        initial_q = self.mind.initial(batch_size, image.device)
        context = self.contexts(initial_q, perception.latents)
        expected0 = self.reality.predict_latents(initial_q, context.embedding)
        drives = DriveState.zeros(batch_size, image.device)
        attach_power = torch.zeros_like(perception.latents)
        sae0 = self.sae(
            perception.latents,
            expected0,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        q, _delta_q = self.mind.infer(
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
        return self.defs.project(sae.eps, context.probs)

    def _definition_state_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        perception = self.perception(image, dataset_id=dataset_id)
        batch_size = image.shape[0]
        initial_q = self.mind.initial(batch_size, image.device)
        context = self.contexts(initial_q, perception.latents)
        expected0 = self.reality.predict_latents(initial_q, context.embedding)
        drives = DriveState.zeros(batch_size, image.device)
        attach_power = torch.zeros_like(perception.latents)
        sae0 = self.sae(
            perception.latents,
            expected0,
            perception.meta.source_confidence,
            attach_power,
            drives,
            context.probs,
        )
        q, _delta_q = self.mind.infer(
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
        return self.defs.raw_scores(sae.eps, context.probs), context.probs

    def _definition_scores_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores, _context_probs = self._definition_state_for_image(image, dataset_id)
        return scores

    @torch.no_grad()
    def _diagnostic_definition_scores_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._definition_scores_for_image(image, dataset_id)

    @torch.no_grad()
    def _diagnostic_definition_state_for_image(
        self,
        image: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._definition_state_for_image(image, dataset_id)

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
        memory_read = self.memory.read_write_object_files(
            batch,
            perception.latents.mean(dim=1),
            step=len(self.memory.records),
        )
        memory_feature = memory_read.feature
        memory_confidence = memory_read.confidence
        memory_prediction_confidence = memory_confidence
        if "occluded_t" in batch:
            occluded_gate = batch["occluded_t"].to(device=image_t.device, dtype=image_t.dtype).unsqueeze(-1)
            memory_prediction_confidence = memory_prediction_confidence * occluded_gate
        if not self.cfg.use_memory_conditioning:
            memory_prediction_confidence = torch.zeros_like(memory_prediction_confidence)
        memory_definition_confidence = memory_prediction_confidence if self.cfg.use_ternary_conditioning else torch.zeros_like(memory_prediction_confidence)
        ternary_base = self.defs.project(sae.eps, context.probs)
        ternary = self.defs.project(sae.eps, context.probs, memory_feature, memory_definition_confidence)
        ternary_condition = ternary if self.cfg.use_ternary_conditioning else None
        ternary_base_condition = ternary_base if self.cfg.use_ternary_conditioning else None
        recon = self.reality.reconstruct_image(q, ternary_condition)
        next_image = self.reality.predict_next_image(
            q,
            context.embedding,
            ternary_condition,
            memory_feature,
            memory_prediction_confidence,
        )
        reappearance_alignment = _zero_like_scalar(image_t)
        object_cycle_consistency = _zero_like_scalar(image_t)
        reappearance_file_query = _zero_like_scalar(image_t)
        needs_reappearance_target = (
            self.cfg.reappearance_alignment_weight > 0.0
            or self.cfg.object_cycle_weight > 0.0
            or self.cfg.reappearance_file_query_weight > 0.0
        )
        if needs_reappearance_target and "visible_tp1" in batch and "occluded_t" in batch:
            visible_tp1_for_alignment = batch["visible_tp1"].to(device=image_t.device, dtype=image_t.dtype)
            occluded_t_for_alignment = batch["occluded_t"].to(device=image_t.device, dtype=image_t.dtype)
            reappeared_for_alignment = (occluded_t_for_alignment > 0.5) & (visible_tp1_for_alignment > 0.5)
            if bool(reappeared_for_alignment.any().item()):
                target_dataset_id = dataset_id[reappeared_for_alignment] if dataset_id is not None else None
                source_scores = self.defs.raw_scores(
                    sae.eps,
                    context.probs,
                    memory_feature,
                    memory_definition_confidence,
                )
                target_scores, target_context_probs = self._definition_state_for_image(
                    image_tp1[reappeared_for_alignment],
                    dataset_id=target_dataset_id,
                )
                source_reappeared_scores = source_scores[reappeared_for_alignment]
                if self.cfg.reappearance_alignment_weight > 0.0:
                    reappearance_alignment = _paired_contrastive_loss(
                        source_reappeared_scores,
                        target_scores,
                        self.cfg.reappearance_alignment_temperature,
                    )
                needs_file_anchor = (
                    self.cfg.object_cycle_weight > 0.0 or self.cfg.reappearance_file_query_weight > 0.0
                )
                if needs_file_anchor:
                    source_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_for_alignment],
                        context.probs[reappeared_for_alignment],
                        memory_definition_confidence[reappeared_for_alignment],
                    )
                    target_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_for_alignment],
                        target_context_probs,
                        memory_definition_confidence[reappeared_for_alignment],
                    )
                if self.cfg.object_cycle_weight > 0.0:
                    object_cycle_consistency = _object_cycle_loss(
                        source_reappeared_scores,
                        target_scores,
                        source_file_scores,
                        target_file_scores,
                        self.cfg.object_cycle_temperature,
                        self.cfg.object_cycle_pair_weight,
                        self.cfg.object_cycle_file_weight,
                    )
                if self.cfg.reappearance_file_query_weight > 0.0:
                    target_query_scores = self.defs.file_query_scores(target_scores)
                    file_query_pair = _bidirectional_paired_contrastive_loss(
                        target_query_scores,
                        target_file_scores,
                        self.cfg.reappearance_file_query_temperature,
                    )
                    file_query_hard = _zero_like_scalar(image_t)
                    if "sequence_id" in batch and "object_id" in batch:
                        sequence_id = batch["sequence_id"].to(device=image_t.device, dtype=torch.long)
                        object_id = batch["object_id"].to(device=image_t.device, dtype=torch.long)
                        file_query_hard = _grouped_contrastive_query_loss(
                            target_query_scores,
                            target_file_scores,
                            sequence_id[reappeared_for_alignment],
                            object_id[reappeared_for_alignment],
                            self.cfg.reappearance_file_query_temperature,
                        )
                    reappearance_file_query = (
                        file_query_pair + self.cfg.reappearance_file_query_hard_weight * file_query_hard
                    )
        prior = self.reality.context_prior(context.probs)
        free_energy = variational_free_energy(sae.raw, sae.precision, q, prior)
        ternary_nonzero = ternary.ne(0)
        ternary_base_nonzero = ternary_base.ne(0)
        memory_definition_flip = ternary.ne(ternary_base)
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
            "reappearance_alignment": reappearance_alignment,
            "object_cycle_consistency": object_cycle_consistency,
            "reappearance_file_query": reappearance_file_query,
        }
        diagnostics = {
            "context_max_probability": context.probs.max(dim=-1).values.mean(),
            "context_effective_count": torch.exp(context_entropy_value.detach()),
            "context_used_count": context_used,
            "ternary_zero_fraction": ternary.eq(0).float().mean(),
            "ternary_nonzero_fraction": ternary_nonzero.float().mean(),
            "ternary_positive_fraction": ternary.gt(0).float().mean(),
            "ternary_negative_fraction": ternary.lt(0).float().mean(),
            "memory_definition_flip_fraction": memory_definition_flip.to(image_t.dtype).mean(),
            "memory_definition_activation_delta": (
                ternary_nonzero.to(image_t.dtype).mean() - ternary_base_nonzero.to(image_t.dtype).mean()
            ),
            "ternary_condition_norm": (
                self.reality.condition_latents(q, ternary_condition) - q
            ).detach().norm(dim=-1).mean(),
            "memory_definition_condition_norm": (
                self.reality.condition_latents(q, ternary_condition)
                - self.reality.condition_latents(q, ternary_base_condition)
            ).detach().norm(dim=-1).mean(),
            "memory_condition_norm": (
                self.reality.condition_latents(q, ternary_condition, memory_feature, memory_prediction_confidence)
                - self.reality.condition_latents(q, ternary_condition)
            ).detach().norm(dim=-1).mean(),
            "memory_object_file_count": torch.tensor(
                len(self.memory.object_files),
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_object_read_count": torch.tensor(
                self.memory.object_read_count,
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_object_write_count": torch.tensor(
                self.memory.object_write_count,
                dtype=image_t.dtype,
                device=image_t.device,
            ),
            "memory_object_hit_fraction": memory_read.hit.to(image_t.dtype).mean(),
            "memory_object_write_fraction": memory_read.write.to(image_t.dtype).mean(),
            "memory_object_confidence_mean": memory_confidence.mean(),
            "memory_object_prediction_confidence_mean": memory_prediction_confidence.mean(),
            "memory_object_age_mean": memory_read.age.mean(),
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
                active_memory = memory_read.hit.to(device=image_t.device)
                if bool(active_memory.any().item()):
                    diagnostics.update(_prefix_metrics(
                        "memory_object_",
                        feature_label_diagnostics(memory_feature[active_memory], object_id[active_memory]),
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
                "temporal_memory_visible_hit_fraction": _masked_mean(
                    memory_read.hit.to(image_t.dtype),
                    visible_mask,
                    image_t.dtype,
                ),
                "temporal_memory_occluded_hit_fraction": _masked_mean(
                    memory_read.hit.to(image_t.dtype),
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_memory_occluded_confidence_mean": _masked_mean(
                    memory_prediction_confidence.squeeze(-1),
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_memory_definition_occluded_flip_fraction": _masked_mean(
                    memory_definition_flip.to(image_t.dtype).mean(dim=1),
                    occluded_mask,
                    image_t.dtype,
                ),
                "temporal_memory_definition_occluded_activation_delta": _masked_mean(
                    ternary_nonzero.to(image_t.dtype).mean(dim=1)
                    - ternary_base_nonzero.to(image_t.dtype).mean(dim=1),
                    occluded_mask,
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
            with torch.no_grad():
                next_without_memory = self.reality.predict_next_image(q, context.embedding, ternary_condition)
                next_without_any_memory = self.reality.predict_next_image(q, context.embedding, ternary_base_condition)
                no_memory_error = (next_without_memory - image_tp1).square().flatten(1).mean(dim=1)
                no_any_memory_error = (next_without_any_memory - image_tp1).square().flatten(1).mean(dim=1)
                memory_impact = no_memory_error - prediction_error_per_sample.detach()
                memory_total_impact = no_any_memory_error - prediction_error_per_sample.detach()
                memory_definition_impact = no_any_memory_error - no_memory_error
            diagnostics.update({
                "memory_prediction_impact_mean": memory_impact.mean(),
                "memory_prediction_occluded_impact_mean": _masked_mean(memory_impact, occluded_mask, image_t.dtype),
                "memory_prediction_reappeared_impact_mean": _masked_mean(memory_impact, reappeared, image_t.dtype),
                "memory_prediction_disappearance_impact_mean": _masked_mean(
                    memory_impact,
                    disappearance_mask,
                    image_t.dtype,
                ),
                "memory_total_prediction_impact_mean": memory_total_impact.mean(),
                "memory_total_prediction_occluded_impact_mean": _masked_mean(
                    memory_total_impact,
                    occluded_mask,
                    image_t.dtype,
                ),
                "memory_total_prediction_reappeared_impact_mean": _masked_mean(
                    memory_total_impact,
                    reappeared,
                    image_t.dtype,
                ),
                "memory_definition_prediction_impact_mean": memory_definition_impact.mean(),
                "memory_definition_prediction_occluded_impact_mean": _masked_mean(
                    memory_definition_impact,
                    occluded_mask,
                    image_t.dtype,
                ),
                "memory_definition_prediction_reappeared_impact_mean": _masked_mean(
                    memory_definition_impact,
                    reappeared,
                    image_t.dtype,
                ),
            })
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
                diagnostics.update(_prefix_metrics(
                    "occluded_base_",
                    ternary_label_diagnostics(
                        ternary_base[occluded_mask],
                        object_id[occluded_mask],
                        context_hard[occluded_mask],
                    ),
                ))
                diagnostics["occluded_memory_definition_object_probe_delta"] = (
                    diagnostics["occluded_object_ternary_mode_probe_accuracy"]
                    - diagnostics["occluded_base_ternary_mode_probe_accuracy"]
                )
                occluded_active = occluded_mask & memory_read.hit.to(device=image_t.device)
                if bool(occluded_active.any().item()):
                    diagnostics.update(_prefix_metrics(
                        "occluded_memory_object_",
                        feature_label_diagnostics(memory_feature[occluded_active], object_id[occluded_active]),
                    ))
                reappeared_active = reappeared & (object_id >= 0)
                if bool(reappeared_active.any().item()):
                    target_dataset_id = dataset_id[reappeared_active] if dataset_id is not None else None
                    target_ternary = self._diagnostic_ternary_for_image(
                        image_tp1[reappeared_active],
                        dataset_id=target_dataset_id,
                    )
                    target_definition_scores, target_context_probs = self._diagnostic_definition_state_for_image(
                        image_tp1[reappeared_active],
                        dataset_id=target_dataset_id,
                    )
                    source_definition_scores = self.defs.raw_scores(
                        sae.eps,
                        context.probs,
                        memory_feature,
                        memory_definition_confidence,
                    )[reappeared_active]
                    object_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_active],
                        context.probs[reappeared_active],
                        memory_definition_confidence[reappeared_active],
                    )
                    target_object_file_scores = self.defs.memory_scores(
                        memory_feature[reappeared_active],
                        target_context_probs,
                        memory_definition_confidence[reappeared_active],
                    )
                    target_query_scores = self.defs.file_query_scores(target_definition_scores)
                    source_labels = object_id[reappeared_active]
                    diagnostics.update(_prefix_metrics(
                        "reappeared_",
                        feature_match_diagnostics(
                            ternary[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                            source_labels,
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_base_",
                        feature_match_diagnostics(
                            ternary_base[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                            source_labels,
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_",
                        paired_feature_match_diagnostics(
                            ternary[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_base_",
                        paired_feature_match_diagnostics(
                            ternary_base[reappeared_active].sign().to(image_t.dtype),
                            target_ternary.sign().to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_file_",
                        paired_feature_match_diagnostics(
                            source_definition_scores.to(image_t.dtype),
                            object_file_scores.to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_target_file_",
                        paired_feature_match_diagnostics(
                            target_definition_scores.to(image_t.dtype),
                            target_object_file_scores.to(image_t.dtype),
                        ),
                    ))
                    diagnostics.update(_prefix_metrics(
                        "reappeared_query_file_",
                        paired_feature_match_diagnostics(
                            target_query_scores.to(image_t.dtype),
                            target_object_file_scores.to(image_t.dtype),
                        ),
                    ))
                    if "sequence_id" in batch:
                        sequence_id = batch["sequence_id"].to(device=image_t.device, dtype=torch.long)
                        source_sequences = sequence_id[reappeared_active]
                        diagnostics.update(_prefix_metrics(
                            "reappeared_file_",
                            grouped_instance_match_diagnostics(
                                source_definition_scores.to(image_t.dtype),
                                object_file_scores.to(image_t.dtype),
                                source_sequences,
                                source_sequences,
                                source_labels,
                                source_labels,
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_target_file_",
                            grouped_instance_match_diagnostics(
                                target_definition_scores.to(image_t.dtype),
                                target_object_file_scores.to(image_t.dtype),
                                source_sequences,
                                source_sequences,
                                source_labels,
                                source_labels,
                            ),
                        ))
                        diagnostics.update(_prefix_metrics(
                            "reappeared_query_file_",
                            grouped_instance_match_diagnostics(
                                target_query_scores.to(image_t.dtype),
                                target_object_file_scores.to(image_t.dtype),
                                source_sequences,
                                source_sequences,
                                source_labels,
                                source_labels,
                            ),
                        ))
                    diagnostics["reappeared_memory_definition_match_delta"] = (
                        diagnostics["reappeared_feature_match_accuracy"]
                        - diagnostics["reappeared_base_feature_match_accuracy"]
                    )
                    diagnostics["reappeared_paired_memory_definition_match_delta"] = (
                        diagnostics["reappeared_paired_feature_match_accuracy"]
                        - diagnostics["reappeared_base_paired_feature_match_accuracy"]
                    )
                elif include_label_diagnostics:
                    diagnostics.update({
                        "reappeared_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_file_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_file_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_file_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_file_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_target_file_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_match_margin": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_same_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_margin": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_same_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_nearest_same_group_other_distance": _zero_like_scalar(image_t),
                        "reappeared_query_file_instance_hard_valid_fraction": _zero_like_scalar(image_t),
                        "reappeared_base_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_base_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_base_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_base_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_match_accuracy": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_match_margin": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_same_distance": _zero_like_scalar(image_t),
                        "reappeared_base_paired_feature_nearest_other_distance": _zero_like_scalar(image_t),
                        "reappeared_memory_definition_match_delta": _zero_like_scalar(image_t),
                        "reappeared_paired_memory_definition_match_delta": _zero_like_scalar(image_t),
                    })
        total = (
            self.cfg.recon_weight * losses["reconstruction"]
            + self.cfg.pred_weight * losses["prediction"]
            + self.cfg.free_energy_weight * losses["free_energy"]
            + self.cfg.complexity_weight * losses["complexity"]
            + self.cfg.context_entropy_weight * losses["context_entropy"]
            + self.cfg.context_balance_weight * losses["context_balance"]
            + self.cfg.ternary_activation_weight * losses["ternary_activation_l1"]
            + self.cfg.bit_cost_weight * losses["bit_cost"]
            + self.cfg.reappearance_alignment_weight * losses["reappearance_alignment"]
            + self.cfg.object_cycle_weight * losses["object_cycle_consistency"]
            + self.cfg.reappearance_file_query_weight * losses["reappearance_file_query"]
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
            memory_feature=memory_feature,
            memory_confidence=memory_prediction_confidence,
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
            prediction = self.reality.predict_next_image(
                latent,
                context_embedding,
                ablated,
                output.memory_feature,
                output.memory_confidence,
            )
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
