from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TsmConfig:
    d_model: int = 256
    workspace_latents: int = 64
    hierarchy_depth: int = 2
    contexts: int = 8
    definitions_per_context: int = 16
    image_channels: int = 1
    image_size: int = 28
    patch_size: int = 4
    attention_heads: int = 4
    inference_steps: int = 2
    dropout: float = 0.0
    apathy_floor: float = 0.02
    recon_weight: float = 1.0
    pred_weight: float = 1.0
    free_energy_weight: float = 0.05
    complexity_weight: float = 0.01
    context_entropy_weight: float = 0.001
    context_balance_weight: float = 0.01
    ternary_activation_weight: float = 0.001
    bit_cost_weight: float = 0.001
    use_ternary_conditioning: bool = True
    ternary_condition_scale: float = 0.25
    use_memory_conditioning: bool = True
    memory_condition_scale: float = 0.2
    memory_definition_scale: float = 0.75
    reappearance_alignment_weight: float = 0.0
    reappearance_alignment_temperature: float = 0.25
    object_cycle_weight: float = 0.0
    object_cycle_temperature: float = 0.25
    object_cycle_pair_weight: float = 0.25
    object_cycle_file_weight: float = 0.25
    reappearance_file_query_weight: float = 0.0
    reappearance_file_query_temperature: float = 0.25
    reappearance_file_query_hard_weight: float = 1.0
    active_file_query_weight: float = 0.0
    active_file_query_temperature: float = 0.25
    active_file_expectation_weight: float = 0.0
    active_file_expectation_temperature: float = 0.25
    active_file_expectation_hard_weight: float = 0.0
    active_file_expectation_detach_inputs: bool = True
    active_file_candidate_radius: float = 10.0
    active_file_candidate_max_age: float = 8.0
    active_file_candidate_wrap: bool = False
    learned_active_file_gate_weight: float = 0.0
    learned_active_file_gate_topk: int = 8
    learned_active_file_gate_threshold: float = 0.5
    learned_active_file_gate_detach_inputs: bool = True
    learned_active_file_gate_context_features: bool = False
    learned_active_file_gate_expectation_features: bool = False


@dataclass
class DatasetConfig:
    name: str = "mnist"
    split: str = "train"
    cache_dir: str = "data/hf"
    limit: int | None = None
    seed: int = 17
    variant: str | None = None


@dataclass
class DefinitionHardeningConfig:
    enabled: bool = True
    min_usage: float = 0.05
    max_usage: float = 0.95
    min_stability: float = 0.9
    min_mode_mutual_information: float = 0.02
    min_prediction_impact: float = 1e-7
    harden_after_windows: int = 3
    soften_after_windows: int = 3
    reject_after_windows: int = 6
    recent_window_limit: int = 8
    min_valid_modes: int = 2
    quarantine_loss_z: float = 3.0
    min_loss_history: int = 5


@dataclass
class TrainConfig:
    run_name: str = "tsm_run"
    model: TsmConfig = field(default_factory=TsmConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    definition_hardening: DefinitionHardeningConfig = field(default_factory=DefinitionHardeningConfig)
    batch_size: int = 64
    max_steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    log_interval: int = 10
    checkpoint_interval: int = 100
    sample_interval: int = 100
    grad_clip: float = 1.0
    runs_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        raw = dict(raw)
        model_raw = raw.pop("model", {})
        if "ternary_sparsity_weight" in model_raw and "ternary_activation_weight" not in model_raw:
            model_raw["ternary_activation_weight"] = model_raw.pop("ternary_sparsity_weight")
        model = TsmConfig(**model_raw)
        dataset = DatasetConfig(**raw.pop("dataset", {}))
        definition_hardening = DefinitionHardeningConfig(**raw.pop("definition_hardening", {}))
        return cls(model=model, dataset=dataset, definition_hardening=definition_hardening, **raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
