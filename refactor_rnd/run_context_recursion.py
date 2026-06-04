from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .context_recursion import aggregate_results, run_all, to_jsonable


def _parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m refactor_rnd.run_context_recursion")
    parser.add_argument("--seeds", default="31,37,43,47,53")
    parser.add_argument("--out", default="refactor_rnd/runs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    results = [run_all(seed) for seed in seeds]
    aggregate = aggregate_results(results)
    payload = {
        "summary": "TRIT -> CONTEXT -> ABSTRACTION context-recursion bench",
        "seeds": seeds,
        "aggregate": aggregate,
        "per_seed": results,
    }
    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S_context_recursion")
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.md"
    metrics_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(_summary_markdown(payload), encoding="utf-8")
    print(run_dir)


def _summary_markdown(payload: dict) -> str:
    aggregate = payload["aggregate"]
    acceptance = aggregate["acceptance"]
    diagnostics = payload["per_seed"][0]["collapse_diagnostics"]
    probe_d = payload["per_seed"][0]["probe_d_stress"]
    probe_d_diagnosis = probe_d["diagnosis"]
    probe_e = payload["per_seed"][0]["probe_e_decomposition"]
    probe_e_diagnosis = probe_e["diagnosis"]
    probe_f = payload["per_seed"][0]["probe_f_order_loss_split"]
    probe_f_diagnosis = probe_f["diagnosis"]
    lines = [
        "# Context-Recursion Bench",
        "",
        "This R&D bench tests whether higher levels need order/lag-sensitive CONTEXTS rather than histogram summaries, and whether stable Knowledge can be cut into reusable ABSTRACTIONS.",
        "",
        "## Acceptance",
        "",
        f"- same_marginal_pass_fraction: {acceptance['same_marginal_pass_fraction']:.3f}",
        f"- shared_anchor_pass_fraction: {acceptance['shared_anchor_pass_fraction']:.3f}",
        f"- novelty_pass_fraction: {acceptance['novelty_pass_fraction']:.3f}",
        "",
        "## Aggregate Metrics",
        "",
        "| metric | mean | min | max |",
        "|---|---:|---:|---:|",
    ]
    for key, value in aggregate.items():
        if key in {"seed_count", "acceptance"}:
            continue
        lines.append(f"| {key} | {value['mean']:.3f} | {value['min']:.3f} | {value['max']:.3f} |")

    lines.extend([
        "",
        "## Probe C1/C2 - Level Information Audit",
        "",
        "Compare encoder choice against both abstraction NMI and a leave-one-out supervised probe on the raw level vectors.",
        "",
        "| encoder | level | abstraction_nmi | abstraction_ari | supervised_probe_accuracy | supervised_probe_nmi | abstraction_count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in diagnostics["encoder_level_table"]:
        lines.append(
            f"| {row['encoder']} | {row['level']} | {row['abstraction_nmi']:.3f} | {row['abstraction_ari']:.3f} | "
            f"{row['supervised_probe_accuracy']:.3f} | {row['supervised_probe_nmi']:.3f} | {row['abstraction_count']} |"
        )

    lines.extend([
        "",
        "## Probe C3 - Classify Before Recursion",
        "",
        "Compare direct raw-context recursion against the full Context -> Classify -> Abstraction handoff before the next level.",
        "",
        "| mode | level | abstraction_nmi | abstraction_ari | supervised_probe_accuracy | supervised_probe_nmi | abstraction_count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in diagnostics["classify_before_recursion"]["table"]:
        lines.append(
            f"| {row['mode']} | {row['level']} | {row['abstraction_nmi']:.3f} | {row['abstraction_ari']:.3f} | "
            f"{row['supervised_probe_accuracy']:.3f} | {row['supervised_probe_nmi']:.3f} | {row['abstraction_count']} |"
        )

    diagnosis = diagnostics["diagnosis"]
    lines.extend([
        "",
        "## Diagnosis",
        "",
        f"- primary_cause: {diagnosis['primary_cause']}",
        f"- summary: {diagnosis['summary']}",
        f"- recursion_takeaway: {diagnosis['recursion_takeaway']}",
        "",
        "## Probe D - Context Degradation Stress",
        "",
        "Stress the ordered Context definition by degrading lag, order, anchors, repeated symbols, missing events, and frame boundaries.",
        "",
        f"- summary: {probe_d_diagnosis['summary']}",
        f"- transition_lag_order_beats_transition_case_count: {probe_d_diagnosis['transition_lag_order_beats_transition_case_count']}",
        f"- degraded_transition_lag_order_weaker_case_count: {probe_d_diagnosis['degraded_transition_lag_order_weaker_case_count']}",
        "- degraded_transition_lag_order_weaker_cases: "
        + (", ".join(probe_d_diagnosis["degraded_transition_lag_order_weaker_cases"]) or "none"),
        f"- most_damaging_case: {probe_d_diagnosis['most_damaging_case']}",
        f"- least_damaging_case: {probe_d_diagnosis['least_damaging_case']}",
        "",
        "| stress_case | level3_damage_score | level3_nmi_damage | level3_probe_damage |",
        "|---|---:|---:|---:|",
    ])
    for row in probe_d_diagnosis["level3_damage_ranking"]:
        lines.append(
            f"| {row['stress_case']} | {row['level3_damage_score']:.3f} | {row['level3_nmi_damage']:.3f} | {row['level3_probe_damage']:.3f} |"
        )

    lines.extend([
        "",
        "| stress_case | degradation | encoder | level1_nmi | level1_probe | level1_count | level2_nmi | level2_probe | level2_count | level3_nmi | level3_probe | level3_count | abstain_spike_rate | collapse_level | degradation_threshold |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in probe_d["table"]:
        threshold = "survives_max" if row["degradation_threshold"] is None else f"{row['degradation_threshold']:.2f}"
        lines.append(
            f"| {row['stress_case']} | {row['degradation']:.2f} | {row['encoder']} | "
            f"{row['level1_abstraction_nmi']:.3f} | {row['level1_supervised_probe_accuracy']:.3f} | {row['level1_abstraction_count']} | "
            f"{row['level2_abstraction_nmi']:.3f} | {row['level2_supervised_probe_accuracy']:.3f} | {row['level2_abstraction_count']} | "
            f"{row['level3_abstraction_nmi']:.3f} | {row['level3_supervised_probe_accuracy']:.3f} | {row['level3_abstraction_count']} | "
            f"{row['abstain_spike_rate']:.3f} | {row['collapse_level']} | {threshold} |"
        )

    lines.extend([
        "",
        "## Probe E - Damage Decomposition",
        "",
        "Split level-3 damage into semantic confusion and order loss for the degraded order-aware encoder.",
        "",
        f"- summary: {probe_e_diagnosis['summary']}",
        f"- most_total_damage_case: {probe_e_diagnosis['most_total_damage_case']}",
        f"- least_total_damage_case: {probe_e_diagnosis['least_total_damage_case']}",
        f"- most_order_loss_case: {probe_e_diagnosis['most_order_loss_case']}",
        f"- most_semantic_confusion_case: {probe_e_diagnosis['most_semantic_confusion_case']}",
        "",
        "| stress_case | semantic_confusion_score | order_loss_score | total_damage_score | supervised_probe_accuracy | abstraction_nmi | order_recovery_accuracy | failure_mode |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in probe_e["table"]:
        lines.append(
            f"| {row['stress_case']} | {row['semantic_confusion_score']:.3f} | {row['order_loss_score']:.3f} | "
            f"{row['total_damage_score']:.3f} | {row['supervised_probe_accuracy']:.3f} | {row['abstraction_nmi']:.3f} | "
            f"{row['order_recovery_accuracy']:.3f} | {row['failure_mode']} |"
        )

    lines.extend([
        "",
        "## Probe F - Ordered Implication Split",
        "",
        "Split Probe E ordered implication damage into phase drift versus residual rank instability.",
        "Phase drift means the same ranked implication survives but is shifted in time; rank instability means candidate continuations no longer keep lawful ordering.",
        "Alignment errors below are measured on the stressed full ordered context and used to apportion degraded ordered implication damage.",
        "",
        f"- summary: {probe_f_diagnosis['summary']}",
        f"- most_total_order_loss_case: {probe_f_diagnosis['most_total_order_loss_case']}",
        f"- most_ordered_implication_damage_case: {probe_f_diagnosis['most_ordered_implication_damage_case']}",
        f"- least_total_order_loss_case: {probe_f_diagnosis['least_total_order_loss_case']}",
        f"- synthetic_phase_control_case: {probe_f_diagnosis['synthetic_phase_control_case']}",
        f"- phase_control_validated: {probe_f_diagnosis['phase_control_validated']}",
        f"- overall_most_phase_error_case: {probe_f_diagnosis['overall_most_phase_error_case']}",
        f"- overall_most_rank_instability_case: {probe_f_diagnosis['overall_most_rank_instability_case']}",
        f"- natural_stressor_most_phase_like_case: {probe_f_diagnosis['most_natural_phase_error_case']}",
        f"- natural_stressor_most_rank_instability_case: {probe_f_diagnosis['most_natural_rank_instability_case']}",
        f"- ordered_implication_interpretation: {probe_f_diagnosis['ordered_implication_interpretation']}",
        f"- phase_pov_interpretation: {probe_f_diagnosis['phase_pov_interpretation']}",
        f"- rank_instability_pov_interpretation: {probe_f_diagnosis['rank_instability_pov_interpretation']}",
        f"- pure_phase_offset_interpretation: {probe_f_diagnosis['pure_phase_offset_interpretation']}",
        f"- variable_lag_interpretation: {probe_f_diagnosis['variable_lag_interpretation']}",
        "",
        "| stress_case | phase_error_score | rank_instability_score | ordered_implication_damage_score | strict_order_alignment_error | phase_tolerant_order_alignment_error | order_failure_mode |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in probe_f["table"]:
        lines.append(
            f"| {row['stress_case']} | {row['phase_error_score']:.3f} | {row['rank_instability_score']:.3f} | "
            f"{row['ordered_implication_damage_score']:.3f} | {row['strict_order_alignment_error']:.3f} | "
            f"{row['phase_tolerant_order_alignment_error']:.3f} | {row['order_failure_mode']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
