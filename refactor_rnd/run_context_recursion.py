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
    lines = [
        "# Context-Recursion Bench",
        "",
        "This R&D bench tests whether higher levels need relation-space CONTEXTS rather than histogram summaries.",
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
        "## Depth Probe",
        "",
        "Depth probe has no pass criterion. It reports whether abstractions can re-enter as next-octave units without collapse.",
    ])
    first = payload["per_seed"][0]["depth_probe"]
    lines.append("")
    lines.append("| level | nmi | accuracy | abstraction_count | abstain_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for level_name, metrics in first.items():
        lines.append(
            f"| {level_name} | {metrics['nmi']:.3f} | {metrics['accuracy']:.3f} | "
            f"{metrics['abstraction_count']} | {metrics['abstain_rate']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
