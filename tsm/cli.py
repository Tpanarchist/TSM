from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainConfig
from .data import canonical_dataset_name, load_public_dataset
from .trainer import axis_report, evaluate, run_seed_sweep, sample, smoke, train


def _parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_compare_configs(values: list[str] | None) -> list[tuple[str, TrainConfig]]:
    comparisons: list[tuple[str, TrainConfig]] = []
    for value in values or []:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        comparisons.append((name.strip(), TrainConfig.from_yaml(path.strip())))
    return comparisons


def _cmd_data_pull(args: argparse.Namespace) -> None:
    name = canonical_dataset_name(args.dataset)
    ds = load_public_dataset(name, args.split, args.cache)
    print(json.dumps({"dataset": name, "split": args.split, "rows": len(ds), "cache": args.cache}, indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig.from_yaml(args.config)
    run_dir = train(cfg, device_name=args.device, resume=args.resume)
    print(run_dir)


def _cmd_eval(args: argparse.Namespace) -> None:
    metrics = evaluate(args.checkpoint, device_name=args.device, split=args.split, limit=args.limit)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _cmd_sample(args: argparse.Namespace) -> None:
    path = sample(args.checkpoint, args.out, device_name=args.device, split=args.split)
    print(path)


def _cmd_axis_report(args: argparse.Namespace) -> None:
    path = axis_report(
        args.checkpoint,
        out_path=args.out,
        device_name=args.device,
        split=args.split,
        limit=args.limit,
        min_usage=args.min_usage,
        label_key=args.label_key,
    )
    print(path)


def _cmd_smoke(args: argparse.Namespace) -> None:
    run_dir = smoke(device_name=args.device, steps=args.steps)
    print(run_dir)


def _cmd_seed_sweep(args: argparse.Namespace) -> None:
    cfg = TrainConfig.from_yaml(args.config)
    disabled_cfg = TrainConfig.from_yaml(args.disabled_config) if args.disabled_config else None
    comparison_cfgs = _parse_compare_configs(args.compare_config)
    sweep_dir = run_seed_sweep(
        cfg,
        device_name=args.device,
        seeds=_parse_seeds(args.seeds),
        steps=args.steps,
        disabled_cfg=disabled_cfg,
        comparison_cfgs=comparison_cfgs,
        eval_split=args.eval_split,
        eval_limit=args.eval_limit,
        out_dir=args.out,
    )
    print(sweep_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsm")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    pull = data_sub.add_parser("pull")
    pull.add_argument("--dataset", default="mnist")
    pull.add_argument("--split", default="train")
    pull.add_argument("--cache", default="data/hf")
    pull.set_defaults(func=_cmd_data_pull)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument("--device", default="cuda")
    train_cmd.add_argument("--resume")
    train_cmd.set_defaults(func=_cmd_train)

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--device", default="cuda")
    eval_cmd.add_argument("--split", default="test")
    eval_cmd.add_argument("--limit", type=int)
    eval_cmd.set_defaults(func=_cmd_eval)

    sample_cmd = sub.add_parser("sample")
    sample_cmd.add_argument("--checkpoint", required=True)
    sample_cmd.add_argument("--out", default="runs/samples")
    sample_cmd.add_argument("--device", default="cuda")
    sample_cmd.add_argument("--split", default="test")
    sample_cmd.set_defaults(func=_cmd_sample)

    axis_cmd = sub.add_parser("axis-report")
    axis_cmd.add_argument("--checkpoint", required=True)
    axis_cmd.add_argument("--out")
    axis_cmd.add_argument("--device", default="cuda")
    axis_cmd.add_argument("--split", default="test")
    axis_cmd.add_argument("--limit", type=int)
    axis_cmd.add_argument("--min-usage", type=float, default=1e-6)
    axis_cmd.add_argument("--label-key", default="mode")
    axis_cmd.set_defaults(func=_cmd_axis_report)

    sweep_cmd = sub.add_parser("seed-sweep")
    sweep_cmd.add_argument("--config", required=True)
    sweep_cmd.add_argument("--disabled-config")
    sweep_cmd.add_argument("--compare-config", action="append", help="Additional NAME=path config to include")
    sweep_cmd.add_argument("--seeds", default="31,37,43")
    sweep_cmd.add_argument("--steps", type=int)
    sweep_cmd.add_argument("--device", default="cuda")
    sweep_cmd.add_argument("--eval-split", default="test")
    sweep_cmd.add_argument("--eval-limit", type=int)
    sweep_cmd.add_argument("--out")
    sweep_cmd.set_defaults(func=_cmd_seed_sweep)

    smoke_cmd = sub.add_parser("smoke")
    smoke_cmd.add_argument("--device", default="cuda")
    smoke_cmd.add_argument("--steps", type=int, default=20)
    smoke_cmd.set_defaults(func=_cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
