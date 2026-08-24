#!/usr/bin/env python3
"""CLI for streaming evaluation of raw-BF16 real-capture shards.

Examples::

    # evaluate every mode and the default candidate against the baseline tag
    python tools/evaluate_real.py --dataset 2eff88265a618c5f

    # one mode, two candidates, restricted groups, bounded by --limit
    python tools/evaluate_real.py --dataset data/real-captures/2eff88265a618c5f \
      --modes stochastic --candidate solution.py \
      --candidate solution/v004-calibration-weighted \
      --group-filter linear --group-filter role:q_proj --limit 2

    # recompute everything, keep records under a custom directory
    python tools/evaluate_real.py --dataset 2eff88265a618c5f --force \
      --output benchmarks/realdata

``--dataset`` accepts either the dataset directory itself or its 16-hex id
resolved under ``--captures-root`` (default ``data/real-captures``).  One
group shard is loaded, validated, evaluated and released at a time; per-case
JSON records, a per-(group, mode) baseline cache and a run manifest are all
written atomically under the output directory (see
``tools/realdata/evaluate.py`` for the resume semantics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python tools/evaluate_real.py` too
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.realdata import evaluate
from tools.realdata.capture import set_oom_score
from tools.realdata.evaluate import (
    DEFAULT_CAPTURES_ROOT,
    DEFAULT_OUTPUT_DIR,
    parse_filters,
    resolve_dataset,
)
from tools.realdata import shards


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("empty comma-separated list")
    try:
        return [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid integer list {value!r}; use e.g. --layers 0,14,27"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset directory (containing manifest.json) or a dataset id "
        "resolved under --captures-root",
    )
    parser.add_argument(
        "--captures-root",
        type=Path,
        default=DEFAULT_CAPTURES_ROOT,
        help="root under which dataset ids are resolved (default: data/real-captures)",
    )
    parser.add_argument(
        "--baseline",
        default="solution/v000-baseline",
        help="baseline solution: a .py path, a directory containing "
        "solution.py, or a Git ref (git show <ref>:solution.py)",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="SPEC",
        help="candidate solution (same spec forms as --baseline); repeatable",
    )
    parser.add_argument(
        "--modes",
        action="append",
        default=[],
        metavar="MODE[,MODE...]",
        help=f"NVFP4 source modes to evaluate: {', '.join(shards.SOURCE_MODES)} "
        "(comma lists allowed; repeatable; default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="stochastic draw seed")
    parser.add_argument(
        "--threads", type=int, default=1, help="torch CPU threads (default 1)"
    )
    parser.add_argument(
        "--group-filter",
        action="append",
        default=[],
        metavar="PRED",
        help="group predicate: linear|attention|kind:K|layer:N|role:R|id:G; "
        "comma lists inside one value are OR'd, values are AND'd; repeatable",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of groups to evaluate (after filtering)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory for records / baseline cache / run manifests "
        "(default: benchmarks/realdata)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute existing case records and baseline cache instead of resuming",
    )
    return parser


def _normalize_modes(values: list[str]) -> list[str]:
    if not values:
        return list(shards.SOURCE_MODES)
    modes: list[str] = []
    for value in values:
        for raw in value.split(","):
            mode = raw.strip()
            if not mode:
                continue
            if mode not in shards.SOURCE_MODES:
                raise ValueError(
                    f"unknown source mode {mode!r}; choose from {shards.SOURCE_MODES}"
                )
            if mode not in modes:
                modes.append(mode)
    if not modes:
        raise ValueError("no source modes selected")
    return modes


def main(argv: list[str] | None = None) -> int:
    set_oom_score(500)  # best effort; evaluator dies before the runner on OOM
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    torch.set_num_threads(args.threads)
    try:
        modes = _normalize_modes(args.modes)
        filters = parse_filters(args.group_filter)
    except ValueError as error:
        parser.error(str(error))
    candidates = args.candidate or ["solution.py"]
    filters_raw = args.group_filter

    dataset_dir, manifest = resolve_dataset(args.dataset, args.captures_root)
    run_manifest = evaluate.evaluate_dataset(
        {
            "dataset_dir": dataset_dir,
            "manifest": manifest,
            "dataset_spec": args.dataset,
            "modes": modes,
            "seed": args.seed,
            "baseline_spec": args.baseline,
            "candidate_specs": candidates,
            "output_dir": args.output,
            "filters": filters,
            "filters_raw": filters_raw,
            "limit": args.limit,
            "force": args.force,
            "threads": args.threads,
        }
    )
    print(evaluate.summarize(run_manifest))
    status = run_manifest["status"]
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
