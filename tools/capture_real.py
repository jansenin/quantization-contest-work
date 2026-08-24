"""CLI for the real-model raw-BF16 capture pipeline.

Examples::

    python tools/capture_real.py --smoke                      # tiny lengths
    python tools/capture_real.py --model qwen3-0.6b --threads 4
    python tools/capture_real.py --layers 0,14,27 --force     # restart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python tools/capture_real.py` too
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.realdata import capture
from tools.realdata.capture import (
    DEFAULT_DOWNLOAD_STATE,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_OUTPUT_ROOT,
    RealLoader,
    run_capture,
    set_oom_score,
)


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


def _parse_name_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("empty comma-separated list")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ALIAS,
        help=f"model alias from tools/model_downloads.json (default: {DEFAULT_MODEL_ALIAS})",
    )
    parser.add_argument(
        "--download-state",
        type=Path,
        default=DEFAULT_DOWNLOAD_STATE,
        help="download-state JSON written by tools/download_models.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root under which data/real-captures/<dataset_id>/ is created",
    )
    parser.add_argument(
        "--layers",
        type=_parse_int_list,
        default=None,
        metavar="L0,L1,...",
        help="comma list of layer indices (default: 0, midpoint, last)",
    )
    parser.add_argument(
        "--linear-roles",
        type=_parse_name_list,
        default=None,
        metavar="ROLE,ROLE,...",
        help=(
            "comma list of linear roles to capture "
            "(default: q_proj,o_proj,gate_proj,up_proj,down_proj)"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use short sequence lengths [8,16,24,32,40] instead of [10,128,512,1024,1024]",
    )
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--seed", type=int, default=0, help="capture seed (part of dataset id)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard any existing capture for this dataset id and restart",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    set_oom_score(500)  # best effort; capture dies before the runner on OOM
    config = {
        "alias": args.model,
        "download_state_path": args.download_state,
        "output_root": args.output_root,
        "layers": args.layers,
        "linear_roles": args.linear_roles,
        "smoke": args.smoke,
        "threads": args.threads,
        "seed": args.seed,
        "force": args.force,
    }
    loader = RealLoader(
        download_state_path=args.download_state, threads=args.threads
    )
    summary = run_capture(config, loader=loader)
    print(f"dataset_id={summary['dataset_id']}")
    print(f"output_dir={summary['output_dir']}")
    print(f"status={summary['status']}")
    print(
        f"groups: linear={summary['groups']['linear']} "
        f"attention={summary['groups']['attention']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
