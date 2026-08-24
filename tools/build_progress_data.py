#!/usr/bin/env python3
"""Build the compact, deterministic ``progress/dashboard-data.json``.

Reads the ignored real-model benchmark outputs one file at a time --
``benchmarks/realdata/runs/*.json`` (run manifests) and
``benchmarks/realdata/records/*.json`` (per-case records) -- selects the latest
complete/ok full run per canonical dataset, validates every record against the
selected run's candidate-commit mapping, and emits a bounded schema-v1 JSON
document with one compact row per scored case plus competition, official, and
synthetic metadata.  The browser computes any judge-scale proxy after
filtering; the file only carries the proxy definition and constants.

Only the Python stdlib is used (no torch/transformers).  The output is written
atomically (temp file + ``os.replace``) and is byte-deterministic for a fixed
``--generated-at``.

Examples::

    python tools/build_progress_data.py
    python tools/build_progress_data.py --output /tmp/dashboard-data.json \\
        --generated-at 2026-08-25T00:00:00+00:00 --print
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# The four canonical real-model datasets currently documented in
# docs/real-model-benchmarks.md.  Override with --datasets for partial builds.
CANONICAL_DATASETS = (
    "b7925ee95f17f32b",  # qwen3-0.6b
    "834a78afb85e5d5a",  # deepseek-r1-qwen-1.5b
    "097430592fe2f4d6",  # smollm2-1.7b
    "4405c7fadc8bef16",  # qwen3.5-2b
)

# Every case row carries exactly these fields (bounded schema; no timing blobs
# and no hashes per row).  The dashboard groups/filters on all of them.
CASE_FIELDS = (
    "source",
    "model",
    "arch",
    "dataset",
    "variant",
    "spec",
    "mode",
    "kind",
    "layer",
    "role",
    "case",
    "seq_len",
    "valid",
    "improvement_percent",
    "baseline_mse",
    "candidate_mse",
    "threads",
)

# Competition configuration from docs/problem-contract.md section 1.7.
COMPETITION = {
    "linear_groups": 50,
    "attention_groups": 50,
    "cases_per_group": 5,
    "total_cases": 500,
    "score_per_case_max": 100,
    "theoretical_max_score": 50000,
}

# Judge-scale proxy: definition only; the dashboard computes the value after
# filtering so arbitrary filter/grouping choices stay consistent.  The raw
# computed value is never silently clamped (clamping is display-only).
JUDGE_SCALE_PROXY = {
    "definition": (
        "Linear rescale of a filtered local mean improvement percent onto the "
        "observed official score range.  The dashboard computes this value "
        "after filtering; this file carries only the definition and constants."
    ),
    "baseline_official_score": 1240,
    "maximum_official_score": 50000,
    "formula": "1240 + (50000 - 1240) * local_mean_improvement_percent / 100",
    "disclaimer": (
        "Proxy only; not an official prediction.  The organizer's standard "
        "baseline and hidden cases are unavailable locally."
    ),
    "display_note": (
        "Clamp to [0, 50000] for display only if desired; do not silently "
        "clamp the raw computed proxy value."
    ),
}

# Official user-reported platform measurements (docs/official-submission-results.md),
# copied exactly: score and runtime seconds per variant.
OFFICIAL_MEASUREMENTS = (
    {
        "variant": "v000",
        "spec": "solution/v000-baseline",
        "name": "ceil(max/7) E6M2 scale; exact local hierarchy selection; no calibration",
        "score": 1240,
        "runtimes_s": [105, 92],
    },
    {
        "variant": "v001",
        "spec": "solution/v001-bf16-target",
        "name": "baseline with BF16-rounded NVFP4 target",
        "score": 1240,
        "runtimes_s": [111],
    },
    {
        "variant": "v002",
        "spec": "solution/v002-e6m2-neighbors",
        "name": "search anchor and adjacent E6M2 ticks for every tensor",
        "score": 5025,
        "runtimes_s": [113],
    },
    {
        "variant": "v003",
        "spec": "solution/v003-role-gated",
        "name": "baseline weight; adjacent-scale search for activations/Q/K/V",
        "score": 3600,
        "runtimes_s": [111],
    },
    {
        "variant": "v004",
        "spec": "solution/v004-calibration-weighted",
        "name": "v003 plus fourth-root calibration-energy weighting for the fixed weight hierarchy",
        "score": 4340,
        "runtimes_s": [117],
    },
    {
        "variant": "v005",
        "spec": "solution/v005-activation-weighted",
        "name": "v004 plus weight-column-energy weighting for dynamic activation hierarchy selection",
        "score": 4600,
        "runtimes_s": [111],
    },
    {
        "variant": "v006",
        "spec": "solution/v006-qk-weighted",
        "name": "v005 plus counterpart-energy weighting for Q/K hierarchy selection",
        "score": 4750,
        "runtimes_s": [123],
    },
    {
        "variant": "v007",
        "spec": "solution/v007-akv-wide-search",
        "name": "v006 plus E6M2 offsets +2,+3 for activations/K/V, retaining adjacent Q search",
        "score": 5276,
        "runtimes_s": [98],
    },
    {
        "variant": "v008",
        "spec": "solution/v008-gated-hadamard",
        "name": "v007 plus calibration-gated block-local H64 rotation for Linear only",
        "score": 5326,
        "runtimes_s": [120],
    },
)

_VARIANT_RE = re.compile(r"^v(\d+)(?=[-_]|$)")


class DashboardError(Exception):
    """Raised when benchmark inputs are absent, malformed, or unusable."""


def variant_short(spec):
    """Return the short variant tag (``v001``) from a spec (``solution/v001-...``)."""
    if not spec:
        return ""
    seg = str(spec).rstrip("/").split("/")[-1]
    match = _VARIANT_RE.match(seg)
    return "v" + match.group(1) if match else ""


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise DashboardError(f"cannot read {path}: {exc}") from None
    except ValueError as exc:
        raise DashboardError(f"{path}: invalid JSON: {exc}") from None


def _iter_json_files(directory):
    if not os.path.isdir(directory):
        raise DashboardError(f"directory not found: {directory}")
    names = sorted(name for name in os.listdir(directory) if name.endswith(".json"))
    if not names:
        raise DashboardError(f"no *.json files under {directory}")
    for name in names:
        yield name, os.path.join(directory, name)


def _run_sort_key(run):
    """Deterministic recency key: updated_at, then created_at, then file name."""
    updated = str(run.get("updated_at") or "")
    created = str(run.get("created_at") or "")
    return (updated, created, str(run.get("_file") or ""))


def load_runs(runs_dir, datasets):
    """Load complete/ok run manifests, keeping the latest one per dataset.

    Returns ``(selected_runs, counters)`` where ``selected_runs`` maps dataset
    id to a normalized run dict.  A run is usable only when ``status == "ok"``,
    no case failed, and no case was invalid (a complete full run), and it must
    list at least one candidate.
    """
    if not os.path.isdir(runs_dir):
        raise DashboardError(f"runs directory not found: {runs_dir}")

    candidates_by_dataset = {}
    skipped = {"status_not_ok": 0, "incomplete": 0, "no_candidates": 0, "other_dataset": 0}
    for name, path in _iter_json_files(runs_dir):
        run = _load_json(path)
        if not isinstance(run, dict):
            skipped["other_dataset"] += 1
            continue
        run["_file"] = name
        dataset = (run.get("dataset") or {}).get("id")
        if dataset not in datasets:
            skipped["other_dataset"] += 1
            continue
        if run.get("status") != "ok":
            skipped["status_not_ok"] += 1
            continue
        counts = run.get("counts") or {}
        if counts.get("cases_failed") or counts.get("cases_invalid"):
            skipped["incomplete"] += 1
            continue
        candidates = run.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            skipped["no_candidates"] += 1
            continue
        candidates_by_dataset.setdefault(dataset, []).append(run)

    selected = {}
    missing = []
    for dataset in datasets:
        pool = candidates_by_dataset.get(dataset)
        if not pool:
            missing.append(dataset)
            continue
        pool.sort(key=_run_sort_key)
        selected[dataset] = pool[-1]

    if not selected:
        raise DashboardError(
            f"no complete/ok run manifest found for any requested dataset "
            f"under {runs_dir} (requested: {sorted(datasets)})"
        )
    if missing:
        raise DashboardError(
            "no complete/ok run manifest for dataset(s): "
            f"{', '.join(sorted(missing))} under {runs_dir}"
        )
    return selected, skipped


def _normalized_run(dataset, run):
    """Keep only the small bounded slice of a run manifest needed downstream."""
    counts = run.get("counts") or {}
    return {
        "file": run.get("_file"),
        "dataset": dataset,
        "status": run.get("status"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "groups_total": counts.get("groups_total"),
        "case_slots_total": counts.get("cases_slots_total"),
        "threads": run.get("threads"),
        "baseline_commit": (run.get("baseline") or {}).get("commit"),
        "specs_by_commit": {
            (cand.get("commit") if isinstance(cand, dict) else None): (
                cand.get("spec") if isinstance(cand, dict) else None
            )
            for cand in (run.get("candidates") or [])
        },
        "model": run.get("model") or {},
    }


def _case_row(d, spec):
    """Map one record dict onto the bounded case-row schema (no timing/hashes)."""
    geometry = d.get("geometry") or {}
    model = d.get("model") or {}
    kind = d.get("kind") or "linear"
    role = d.get("role")
    if not role:
        group = d.get("group") or ""
        role = group.split(".", 1)[1] if "." in group else ("self_attn" if kind == "attention" else "")
    return {
        "source": "realdata",
        "model": model.get("alias") or "",
        "arch": model.get("arch") or "",
        "dataset": d.get("dataset_id"),
        "variant": variant_short(spec),
        "spec": spec,
        "mode": d.get("mode"),
        "kind": kind,
        "layer": d.get("layer"),
        "role": role,
        "case": d.get("case"),
        "seq_len": geometry.get("seq_len"),
        "valid": bool(d.get("valid")),
        "improvement_percent": d.get("improvement_percent"),
        "baseline_mse": d.get("baseline_mse"),
        "candidate_mse": d.get("candidate_mse"),
        "threads": d.get("threads"),
    }


def select_case_rows(records_dir, runs):
    """Read records one file at a time; keep those consistent with a run.

    A record is kept only when its dataset has a selected run, its candidate
    commit maps to the record's own spec under that run, and its baseline
    commit matches the run's baseline commit.  Returns ``(rows, counters)``.
    """
    if not os.path.isdir(records_dir):
        raise DashboardError(f"records directory not found: {records_dir}")

    rows = []
    counters = {
        "read": 0,
        "selected": 0,
        "skipped_parse": 0,
        "skipped_dataset": 0,
        "skipped_candidate_commit": 0,
        "skipped_baseline_commit": 0,
    }
    for name, path in _iter_json_files(records_dir):
        try:
            d = _load_json(path)
        except DashboardError:
            counters["skipped_parse"] += 1
            continue
        counters["read"] += 1
        if not isinstance(d, dict):
            counters["skipped_parse"] += 1
            continue

        dataset = d.get("dataset_id")
        run = runs.get(dataset)
        if run is None:
            counters["skipped_dataset"] += 1
            continue

        candidate = d.get("candidate") or {}
        spec = candidate.get("spec")
        commit = candidate.get("commit")
        if spec is None or run["specs_by_commit"].get(commit) != spec:
            counters["skipped_candidate_commit"] += 1
            continue
        baseline = d.get("baseline") or {}
        if baseline.get("commit") != run["baseline_commit"]:
            counters["skipped_baseline_commit"] += 1
            continue

        rows.append(_case_row(d, spec))
        counters["selected"] += 1

    if not rows:
        raise DashboardError(
            f"no usable case records selected from {records_dir} "
            f"(read={counters['read']}, skipped={counters['read'] - counters['selected']})"
        )
    return rows, counters


def _case_sort_key(row):
    match = _VARIANT_RE.match(str(row["variant"]))
    return (
        str(row["model"]),
        int(match.group(1)) if match else -1,
        str(row["mode"]),
        str(row["kind"]),
        int(row["layer"]) if row["layer"] is not None else -1,
        str(row["role"]),
        int(row["case"]) if row["case"] is not None else -1,
    )


def sort_case_rows(rows):
    """Deterministic order: model, variant numeric, mode, kind, layer, role, case."""
    return sorted(rows, key=_case_sort_key)


def _synthetic_sort_key(row):
    match = _VARIANT_RE.match(str(row["variant"]))
    return (int(match.group(1)) if match else -1, str(row["spec"]))


def load_synthetic(results_path):
    """Parse ``progress/results.jsonl`` into labeled synthetic summary rows."""
    if not os.path.isfile(results_path):
        raise DashboardError(f"synthetic results file not found: {results_path}")

    rows = []
    with open(results_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError as exc:
                raise DashboardError(
                    f"{results_path}:{lineno}: invalid JSONL row: {exc}"
                ) from None
            if not isinstance(d, dict):
                raise DashboardError(f"{results_path}:{lineno}: row is not a JSON object")
            rows.append(
                {
                    "source": "synthetic",
                    "variant": variant_short(d.get("variant")),
                    "spec": d.get("variant"),
                    "mean_improvement_percent": d.get("mean_improvement_percent"),
                    "case_count": d.get("case_count"),
                    "runtime_s": d.get("runtime"),
                    "status": d.get("status"),
                    "suite": d.get("suite"),
                    "timestamp": d.get("timestamp") or d.get("ts"),
                }
            )
    if not rows:
        raise DashboardError(f"no synthetic rows in {results_path}")
    rows.sort(key=_synthetic_sort_key)
    return rows


def _models_metadata(runs):
    models = []
    seen = set()
    for dataset in sorted(runs):
        model = runs[dataset]["model"]
        alias = model.get("alias") or ""
        if not alias or alias in seen:
            continue
        seen.add(alias)
        models.append(
            {
                "alias": alias,
                "arch": model.get("arch"),
                "repo_id": model.get("repo_id"),
                "num_layers": model.get("num_layers"),
                "hidden_size": model.get("hidden_size"),
                "num_attention_heads": model.get("num_attention_heads"),
                "num_key_value_heads": model.get("num_key_value_heads"),
                "head_dim": model.get("head_dim"),
            }
        )
    return sorted(models, key=lambda m: str(m["alias"]))


def _num_range(values):
    finite = [v for v in values if isinstance(v, (int, float)) and v == v]  # noqa: PLR0124 (NaN check)
    return [min(finite), max(finite)] if finite else []


def _dimensions(rows, runs):
    roles = {}
    for kind in sorted({r["kind"] for r in rows}):
        roles[kind] = sorted({r["role"] for r in rows if r["kind"] == kind})
    variants = sorted({r["variant"] for r in rows}, key=lambda v: _case_sort_key({"variant": v, "model": "", "mode": "", "kind": "", "layer": None, "role": "", "case": None}))
    return {
        "modes": sorted({r["mode"] for r in rows}),
        "kinds": sorted({r["kind"] for r in rows}),
        "variants": variants,
        "roles": roles,
        "case_indices": sorted({r["case"] for r in rows}),
        "seq_len_values": sorted({r["seq_len"] for r in rows}),
        "layer_range": _num_range([r["layer"] for r in rows]),
        "improvement_percent_range": _num_range([r["improvement_percent"] for r in rows]),
        "baseline_mse_range": _num_range([r["baseline_mse"] for r in rows]),
        "candidate_mse_range": _num_range([r["candidate_mse"] for r in rows]),
        "thread_values": sorted({r["threads"] for r in rows}),
        "datasets": sorted({r["dataset"] for r in rows}),
    }


def build_dashboard_data(
    runs_dir,
    records_dir,
    results_path,
    datasets=CANONICAL_DATASETS,
    generated_at=None,
    source_root=None,
):
    """Load benchmark inputs one file at a time and return the schema-v1 dict."""
    runs_raw, run_skipped = load_runs(runs_dir, datasets)
    runs = {dataset: _normalized_run(dataset, run) for dataset, run in runs_raw.items()}
    rows, record_counters = select_case_rows(records_dir, runs)
    rows = sort_case_rows(rows)
    synthetic = load_synthetic(results_path)

    generated_at = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    def source_label(path):
        path = Path(path)
        if source_root is not None:
            try:
                return path.resolve().relative_to(Path(source_root).resolve()).as_posix()
            except ValueError:
                pass
        return str(path)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "competition": COMPETITION,
        "judge_scale_proxy": JUDGE_SCALE_PROXY,
        "official": list(OFFICIAL_MEASUREMENTS),
        "sources": {
            "official_source": "docs/official-submission-results.md",
            "runs": [
                {
                    "file": runs[dataset]["file"],
                    "dataset": dataset,
                    "status": runs[dataset]["status"],
                    "created_at": runs[dataset]["created_at"],
                    "updated_at": runs[dataset]["updated_at"],
                    "groups_total": runs[dataset]["groups_total"],
                    "case_slots_total": runs[dataset]["case_slots_total"],
                    "threads": runs[dataset]["threads"],
                }
                for dataset in sorted(runs)
            ],
            "run_files_skipped": run_skipped,
            "records": {
                "dir": source_label(records_dir),
                **record_counters,
            },
            "synthetic": {
                "file": source_label(results_path),
                "rows": len(synthetic),
            },
            "note": (
                "Per-case rows are filtered to the latest complete/ok full run "
                "per dataset and validated against that run's candidate-commit "
                "mapping; rows carry no timing blobs or hashes."
            ),
        },
        "models": _models_metadata(runs),
        "dimensions": _dimensions(rows, runs),
        "synthetic": synthetic,
        "cases": rows,
    }


def write_atomic(path, text):
    """Atomically replace ``path`` with ``text`` (same-directory temp + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".dashboard-data-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def serialize(document):
    """Compact, deterministic JSON (sorted keys, no insignificant whitespace)."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root (default: parent of this script)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None, help="run manifest directory")
    parser.add_argument("--records-dir", type=Path, default=None, help="per-case record directory")
    parser.add_argument("--results", type=Path, default=None, help="progress/results.jsonl path")
    parser.add_argument("--output", type=Path, default=None, help="output JSON path")
    parser.add_argument(
        "--datasets",
        default=",".join(CANONICAL_DATASETS),
        help="comma-separated dataset ids to include (default: the four canonical)",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="ISO-8601 timestamp for metadata; fixed values keep the output "
        "byte-deterministic (default: now, UTC)",
    )
    parser.add_argument("--print", action="store_true", help="also write the document to stdout")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.repo_root or Path(__file__).resolve().parents[1]
    runs_dir = args.runs_dir or root / "benchmarks" / "realdata" / "runs"
    records_dir = args.records_dir or root / "benchmarks" / "realdata" / "records"
    results_path = args.results or root / "progress" / "results.jsonl"
    output = args.output or root / "progress" / "dashboard-data.json"
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    if not datasets:
        parser_error = "no dataset ids given to --datasets"
        print(f"build_progress_data: error: {parser_error}", file=sys.stderr)
        return 2

    try:
        document = build_dashboard_data(
            runs_dir,
            records_dir,
            results_path,
            datasets=datasets,
            generated_at=args.generated_at,
            source_root=root,
        )
        text = serialize(document)
        write_atomic(output, text)
    except DashboardError as exc:
        print(f"build_progress_data: error: {exc}", file=sys.stderr)
        return 2

    records = document["sources"]["records"]
    print(
        f"wrote {output} ({len(text)} bytes): schema v{document['schema_version']}, "
        f"{records['selected']} case rows (read {records['read']}, skipped "
        f"{records['read'] - records['selected']}), {document['sources']['synthetic']['rows']} "
        f"synthetic rows, {len(document['models'])} models"
    )
    if args.print:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
