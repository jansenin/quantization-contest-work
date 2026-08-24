"""Unit tests for ``tools/build_progress_data.py``.

Uses only tiny temporary JSON fixtures (no torch, no model weights).  Verifies:

* the schema-v1 document shape, competition constants, and the exact official
  measurements (scores and runtimes);
* the judge-scale proxy definition (formula, constants, disclaimer);
* per-case row fields (no timing blobs or hashes) and role normalization
  (attention -> ``self_attn``);
* deterministic output bytes for a fixed ``--generated-at``;
* latest-complete/ok-run selection per dataset and record validation against
  the selected run's candidate-commit mapping;
* atomic output (valid JSON, no leftover temp files) and clear errors when
  inputs are absent;
* the CLI end-to-end, including a failing invocation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import build_progress_data
from tools.build_progress_data import (
    CANONICAL_DATASETS,
    CASE_FIELDS,
    COMPETITION,
    DashboardError,
    build_dashboard_data,
    serialize,
    variant_short,
    write_atomic,
)

DS_A, DS_B = "b7925ee95f17f32b", "834a78afb85e5d5a"
FIXED_TS = "2026-08-25T00:00:00+00:00"

V001 = "solution/v001-bf16-target"
V002 = "solution/v002-e6m2-neighbors"
C_B0 = "b0"


def make_run(
    dataset,
    *,
    status="ok",
    cases_failed=0,
    cases_invalid=0,
    created="2026-08-24T00:00:00+00:00",
    updated="2026-08-24T01:00:00+00:00",
    candidates=None,
    baseline_commit=C_B0,
):
    candidates = candidates or [
        {"commit": "c1", "kind": "git", "source_sha256": None, "spec": V001},
        {"commit": "c2", "kind": "git", "source_sha256": None, "spec": V002},
    ]
    return {
        "baseline": {"commit": baseline_commit, "kind": "git", "spec": "solution/v000-baseline"},
        "candidates": candidates,
        "counts": {
            "cases_failed": cases_failed,
            "cases_invalid": cases_invalid,
            "cases_new": 2,
            "cases_skipped": 0,
            "cases_slots_total": 2,
            "groups_failed": 0,
            "groups_selected": 1,
            "groups_total": 1,
        },
        "created_at": created,
        "updated_at": updated,
        "dataset": {"dir": f"data/real-captures/{dataset}", "id": dataset, "spec": dataset},
        "model": {
            "alias": "tiny-model",
            "arch": "qwen3",
            "head_dim": 64,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_layers": 2,
            "repo_id": "test/tiny",
        },
        "modes": ["ceil", "nearest", "stochastic"],
        "status": status,
        "threads": 1,
        "schema_version": 1,
    }


def make_record(
    dataset,
    spec,
    commit,
    *,
    case=0,
    mode="ceil",
    kind="linear",
    layer=0,
    role=None,
    seq_len=10,
    valid=True,
    improvement=1.5,
    baseline_mse=0.01,
    candidate_mse=0.00985,
    threads=1,
    baseline_commit=C_B0,
):
    if role is None:
        role = "q_proj" if kind == "linear" else None
    group = f"{layer}.{role}" if kind == "linear" else f"{layer}.self_attn"
    return {
        "baseline": {"commit": baseline_commit, "kind": "git", "mse": baseline_mse, "spec": "solution/v000-baseline"},
        "baseline_mse": baseline_mse,
        "candidate": {"commit": commit, "kind": "git", "mse": candidate_mse, "spec": spec},
        "candidate_mse": candidate_mse,
        "case": case,
        "created_at": "2026-08-24T02:00:00+00:00",
        "dataset_id": dataset,
        "geometry": {
            "seq_len": seq_len,
            "kind": kind,
            "in_features": 64,
            "out_features": 64,
            "activation_shape": [seq_len, 64],
            "output_shape": [seq_len, 64],
        },
        "group": group,
        "improvement_percent": improvement,
        "invalid_reason": None,
        "kind": kind,
        "layer": layer,
        "mode": mode,
        "model": {"alias": "tiny-model", "arch": "qwen3"},
        "record_key": "HASH-HASH-HASH",
        "role": role,
        "schema_version": 1,
        "seed": 0,
        "threads": threads,
        "timing": {"baseline_wall_s": 0.1, "candidate_wall_s": 0.2, "case_s": 0.05},
        "valid": valid,
    }


class Fixture:
    """Build a tiny runs/records/results tree under a temp directory."""

    def __init__(self, tmpdir):
        self.root = Path(tmpdir)
        self.runs = self.root / "benchmarks" / "realdata" / "runs"
        self.records = self.root / "benchmarks" / "realdata" / "records"
        self.results = self.root / "progress" / "results.jsonl"
        self.runs.mkdir(parents=True)
        self.records.mkdir(parents=True)
        (self.root / "progress").mkdir(exist_ok=True)

    def add_run(self, name, run):
        with open(self.runs / name, "w", encoding="utf-8") as fh:
            json.dump(run, fh)

    def add_record(self, name, record):
        with open(self.records / name, "w", encoding="utf-8") as fh:
            json.dump(record, fh)

    def add_results(self, rows):
        with open(self.results, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def build(self, datasets=(DS_A, DS_B), generated_at=FIXED_TS):
        return build_dashboard_data(
            self.runs, self.records, self.results, datasets=datasets, generated_at=generated_at
        )


def default_fixture(tmpdir):
    fx = Fixture(tmpdir)
    # Dataset A: an old ok run (superseded), a latest ok run with a different
    # candidate set, a failed run, and an incomplete run.
    fx.add_run(
        "old-a.json",
        make_run(DS_A, created="2026-08-24T10:00:00+00:00", updated="2026-08-24T10:00:00+00:00"),
    )
    fx.add_run(
        "latest-a.json",
        make_run(
            DS_A,
            created="2026-08-24T11:00:00+00:00",
            updated="2026-08-24T11:00:00+00:00",
            candidates=[
                {"commit": "c1", "kind": "git", "source_sha256": None, "spec": V001},
                {"commit": "c3", "kind": "git", "source_sha256": None, "spec": V002},
            ],
        ),
    )
    fx.add_run("failed-a.json", make_run(DS_A, status="failed", updated="2026-08-24T12:00:00+00:00"))
    fx.add_run("incomplete-a.json", make_run(DS_A, cases_failed=1, updated="2026-08-24T13:00:00+00:00"))
    # Dataset B: one ok run.
    fx.add_run(
        "b.json",
        make_run(DS_B, created="2026-08-24T20:00:00+00:00", updated="2026-08-24T20:00:00+00:00"),
    )

    # Records for dataset A under the latest run's candidate set.
    fx.add_record("a-linear-v001.json", make_record(DS_A, V001, "c1", case=0, mode="ceil", kind="linear", layer=0, role="q_proj", seq_len=10, improvement=2.5))
    fx.add_record("a-attn-v002.json", make_record(DS_A, V002, "c3", case=4, mode="nearest", kind="attention", layer=1, seq_len=512, improvement=9.0))
    fx.add_record("a-linear-v002.json", make_record(DS_A, V002, "c3", case=2, mode="stochastic", kind="linear", layer=0, role="down_proj", seq_len=128, improvement=-0.5))
    fx.add_record("a-invalid.json", make_record(DS_A, V001, "c1", case=1, mode="ceil", valid=False, improvement=None, baseline_mse=0.01, candidate_mse=0.013))
    # Orphan: candidate commit only exists in the superseded old run -> skipped.
    fx.add_record("a-orphan-v002.json", make_record(DS_A, V002, "c2", case=3, mode="ceil", layer=1, role="up_proj"))
    # Baseline mismatch -> skipped.
    fx.add_record("a-bad-baseline.json", make_record(DS_A, V001, "c1", case=3, mode="ceil", baseline_commit="OTHER"))
    # Dataset B records.
    fx.add_record("b-linear-v001.json", make_record(DS_B, V001, "c1", case=0, mode="ceil", kind="linear", layer=0, role="gate_proj", seq_len=1024, improvement=4.25))
    fx.add_record("b-attn-v001.json", make_record(DS_B, V001, "c1", case=2, mode="stochastic", kind="attention", layer=0, seq_len=10, improvement=7.75))
    # Dataset outside the requested list -> skipped.
    fx.add_record("other-ds.json", make_record("4405c7fadc8bef16", V001, "c1", case=0, mode="ceil", improvement=1.0))
    # Unparseable record file -> skipped_parse.
    with open(fx.records / "bad.json", "w", encoding="utf-8") as fh:
        fh.write("{not json")

    fx.add_results(
        [
            {"variant": "solution/v000-baseline", "suite": "synthetic-v1", "mean_improvement_percent": 0.0, "case_count": 40, "runtime": 0.26, "status": "ok", "timestamp": "2026-08-22T14:56:15+00:00"},
            {"variant": "solution/v001-bf16-target", "suite": "synthetic-v1", "mean_improvement_percent": -0.91, "case_count": 40, "runtime": 0.26, "status": "ok", "timestamp": "2026-08-22T14:58:51+00:00"},
        ]
    )
    return fx


class BuildDataTests(unittest.TestCase):
    def test_schema_shape_competition_and_official(self):
        with tempfile.TemporaryDirectory() as td:
            doc = default_fixture(td).build()
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["generated_at"], FIXED_TS)
        self.assertEqual(doc["competition"], COMPETITION)
        self.assertEqual(doc["competition"]["theoretical_max_score"], 50000)
        self.assertEqual(doc["competition"]["score_per_case_max"], 100)
        self.assertEqual(doc["competition"]["total_cases"], 500)
        self.assertEqual(doc["competition"]["linear_groups"], 50)
        self.assertEqual(doc["competition"]["attention_groups"], 50)

        official = {o["variant"]: o for o in doc["official"]}
        self.assertEqual(len(official), 9)
        expected = {
            "v000": (1240, [105, 92]),
            "v001": (1240, [111]),
            "v002": (5025, [113]),
            "v003": (3600, [111]),
            "v004": (4340, [117]),
            "v005": (4600, [111]),
            "v006": (4750, [123]),
            "v007": (5276, [98]),
            "v008": (5326, [120]),
        }
        for variant, (score, runtimes) in expected.items():
            self.assertEqual(official[variant]["score"], score, variant)
            self.assertEqual(official[variant]["runtimes_s"], runtimes, variant)
        self.assertTrue(official["v008"]["spec"].startswith("solution/v008"))
        self.assertTrue(all("spec" in o and "name" in o for o in doc["official"]))

    def test_judge_scale_proxy_definition(self):
        with tempfile.TemporaryDirectory() as td:
            proxy = default_fixture(td).build()["judge_scale_proxy"]
        self.assertEqual(proxy["baseline_official_score"], 1240)
        self.assertEqual(proxy["maximum_official_score"], 50000)
        self.assertEqual(proxy["formula"], "1240 + (50000 - 1240) * local_mean_improvement_percent / 100")
        self.assertIn("not an official prediction", proxy["disclaimer"])
        self.assertIn("clamp", proxy["display_note"].lower())

    def test_case_row_fields_and_role_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            doc = default_fixture(td).build()
        rows = doc["cases"]
        for row in rows:
            self.assertEqual(set(row.keys()), set(CASE_FIELDS))
            for banned in ("timing", "record_key", "seed", "commit", "source_sha256"):
                self.assertNotIn(banned, row)
            self.assertEqual(row["source"], "realdata")

        attn = [r for r in rows if r["kind"] == "attention"]
        self.assertTrue(attn)
        for row in attn:
            self.assertEqual(row["role"], "self_attn")
        linear = [r for r in rows if r["kind"] == "linear"]
        self.assertTrue(linear)
        roles = {r["role"] for r in linear}
        self.assertTrue(roles <= {"q_proj", "down_proj", "gate_proj", "up_proj", "o_proj"})
        # The explicitly-invalid record is kept and marked.
        invalid = [r for r in rows if not r["valid"]]
        self.assertEqual(len(invalid), 1)
        self.assertIsNone(invalid[0]["improvement_percent"])

    def test_run_and_commit_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            doc = default_fixture(td).build()
        counters = doc["sources"]["records"]
        self.assertEqual(counters["read"], 9)  # 8 valid JSON + 1 unparseable
        self.assertEqual(counters["skipped_parse"], 1)
        self.assertEqual(counters["skipped_dataset"], 1)
        self.assertEqual(counters["skipped_candidate_commit"], 1)
        self.assertEqual(counters["skipped_baseline_commit"], 1)
        self.assertEqual(counters["selected"], 6)  # 5 dataset-A rows + ... see below
        # Latest run per dataset only: old-a.json must not appear; failed and
        # incomplete runs must not appear.
        run_files = {r["file"] for r in doc["sources"]["runs"]}
        self.assertEqual(run_files, {"latest-a.json", "b.json"})
        self.assertEqual(doc["sources"]["run_files_skipped"]["status_not_ok"], 1)
        self.assertEqual(doc["sources"]["run_files_skipped"]["incomplete"], 1)
        # The orphan c2 record (old run only) was skipped; the c3 record was kept.
        specs = {r["spec"] for r in doc["cases"]}
        self.assertIn(V002, specs)

    def test_deterministic_ordering_and_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            fx = default_fixture(td)
            doc = fx.build()
            doc2 = fx.build()
        self.assertEqual(serialize(doc), serialize(doc2))

        rows = doc["cases"]
        keys = [
            (r["model"], r["variant"], r["mode"], r["kind"], r["layer"], r["role"], r["case"])
            for r in rows
        ]
        self.assertEqual(keys, sorted(keys, key=lambda k: (k[0], int(k[1][1:]), k[2], k[3], k[4], k[5], k[6])))

        # Same fixture on disk produces identical bytes, too.
        with tempfile.TemporaryDirectory() as td:
            fx = default_fixture(td)
            fx.build()
            out1 = (fx.root / "out.json")
            write_atomic(out1, serialize(fx.build()))
            out2 = (fx.root / "out2.json")
            write_atomic(out2, serialize(fx.build()))
            self.assertEqual(out1.read_bytes(), out2.read_bytes())

    def test_synthetic_summary(self):
        with tempfile.TemporaryDirectory() as td:
            doc = default_fixture(td).build()
        syn = doc["synthetic"]
        self.assertEqual(len(syn), 2)
        for row in syn:
            self.assertEqual(row["source"], "synthetic")
        by_variant = {r["variant"]: r for r in syn}
        self.assertEqual(by_variant["v000"]["mean_improvement_percent"], 0.0)
        self.assertEqual(by_variant["v001"]["case_count"], 40)
        self.assertEqual(by_variant["v001"]["runtime_s"], 0.26)
        self.assertEqual(by_variant["v001"]["status"], "ok")
        self.assertEqual(by_variant["v001"]["spec"], "solution/v001-bf16-target")
        self.assertIn("timestamp", by_variant["v001"])
        self.assertEqual(doc["sources"]["synthetic"]["rows"], 2)

    def test_models_and_dimensions(self):
        with tempfile.TemporaryDirectory() as td:
            doc = default_fixture(td).build()
        self.assertEqual(len(doc["models"]), 1)
        self.assertEqual(doc["models"][0]["alias"], "tiny-model")
        dims = doc["dimensions"]
        self.assertEqual(dims["modes"], ["ceil", "nearest", "stochastic"])
        self.assertEqual(dims["kinds"], ["attention", "linear"])
        self.assertIn("self_attn", dims["roles"]["attention"])
        self.assertEqual(dims["case_indices"], [0, 1, 2, 4])
        self.assertEqual(dims["seq_len_values"], [10, 128, 512, 1024])
        self.assertEqual(dims["datasets"], sorted([DS_A, DS_B]))

    def test_atomic_output_and_no_temp_leftovers(self):
        with tempfile.TemporaryDirectory() as td:
            fx = default_fixture(td)
            out = fx.root / "out" / "dashboard-data.json"
            doc = fx.build(generated_at=FIXED_TS)
            write_atomic(out, serialize(doc))
            self.assertTrue(out.is_file())
            with open(out, encoding="utf-8") as fh:
                reloaded = json.load(fh)
            self.assertEqual(reloaded["schema_version"], 1)
            self.assertEqual(reloaded["generated_at"], FIXED_TS)
            self.assertEqual(len(reloaded["cases"]), len(doc["cases"]))
            leftovers = [p for p in fx.root.rglob("*.tmp") if ".dashboard-data-" in p.name]
            self.assertEqual(leftovers, [])

    def test_clear_errors_for_missing_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            fx = default_fixture(td)
            missing = Path(td) / "nope"
            with self.assertRaises(DashboardError) as ctx:
                build_dashboard_data(missing, fx.records, fx.results)
            self.assertIn("runs directory not found", str(ctx.exception))
            with self.assertRaises(DashboardError) as ctx:
                build_dashboard_data(fx.runs, missing, fx.results, datasets=(DS_A, DS_B))
            self.assertIn("records directory not found", str(ctx.exception))
            with self.assertRaises(DashboardError) as ctx:
                build_dashboard_data(fx.runs, fx.records, missing, datasets=(DS_A, DS_B))
            self.assertIn("synthetic results file not found", str(ctx.exception))

        # Empty runs directory -> error naming the directory.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "benchmarks" / "realdata" / "runs").mkdir(parents=True)
            (root / "benchmarks" / "realdata" / "records").mkdir(parents=True)
            results = root / "progress" / "results.jsonl"
            results.parent.mkdir(parents=True)
            with open(results, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"variant": "solution/v000-baseline"}) + "\n")
            # Only a failed run exists -> no complete/ok run manifest.
            with open(root / "benchmarks" / "realdata" / "runs" / "failed.json", "w", encoding="utf-8") as fh:
                json.dump(make_run(DS_A, status="failed"), fh)
            with self.assertRaises(DashboardError) as ctx:
                build_dashboard_data(
                    root / "benchmarks" / "realdata" / "runs",
                    root / "benchmarks" / "realdata" / "records",
                    results,
                )
            self.assertIn("no complete/ok run manifest", str(ctx.exception))

        # A requested dataset without any ok run -> error listing it.
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(td)
            fx.add_run("bad.json", make_run(DS_A, status="failed"))
            fx.add_record("r.json", make_record(DS_A, V001, "c1"))
            fx.add_results([{"variant": "solution/v000-baseline"}])
            with self.assertRaises(DashboardError) as ctx:
                fx.build(datasets=(DS_A,))
            self.assertIn(DS_A, str(ctx.exception))

    def test_variant_short(self):
        self.assertEqual(variant_short("solution/v008-gated-hadamard"), "v008")
        self.assertEqual(variant_short("v002-e6m2-neighbors"), "v002")
        self.assertEqual(variant_short("solution/v000-baseline"), "v000")
        self.assertEqual(variant_short(""), "")
        self.assertEqual(variant_short(None), "")


class CliTests(unittest.TestCase):
    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(_ROOT / "tools" / "build_progress_data.py"), *map(str, args)],
            capture_output=True,
            text=True,
        )

    def test_cli_success_and_failure(self):
        with tempfile.TemporaryDirectory() as td:
            fx = default_fixture(td)
            out = fx.root / "dashboard-data.json"
            proc = self._run_cli(
                "--runs-dir", fx.runs,
                "--records-dir", fx.records,
                "--results", fx.results,
                "--output", out,
                "--datasets", f"{DS_A},{DS_B}",
                "--generated-at", FIXED_TS,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("case rows", proc.stdout)
            with open(out, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual(doc["schema_version"], 1)
            self.assertEqual(doc["generated_at"], FIXED_TS)

            # Missing records directory -> non-zero exit with a clear message.
            proc2 = self._run_cli(
                "--runs-dir", fx.runs,
                "--records-dir", fx.root / "missing-records",
                "--results", fx.results,
                "--output", fx.root / "out2.json",
                "--datasets", f"{DS_A},{DS_B}",
            )
            self.assertNotEqual(proc2.returncode, 0)
            self.assertIn("records directory not found", proc2.stderr)

    def test_cli_defaults_repo_root(self):
        with tempfile.TemporaryDirectory() as td:
            fx = default_fixture(td)
            proc = self._run_cli(
                "--repo-root", fx.root,
                "--datasets", f"{DS_A},{DS_B}",
                "--generated-at", FIXED_TS,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            # Defaults resolved under --repo-root.
            output = fx.root / "progress" / "dashboard-data.json"
            self.assertTrue(output.is_file())
            with open(output, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual(doc["sources"]["records"]["dir"], "benchmarks/realdata/records")
            self.assertEqual(doc["sources"]["synthetic"]["file"], "progress/results.jsonl")


if __name__ == "__main__":
    unittest.main()
