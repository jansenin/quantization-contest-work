"""Unit tests for the streaming real-capture evaluator.

Covers ``tools/realdata/evaluate.py`` and the ``tools/evaluate_real.py`` CLI
using only tiny synthetic raw-BF16 shards and fake solution files/tags: no
model weights and no ``transformers`` import.  Verifies:

* dataset path / dataset-id resolution and manifest validation;
* NVFP4 source-mode derivation (ceil / nearest / stochastic) with stable
  unique ``tensor_id`` strings and a seed (determinism, semantics, seed and
  tensor-id sensitivity);
* one-group-at-a-time streaming (lazy shard loading, filters, limit);
* Linear (one weight calibration + five dynamic activations) and Attention
  (one calibration + five Q/K/V triples) call semantics via a logging fake;
* detailed disaggregated record metadata, MSE/improvement scoring and robust
  zero-denominator handling;
* deterministic record keys/results across reruns;
* invalid shards/refs (partial vs hard failure);
* atomic output and resume/skip behavior (unit granularity, ``--force``);
* CLI parsing defaults and repeatable modes/candidates;
* a subprocess smoke test that best-effort sets ``oom_score_adj=500`` and
  asserts the peak RSS beyond the unavoidable torch import stays below 300 MB.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import evaluate_real
from tools.realdata import evaluate, shards
from tools.realdata.evaluate import (
    DatasetError,
    GroupLoadError,
    case_record_key,
    derive_source,
    evaluate_dataset,
    iter_groups,
    parse_filters,
    resolve_dataset,
)
from tools.realdata.shards import SAMPLES_PER_SPLIT

TESTS = SAMPLES_PER_SPLIT
SEQ = 4
IN_F = 64
OUT_F = 64
QH, KVH, HD = 2, 1, 64
Q_HIDDEN = QH * HD
KV_HIDDEN = KVH * HD
FAKE_REVISION = "b" * 40


# ---------------------------------------------------------------------------
# Fake solution sources (self-contained modules, torch only)
# ---------------------------------------------------------------------------

_ZERO_SOLUTION = '''
import torch


def _dq(q, s):
    return (q.unflatten(-1, (-1, 16)) * s.unsqueeze(-1)).flatten(-2, -1).float()


def _params(x):
    x = x.float()
    C = x.shape[-1]
    nb = C // 64
    blocks = x.unflatten(-1, (nb, 64))
    return {
        "scale_factor": torch.ones(x.shape[:-1] + (nb, 1, 1, 1)),
        "scale_lv2": torch.ones(x.shape[:-1] + (nb, 8, 1, 1)),
        "scale_lv3": torch.ones(x.shape[:-1] + (nb, 8, 2, 1)),
        "sign": torch.ones(x.shape[:-1] + (nb, 8, 2, 4)),
        "mant": torch.zeros(x.shape[:-1] + (nb, 8, 2, 4)),
    }


def hif4_calibration_and_quantize_weight(wq, ws, calib):
    return {"weight_params": _params(_dq(wq, ws)), "activation_state": None}


def hif4_dynamic_quantize_activation(aq, as_, state):
    return _params(_dq(aq, as_))


def hif4_calibration_attention(calib, qh, kvh, hd):
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(q, s, nh, hd, state):
    return _params(_dq(q, s))


def hif4_dynamic_quantize_k(q, s, nh, hd, state):
    return _params(_dq(q, s))


def hif4_dynamic_quantize_v(q, s, nh, hd, state):
    return _params(_dq(q, s))
'''

_COPY_SOLUTION = '''
import torch


def _dq(q, s):
    return (q.unflatten(-1, (-1, 16)) * s.unsqueeze(-1)).flatten(-2, -1).float()


def _params(x):
    x = x.float()
    C = x.shape[-1]
    nb = C // 64
    blocks = x.unflatten(-1, (nb, 64))
    bm = blocks.abs().amax(dim=-1)
    pos = bm > 0
    sfv = torch.where(
        pos,
        torch.pow(2.0, torch.floor(torch.log2(torch.clamp(bm, min=2.0 ** -100)))),
        torch.ones_like(bm),
    )
    sfv = torch.clamp(sfv, min=2.0 ** -48, max=49152.0)
    mant = torch.clamp(
        torch.round((blocks / sfv.unsqueeze(-1)).abs() * 4.0) / 4.0, max=1.75
    )
    sg = torch.sign(blocks)
    sg = torch.where(sg == 0, torch.ones_like(sg), sg)
    return {
        "scale_factor": sfv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        .expand(x.shape[:-1] + (nb, 1, 1, 1)).contiguous(),
        "scale_lv2": torch.ones(x.shape[:-1] + (nb, 8, 1, 1)),
        "scale_lv3": torch.ones(x.shape[:-1] + (nb, 8, 2, 1)),
        "sign": sg.reshape(x.shape[:-1] + (nb, 8, 2, 4)),
        "mant": mant.reshape(x.shape[:-1] + (nb, 8, 2, 4)),
    }


def hif4_calibration_and_quantize_weight(wq, ws, calib):
    return {"weight_params": _params(_dq(wq, ws)), "activation_state": None}


def hif4_dynamic_quantize_activation(aq, as_, state):
    return _params(_dq(aq, as_))


def hif4_calibration_attention(calib, qh, kvh, hd):
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(q, s, nh, hd, state):
    return _params(_dq(q, s))


def hif4_dynamic_quantize_k(q, s, nh, hd, state):
    return _params(_dq(q, s))


def hif4_dynamic_quantize_v(q, s, nh, hd, state):
    return _params(_dq(q, s))
'''

_LOGGING_PROLOGUE = '''
import json as _json
import os as _os

_LOG = _os.environ.get("RD_FAKE_LOG")

def _log(name, *shapes):
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps({"func": name, "shapes": [list(s) for s in shapes]}) + "\\n")
'''


def _write_fake_solution(path: Path, flavor: str) -> Path:
    """Write a fake solution module; ``flavor`` is 'zero' or 'copy'."""
    body = _ZERO_SOLUTION if flavor == "zero" else _COPY_SOLUTION
    path.write_text(body, encoding="utf-8")
    return path


def _write_logging_solution(path: Path) -> Path:
    """Zero-flavor solution whose six APIs append to RD_FAKE_LOG."""
    text = _ZERO_SOLUTION.replace(
        "def hif4_calibration_and_quantize_weight(wq, ws, calib):",
        f"{_LOGGING_PROLOGUE}\n\ndef hif4_calibration_and_quantize_weight(wq, ws, calib):\n"
        '    _log("weight", wq.shape, ws.shape)',
    ).replace(
        "def hif4_dynamic_quantize_activation(aq, as_, state):",
        "def hif4_dynamic_quantize_activation(aq, as_, state):\n"
        '    _log("activation", aq.shape, as_.shape)',
    ).replace(
        "def hif4_calibration_attention(calib, qh, kvh, hd):",
        "def hif4_calibration_attention(calib, qh, kvh, hd):\n"
        '    _log("attention", [qh, kvh, hd])',
    ).replace(
        "def hif4_dynamic_quantize_q(q, s, nh, hd, state):",
        "def hif4_dynamic_quantize_q(q, s, nh, hd, state):\n"
        '    _log("q", q.shape, s.shape)',
    ).replace(
        "def hif4_dynamic_quantize_k(q, s, nh, hd, state):",
        "def hif4_dynamic_quantize_k(q, s, nh, hd, state):\n"
        '    _log("k", q.shape, s.shape)',
    ).replace(
        "def hif4_dynamic_quantize_v(q, s, nh, hd, state):",
        "def hif4_dynamic_quantize_v(q, s, nh, hd, state):\n"
        '    _log("v", q.shape, s.shape)',
    )
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tiny dataset builder
# ---------------------------------------------------------------------------


class TinyDataset:
    """A minimal valid capture dataset: one Linear and one Attention group."""

    def __init__(self, root: Path, dataset_id: str = "0" * 16) -> None:
        self.root = root
        self.dataset_id = dataset_id
        self.dir = root / "dataset"
        (self.dir / "linear").mkdir(parents=True)
        (self.dir / "attention").mkdir(parents=True)
        self.entries: dict[str, dict] = {}
        self._counter = 0
        self.linear_meta = {
            "layer_idx": 0,
            "role": "q_proj",
            "in_features": IN_F,
            "out_features": OUT_F,
            "has_bias": False,
            "sample_lengths": [SEQ] * (2 * TESTS),
            "sample_token_hashes": [f"h{i:064x}" for i in range(2 * TESTS)],
        }
        self.attn_meta = {
            "layer_idx": 0,
            "arch": "fake",
            "sample_lengths": [SEQ] * (2 * TESTS),
            "sample_token_hashes": [f"h{i:064x}" for i in range(2 * TESTS)],
        }

    def _tensor(self, *shape: int, zero: bool = False) -> torch.Tensor:
        if zero:
            value = torch.zeros(*shape, dtype=torch.bfloat16)
        else:
            # Deterministic data (fixed generator) so every assertion in this
            # suite is reproducible across runs and interpreters.
            self._counter += 1
            generator = torch.Generator().manual_seed(0x5EED + self._counter)
            value = torch.randn(*shape, dtype=torch.bfloat16, generator=generator)
        return value.contiguous()

    def add_linear(
        self,
        group_id: str = "0.q_proj",
        zero: bool = False,
        layer_idx: int = 0,
        role: str = "q_proj",
        out_features: int = OUT_F,
        in_features: int = IN_F,
    ) -> "TinyDataset":
        weight = self._tensor(out_features, in_features, zero=zero)
        calib = [self._tensor(SEQ, in_features, zero=zero) for _ in range(TESTS)]
        test = [self._tensor(SEQ, in_features, zero=zero) for _ in range(TESTS)]
        metadata = dict(self.linear_meta, layer_idx=layer_idx, role=role,
                        in_features=in_features, out_features=out_features)
        shard = shards.build_linear_shard(metadata, weight, calib, test)
        shards.validate_linear_shard(shard)
        relative = Path("linear") / f"{group_id}.pt"
        digest = shards.atomic_save_torch(self.dir / relative, shard)
        self.entries[group_id] = {
            "kind": "linear",
            "id": group_id,
            "path": str(relative),
            "sha256": digest,
            "metadata": metadata,
            "status": "complete",
        }
        return self

    def add_attention(
        self,
        group_id: str = "0.self_attn",
        zero: bool = False,
        layer_idx: int = 0,
    ) -> "TinyDataset":
        def sample() -> dict:
            return {
                "q": self._tensor(SEQ, Q_HIDDEN, zero=zero),
                "k": self._tensor(SEQ, KV_HIDDEN, zero=zero),
                "v": self._tensor(SEQ, KV_HIDDEN, zero=zero),
            }

        metadata = dict(self.attn_meta, layer_idx=layer_idx)
        shard = shards.build_attention_shard(
            metadata, QH, KVH, HD,
            [sample() for _ in range(TESTS)],
            [sample() for _ in range(TESTS)],
        )
        shards.validate_attention_shard(shard)
        relative = Path("attention") / f"{group_id}.pt"
        digest = shards.atomic_save_torch(self.dir / relative, shard)
        self.entries[group_id] = {
            "kind": "attention",
            "id": group_id,
            "path": str(relative),
            "sha256": digest,
            "metadata": metadata,
            "status": "complete",
        }
        return self

    def set_group(self, group_id: str, entry: dict) -> None:
        self.entries[group_id] = entry

    def write_manifest(self, status: str = "complete") -> None:
        manifest = {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "status": status,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "raw_bf16": shards.RAW_BF16_STORAGE,
            "source_modes": shards.SOURCE_MODES,
            "model": {
                "alias": "tiny-model",
                "repo_id": "tests/tiny-model",
                "resolved_revision": FAKE_REVISION,
                "arch": "fake",
                "transformers_version": "4.57.6-test",
                "num_layers": 1,
                "hidden_size": IN_F,
                "num_attention_heads": QH,
                "num_key_value_heads": KVH,
                "head_dim": HD,
            },
            "corpus": {
                "schema_version": 1,
                "sha256": "c" * 64,
                "tokenization": {"special_tokens": False},
            },
            "samples": [
                {
                    "index": i,
                    "split": "calib" if i < TESTS else "test",
                    "length": SEQ,
                    "prompt": f"prompt-{i}",
                    "prompt_hash": f"p{i:064x}",
                    "token_hash": f"h{i:064x}",
                }
                for i in range(2 * TESTS)
            ],
            "layers": {"selected": [0], "linear_roles": ["q_proj"]},
            "groups": [self.entries[key] for key in sorted(self.entries)],
            "seed": 0,
            "runtime": {"torch_version": torch.__version__, "threads": 1},
        }
        shards.write_manifest(self.dir / "manifest.json", manifest)

    def manifest(self) -> dict:
        return shards.load_manifest(self.dir / "manifest.json")


def _default_config(dataset: TinyDataset, tmp: Path, **overrides) -> dict:
    config = {
        "dataset_dir": dataset.dir,
        "manifest": dataset.manifest(),
        "dataset_spec": str(dataset.dir),
        "modes": ["ceil", "nearest", "stochastic"],
        "seed": 0,
        "baseline_spec": str(tmp / "base.py"),
        "candidate_specs": [str(tmp / "cand.py")],
        "output_dir": tmp / "out",
        "force": False,
        "threads": 1,
        "limit": None,
    }
    config.update(overrides)
    return config


def _stable_record_fields(record: dict) -> dict:
    """Record content minus wall-clock timings and creation timestamps."""
    stable = json.loads(json.dumps(record, sort_keys=True))
    stable.pop("created_at", None)
    if isinstance(stable.get("timing"), dict):
        stable["timing"] = {k: None for k in sorted(stable["timing"])}
    for key in ("baseline", "candidate"):
        if isinstance(stable.get(key), dict):
            for timing_key in ("calibration_s", "online_s", "scoring_s"):
                stable[key][timing_key] = None
    return stable


class EvaluateTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dataset = TinyDataset(self.root)
        self.dataset.add_linear().add_attention()
        self.dataset.write_manifest()
        self.base = _write_fake_solution(self.root / "base.py", "zero")
        self.cand = _write_fake_solution(self.root / "cand.py", "copy")

    def run_default(self, **overrides) -> dict:
        config = _default_config(self.dataset, self.root, **overrides)
        return evaluate_dataset(config)


# ---------------------------------------------------------------------------
# Dataset resolution and streaming
# ---------------------------------------------------------------------------


class ResolveDatasetTest(unittest.TestCase):
    def test_path_and_id_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = TinyDataset(root)
            dataset.add_linear()
            dataset.write_manifest()
            by_path = resolve_dataset(str(dataset.dir))
            self.assertEqual(by_path[0], dataset.dir)
            self.assertEqual(by_path[1]["dataset_id"], dataset.dataset_id)
            captures = root / "captures"
            captures.mkdir()
            dataset.dir.rename(captures / dataset.dataset_id)
            by_id = resolve_dataset(dataset.dataset_id, captures_root=captures)
            self.assertEqual(by_id[0], captures / dataset.dataset_id)
            with self.assertRaises(DatasetError):
                resolve_dataset("f" * 16, captures_root=root)
            with self.assertRaises(DatasetError):
                resolve_dataset(dataset.dataset_id, captures_root=root / "missing")

    def test_incomplete_manifest_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = TinyDataset(Path(td))
            dataset.add_linear()
            dataset.write_manifest(status="in_progress")
            with self.assertRaisesRegex(DatasetError, "expected 'complete'"):
                resolve_dataset(str(dataset.dir))


class GroupStreamingTest(unittest.TestCase):
    def test_groups_loaded_one_at_a_time(self) -> None:
        """Shards load lazily: deleting a later shard only breaks later pulls."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = TinyDataset(root)
            dataset.add_linear("0.q_proj", layer_idx=0)
            dataset.add_attention("1.self_attn", layer_idx=1)
            dataset.add_linear("1.q_proj", layer_idx=1)
            dataset.write_manifest()
            manifest = dataset.manifest()
            iterator = iter_groups(dataset.dir, manifest)
            first_entry, first_shard = next(iterator)
            self.assertEqual(first_entry["id"], "0.q_proj")
            self.assertEqual(first_shard["kind"], "linear")
            # The remaining shards are not in memory yet: deleting the second
            # group's file must only raise when its turn comes.
            (dataset.dir / manifest["groups"][1]["path"]).unlink()
            with self.assertRaises(GroupLoadError):
                next(iterator)

    def test_filters_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = TinyDataset(root)
            dataset.add_linear("0.q_proj", layer_idx=0)
            dataset.add_linear("0.o_proj", layer_idx=0, role="o_proj")
            dataset.add_attention("0.self_attn", layer_idx=0)
            dataset.add_attention("1.self_attn", layer_idx=1)
            dataset.write_manifest()
            manifest = dataset.manifest()

            def ids(**kwargs):
                return [
                    entry["id"]
                    for entry, _ in iter_groups(dataset.dir, manifest, **kwargs)
                ]

            self.assertEqual(ids(filters=parse_filters(["linear"])), ["0.o_proj", "0.q_proj"])
            self.assertEqual(
                ids(filters=parse_filters(["role:q_proj"])), ["0.q_proj"]
            )
            self.assertEqual(
                ids(filters=parse_filters(["attention", "layer:1"])), ["1.self_attn"]
            )
            self.assertEqual(
                ids(filters=parse_filters(["linear", "role:o_proj,role:q_proj"])),
                ["0.o_proj", "0.q_proj"],
            )
            self.assertEqual(ids(limit=2), ["0.o_proj", "0.q_proj"])
            with self.assertRaises(ValueError):
                parse_filters(["bogus:1"])

    def test_invalid_shard_raises_at_its_turn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = TinyDataset(Path(td))
            dataset.add_linear("0.q_proj")
            dataset.add_linear("1.q_proj")
            dataset.write_manifest()
            bad = {
                "kind": "linear",
                "schema_version": 1,
                "metadata": {},
                "weight": torch.zeros(8, 8, dtype=torch.bfloat16),
                "calib_activation_list": [],
                "test_activation_list": [],
            }
            shards.atomic_save_torch(dataset.dir / "linear" / "1.q_proj.pt", bad)
            manifest = dataset.manifest()
            iterator = iter_groups(dataset.dir, manifest)
            entry, _ = next(iterator)
            self.assertEqual(entry["id"], "0.q_proj")
            with self.assertRaisesRegex(GroupLoadError, "invalid shard"):
                next(iterator)


# ---------------------------------------------------------------------------
# NVFP4 source-mode derivation
# ---------------------------------------------------------------------------


class SourceModesTest(unittest.TestCase):
    def _shard(self, zero: bool = False) -> dict:
        with tempfile.TemporaryDirectory() as td:
            dataset = TinyDataset(Path(td))
            dataset.add_linear(zero=zero)
            dataset.write_manifest()
            manifest = dataset.manifest()
            entry = manifest["groups"][0]
            return shards.load_tensor(dataset.dir / entry["path"])

    def test_derive_deterministic_and_legal(self) -> None:
        from tools.nvfp4_quantize import E4M3_POSITIVE_VALUES
        from tools.nvfp4_quantize import NVFP4_CARRIERS  # re-exported

        shard = self._shard()
        for mode in ("ceil", "nearest", "stochastic"):
            first = derive_source(shard, "d" * 16, "0.q_proj", mode, seed=7)
            second = derive_source(shard, "d" * 16, "0.q_proj", mode, seed=7)
            wq1, ws1 = first["weight"]
            wq2, ws2 = second["weight"]
            self.assertTrue(torch.equal(wq1, wq2), mode)
            self.assertTrue(torch.equal(ws1, ws2), mode)
            for value in wq1.unique().tolist():
                self.assertIn(value, NVFP4_CARRIERS, f"{mode}: bad carrier")
            self.assertTrue((ws1 > 0).all(), f"{mode}: scales must be positive")
            self.assertTrue(torch.isfinite(ws1).all(), f"{mode}")
            for grid_value in ws1.unique().tolist():
                self.assertIn(grid_value, E4M3_POSITIVE_VALUES, f"{mode}: bad scale")
            # last-dim/scale-shape contract
            self.assertEqual(tuple(ws1.shape), tuple(wq1.shape[:-1]) + (wq1.shape[-1] // 16,))
            # mode is not part of the tensor identity: same id, same seed
            for ti, pair in enumerate(first["test"]):
                self.assertTrue(
                    torch.equal(pair[0], first["test"][ti][0]), "deterministic test pair"
                )

    def test_ceil_scale_is_minimal_grid_upper_bound(self) -> None:
        from tools.nvfp4_quantize import E4M3_POSITIVE_VALUES

        shard = self._shard()
        source = derive_source(shard, "d" * 16, "0.q_proj", "ceil", seed=0)
        grid = torch.tensor(E4M3_POSITIVE_VALUES, dtype=torch.float64)
        pairs = [
            (shard["weight"], source["weight"]),
            *zip(shard["calib_activation_list"], source["calib"]),
            *zip(shard["test_activation_list"], source["test"]),
        ]
        for raw, (_, scale) in pairs:
            blocks = raw.to(torch.float64).unflatten(-1, (-1, 16)).abs().amax(dim=-1)
            target = blocks / 6.0
            expected = grid[torch.searchsorted(grid, target, right=False).clamp(max=len(grid) - 1)]
            self.assertTrue(
                torch.equal(scale.to(torch.float64), expected),
                "ceil scale must be the smallest grid value >= max_abs/6",
            )
            self.assertTrue((scale.to(torch.float64) >= target).all())

    def test_nearest_is_adjacent_grid_value(self) -> None:
        from tools.nvfp4_quantize import E4M3_POSITIVE_VALUES

        shard = self._shard()
        source = derive_source(shard, "d" * 16, "0.q_proj", "nearest", seed=0)
        grid = torch.tensor(E4M3_POSITIVE_VALUES, dtype=torch.float64)
        _, scale = source["weight"]
        target = shard["weight"].to(torch.float64).unflatten(-1, (-1, 16)).abs().amax(dim=-1) / 6.0
        scale64 = scale.to(torch.float64)
        # every scale is one of the two grid values bracketing the target
        upper = grid[torch.searchsorted(grid, target, right=True).clamp(max=len(grid) - 1)]
        lower = grid[(torch.searchsorted(grid, target, right=True) - 1).clamp(min=0)]
        self.assertTrue(((scale64 == lower) | (scale64 == upper)).all())

    def test_stochastic_seed_and_tensor_id_sensitivity(self) -> None:
        from tools.nvfp4_quantize import _tensor_key

        # distinct tensor ids yield distinct draw keys (decorrelation mechanism)
        self.assertNotEqual(_tensor_key(0, "d/0.q_proj/weight"), _tensor_key(0, "d/0.q_proj/calib/0"))
        self.assertEqual(_tensor_key(5, "x"), _tensor_key(5, "x"))
        self.assertNotEqual(_tensor_key(5, "x"), _tensor_key(6, "x"))

        # same tensor, same id, different seed -> different stochastic scales
        # for at least one seed in a fixed deterministic range
        shard = self._shard()
        seed_a = derive_source(shard, "d" * 16, "0.q_proj", "stochastic", seed=1)
        seed_b = derive_source(shard, "d" * 16, "0.q_proj", "stochastic", seed=2)
        # For a random tensor with 64+ blocks the draw sequences virtually
        # always differ; assert difference over a fixed seed sweep instead of
        # relying on any single seed's collision-free outcome.
        differ = any(
            not torch.equal(
                derive_source(shard, "d" * 16, "0.q_proj", "stochastic", seed=seed)["weight"][1],
                seed_a["weight"][1],
            )
            for seed in range(1, 5)
        )
        self.assertTrue(differ)
        # same seed but distinct tensor_id gives a distinct draw sequence
        id_a = derive_source(shard, "d" * 16, "0.q_proj", "stochastic", seed=1)
        id_b = derive_source(shard, "d" * 16, "0.q_proj2", "stochastic", seed=1)
        self.assertNotEqual(
            tuple(id_a["weight"][1].tolist()), tuple(id_b["weight"][1].tolist())
        )

    def test_zero_shard_derivation_is_stable(self) -> None:
        shard = self._shard(zero=True)
        for mode in ("ceil", "nearest", "stochastic"):
            source = derive_source(shard, "d" * 16, "0.q_proj", mode, seed=3)
            self.assertTrue(torch.equal(source["weight"][0], torch.zeros_like(source["weight"][0])))
            self.assertTrue((source["weight"][1] > 0).all(), mode)


# ---------------------------------------------------------------------------
# Call semantics
# ---------------------------------------------------------------------------


class CallSemanticsTest(EvaluateTestBase):
    def test_linear_and_attention_call_counts(self) -> None:
        log_path = self.root / "calls.jsonl"
        os.environ["RD_FAKE_LOG"] = str(log_path)
        self.addCleanup(os.environ.pop, "RD_FAKE_LOG", None)
        logging_solution = _write_logging_solution(self.root / "logging.py")
        run = self.run_default(baseline_spec=str(logging_solution),
                               candidate_specs=[str(logging_solution)],
                               modes=["ceil"])
        self.assertEqual(run["status"], "ok")
        calls = [json.loads(line) for line in log_path.read_text().splitlines()]
        names = [call["func"] for call in calls]
        self.assertEqual(names.count("weight"), 2)  # one per group per mode
        self.assertEqual(names.count("activation"), 2 * TESTS)
        self.assertEqual(names.count("attention"), 2)
        self.assertEqual(names.count("q"), 2 * TESTS)
        self.assertEqual(names.count("k"), 2 * TESTS)
        self.assertEqual(names.count("v"), 2 * TESTS)
        # baseline and candidate each ran one calibration + five tests
        for weight_call in [call for call in calls if call["func"] == "weight"]:
            self.assertEqual(weight_call["func"], "weight")
            self.assertEqual(weight_call["shapes"], [[OUT_F, IN_F], [OUT_F, IN_F // 16]])
        attn_calls = [c for c in calls if c["func"] == "attention"]
        for call in attn_calls:
            self.assertEqual(call["shapes"], [[QH, KVH, HD]])


# ---------------------------------------------------------------------------
# Records / scoring / determinism
# ---------------------------------------------------------------------------


class RecordsMetadataTest(EvaluateTestBase):
    def test_records_contain_expected_fields(self) -> None:
        run = self.run_default(modes=["ceil"])
        self.assertEqual(run["status"], "ok")
        records = list((self.root / "out" / "records").glob("*.json"))
        self.assertEqual(len(records), 2 * 1 * 1 * TESTS)  # groups*modes*cands*tests
        by_group: dict[str, list[dict]] = {}
        for path in records:
            record = json.loads(path.read_text())
            by_group.setdefault(record["group"], []).append(record)
        self.assertEqual(len(by_group["0.q_proj"]), TESTS)
        self.assertEqual(len(by_group["0.self_attn"]), TESTS)

        linear = by_group["0.q_proj"][0]
        self.assertEqual(linear["dataset_id"], "0" * 16)
        self.assertEqual(linear["model"]["alias"], "tiny-model")
        self.assertEqual(linear["model"]["arch"], "fake")
        self.assertEqual(linear["model"]["resolved_revision"], FAKE_REVISION)
        self.assertEqual(linear["kind"], "linear")
        self.assertEqual(linear["layer"], 0)
        self.assertEqual(linear["role"], "q_proj")
        self.assertEqual(linear["mode"], "ceil")
        self.assertEqual(linear["seed"], 0)
        self.assertIn("case", linear)
        self.assertEqual(linear["geometry"]["in_features"], IN_F)
        self.assertEqual(linear["geometry"]["out_features"], OUT_F)
        self.assertEqual(linear["geometry"]["seq_len"], SEQ)
        self.assertEqual(linear["geometry"]["output_shape"], [SEQ, OUT_F])
        self.assertEqual(linear["geometry"]["calib_samples"], TESTS)
        self.assertEqual(linear["geometry"]["test_samples"], TESTS)
        self.assertTrue(linear["valid"])
        self.assertIsNone(linear["invalid_reason"])
        self.assertGreater(linear["improvement_percent"], 0.0)
        self.assertGreater(linear["baseline_mse"], 0.0)
        self.assertLess(linear["candidate_mse"], linear["baseline_mse"])
        self.assertEqual(linear["baseline"]["kind"], "path")
        self.assertEqual(linear["candidate"]["kind"], "path")
        self.assertIsNotNone(linear["timing"]["case_s"])
        self.assertTrue(
            (self.root / "out" / "records" / f"{linear['record_key']}.json").is_file()
        )

        attention = by_group["0.self_attn"][0]
        self.assertEqual(attention["kind"], "attention")
        self.assertIsNone(attention["role"])
        self.assertEqual(attention["geometry"]["q_num_heads"], QH)
        self.assertEqual(attention["geometry"]["kv_num_heads"], KVH)
        self.assertEqual(attention["geometry"]["head_dim"], HD)
        self.assertEqual(attention["geometry"]["q_hidden"], Q_HIDDEN)
        self.assertEqual(attention["geometry"]["kv_hidden"], KV_HIDDEN)
        self.assertEqual(attention["geometry"]["output_shape"], [SEQ, Q_HIDDEN])
        self.assertGreater(attention["improvement_percent"], 0.0)

    def test_identical_variants_score_zero_improvement(self) -> None:
        run = self.run_default(candidate_specs=[str(self.base)], modes=["ceil"])
        records = list((self.root / "out" / "records").glob("*.json"))
        for path in records:
            record = json.loads(path.read_text())
            self.assertTrue(record["valid"])
            self.assertAlmostEqual(record["improvement_percent"], 0.0, places=9)
            self.assertAlmostEqual(
                record["candidate_mse"], record["baseline_mse"], places=12
            )

    def test_zero_denominator_marks_case_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = TinyDataset(root)
            dataset.add_linear(zero=True)
            dataset.write_manifest()
            base = _write_fake_solution(root / "base.py", "zero")
            cand = _write_fake_solution(root / "cand.py", "zero")
            run = evaluate_dataset(_default_config(dataset, root, modes=["ceil"]))
            self.assertEqual(run["status"], "invalid")
            self.assertEqual(run["counts"]["cases_invalid"], TESTS)
            record = json.loads(
                list((root / "out" / "records").glob("*.json"))[0].read_text()
            )
            self.assertFalse(record["valid"])
            self.assertEqual(record["invalid_reason"], "baseline_mse_zero")
            self.assertIsNone(record["improvement_percent"])
            self.assertEqual(record["baseline_mse"], 0.0)
            self.assertIsNone(
                run["per_mode"]["ceil"]["mean_improvement_percent"]
            )

    def test_deterministic_results_across_runs(self) -> None:
        first = self.run_default()
        second = self.run_default(force=True)  # recompute everything
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        first_records = {
            p.stem: json.loads(p.read_text()) for p in (self.root / "out" / "records").glob("*.json")
        }
        second_records = {
            p.stem: json.loads(p.read_text()) for p in (self.root / "out" / "records").glob("*.json")
        }
        self.assertEqual(sorted(first_records), sorted(second_records))
        for key in first_records:
            self.assertEqual(
                _stable_record_fields(first_records[key]),
                _stable_record_fields(second_records[key]),
                key,
            )

    def test_record_keys_are_deterministic_and_discriminating(self) -> None:
        key = case_record_key("d", 0, "b", "c", "0.q_proj", "ceil", 2)
        self.assertEqual(key, case_record_key("d", 0, "b", "c", "0.q_proj", "ceil", 2))
        for different in (
            case_record_key("e", 0, "b", "c", "0.q_proj", "ceil", 2),
            case_record_key("d", 1, "b", "c", "0.q_proj", "ceil", 2),
            case_record_key("d", 0, "x", "c", "0.q_proj", "ceil", 2),
            case_record_key("d", 0, "b", "x", "0.q_proj", "ceil", 2),
            case_record_key("d", 0, "b", "c", "0.q_proj", "nearest", 2),
            case_record_key("d", 0, "b", "c", "0.q_proj", "ceil", 3),
        ):
            self.assertNotEqual(key, different)


# ---------------------------------------------------------------------------
# Failures: invalid shards / refs
# ---------------------------------------------------------------------------


class InvalidInputTest(EvaluateTestBase):
    def test_dynamic_candidate_failure_is_isolated(self) -> None:
        crashing = self.root / "crashing.py"
        crashing.write_text(
            _ZERO_SOLUTION.replace(
                "def hif4_dynamic_quantize_activation(aq, as_, state):\n"
                "    return _params(_dq(aq, as_))",
                "def hif4_dynamic_quantize_activation(aq, as_, state):\n"
                "    raise RuntimeError('dynamic crash')",
            ),
            encoding="utf-8",
        )
        run = self.run_default(
            candidate_specs=[str(crashing)], modes=["ceil"]
        )
        self.assertEqual(run["status"], "partial")
        self.assertEqual(run["counts"]["cases_failed"], TESTS)
        self.assertTrue(
            any("dynamic crash" in item["error"] for item in run["failed_groups"])
        )
        records = list((self.root / "out" / "records").glob("*.json"))
        self.assertEqual(len(records), TESTS)  # healthy Attention group survives

    def test_corrupt_shard_records_partial_failure(self) -> None:
        # Corrupt the attention shard payload (bad kind + tiny weight).
        bad = {
            "kind": "linear",
            "schema_version": 1,
            "metadata": {},
            "weight": torch.zeros(8, 8, dtype=torch.bfloat16),
            "calib_activation_list": [],
            "test_activation_list": [],
        }
        shards.atomic_save_torch(self.dataset.dir / "attention" / "0.self_attn.pt", bad)
        run = self.run_default(modes=["ceil"])
        self.assertEqual(run["status"], "partial")
        self.assertEqual(run["counts"]["groups_failed"], 1)
        self.assertEqual(run["counts"]["groups_selected"], 1)
        self.assertTrue(any(g["group"] == "0.self_attn" for g in run["failed_groups"]))
        # the healthy linear group still produced records
        records = list((self.root / "out" / "records").glob("*.json"))
        self.assertEqual(len(records), TESTS)
        # CLI exit code reflects the partial run
        args = [
            "--dataset", str(self.dataset.dir), "--modes", "ceil",
            "--baseline", str(self.base), "--candidate", str(self.cand),
            "--output", str(self.root / "out2"),
        ]
        self.assertEqual(evaluate_real.main(args), 2)

    def test_invalid_baseline_ref_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "neither an existing"):
            self.run_default(baseline_spec="no/such-ref-anywhere")
        # a directory without solution.py also fails
        with self.assertRaises(ValueError):
            self.run_default(baseline_spec=str(self.root))


# ---------------------------------------------------------------------------
# Fake git tags
# ---------------------------------------------------------------------------


class GitRefTest(EvaluateTestBase):
    def _make_git_repo(self) -> Path:
        repo = self.root / "repo"
        repo.mkdir()
        (repo / "solution.py").write_text(_ZERO_SOLUTION, encoding="utf-8")
        commands = (
            ["git", "init", "-q"],
            ["git", "add", "solution.py"],
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
            ["git", "tag", "solution/v000-baseline"],
            ["git", "tag", "solution/v999-fake"],
        )
        for command in commands:
            result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{command}: {result.stderr}")
        return repo

    def test_fake_tag_solution_loading_and_run(self) -> None:
        repo = self._make_git_repo()
        info = evaluate._load_solution("solution/v000-baseline", str(repo))
        self.assertEqual(info["kind"], "git")
        self.assertIsNotNone(info["commit"])
        run = self.run_default(baseline_spec="solution/v000-baseline", repo_root=repo,
                               candidate_specs=["solution/v999-fake"], modes=["ceil"])
        self.assertEqual(run["status"], "ok")
        self.assertEqual(run["baseline"]["kind"], "git")
        self.assertEqual(run["candidates"][0]["kind"], "git")
        record = json.loads(
            next((self.root / "out" / "records").glob("*.json")).read_text()
        )
        self.assertEqual(record["baseline"]["kind"], "git")
        self.assertEqual(record["candidate"]["kind"], "git")
        # unknown ref in the fake repo fails
        with self.assertRaises(ValueError):
            self.run_default(baseline_spec="no/such/tag", repo_root=repo)


# ---------------------------------------------------------------------------
# Resume / atomicity
# ---------------------------------------------------------------------------


class ResumeTest(EvaluateTestBase):
    def test_path_edit_changes_identity_and_does_not_reuse_records(self) -> None:
        first = self.run_default(modes=["ceil"])
        self.assertEqual(first["counts"]["cases_new"], 2 * TESTS)
        first_label = first["candidates"][0]["label"]
        self.cand.write_text(
            self.cand.read_text(encoding="utf-8") + "\n# edited\n",
            encoding="utf-8",
        )
        second = self.run_default(modes=["ceil"])
        self.assertNotEqual(first_label, second["candidates"][0]["label"])
        self.assertEqual(second["counts"]["cases_new"], 2 * TESTS)
        self.assertEqual(second["counts"]["cases_skipped"], 0)
        self.assertEqual(
            len(list((self.root / "out" / "records").glob("*.json"))),
            4 * TESTS,
        )

    def test_rerun_skips_and_partial_resume(self) -> None:
        first = self.run_default()
        self.assertEqual(first["counts"]["cases_new"], 2 * 3 * 1 * TESTS)
        records_dir = self.root / "out" / "records"

        # identical rerun: nothing new, everything skipped
        second = self.run_default()
        self.assertEqual(second["counts"]["cases_new"], 0)
        self.assertEqual(second["counts"]["cases_skipped"], 2 * 3 * 1 * TESTS)
        self.assertEqual(second["counts"]["cases_slots_total"], 2 * 3 * TESTS)
        for stats in second["per_mode"].values():
            self.assertIsNotNone(stats["mean_improvement_percent"])

        # deleting one case record recomputes exactly its (group, mode, candidate)
        # unit: calibration is shared, so all five cases of that unit rerun.
        victim = next(records_dir.glob("*.json"))
        victim.unlink()
        third = self.run_default()
        self.assertEqual(third["counts"]["cases_new"], TESTS)
        self.assertEqual(third["counts"]["cases_skipped"], 2 * 3 * 1 * TESTS - TESTS)

        # --force recomputes everything
        fourth = self.run_default(force=True)
        self.assertEqual(fourth["counts"]["cases_new"], 2 * 3 * 1 * TESTS)
        self.assertEqual(fourth["counts"]["cases_skipped"], 0)

    def test_no_temp_leftovers_and_single_slot_per_case(self) -> None:
        self.run_default()
        out = self.root / "out"
        self.assertEqual(list(out.rglob(".tmp-*")), [])
        # a second run does not add new files (bounded retention)
        count_before = len(list((out / "records").glob("*.json")))
        self.run_default()
        self.assertEqual(len(list((out / "records").glob("*.json"))), count_before)

    def test_baseline_cache_reused_across_candidates(self) -> None:
        first = self.run_default()
        baseline_files = list((self.root / "out" / "baseline").glob("*.json"))
        self.assertEqual(len(baseline_files), 2 * 3)  # per (group, mode)
        # adding a second candidate must not re-run the baseline module: the
        # baseline cache files are untouched and only the new slots appear.
        second = self.run_default(candidate_specs=[str(self.cand), str(self.base)])
        self.assertEqual(len(list((self.root / "out" / "baseline").glob("*.json"))),
                         len(baseline_files))
        self.assertEqual(second["counts"]["cases_new"], 2 * 3 * 1 * TESTS)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


class CliParserTest(unittest.TestCase):
    def test_defaults(self) -> None:
        args = evaluate_real.build_parser().parse_args(["--dataset", "d" * 16])
        self.assertEqual(args.baseline, "solution/v000-baseline")
        self.assertEqual(args.threads, 1)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.candidate, [])
        self.assertEqual(args.modes, [])
        self.assertIsNone(args.limit)
        self.assertFalse(args.force)
        self.assertEqual(args.output, evaluate_real.DEFAULT_OUTPUT_DIR)
        self.assertEqual(args.captures_root, evaluate_real.DEFAULT_CAPTURES_ROOT)

    def test_modes_normalization(self) -> None:
        self.assertEqual(
            evaluate_real._normalize_modes(["ceil,nearest", "stochastic"]),
            ["ceil", "nearest", "stochastic"],
        )
        self.assertEqual(evaluate_real._normalize_modes([]), ["ceil", "nearest", "stochastic"])
        self.assertEqual(evaluate_real._normalize_modes(["nearest", "nearest"]), ["nearest"])
        with self.assertRaises(ValueError):
            evaluate_real._normalize_modes(["bogus"])

    def test_repeatable_candidates_and_filters(self) -> None:
        args = evaluate_real.build_parser().parse_args(
            ["--dataset", "d" * 16, "--candidate", "a.py", "--candidate", "solution/v004-x",
             "--group-filter", "linear", "--group-filter", "role:q_proj,role:o_proj",
             "--limit", "2", "--modes", "ceil"]
        )
        self.assertEqual(args.candidate, ["a.py", "solution/v004-x"])
        self.assertEqual(args.limit, 2)
        predicates = parse_filters(args.group_filter)
        self.assertEqual(predicates, [[("kind", "linear")], [("role", "q_proj"), ("role", "o_proj")]])
        with self.assertRaises(ValueError):
            parse_filters(["layer:x"])

    def test_missing_dataset_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            evaluate_real.build_parser().parse_args([])


# ---------------------------------------------------------------------------
# Subprocess smoke (OOM score + RAM budget)
# ---------------------------------------------------------------------------


class SubprocessSmokeTest(unittest.TestCase):
    def test_subprocess_smoke_with_oom_score_and_ram_budget(self) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(__file__)), "--subprocess-smoke"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("SUBPROCESS_SMOKE_OK", result.stdout)
        self.assertIn("OOM_ADJ_OK", result.stdout)
        self.assertIn("RAM_DELTA_KB", result.stdout)


def _run_subprocess_smoke() -> int:
    import traceback

    oom_ok = evaluate_real.set_oom_score(500)
    try:
        import torch  # noqa: F401  (the unavoidable import)
        base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = TinyDataset(root)
            dataset.add_linear().add_attention()
            dataset.write_manifest()
            base = _write_fake_solution(root / "base.py", "zero")
            cand = _write_fake_solution(root / "cand.py", "copy")
            output = root / "out"
            argv = [
                "--dataset", str(dataset.dir),
                "--modes", "ceil,nearest,stochastic",
                "--seed", "5",
                "--baseline", str(base),
                "--candidate", str(cand),
                "--candidate", str(base),
                "--output", str(output),
                "--threads", "1",
                "--group-filter", "linear",
                "--limit", "1",
            ]
            exit_code = evaluate_real.main(argv)
            assert exit_code == 0, exit_code
            records = list((output / "records").glob("*.json"))
            assert len(records) == 1 * 3 * 2 * TESTS, len(records)
            run_manifest = json.loads(
                next((output / "runs").glob("*.json")).read_text()
            )
            assert run_manifest["status"] == "ok"
            assert run_manifest["counts"]["cases_new"] == 1 * 3 * 2 * TESTS
            assert "transformers" not in sys.modules, (
                "subprocess must not import transformers"
            )
            assert not list((output).rglob(".tmp-*"))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_kb = peak - base_rss
        # tiny shards: peak growth beyond the torch import must stay small
        assert delta_kb < 300 * 1024, f"RAM delta {delta_kb} KB exceeds 300 MB"
        print(f"OOM_ADJ_OK={oom_ok}")
        print(f"RAM_DELTA_KB={delta_kb}")
        print("SUBPROCESS_SMOKE_OK")
        return 0
    except Exception:  # pragma: no cover - diagnostic path
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if "--subprocess-smoke" in sys.argv:
        raise SystemExit(_run_subprocess_smoke())
    unittest.main()
