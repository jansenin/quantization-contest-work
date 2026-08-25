"""Tests for the genuine ModelOpt NVFP4 checkpoint validator.

Pure helpers are exercised with tiny in-memory tensors (no safetensors
required).  Snapshot-level tests use a tiny temporary safetensors fixture and
are skipped when safetensors is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    _HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None
except Exception:
    _HAS_SAFETENSORS = False

from tools import validate_genuine_nvfp4 as validator  # noqa: E402

from tools.validate_genuine_nvfp4 import (  # noqa: E402
    E2M1_TABLE,
    E2M1_TENSOR,
    atomic_write,
    build_contest_pair,
    dequantize_contest,
    dequantize_modelopt,
    evaluate_group,
    render_markdown,
    validate_contest_pair,
    validate_group_inputs,
)


def _nibble(value: float) -> int:
    """Map an E2M1 table value to its nibble (positive zero is canonical)."""
    if value == 0.0:
        return 0
    return {table_value: index for index, table_value in enumerate(E2M1_TABLE)}[value]


def _pack_row(carriers: list[float]) -> torch.Tensor:
    """Pack one row of K carriers into uint8 bytes (low nibble = even K)."""
    if len(carriers) % 2:
        raise ValueError("row must have an even number of carriers")
    bytes_list = []
    for even in range(0, len(carriers), 2):
        low = _nibble(carriers[even])
        high = _nibble(carriers[even + 1])
        bytes_list.append(low | (high << 4))
    return torch.tensor(bytes_list, dtype=torch.uint8)


def _sample_group():
    """A tiny genuine-style ModelOpt group: N=2, K=16, scale1=8/16, g=0.25."""
    carriers = [
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0],
        [1.0, 2.0, 3.0, 4.0, 6.0, 0.5, 0.0, -0.5,
         -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 4.0, 2.0],
    ]
    packed = torch.stack([_pack_row(row) for row in carriers])
    scale1 = torch.tensor([[8.0], [16.0]], dtype=torch.float32)
    global_scale = torch.tensor(0.25, dtype=torch.float32)
    return packed, scale1, global_scale, carriers


def _f8(values):
    """Build a scale tensor; prefer float8_e4m3fn, fall back to float32."""
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        dtype = torch.float32
    if isinstance(values, torch.Tensor):
        return values.detach().to(dtype=dtype)
    return torch.tensor(values, dtype=dtype)


class PackDecodeTests(unittest.TestCase):
    def test_all_table_entries_decode_exactly(self) -> None:
        packed = torch.arange(16, dtype=torch.uint8)
        carriers = validator.decode_packed_nvfp4(packed)
        self.assertEqual(carriers.shape, (32,))
        self.assertEqual(
            carriers[0::2].tolist(),
            [float(E2M1_TABLE[index]) for index in range(16)],
        )
        self.assertEqual(carriers[1::2].tolist(), [0.0] * 16)

        packed = torch.arange(16, dtype=torch.uint8) << 4
        carriers = validator.decode_packed_nvfp4(packed)
        self.assertEqual(
            carriers[1::2].tolist(),
            [float(E2M1_TABLE[index]) for index in range(16)],
        )
        self.assertEqual(carriers[0::2].tolist(), [0.0] * 16)

    def test_low_nibble_is_even_k(self) -> None:
        packed = torch.tensor([0x12, 0x93], dtype=torch.uint8)
        carriers = validator.decode_packed_nvfp4(packed)
        self.assertEqual(carriers.tolist(), [1.0, 0.5, 1.5, -0.5])

    def test_swapped_flag_exchanges_nibbles(self) -> None:
        packed = torch.tensor([0x12, 0x93], dtype=torch.uint8)
        carriers = validator.decode_packed_nvfp4(packed, swapped=True)
        self.assertEqual(carriers.tolist(), [0.5, 1.0, -0.5, 1.5])

    def test_negative_zero_policy(self) -> None:
        # Nibble 8 is NVFP4's sign-encoded zero; it decodes to +0.0 per the
        # table and never produces a negative-zero bit pattern.
        carriers = validator.decode_packed_nvfp4(torch.tensor([0x88], dtype=torch.uint8))
        self.assertEqual(carriers.tolist(), [0.0, 0.0])
        self.assertEqual(
            int((torch.signbit(carriers) & (carriers == 0.0)).sum().item()), 0
        )
        mixed = validator.decode_packed_nvfp4(torch.tensor([0x8F], dtype=torch.uint8))
        self.assertEqual(mixed.tolist(), [-6.0, 0.0])


class InputValidationTests(unittest.TestCase):
    def test_rejects_non_uint8_packed(self) -> None:
        with self.assertRaises(ValueError):
            validator.decode_packed_nvfp4(torch.zeros(4, dtype=torch.float32))
        with self.assertRaises(ValueError):
            validator.decode_packed_nvfp4(torch.zeros(4, dtype=torch.int32))

    def test_rejects_empty_packed(self) -> None:
        with self.assertRaises(ValueError):
            validator.decode_packed_nvfp4(torch.tensor([], dtype=torch.uint8))

    def test_rejects_k_not_multiple_of_16(self) -> None:
        packed = torch.zeros(4, dtype=torch.uint8)  # K = 8
        scale = torch.zeros(1, dtype=torch.float32)
        with self.assertRaises(ValueError):
            validate_group_inputs(packed, scale)

    def test_rejects_scale_shape_mismatch(self) -> None:
        packed = torch.zeros((2, 8), dtype=torch.uint8)  # K = 16
        with self.assertRaises(ValueError):
            validate_group_inputs(packed, torch.zeros((2, 2), dtype=torch.float32))
        with self.assertRaises(ValueError):
            validate_group_inputs(packed, torch.zeros(1, dtype=torch.float32))

    def test_rejects_non_float_weight_scale(self) -> None:
        packed = torch.zeros((2, 8), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            validate_group_inputs(packed, torch.zeros((2, 1), dtype=torch.int32))

    def test_rejects_nonfinite_weight_scale(self) -> None:
        packed = torch.zeros((2, 8), dtype=torch.uint8)
        for value in (float("nan"), float("inf"), float("-inf")):
            scale = torch.zeros((2, 1), dtype=torch.float32)
            scale[1, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_group_inputs(packed, scale)

    def test_rejects_non_scalar_global_scale(self) -> None:
        packed = torch.zeros((2, 8), dtype=torch.uint8)
        scale = torch.zeros((2, 1), dtype=torch.float32)
        with self.assertRaises(ValueError):
            validate_group_inputs(packed, scale, torch.zeros(2, dtype=torch.float32))
        with self.assertRaises(ValueError):
            validate_group_inputs(packed, scale, torch.tensor(1, dtype=torch.int64))

    def test_rejects_nonfinite_global_scale(self) -> None:
        packed = torch.zeros((2, 8), dtype=torch.uint8)
        scale = torch.zeros((2, 1), dtype=torch.float32)
        with self.assertRaises(ValueError):
            validate_group_inputs(packed, scale, torch.tensor(float("nan")))


class ContestPairTests(unittest.TestCase):
    def test_global_scale_is_folded_into_per16_scale(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        carrier, scale = build_contest_pair(packed, scale1, global_scale)
        self.assertEqual(carrier.dtype, torch.bfloat16)
        self.assertEqual(scale.dtype, torch.bfloat16)
        self.assertEqual(tuple(carrier.shape), (2, 16))
        self.assertEqual(tuple(scale.shape), (2, 1))
        expected = (scale1 * 0.25).to(torch.bfloat16)
        self.assertTrue(torch.equal(scale, expected))

    def test_global_scale_necessity(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        parent = dequantize_modelopt(packed, scale1, global_scale)
        result = evaluate_group(packed, scale1, global_scale, parent=parent, name="t")
        metrics = result["metrics"]
        self.assertEqual(metrics["best_mode"], "canonical")
        self.assertTrue(metrics["is_genuine"])
        self.assertLess(metrics["canonical"]["normalized_mse"], 1e-9)
        self.assertGreater(metrics["no_global"]["normalized_mse"], 1.0)
        self.assertLess(metrics["canonical"]["normalized_mse"], metrics["no_global"]["normalized_mse"])

    def test_decode_closeness_and_fold_agreement(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        parent = dequantize_modelopt(packed, scale1, global_scale)
        carrier, scale = build_contest_pair(packed, scale1, global_scale)
        # The contest BF16 pair reproduces the ModelOpt BF16 dequant exactly
        # for these power-of-two values.
        self.assertTrue(torch.equal(dequantize_contest(carrier, scale), parent))
        result = evaluate_group(packed, scale1, global_scale, parent=parent, name="t")
        self.assertEqual(result["contest_fold_agreement"]["exact_fraction"], 1.0)
        self.assertEqual(result["contest_fold_agreement"]["max_abs_diff"], 0.0)

    def test_contest_pair_legality(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        carrier, scale = build_contest_pair(packed, scale1, global_scale)
        legality = validate_contest_pair(carrier, scale)
        self.assertTrue(legality["carrier_legal"])
        self.assertEqual(legality["illegal_carrier_count"], 0)
        self.assertEqual(legality["negative_zero_count"], 0)
        self.assertTrue(legality["scale_shape_ok"])

        bad = carrier.clone()
        bad[0, 0] = torch.tensor(0.25, dtype=torch.bfloat16)
        self.assertFalse(validate_contest_pair(bad, scale)["carrier_legal"])

    def test_contest_pair_shape_rejection(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        carrier, scale = build_contest_pair(packed, scale1, global_scale)
        with self.assertRaises(ValueError):
            validate_contest_pair(carrier, torch.zeros((2, 2), dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            validate_contest_pair(carrier.float(), scale)

    def test_swapped_discrimination(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        parent = dequantize_modelopt(packed, scale1, global_scale)
        result = evaluate_group(packed, scale1, global_scale, parent=parent, name="t")
        metrics = result["metrics"]
        self.assertLess(
            metrics["canonical"]["normalized_mse"],
            metrics["swapped"]["normalized_mse"],
        )
        self.assertEqual(result["legality"]["illegal_carrier_count"], 0)

    def test_bad_scale_alignment_fails_absolute_fit_gate(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        packed = torch.cat((packed, packed), dim=-1)
        scale1 = torch.tensor([[8.0, 24.0], [16.0, 40.0]])
        parent = dequantize_modelopt(packed, scale1, global_scale)
        shifted = scale1.flip(-1)

        result = evaluate_group(
            packed, shifted, global_scale, parent=parent, name="misaligned"
        )

        self.assertEqual(result["metrics"]["best_mode"], "canonical")
        self.assertFalse(result["metrics"]["is_genuine"])
        self.assertGreater(
            result["metrics"]["canonical"]["normalized_mse"],
            validator.MAX_CANONICAL_NMSE,
        )

    def test_missing_global_scale_is_unit_factor(self) -> None:
        packed, scale1, _, _ = _sample_group()
        parent = dequantize_modelopt(packed, scale1, None)
        result = evaluate_group(packed, scale1, None, parent=parent, name="t")
        self.assertFalse(result["global_scale_present"])
        self.assertIsNone(result["global_scale"])
        # With g == 1 the canonical and no-global hypotheses coincide; the
        # canonical tie-break wins.
        self.assertTrue(result["metrics"]["is_genuine"])
        self.assertEqual(result["metrics"]["best_mode"], "canonical")
        self.assertEqual(
            result["metrics"]["canonical"]["normalized_mse"],
            result["metrics"]["no_global"]["normalized_mse"],
        )

    def test_one_dimensional_group(self) -> None:
        packed, scale1, global_scale, _ = _sample_group()
        parent = dequantize_modelopt(packed, scale1, global_scale)
        result = evaluate_group(
            packed[0], scale1[0], global_scale, parent=parent[0], name="1d"
        )
        self.assertTrue(result["metrics"]["is_genuine"])
        self.assertEqual(tuple(result["logical_shape"]), (16,))
        self.assertEqual(tuple(result["scale_shape"]), (1,))


class RendererAndOutputTests(unittest.TestCase):
    def _sample_report(self) -> dict:
        packed, scale1, global_scale, _ = _sample_group()
        parent = dequantize_modelopt(packed, scale1, global_scale)
        tensor = evaluate_group(
            packed, scale1, global_scale, parent=parent,
            name="model.layers.0.self_attn.v_proj.weight",
        )
        summary = validator._build_summary([tensor], [], [tensor["name"]])
        return {
            "format": validator.REPORT_FORMAT,
            "provenance": {
                "nvfp4_snapshot": {
                    "kind": "single",
                    "root": "/tmp/fake",
                    "files": [{
                        "path": "/tmp/fake/model.safetensors",
                        "size_bytes": 100,
                        "mtime_iso": "2026-01-01T00:00:00+00:00",
                    }],
                    "total_size_bytes": 100,
                },
                "bf16_snapshot": None,
                "producer": None,
                "tool": {
                    "name": "tools/validate_genuine_nvfp4.py",
                    "report_format": validator.REPORT_FORMAT,
                    "torch": "0.0.0",
                    "safetensors": None,
                    "oom_score_adj": 500,
                    "timestamp_utc": "2026-01-01T00:00:00+00:00",
                },
            },
            "inventory": validator._inventory_summary({}),
            "tensors": [tensor],
            "errors": [],
            "summary": summary,
        }

    def test_render_markdown_is_deterministic(self) -> None:
        report = self._sample_report()
        first = render_markdown(report)
        second = render_markdown(report)
        self.assertEqual(first, second)
        self.assertIn("## Results", first)
        self.assertIn("## Summary", first)
        self.assertIn("## Method and caveats", first)
        self.assertIn("model.layers.0.self_attn.v_proj.weight", first)
        self.assertIn("canonical", first)

    def test_report_exit_code_requires_strong_available_parent_fits(self) -> None:
        base = {
            "validated_tensor_count": 2,
            "tensor_errors": 0,
            "parent_missing_count": 0,
            "genuine_count": 2,
        }
        self.assertEqual(validator._report_exit_code(base), 0)
        self.assertEqual(
            validator._report_exit_code({**base, "genuine_count": 1}), 1
        )
        self.assertEqual(
            validator._report_exit_code(
                {**base, "parent_missing_count": 2, "genuine_count": 0}
            ),
            0,
        )
        self.assertEqual(
            validator._report_exit_code({**base, "tensor_errors": 1}), 1
        )

    def test_json_serialization_is_deterministic(self) -> None:
        report = self._sample_report()
        first = json.dumps(report, indent=2, sort_keys=True)
        second = json.dumps(report, indent=2, sort_keys=True)
        self.assertEqual(first, second)

    def test_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.md"
            atomic_write(target, "hello")
            self.assertEqual(target.read_text(encoding="utf-8"), "hello")
            # No leftover temp files.
            self.assertEqual(list(Path(directory).iterdir()), [target])
            atomic_write(target, "world")
            self.assertEqual(target.read_text(encoding="utf-8"), "world")

    def test_atomic_write_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "out.json"
            atomic_write(target, "{}")
            self.assertEqual(target.read_text(encoding="utf-8"), "{}")


@unittest.skipUnless(_HAS_SAFETENSORS, "safetensors not installed")
class SnapshotFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.nvfp4_dir = self.root / "nvfp4"
        self.parent_dir = self.root / "parent"
        self.nvfp4_dir.mkdir()
        self.parent_dir.mkdir()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _write_group(self, tensors: dict, directory: Path, filename: str) -> None:
        from safetensors.torch import save_file

        save_file(tensors, str(directory / filename))

    def _nvfp4_tensors(self) -> dict:
        v_packed, v_scale1, v_g, _ = _sample_group()
        g_packed, g_scale1, g_g, _ = _sample_group()
        return {
            "model.layers.0.self_attn.v_proj.weight": v_packed,
            "model.layers.0.self_attn.v_proj.weight_scale": _f8(v_scale1),
            "model.layers.0.self_attn.v_proj.weight_scale_2": v_g,
            "model.layers.0.self_attn.v_proj.input_scale": torch.tensor(
                0.5, dtype=torch.float32
            ),
            "model.layers.0.mlp.gate_proj.weight": g_packed,
            "model.layers.0.mlp.gate_proj.weight_scale": _f8(g_scale1),
            "model.layers.0.mlp.gate_proj.weight_scale_2": g_g,
            "model.embed_tokens.weight": torch.zeros((3, 16), dtype=torch.bfloat16),
        }

    def _parent_tensors(self) -> dict:
        v_packed, v_scale1, v_g, _ = _sample_group()
        g_packed, g_scale1, g_g, _ = _sample_group()
        return {
            "model.layers.0.self_attn.v_proj.weight": dequantize_modelopt(
                v_packed, v_scale1, v_g
            ),
            "model.layers.0.mlp.gate_proj.weight": dequantize_modelopt(
                g_packed, g_scale1, g_g
            ),
            "model.embed_tokens.weight": torch.zeros((3, 16), dtype=torch.bfloat16),
        }

    def _write_single_file(self) -> None:
        self._write_group(self._nvfp4_tensors(), self.nvfp4_dir, "model.safetensors")
        self._write_group(self._parent_tensors(), self.parent_dir, "model.safetensors")

    def test_single_file_default_selection_and_genuine(self) -> None:
        self._write_single_file()
        report = validator.validate_snapshot(str(self.nvfp4_dir), str(self.parent_dir))
        self.assertEqual(report["summary"]["validated_tensor_count"], 1)
        self.assertEqual(report["summary"]["genuine_count"], 1)
        self.assertEqual(report["summary"]["tensor_errors"], 0)
        tensor = report["tensors"][0]
        self.assertEqual(
            tensor["name"], "model.layers.0.self_attn.v_proj.weight"
        )
        self.assertEqual(tensor["role"], "v_proj")
        self.assertTrue(tensor["metrics"]["is_genuine"])
        self.assertTrue(tensor["global_scale_present"])
        self.assertLess(tensor["metrics"]["canonical"]["normalized_mse"], 1e-9)
        self.assertLess(
            tensor["metrics"]["canonical"]["normalized_mse"],
            tensor["metrics"]["swapped"]["normalized_mse"],
        )
        self.assertEqual(tensor["legality"]["illegal_carrier_count"], 0)
        self.assertEqual(tensor["legality"]["e4m3_scale_fraction"], 1.0)
        self.assertEqual(tensor["legality"]["nibble8_zero_count"], 0)
        self.assertEqual(tensor["contest_pair"]["carrier_shape"], [2, 16])

    def test_indexed_shards_across_group(self) -> None:
        # The weight lives in shard 1 while its scale/scale_2 live in shard 2.
        v_packed, v_scale1, v_g, _ = _sample_group()
        shard_one = {
            "model.layers.0.self_attn.v_proj.weight": v_packed,
            "model.embed_tokens.weight": torch.zeros((3, 16), dtype=torch.bfloat16),
        }
        shard_two = {
            "model.layers.0.self_attn.v_proj.weight_scale": _f8(v_scale1),
            "model.layers.0.self_attn.v_proj.weight_scale_2": v_g,
        }
        self._write_group(shard_one, self.nvfp4_dir, "model-00001-of-00002.safetensors")
        self._write_group(shard_two, self.nvfp4_dir, "model-00002-of-00002.safetensors")
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                name: (
                    "model-00001-of-00002.safetensors"
                    if name in shard_one
                    else "model-00002-of-00002.safetensors"
                )
                for name in list(shard_one) + list(shard_two)
            },
        }
        (self.nvfp4_dir / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        self._write_group(self._parent_tensors(), self.parent_dir, "model.safetensors")

        report = validator.validate_snapshot(str(self.nvfp4_dir), str(self.parent_dir))
        self.assertEqual(
            report["provenance"]["nvfp4_snapshot"]["kind"], "shards"
        )
        self.assertEqual(report["summary"]["validated_tensor_count"], 1)
        self.assertTrue(report["tensors"][0]["metrics"]["is_genuine"])
        self.assertEqual(
            report["tensors"][0]["legality"]["illegal_carrier_count"], 0
        )

    def test_custom_tensor_selector(self) -> None:
        self._write_single_file()
        report = validator.validate_snapshot(
            str(self.nvfp4_dir), str(self.parent_dir), tensor_selectors=["gate_proj"]
        )
        self.assertEqual(report["summary"]["validated_tensor_count"], 1)
        self.assertEqual(
            report["tensors"][0]["name"],
            "model.layers.0.mlp.gate_proj.weight",
        )

    def test_glob_selector(self) -> None:
        self._write_single_file()
        report = validator.validate_snapshot(
            str(self.nvfp4_dir),
            str(self.parent_dir),
            tensor_selectors=["*.v_proj.weight"],
        )
        self.assertEqual(report["summary"]["validated_tensor_count"], 1)
        self.assertTrue(report["tensors"][0]["metrics"]["is_genuine"])

    def test_chunked_matches_whole_tensor_path(self) -> None:
        packed = torch.randint(0, 256, (64, 8), dtype=torch.uint8)
        scale1 = torch.full((64, 1), 8.0)
        global_scale = torch.tensor(0.25, dtype=torch.float32)
        name = "model.layers.0.self_attn.o_proj.weight"
        tensors = {
            name: packed,
            "model.layers.0.self_attn.o_proj.weight_scale": _f8(scale1),
            "model.layers.0.self_attn.o_proj.weight_scale_2": global_scale,
        }
        parent_tensors = {
            name: dequantize_modelopt(packed, scale1, global_scale),
        }
        self._write_group(tensors, self.nvfp4_dir, "model.safetensors")
        self._write_group(parent_tensors, self.parent_dir, "model.safetensors")

        chunked = validator.validate_snapshot(
            str(self.nvfp4_dir), str(self.parent_dir), chunk_rows=16
        )
        whole = validator.validate_snapshot(
            str(self.nvfp4_dir), str(self.parent_dir), chunk_rows=10_000
        )
        self.assertEqual(chunked["summary"]["validated_tensor_count"], 1)
        self.assertEqual(
            chunked["tensors"][0]["metrics"]["canonical"],
            whole["tensors"][0]["metrics"]["canonical"],
        )
        self.assertEqual(
            chunked["tensors"][0]["legality"],
            whole["tensors"][0]["legality"],
        )

    def test_missing_parent_tensor_yields_structural_only(self) -> None:
        self._write_group(
            self._nvfp4_tensors(), self.nvfp4_dir, "model.safetensors"
        )
        report = validator.validate_snapshot(str(self.nvfp4_dir))
        tensor = report["tensors"][0]
        self.assertFalse(tensor["metrics"]["parent_available"])
        self.assertIsNone(tensor["metrics"]["is_genuine"])
        self.assertEqual(tensor["legality"]["illegal_carrier_count"], 0)
        self.assertIn("structural validation", report["summary"]["conclusion"])


if __name__ == "__main__":
    unittest.main()
