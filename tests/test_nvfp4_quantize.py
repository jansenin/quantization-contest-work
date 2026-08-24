"""Tests for ``tools/nvfp4_quantize.py``.

Covers, with stdlib ``unittest`` only (no pytest, no file I/O, tiny tensors
only -- memory stays far below 300 MB):

* E4M3FN grid endpoints/count and E2M1 carrier grid, reusing the fingerprint
  tool's grid as the single source of truth;
* output legality: shapes, BF16 dtypes, exact E2M1/E4M3FN value membership,
  and error paths (divisibility, dtypes, finiteness, modes, seed, offsets,
  chunk sizes);
* every scale mode around adjacent E4M3FN grid values (on-grid, sub-grid,
  midpoint ties, below-min, overflow saturation, nearest clipping);
* bounded chunked processing: bit-identical outputs for tiny vs. huge
  ``chunk_blocks`` in every mode, including stochastic draws with caller
  block offsets, plus chunk-size validation;
* stochastic reproducibility: seed-only default identity (independent of the
  tensor object), cross-process reproducibility, seed/tensor-id sensitivity,
  chunk-order invariance, global-RNG independence, and int64 index-range
  boundaries for ``block_offset``;
* signed-zero policy: negative sources rounding to zero keep ``-0.0`` while
  exact source zeros normalize to ``+0.0``;
* all-zero blocks, carrier tie rounding, clipping, and dequantization
  roundtrips (including exactness on the carrier/scale lattice);
* fingerprint integration limited to structural guarantees (legal carriers,
  exact E4M3FN scales, exact BF16 products): the true ceil theorem
  ``scale == ceil_E4M3(source_max / 6)`` is verified against known source
  maxima, with a subnormal counterexample showing the *stored-value* fixed
  point is not universal.

Run from the workspace root:

    python3 -m unittest discover -s tests -v

or directly:

    python3 tests/test_nvfp4_quantize.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import fingerprint_nvfp4 as fingerprint  # noqa: E402
from tools import nvfp4_quantize as quant  # noqa: E402
from tools import reference_ops as ref  # noqa: E402


def block(value: float) -> torch.Tensor:
    """One 16-wide block whose sole significant element is ``value``."""
    elements = [float(value)] + [0.0] * 15
    return torch.tensor(elements, dtype=torch.float32).reshape(1, 16)


class GridTests(unittest.TestCase):
    def test_e4m3_grid_endpoints_and_count(self) -> None:
        grid = quant.E4M3_POSITIVE_VALUES
        self.assertEqual(len(grid), 126)
        self.assertEqual(grid[0], 2.0 ** -9)
        self.assertEqual(grid[-1], 448.0)
        self.assertTrue(all(grid[i] < grid[i + 1] for i in range(len(grid) - 1)))
        self.assertIn(1.0, grid)
        self.assertIn(1.125, grid)
        self.assertNotIn(480.0, grid)
        self.assertNotIn(0.0, grid)
        self.assertEqual(quant.E4M3_MIN, 2.0 ** -9)
        self.assertEqual(quant.E4M3_MAX, 448.0)

    def test_grid_matches_fingerprint_helper(self) -> None:
        self.assertEqual(
            tuple(quant.E4M3_POSITIVE_VALUES),
            tuple(fingerprint.positive_e4m3fn_values()),
        )
        # Reuse decision: the quantizer imports the fingerprint helper, so the
        # two modules must agree bit-for-bit.
        self.assertTrue(all(v in fingerprint.E4M3_SET for v in quant.E4M3_POSITIVE_VALUES))

    def test_e2m1_carrier_grid(self) -> None:
        self.assertEqual(quant.E2M1_MAGNITUDES, (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
        self.assertEqual(quant.E2M1_MAX, 6.0)
        self.assertEqual(len(fingerprint.NVFP4_CARRIERS), 15)
        signed = set(fingerprint.NVFP4_CARRIERS)
        for magnitude in quant.E2M1_MAGNITUDES:
            self.assertIn(magnitude, signed)
            if magnitude:
                self.assertIn(-magnitude, signed)
        self.assertEqual(quant.SATURATION, 2688.0)


class LegalityTests(unittest.TestCase):
    def test_basic_2d_shapes_and_dtypes(self) -> None:
        value = torch.randn(2, 32, dtype=torch.float32)
        result = quant.quantize_nvfp4(value, scale_mode="ceil")
        q, s = result
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(q.dtype, torch.bfloat16)
        self.assertEqual(s.dtype, torch.bfloat16)
        self.assertEqual(tuple(q.shape), (2, 32))
        self.assertEqual(tuple(s.shape), (2, 2))
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.isfinite(s).all())
        self.assertTrue(all(v in fingerprint.NVFP4_CARRIERS for v in q.unique().tolist()))
        self.assertTrue(all(v in fingerprint.E4M3_SET for v in s.unique().tolist()))

    def test_1d_and_3d_shapes(self) -> None:
        q1, s1 = quant.quantize_nvfp4(torch.randn(16), scale_mode="nearest")
        self.assertEqual(tuple(s1.shape), (1,))
        q3, s3 = quant.quantize_nvfp4(torch.randn(2, 3, 16), scale_mode="stochastic")
        self.assertEqual(tuple(s3.shape), (2, 3, 1))
        self.assertEqual(tuple(q3.shape), (2, 3, 16))

    def test_all_floating_input_dtypes(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            value = torch.tensor([[3.0, 0.5, -1.5] + [0.0] * 13], dtype=dtype)
            q, s = quant.quantize_nvfp4(value)
            self.assertEqual(q.dtype, torch.bfloat16, dtype)
            self.assertEqual(s.dtype, torch.bfloat16, dtype)

    def test_error_paths(self) -> None:
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4("not a tensor")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16, dtype=torch.int64))
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16, dtype=torch.complex64))
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16, dtype=torch.bool))
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(3, 7))
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(3, 0))
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(4, 16, dtype=torch.float32) * float("nan"))
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(4, 16, dtype=torch.float32) * float("inf"))
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(16), scale_mode="round-half-up")
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(16), block_size=0)
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(16), block_size=7)  # 16 % 7 != 0
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16), seed=True)
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16), seed=1.5)
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16), tensor_id=3.5)
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16), tensor_id=True)  # bool rejected
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.ones(16), block_offset=-1.0)
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.ones(16), block_offset=-1)

    def test_nonfinite_in_late_chunk_raises(self) -> None:
        # Finiteness is validated per chunk (bounded boolean, no whole-input
        # temporary): a NaN in the final chunk must raise even with a tiny
        # chunk size, and no partial output is ever returned (raising discards
        # the function-local output tensors).
        value = torch.zeros(3, 16, dtype=torch.float32)
        value[0, 0] = 3.0
        value[2, 15] = float("nan")  # last chunk when chunk_blocks == 1
        for chunk_blocks in (1, 2, 1 << 30):
            with self.assertRaises(ValueError):
                quant.quantize_nvfp4(value, chunk_blocks=chunk_blocks)

    def test_custom_block_size(self) -> None:
        q, s = quant.quantize_nvfp4(torch.randn(4, 32), block_size=32)
        self.assertEqual(tuple(s.shape), (4, 1))


class ScaleModeTests(unittest.TestCase):
    """Exact dyadic targets around adjacent E4M3FN grid values.

    Grid values near 1.0: ``..., 1.0, 1.125, 1.25, 1.375, ...``.  A block's
    target is ``max_abs / 6``; ``max_abs = 6 * target`` is float32-exact for
    every target below, so ``max_abs / 6`` recovers the target exactly.
    """

    ADJACENT = (1.0, 1.125)
    MIDPOINT = 1.0625
    BELOW_MIDPOINT = 1.03125  # nearer 1.0
    ABOVE_MIDPOINT = 1.09375  # nearer 1.125
    SPANNING_PAIR = (1.875, 2.0)  # adjacent across a power-of-two boundary
    SPANNING_MIDPOINT = 1.9375

    def _scale_of(self, max_abs: float, mode: str, **kwargs) -> float:
        q, s = quant.quantize_nvfp4(block(max_abs), scale_mode=mode, **kwargs)
        return float(s.flatten()[0])

    def test_ceil_mode_adjacent_grid(self) -> None:
        lo, hi = self.ADJACENT
        self.assertEqual(self._scale_of(6.0 * lo, "ceil"), lo)  # on-grid
        self.assertEqual(self._scale_of(6.0 * self.MIDPOINT, "ceil"), hi)
        self.assertEqual(self._scale_of(6.0 * self.BELOW_MIDPOINT, "ceil"), hi)
        self.assertEqual(self._scale_of(6.0 * self.ABOVE_MIDPOINT, "ceil"), hi)
        self.assertEqual(self._scale_of(6.0 * hi, "ceil"), hi)  # on-grid

    def test_nearest_mode_adjacent_grid(self) -> None:
        lo, hi = self.ADJACENT
        self.assertEqual(self._scale_of(6.0 * lo, "nearest"), lo)
        self.assertEqual(self._scale_of(6.0 * self.MIDPOINT, "nearest"), hi)  # tie up
        self.assertEqual(self._scale_of(6.0 * self.BELOW_MIDPOINT, "nearest"), lo)
        self.assertEqual(self._scale_of(6.0 * self.ABOVE_MIDPOINT, "nearest"), hi)
        self.assertEqual(self._scale_of(6.0 * hi, "nearest"), hi)

    def test_stochastic_mode_on_grid_target_is_deterministic(self) -> None:
        lo, hi = self.ADJACENT
        for seed in range(6):
            self.assertEqual(self._scale_of(6.0 * lo, "stochastic", seed=seed), lo)
            self.assertEqual(self._scale_of(6.0 * hi, "stochastic", seed=seed), hi)

    def test_spanning_power_of_two_pair(self) -> None:
        lo, hi = self.SPANNING_PAIR
        target = self.SPANNING_MIDPOINT
        for mode in ("ceil", "nearest"):
            self.assertEqual(self._scale_of(6.0 * target, mode), hi)
        # midpoint: probability of the upper value is exactly 1/2
        self.assertIn(
            self._scale_of(6.0 * target, "stochastic", seed=0), (lo, hi)
        )

    def test_below_minimum_target_saturates_to_e4m3_min(self) -> None:
        for mode in ("ceil", "nearest"):
            self.assertEqual(self._scale_of(1e-9, mode), quant.E4M3_MIN)
            self.assertEqual(self._scale_of(0.0, mode), quant.E4M3_MIN)
        # stochastic endpoint must be reached for every seed (no NaN leakage)
        for seed in range(8):
            self.assertEqual(
                self._scale_of(1e-9, "stochastic", seed=seed), quant.E4M3_MIN
            )
            self.assertEqual(
                self._scale_of(0.0, "stochastic", seed=seed), quant.E4M3_MIN
            )

    def test_overflow_saturates_to_e4m3_max(self) -> None:
        for mode in ("ceil", "nearest"):
            self.assertEqual(self._scale_of(3000.0, mode), 448.0)
            self.assertEqual(self._scale_of(quant.SATURATION, mode), 448.0)
        for seed in range(8):
            self.assertEqual(
                self._scale_of(3000.0, "stochastic", seed=seed), 448.0
            )
            self.assertEqual(
                self._scale_of(quant.SATURATION, "stochastic", seed=seed), 448.0
            )

    def test_nearest_can_choose_lower_scale_and_clip(self) -> None:
        # target 1.03125 is closer to 1.0 than to 1.125: ceil -> 1.125,
        # nearest -> 1.0, which is below max_abs/6, so the max element clips.
        max_abs = 6.0 * self.BELOW_MIDPOINT
        value = block(max_abs)
        q_ceil, s_ceil = quant.quantize_nvfp4(value, "ceil")
        q_near, s_near = quant.quantize_nvfp4(value, "nearest")
        self.assertEqual(float(s_ceil.flatten()[0]), 1.125)
        self.assertEqual(float(s_near.flatten()[0]), 1.0)
        out_near = ref.dequantize_nvfp4(q_near, s_near)
        self.assertEqual(float(out_near.flatten()[0]), 6.0)  # clipped < 6.1875
        out_ceil = ref.dequantize_nvfp4(q_ceil, s_ceil)
        self.assertEqual(float(out_ceil.flatten()[0]), 6.75)  # no clip

    def test_overflow_reconstruction_is_finite_and_clipped(self) -> None:
        q, s = quant.quantize_nvfp4(block(3000.0), scale_mode="ceil")
        out = ref.dequantize_nvfp4(q, s)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(float(s.flatten()[0]), 448.0)
        self.assertEqual(float(out.flatten()[0]), 2688.0)  # carrier 6 * 448

    def test_stochastic_interpolation_probabilities(self) -> None:
        # 512 identical blocks, each with target at the midpoint of (1.0, 1.125)
        # -> P(upper) = 1/2, or at a 3:1 ratio -> P(upper) = 3/4 / 1/4.
        n = 512
        cases = {
            0.5: self.MIDPOINT,
            0.75: self.ABOVE_MIDPOINT,
            0.25: self.BELOW_MIDPOINT,
        }
        for probability, target in cases.items():
            # probability is P(scale == 1.125), the grid-upper value.
            value = torch.tensor(
                [[6.0 * target] + [0.0] * 15] * n, dtype=torch.float32
            )
            q, s = quant.quantize_nvfp4(value, scale_mode="stochastic", seed=7)
            fraction = float((s.flatten() == 1.125).double().mean().item())
            self.assertGreater(fraction, probability - 0.15, (probability, fraction))
            self.assertLess(fraction, probability + 0.15, (probability, fraction))

    def test_modes_agree_on_exact_grid_targets(self) -> None:
        for max_abs in (6.0, 6.0 * 1.125, 6.0 * 0.5, 6.0 * 0.28125, 6.0 * 448.0):
            scales = {
                mode: self._scale_of(max_abs, mode, seed=1)
                for mode in ("ceil", "nearest", "stochastic")
            }
            self.assertEqual(len(set(scales.values())), 1, scales)


class StochasticInvarianceTests(unittest.TestCase):
    def _make(self, n_blocks: int = 256) -> torch.Tensor:
        # Every block has max_abs 6.375 -> target 1.0625, midpoint tie of
        # (1.0, 1.125): p(upper) = 1/2, so scales are draw-sensitive.  Carriers
        # alone cannot detect the draw here (both scales map 6.375 to carrier
        # 6), so sensitivity tests must compare the scale tensor as well.
        return torch.tensor(
            [[6.375] + [0.0] * 15] * n_blocks, dtype=torch.float32
        )

    def test_reproducible_across_calls(self) -> None:
        value = self._make()
        q1, s1 = quant.quantize_nvfp4(value, "stochastic", seed=3, tensor_id="t")
        q2, s2 = quant.quantize_nvfp4(value, "stochastic", seed=3, tensor_id="t")
        self.assertTrue(torch.equal(q1, q2))
        self.assertTrue(torch.equal(s1, s2))

    def test_independent_of_global_rng_state(self) -> None:
        value = self._make()
        torch.manual_seed(1234)
        q1, s1 = quant.quantize_nvfp4(value, "stochastic", seed=3, tensor_id="t")
        torch.manual_seed(98765)
        q2, s2 = quant.quantize_nvfp4(value, "stochastic", seed=3, tensor_id="t")
        self.assertTrue(torch.equal(q1, q2))
        self.assertTrue(torch.equal(s1, s2))

    def test_seed_sensitivity(self) -> None:
        value = self._make()
        q1, s1 = quant.quantize_nvfp4(value, "stochastic", seed=0, tensor_id="t")
        q2, s2 = quant.quantize_nvfp4(value, "stochastic", seed=1, tensor_id="t")
        self.assertFalse(torch.equal(s1, s2))
        self.assertTrue(torch.equal(q1, q2))  # carriers coincide here by design

    def test_tensor_id_sensitivity(self) -> None:
        value = self._make()
        q1, s1 = quant.quantize_nvfp4(value, "stochastic", seed=0, tensor_id="a")
        q2, s2 = quant.quantize_nvfp4(value, "stochastic", seed=0, tensor_id="b")
        self.assertFalse(torch.equal(s1, s2))
        self.assertTrue(torch.equal(q1, q2))

    def test_default_identity_is_seed_only_and_reproducible(self) -> None:
        # The default identity is the empty string: the draw sequence is a
        # pure function of the seed, so distinct tensor objects with the same
        # seed must produce identical outputs (correlated by design), and a
        # different seed decorrelates them.
        value = self._make()
        q1, s1 = quant.quantize_nvfp4(value, "stochastic", seed=0)
        q2, s2 = quant.quantize_nvfp4(value.clone(), "stochastic", seed=0)
        self.assertTrue(torch.equal(q1, q2))
        self.assertTrue(torch.equal(s1, s2))
        q3, s3 = quant.quantize_nvfp4(value, "stochastic", seed=1)
        self.assertFalse(torch.equal(s1, s3))

    def test_cross_process_reproducibility(self) -> None:
        # Same seed + explicit tensor_id must reproduce bit-for-bit in fresh
        # interpreter processes (no id()/hash() in the draw path).
        import os
        import subprocess

        script = (
            "import sys, torch\n"
            "sys.path.insert(0, %r)\n"
            "from tools import nvfp4_quantize as quant\n"
            "x = torch.tensor([[6.375] + [0.0] * 15] * 32, dtype=torch.float32)\n"
            "q, s = quant.quantize_nvfp4(x, 'stochastic', seed=9, tensor_id='proc')\n"
            "print(q.view(torch.int16).tolist())\n"
            "print(s.view(torch.int16).tolist())\n"
        ) % str(ROOT)
        env = dict(os.environ)
        out1 = subprocess.check_output([sys.executable, "-c", script], env=env)
        out2 = subprocess.check_output([sys.executable, "-c", script], env=env)
        self.assertEqual(out1, out2)

        # and both must match the in-process result
        value = torch.tensor(
            [[6.375] + [0.0] * 15] * 32, dtype=torch.float32
        )
        q, s = quant.quantize_nvfp4(value, "stochastic", seed=9, tensor_id="proc")
        expected = (
            str(q.view(torch.int16).tolist()) + "\n"
            + str(s.view(torch.int16).tolist()) + "\n"
        ).encode("utf-8")
        self.assertEqual(out1, expected)

    def test_chunk_order_invariance_with_explicit_tensor_id(self) -> None:
        value = torch.randn(4, 64, dtype=torch.float32)
        full = quant.quantize_nvfp4(
            value, "stochastic", seed=5, tensor_id="chunky"
        )
        blocks_per_chunk = (2 * 64) // 16  # 8
        parts = [value[:2], value[2:]]
        chunks = [
            quant.quantize_nvfp4(
                part,
                "stochastic",
                seed=5,
                tensor_id="chunky",
                block_offset=offset,
            )
            for part, offset in zip(parts, (0, blocks_per_chunk))
        ]
        merged = (torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks]))
        self.assertTrue(torch.equal(full[0], merged[0]))
        self.assertTrue(torch.equal(full[1], merged[1]))

    def test_order_of_chunk_processing_does_not_matter(self) -> None:
        value = torch.randn(3, 64, dtype=torch.float32)
        # 3 chunks of 1 row each; the calls happen in reverse order but the
        # result must be reassembled by original block offset.
        offsets = [i * 4 for i in range(3)]  # 4 blocks per row of 64
        chunks = [
            quant.quantize_nvfp4(
                value[i : i + 1],
                "stochastic",
                seed=11,
                tensor_id="order",
                block_offset=offsets[i],
            )
            for i in reversed(range(3))
        ]
        chunks.reverse()  # back to offset order before merging
        merged = (
            torch.cat([c[0] for c in chunks]),
            torch.cat([c[1] for c in chunks]),
        )
        expected = quant.quantize_nvfp4(
            value, "stochastic", seed=11, tensor_id="order"
        )
        self.assertTrue(torch.equal(expected[0], merged[0]))
        self.assertTrue(torch.equal(expected[1], merged[1]))

    def test_non_stochastic_modes_ignore_seed_and_tensor_id(self) -> None:
        value = torch.randn(2, 32, dtype=torch.float32)
        for mode in ("ceil", "nearest"):
            a = quant.quantize_nvfp4(value, mode, seed=0, tensor_id="x")
            b = quant.quantize_nvfp4(value, mode, seed=99, tensor_id="y")
            self.assertTrue(torch.equal(a[0], b[0]))
            self.assertTrue(torch.equal(a[1], b[1]))

    def test_block_offset_int64_boundary(self) -> None:
        # Tiny tensor: 16 blocks; no large allocations anywhere in this test.
        value = torch.randn(16, 16, dtype=torch.float32)
        # Passing boundary: exclusive index end == 2**63 - 1, which is the
        # largest value torch.arange(dtype=int64) accepts as its end point.
        q, s = quant.quantize_nvfp4(
            value, "stochastic", seed=1, tensor_id="b",
            block_offset=2**63 - 1 - 16,
        )
        self.assertEqual(tuple(s.shape), (16, 1))
        self.assertEqual(tuple(q.shape), (16, 16))
        # Exclusive end exceeding int64 -> clear ValueError before any arange.
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(
                value, "stochastic", seed=1, tensor_id="b",
                block_offset=2**63 - 8,
            )
        # block_offset itself beyond int64.
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(
                value, "stochastic", seed=1, tensor_id="b", block_offset=2**63
            )
        # Non-stochastic modes never allocate the index range, so a huge
        # block_offset is accepted (it only affects stochastic draws).
        q, _ = quant.quantize_nvfp4(value, "ceil", block_offset=2**63)
        self.assertEqual(tuple(q.shape), (16, 16))


class ZeroBlockTests(unittest.TestCase):
    def test_all_zero_blocks_all_modes(self) -> None:
        value = torch.zeros(2, 32, dtype=torch.float32)
        for mode in ("ceil", "nearest", "stochastic"):
            q, s = quant.quantize_nvfp4(value, mode, seed=0)
            self.assertTrue(torch.equal(q, torch.zeros_like(q)))
            self.assertTrue(torch.equal(s, torch.full_like(s, quant.E4M3_MIN)))
            out = ref.dequantize_nvfp4(q, s)
            self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_mixed_zero_and_nonzero_blocks(self) -> None:
        value = torch.zeros(2, 32, dtype=torch.float32)
        value[0, 0] = 3.0
        q, s = quant.quantize_nvfp4(value, scale_mode="ceil")
        self.assertEqual(float(s[0, 0]), 0.5)  # ceil(3/6) = 0.5, on grid
        self.assertEqual(float(s[0, 1]), quant.E4M3_MIN)  # zero block
        self.assertEqual(float(s[1, 0]), quant.E4M3_MIN)
        self.assertEqual(float(q[0, 0]), 6.0)
        self.assertEqual(float(q[1, 0]), 0.0)


class ChunkingTests(unittest.TestCase):
    def _signed_zero_sensitive_tensor(self) -> torch.Tensor:
        # Deterministic tensor covering signed-zero carriers (-0.0 vs +0.0),
        # exact source zeros, magnitude ties, clipping, and overflow, so the
        # int16-view comparisons below would catch any bit-level divergence
        # (value equality alone treats -0.0 == +0.0 as equal).
        value = torch.zeros(3, 64, dtype=torch.float32)
        value[0, 0] = 3.0          # block (0,0): max 3.0 -> scale 0.5
        value[0, 1] = -0.0625      # ratio 0.125 -> -0.0 carrier
        value[0, 2] = 0.0625       # ratio 0.125 -> +0.0 carrier
        value[0, 3] = -0.0         # exact source zero -> +0.0 carrier
        value[0, 5] = 1.5          # ratio 3 -> carrier 3
        value[0, 6] = -1.5         # -> carrier -3
        value[0, 7] = 0.875        # tie 1.75 -> carrier 2
        value[0, 16] = 5.7         # block (0,1): max 5.7 -> scale 1.0
        value[0, 17] = -2.5        # ratio 2.5 tie -> carrier -3
        value[0, 18] = 5.0         # ratio 5.0 tie -> carrier 6
        value[2, 32] = 3000.0      # overflow block -> scale 448, carrier 6
        value[2, 33] = -0.0625     # ratio tiny -> -0.0 carrier
        return value

    def test_all_modes_identical_across_chunk_sizes(self) -> None:
        value = self._signed_zero_sensitive_tensor()
        chunk_sizes = (1, 2, 3, 5, 7, 1000, 1 << 30)
        for mode in ("ceil", "nearest", "stochastic"):
            baseline = quant.quantize_nvfp4(value, mode, seed=4, tensor_id="c")
            q_ref = baseline[0].view(torch.int16)
            s_ref = baseline[1].view(torch.int16)
            for chunk_blocks in chunk_sizes:
                result = quant.quantize_nvfp4(
                    value, mode, seed=4, tensor_id="c", chunk_blocks=chunk_blocks
                )
                self.assertTrue(
                    torch.equal(result[0].view(torch.int16), q_ref),
                    (mode, chunk_blocks),
                )
                self.assertTrue(
                    torch.equal(result[1].view(torch.int16), s_ref),
                    (mode, chunk_blocks),
                )

    def test_2d_and_3d_chunked_match(self) -> None:
        for value in (
            torch.randn(2, 48, dtype=torch.float32),
            torch.randn(2, 3, 16, dtype=torch.float32),
        ):
            whole = quant.quantize_nvfp4(value, "stochastic", seed=2, tensor_id="d")
            chunked = quant.quantize_nvfp4(
                value, "stochastic", seed=2, tensor_id="d", chunk_blocks=1
            )
            self.assertTrue(
                torch.equal(whole[0].view(torch.int16), chunked[0].view(torch.int16))
            )
            self.assertTrue(
                torch.equal(whole[1].view(torch.int16), chunked[1].view(torch.int16))
            )

    def test_stochastic_chunked_with_caller_offset(self) -> None:
        # Splitting the tensor into caller-level chunks that are themselves
        # processed with a tiny internal chunk size must match the whole call.
        value = torch.randn(4, 64, dtype=torch.float32)
        full = quant.quantize_nvfp4(value, "stochastic", seed=6, tensor_id="off")
        parts = []
        for row, offset in ((0, 0), (2, 8)):  # 8 blocks per 2 rows of 64
            parts.append(
                quant.quantize_nvfp4(
                    value[row:row + 2],
                    "stochastic",
                    seed=6,
                    tensor_id="off",
                    block_offset=offset,
                    chunk_blocks=3,
                )
            )
        merged = (
            torch.cat([p[0] for p in parts]),
            torch.cat([p[1] for p in parts]),
        )
        self.assertTrue(torch.equal(full[0], merged[0]))
        self.assertTrue(torch.equal(full[1], merged[1]))

    def test_chunk_blocks_validation(self) -> None:
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.randn(32), chunk_blocks=0)
        with self.assertRaises(ValueError):
            quant.quantize_nvfp4(torch.randn(32), chunk_blocks=-3)
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.randn(32), chunk_blocks=1.5)
        with self.assertRaises(TypeError):
            quant.quantize_nvfp4(torch.randn(32), chunk_blocks=True)


class SignedZeroTests(unittest.TestCase):
    def test_negative_rounding_to_zero_keeps_negative_zero(self) -> None:
        # Block max 3.0 -> scale 0.5 (on grid).  |x| = 0.0625 gives ratio
        # 0.125 < 0.25, whose nearest E2M1 magnitude is 0: the negative source
        # value keeps its sign as -0.0, the positive one becomes +0.0, and
        # exact source zeros (+0.0 / -0.0) normalize to +0.0.
        value = torch.tensor(
            [[3.0, -0.0625, 0.0625, 0.0, -0.0] + [0.0] * 11],
            dtype=torch.float32,
        )
        q, s = quant.quantize_nvfp4(value, scale_mode="ceil")
        self.assertEqual(float(s.flatten()[0]), 0.5)
        as_int = q.view(torch.int16).flatten().tolist()
        self.assertEqual(as_int[1], -32768)  # -0.0 carrier
        self.assertEqual(as_int[2], 0)  # +0.0 carrier
        self.assertEqual(as_int[3], 0)  # exact +0.0 source
        self.assertEqual(as_int[4], 0)  # exact -0.0 source normalized
        self.assertEqual(float(q.flatten()[1]), -0.0)
        self.assertEqual(float(q.flatten()[2]), 0.0)
        # dequantization preserves the carrier's signed zero
        out = ref.dequantize_nvfp4(q, s)
        self.assertEqual(float(out.flatten()[1]), -0.0)

    def test_signed_zero_all_modes(self) -> None:
        value = torch.tensor(
            [[3.0, -0.0625, 0.0625] + [0.0] * 13], dtype=torch.float32
        )
        for mode in ("ceil", "nearest", "stochastic"):
            q, _ = quant.quantize_nvfp4(value, mode, seed=0)
            as_int = q.view(torch.int16).flatten().tolist()
            self.assertEqual(as_int[1], -32768, mode)
            self.assertEqual(as_int[2], 0, mode)

    def test_all_zero_block_has_positive_zeros(self) -> None:
        value = torch.zeros(1, 16, dtype=torch.float32)
        q, _ = quant.quantize_nvfp4(value)
        self.assertEqual(q.view(torch.int16).flatten().tolist(), [0] * 16)


class CarrierTieAndClipTests(unittest.TestCase):
    def test_carrier_magnitude_ties_round_away_from_zero(self) -> None:
        ratios = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
             2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0],
            dtype=torch.float64,
        )
        expected = [0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0,
                    2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0]
        got = quant._carrier_magnitude(ratios).tolist()
        self.assertEqual(got, expected)

    def test_carrier_ties_integration(self) -> None:
        # Block max 3.0 -> target 0.5 -> scale 0.5 (on grid).  Ratios below
        # then hit every exact tie point; magnitudes above 6 clip to 6.
        elements = [3.0, 0.125, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5]
        value = torch.tensor(
            elements + [0.0] * 8, dtype=torch.float32
        ).reshape(1, 16)
        q, s = quant.quantize_nvfp4(value, scale_mode="ceil")
        self.assertEqual(float(s.flatten()[0]), 0.5)
        self.assertEqual(
            q.flatten()[:8].tolist(),
            [6.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        )
        self.assertEqual(q.flatten()[8:].tolist(), [0.0] * 8)

    def test_carrier_clip_integration(self) -> None:
        # max 3.5 -> ceil(3.5/6) = 0.625; ratio 3.5/0.625 = 5.6 clips to 6.
        value = torch.tensor([[3.5] + [0.0] * 15], dtype=torch.float32)
        q, s = quant.quantize_nvfp4(value, scale_mode="ceil")
        self.assertEqual(float(s.flatten()[0]), 0.625)
        self.assertEqual(float(q.flatten()[0]), 6.0)


class DequantizeTests(unittest.TestCase):
    def test_reexport_matches_reference(self) -> None:
        q = torch.tensor([[6.0, -3.0, 0.5, 0.0] + [0.0] * 12], dtype=torch.bfloat16)
        s = torch.tensor([[0.5]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(
            quant.dequantize_nvfp4(q, s),
            ref.dequantize_nvfp4(q, s),
        ))

    def test_roundtrip_exact_on_lattice(self) -> None:
        # scale 0.5, carriers {6, 3, 1, 0.5} reproduce {3, 1.5, 0.5, 0.25}
        # exactly; these inputs are already on the quantizer lattice.
        original = torch.tensor(
            [[3.0, 1.5, 0.5, 0.25, 0.0, -1.5, -3.0, 0.0] + [0.0] * 8],
            dtype=torch.float32,
        )
        q, s = quant.quantize_nvfp4(original, scale_mode="ceil")
        self.assertEqual(float(s.flatten()[0]), 0.5)
        restored = quant.dequantize_nvfp4(q, s)
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertEqual(tuple(restored.shape), tuple(original.shape))
        self.assertTrue(torch.equal(restored.float(), original))

    def test_dequantize_shape_and_value_contract(self) -> None:
        value = torch.randn(3, 32, dtype=torch.float32)
        q, s = quant.quantize_nvfp4(value, "nearest")
        out = quant.dequantize_nvfp4(q, s)
        self.assertEqual(tuple(out.shape), tuple(value.shape))
        self.assertEqual(out.dtype, torch.bfloat16)
        # manual BF16 semantics: unflatten -> multiply -> flatten -> bf16
        manual = (
            q.to(torch.float32).unflatten(-1, (-1, 16))
            * s.to(torch.float32).unsqueeze(-1)
        ).flatten(-2, -1)
        self.assertTrue(torch.allclose(out.float(), manual))

    def test_every_product_exactly_bf16(self) -> None:
        value = torch.randn(4, 64, dtype=torch.float32) * 3.0
        q, s = quant.quantize_nvfp4(value, "ceil")
        # reference semantics: unflatten to (..., block, 16), multiply by the
        # per-block scale, flatten back, cast to bf16
        product_fp64 = (
            q.to(torch.float64).unflatten(-1, (-1, 16))
            * s.unsqueeze(-1).to(torch.float64)
        ).flatten(-2, -1)
        product_bf16 = ref.dequantize_nvfp4(q, s).to(torch.float64)
        self.assertTrue(torch.equal(product_fp64, product_bf16))


class FingerprintIntegrationTests(unittest.TestCase):
    def _ceil_e4m3(self, target: float) -> float:
        """Test-side oracle: smallest positive E4M3FN value >= target."""
        for value in quant.E4M3_POSITIVE_VALUES:
            if value >= target:
                return value
        return quant.E4M3_MAX

    def test_ceil_scale_equals_ceil_e4m3_of_source_max(self) -> None:
        # The actual theorem, verified against the known source maxima: the
        # selected ceil scale equals ceil_E4M3(source_max / 6).  Includes
        # subnormal scales (where the stored-value fixed point can fail, see
        # the counterexample test below), on-grid and between-grid targets,
        # and overflow saturation.
        maxima = [
            6.0 * 1.125,      # on grid
            6.0 * 1.03125,    # between grid values
            6.0 * 0.28125,    # small normal scale
            2.5 * 2.0 ** -9,  # subnormal scale, ratio 2.5
            6.0 * 2.0 ** -9,  # subnormal scale, ratio 6
            0.5 * 2.0 ** -9,  # subnormal scale, ratio 0.5
            3000.0,           # overflow -> 448
            6.0 * 448.0,      # saturation boundary
        ]
        rows = [[max_abs] + [0.0] * 15 for max_abs in maxima]
        value = torch.tensor(rows, dtype=torch.float32)
        _, s = quant.quantize_nvfp4(value, scale_mode="ceil")
        for index, max_abs in enumerate(maxima):
            expected = self._ceil_e4m3(max_abs / 6.0)
            self.assertEqual(float(s[index, 0]), expected, (index, max_abs))

    def test_stored_fixed_point_is_not_universal(self) -> None:
        # The stored-value fixed point scale == ceil_E4M3(max_stored*scale/6)
        # is neither a legality condition nor an invariant of this quantizer.
        # A fully legal pair with stored max 3 and scale 2**-8 gives target
        # 3*2**-8/6 = 2**-9, whose grid ceiling is 2**-9 != 2**-8.
        quant_tensor = torch.tensor(
            [[3.0] + [0.0] * 15], dtype=torch.bfloat16
        )
        scale_tensor = torch.tensor([[2.0 ** -8]], dtype=torch.bfloat16)
        result = fingerprint.analyze_pair(
            "legal-pair", "weight", quant_tensor, scale_tensor
        )
        self.assertEqual(result["legal_carrier_count"], 16)
        self.assertEqual(result["e4m3_scale_count"], 1)
        self.assertEqual(result["exact_bf16_product_count"], 16)
        self.assertEqual(result["fixed_point_matches"], 0)
        self.assertEqual(result["informative_blocks"], 1)
        self.assertEqual(result["informative_fixed_point_matches"], 0)

        # This violating pair is reachable from the quantizer.  A source max
        # 13*2**-10 = 6.5*delta selects ceil scale 2*delta, then normalized
        # max 3.25 rounds to carrier 3.  The stored target ceilings to delta,
        # not the selected 2*delta.
        reachable_q, reachable_s = quant.quantize_nvfp4(
            torch.tensor([[13.0 * 2.0 ** -10] + [0.0] * 15], dtype=torch.float32),
            scale_mode="ceil",
        )
        self.assertEqual(float(reachable_s.flatten()[0]), 2.0 ** -8)
        self.assertEqual(float(reachable_q.flatten()[0]), 3.0)
        reachable = fingerprint.analyze_pair(
            "reachable-output", "weight", reachable_q, reachable_s
        )
        self.assertEqual(reachable["fixed_point_matches"], 0)
        self.assertEqual(reachable["informative_blocks"], 1)
        self.assertEqual(reachable["informative_fixed_point_matches"], 0)

        # A different subnormal output can still satisfy the identity.  Source
        # max 2.5*delta keeps scale delta and stores carrier 3; its stored
        # target is below the minimum E4M3 scale and ceilings back to delta.
        q, s = quant.quantize_nvfp4(
            torch.tensor([[2.5 * 2.0 ** -9] + [0.0] * 15], dtype=torch.float32),
            scale_mode="ceil",
        )
        self.assertEqual(float(s.flatten()[0]), 2.0 ** -9)
        self.assertEqual(float(q.flatten()[0]), 3.0)
        own = fingerprint.analyze_pair("own-output", "weight", q, s)
        self.assertEqual(own["fixed_point_matches"], 1)

    def test_analyze_pair_legality_only(self) -> None:
        # analyze_pair integration is limited to structural guarantees: legal
        # carriers, exact E4M3FN scales, and exact BF16 products.  The stored
        # fixed point is not universal (see the counterexample test) and must
        # not be asserted here.
        torch.manual_seed(0)
        value = (torch.randn(3, 64, dtype=torch.float32) * 2.0).abs()
        value[1, :] = torch.zeros(64)  # one all-zero block row
        q, s = quant.quantize_nvfp4(value, scale_mode="ceil")
        result = fingerprint.analyze_pair("roundtrip", "activation", q, s)
        self.assertEqual(result["legal_carrier_count"], q.numel())
        self.assertEqual(result["e4m3_scale_count"], s.numel())
        self.assertEqual(result["exact_bf16_product_count"], q.numel())


if __name__ == "__main__":
    unittest.main(verbosity=2)
