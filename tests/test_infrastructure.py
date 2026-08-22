"""Infrastructure tests for the NVFP4-to-HiF4 contest workspace.

Covers, with stdlib ``unittest`` only (no pytest, no file I/O):

* ``tools/reference_ops.py``
    - exact NVFP4 BF16 dequant semantics (statement-snippet equality and
      bitwise equivalence to an independent FP64-rounding formulation);
    - HiF4 reshape/dequant against an independent per-element FP64 loop;
    - Linear (2-D and batched) against independent einsum/loop formulations;
    - MHA / MQA / GQA Attention against an independent per-head formulation,
      plus the causal mask (including the structural ``out[0] == v[0]``
      consequence for ``seq_k >= seq_q``);
    - ``scalar_mse`` values and error paths.
* ``tools/synthetic_data.py``
    - determinism (repeat calls bitwise identical, all distributions);
    - NVFP4 carrier and per-16-block scale legality;
    - quantization reconstruction error bounded by the block scale;
    - distribution properties (sparse fraction, heavy tails, channel
      outliers, mixed block magnitudes) and builder layouts/validations.
* ``solution.py`` (current candidate)
    - all six public APIs exist and return contract-legal results
      (dict keys, HiF4 param shapes/values incl. exact E6M2, frozen-state
      format, finite outputs);
    - no input mutation and deterministic repeated calls for all six APIs;
    - end-to-end MSE sanity (Linear and MHA/MQA/GQA Attention) against the
      reference NVFP4 outputs.

Run from the workspace root:

    python3 -m unittest discover -s tests -v

or directly:

    python3 tests/test_infrastructure.py

Runtime on a CPU-only machine is a few seconds; no workspace files are
generated (bytecode writing is disabled at import time).
"""

from __future__ import annotations

import atexit
import glob
import math
import os
import sys
import unittest

# Never write .pyc files into the workspace during the test run. The import
# machinery writes this module's own bytecode before the body runs, so also
# remove that single artifact at interpreter shutdown (leave other modules'
# caches untouched).
sys.dont_write_bytecode = True

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def _cleanup_test_bytecode() -> None:
    for path in glob.glob(
        os.path.join(_TEST_DIR, "__pycache__", "test_infrastructure*.pyc")
    ):
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(os.path.join(_TEST_DIR, "__pycache__"))  # only if empty
    except OSError:
        pass


atexit.register(_cleanup_test_bytecode)

_ROOT = os.path.dirname(_TEST_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from tools import reference_ops as ref  # noqa: E402
from tools import synthetic_data as syn  # noqa: E402
from tools import evaluate as evaluator  # noqa: E402
import solution  # noqa: E402

NVFP4_BLOCK = 16
HIF4_BLOCK = 64

SOLUTION_APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)

# ===========================================================================
# Independent reference formulations (kept intentionally separate from the
# implementation under test so equivalence is a real cross-check).
# ===========================================================================


def _indep_hif4_dequant(params, logical_shape):
    """Per-element FP64 reconstruction of the HiF4 five-tensor layout.

    Logical index ``i`` maps to intra-block index ``r = i % 64`` with
    ``a = r // 8`` (8 groups), ``c = (r % 8) // 4`` (2 sub-groups),
    ``d = r % 4`` (4 mantissa slots).
    """
    shape = tuple(int(s) for s in logical_shape)
    channels = shape[-1]
    prefix = shape[:-1]
    nblocks = channels // HIF4_BLOCK
    sf = params["scale_factor"].to(torch.float64).reshape(prefix + (nblocks,))
    lv2 = params["scale_lv2"].to(torch.float64).reshape(prefix + (nblocks, 8))
    lv3 = params["scale_lv3"].to(torch.float64).reshape(
        prefix + (nblocks, 8, 2)
    )
    sg = params["sign"].to(torch.float64).reshape(prefix + (nblocks, 8, 2, 4))
    mn = params["mant"].to(torch.float64).reshape(prefix + (nblocks, 8, 2, 4))
    out = torch.zeros(shape, dtype=torch.float64)
    for i in range(channels):
        b, r = i // HIF4_BLOCK, i % HIF4_BLOCK
        a, c, d = r // 8, (r % 8) // 4, r % 4
        out[..., i] = (
            sf[..., b] * lv2[..., b, a] * lv3[..., b, a, c]
            * sg[..., b, a, c, d] * mn[..., b, a, c, d]
        )
    return out


def _simple_attention(q, k, v, q_num_heads, kv_num_heads, head_dim, causal=False):
    """Per-head loop Attention: contiguous-head GQA mapping ``h -> h // rep``."""
    seq_q, seq_k = int(q.shape[-2]), int(k.shape[-2])
    rep = q_num_heads // kv_num_heads
    qq = q.to(torch.float32).unflatten(-1, (q_num_heads, head_dim))
    kk = k.to(torch.float32).unflatten(-1, (kv_num_heads, head_dim))
    vv = v.to(torch.float32).unflatten(-1, (kv_num_heads, head_dim))
    out = torch.zeros(seq_q, q_num_heads, head_dim, dtype=torch.float32)
    for h in range(q_num_heads):
        j = h // rep
        scores = qq[:, h, :] @ kk[:, j, :].t() * (head_dim ** -0.5)
        if causal:
            mask = torch.triu(
                torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, h, :] = probs @ vv[:, j, :]
    return out.reshape(seq_q, q_num_heads * head_dim)


def _expected_hif4_shapes(shape):
    """HiF4 param shapes mandated by ``example/self_check.py``."""
    shape = tuple(int(s) for s in shape)
    if not shape:
        raise ValueError("shape must have at least one dimension")
    channels = shape[-1]
    if channels % HIF4_BLOCK != 0:
        raise ValueError(
            f"last dimension {channels} not divisible by HiF4 block size 64"
        )
    prefix = shape[:-1] + (channels // HIF4_BLOCK,)
    return {
        "scale_factor": prefix + (1, 1, 1),
        "scale_lv2": prefix + (8, 1, 1),
        "scale_lv3": prefix + (8, 2, 1),
        "sign": prefix + (8, 2, 4),
        "mant": prefix + (8, 2, 4),
    }


_STATE_TENSOR_DTYPES = frozenset({
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
})


def _clone_pair(pair):
    return [pair[0].detach().clone(), pair[1].detach().clone()]


def _clone_qkv_sample(sample):
    return {role: _clone_pair(sample[role]) for role in ("q", "k", "v")}


def _clone_calib_qkv_list(calib):
    return [_clone_qkv_sample(s) for s in calib]


# ===========================================================================
# Shared contract validators (mirroring example/self_check.py)
# ===========================================================================


def assert_state_legal(test, state, tag=""):
    """Validate the frozen calibration-state contract (types, tensors, size)."""
    errors = []
    nodes = [0]

    def visit(value, depth):
        nodes[0] += 1
        if depth > 8:
            errors.append(f"{tag}: nesting depth exceeds 8")
            return
        if type(value) is torch.Tensor:
            if value.device.type != "cpu":
                errors.append(f"{tag}: tensor must be on CPU")
            if value.layout is not torch.strided:
                errors.append(f"{tag}: tensor must be strided")
            if value.dtype not in _STATE_TENSOR_DTYPES:
                errors.append(f"{tag}: tensor dtype {value.dtype} not allowed")
            if value.requires_grad:
                errors.append(f"{tag}: tensor requires_grad must be False")
            if torch.is_complex(value):
                errors.append(f"{tag}: complex tensor not allowed")
            if value.is_floating_point():
                if not torch.isfinite(value.detach()).all():
                    errors.append(f"{tag}: tensor has non-finite values")
            return
        if value is None or type(value) is bool or type(value) is int:
            return
        if type(value) is float:
            if not math.isfinite(value):
                errors.append(f"{tag}: non-finite float")
            return
        if type(value) is str:
            if len(value.encode("utf-8")) > 4096:
                errors.append(f"{tag}: string exceeds 4096 bytes")
            return
        if type(value) is list or type(value) is tuple:
            for item in value:
                visit(item, depth + 1)
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    errors.append(f"{tag}: dict key must be str")
                visit(item, depth + 1)
            return
        errors.append(f"{tag}: unsupported type {type(value).__name__}")

    visit(state, 0)
    test.assertLessEqual(nodes[0], 4096, f"{tag}: state node count too large")
    test.assertEqual(errors, [], "; ".join(errors[:5]))


def assert_state_equal(test, left, right):
    """Compare deterministic calibration states, including nested tensors."""
    test.assertIs(type(left), type(right))
    if type(left) is torch.Tensor:
        test.assertTrue(torch.equal(left, right))
    elif type(left) is dict:
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_state_equal(test, left[key], right[key])
    elif type(left) is list or type(left) is tuple:
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            assert_state_equal(test, left_item, right_item)
    else:
        test.assertEqual(left, right)


def assert_hif4_params_legal(test, params, shape, tag=""):
    """Full HiF4 output-format validation (shapes, values, finiteness)."""
    expected = _expected_hif4_shapes(shape)
    test.assertIsInstance(params, dict, f"{tag}: params must be a dict")
    for name, exp in expected.items():
        test.assertIn(name, params, f"{tag}: missing {name!r}")
        value = params[name]
        test.assertIsInstance(
            value, torch.Tensor, f"{tag}.{name}: must be a torch.Tensor"
        )
        test.assertEqual(
            tuple(value.shape), exp,
            f"{tag}.{name}: shape {tuple(value.shape)} != {exp}",
        )
        test.assertFalse(torch.is_complex(value), f"{tag}.{name}: complex")
        test.assertTrue(
            torch.isfinite(value.detach().to(torch.float64)).all(),
            f"{tag}.{name}: non-finite values",
        )

    sf = params["scale_factor"].detach().to(torch.float64)
    test.assertTrue((sf >= 2.0 ** -48).all(), f"{tag}.scale_factor: below 2^-48")
    test.assertTrue((sf <= 49152.0).all(), f"{tag}.scale_factor: above 49152")
    sf_clamped = sf.clamp(min=2.0 ** -126)
    sf_exp = torch.floor(torch.log2(sf_clamped))
    sf_e6m2 = (
        torch.round(sf * (2.0 ** (2 - sf_exp))) * (2.0 ** (sf_exp - 2))
    )
    test.assertTrue(
        torch.equal(sf, sf_e6m2), f"{tag}.scale_factor: not exact E6M2"
    )

    for name in ("scale_lv2", "scale_lv3"):
        value = params[name].detach().to(torch.float64)
        ok = (value == 1.0) | (value == 2.0)
        test.assertTrue(ok.all(), f"{tag}.{name}: must be exactly {{1, 2}}")

    sign = params["sign"].detach().to(torch.float64)
    ok = (sign == -1.0) | (sign == 0.0) | (sign == 1.0)
    test.assertTrue(ok.all(), f"{tag}.sign: must be exactly {{-1, 0, 1}}")

    mant = params["mant"].detach().to(torch.float64)
    test.assertTrue((mant >= 0.0).all(), f"{tag}.mant: negative mantissa")
    test.assertTrue((mant <= 1.75).all(), f"{tag}.mant: above 1.75")
    test.assertTrue(
        torch.equal(mant * 4.0, torch.round(mant * 4.0)),
        f"{tag}.mant: must be exact multiples of 0.25",
    )

    dequant = ref.dequantize_hif4_params(params, shape)
    test.assertEqual(tuple(dequant.shape), tuple(shape), f"{tag}: reshape")
    test.assertEqual(int(dequant.numel()), math.prod(shape), f"{tag}: numel")
    test.assertTrue(torch.isfinite(dequant).all(), f"{tag}: non-finite dequant")


# ===========================================================================
# reference_ops: NVFP4 dequantization
# ===========================================================================


class TestReferenceNVFP4Dequant(unittest.TestCase):
    """Exact NVFP4 BF16 dequantization semantics."""

    def _make_pair(self, rows=4, cols=64):
        quant = (
            (torch.rand(rows, cols) * 14 - 7)
            .round()
            .mul(0.5)
            .clamp(-6.0, 6.0)
            .to(torch.bfloat16)
        )
        scale = torch.pow(2.0, torch.randint(-8, 3, (rows, cols // 16))).to(
            torch.bfloat16
        )
        return quant, scale

    def test_matches_statement_snippet(self):
        """Bitwise equality with the statement's canonical helper body."""
        quant, scale = self._make_pair()
        blk = 16
        x = quant.unflatten(-1, (-1, blk))
        sf = scale.unsqueeze(-1)
        result = x * sf
        result = result.flatten(-2, -1)
        snippet = result.to(torch.bfloat16)

        got = ref.dequantize_nvfp4(quant, scale)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(tuple(got.shape), tuple(quant.shape))
        self.assertTrue(torch.equal(got, snippet))

    def test_bf16_rounding_is_round_to_nearest_even(self):
        """Reference equals per-element FP64 product rounded to BF16.

        The product of two BF16 values is exactly representable in FP32, so the
        only rounding is the final round-to-nearest-even cast to BF16; the
        FP64-then-round formulation must therefore be bitwise identical.
        """
        quant, scale = self._make_pair()
        indep = (
            quant.double().unflatten(-1, (-1, NVFP4_BLOCK))
            * scale.double().unsqueeze(-1)
        ).flatten(-2, -1).to(torch.bfloat16)
        self.assertTrue(torch.equal(ref.dequantize_nvfp4(quant, scale), indep))

    def test_dtype_and_block_size_options(self):
        quant, scale = self._make_pair(cols=64)
        got = ref.dequantize_nvfp4(quant, scale, dtype=torch.float32)
        self.assertEqual(got.dtype, torch.float32)
        # blk_size=32: scale has one entry per 32 values.
        scale32 = torch.pow(2.0, torch.randint(-8, 3, (4, 2))).to(
            torch.bfloat16
        )
        got32 = ref.dequantize_nvfp4(quant, scale32, blk_size=32)
        self.assertEqual(got32.dtype, torch.bfloat16)
        self.assertEqual(tuple(got32.shape), tuple(quant.shape))

    def test_error_paths(self):
        quant, scale = self._make_pair()
        with self.assertRaises(TypeError):
            ref.dequantize_nvfp4(quant.tolist(), scale)
        with self.assertRaises(ValueError):
            ref.dequantize_nvfp4(quant, scale[:, :1])  # wrong scale shape
        bad = torch.zeros(4, 10).to(torch.bfloat16)  # 10 % 16 != 0
        with self.assertRaises(ValueError):
            ref.dequantize_nvfp4(bad, torch.zeros(4, 1).to(torch.bfloat16))


# ===========================================================================
# reference_ops: HiF4 dequantization / reshape
# ===========================================================================


class TestReferenceHiF4Dequant(unittest.TestCase):
    def _make_params(self, shape):
        shape = tuple(shape)
        expected = _expected_hif4_shapes(shape)
        params = {}
        for name, exp in expected.items():
            params[name] = torch.randn(exp).abs() * 2.0
        params["sign"] = torch.where(torch.randn(expected["sign"]) > 0, 1.0, -1.0)
        return params

    def test_values_match_independent_fp64_loop(self):
        for shape in ((3, 64), (2, 3, 128), (5, 256)):
            params = self._make_params(shape)
            got = ref.dequantize_hif4(
                params["scale_factor"],
                params["scale_lv2"],
                params["scale_lv3"],
                params["sign"],
                params["mant"],
                shape,
                dtype=torch.float64,
            )
            expected = _indep_hif4_dequant(params, shape)
            self.assertEqual(got.dtype, torch.float64)
            self.assertTrue(
                torch.allclose(got, expected, rtol=1e-5, atol=1e-7),
                f"mismatch for shape {shape}",
            )

    def test_default_dtype_is_fp32(self):
        params = self._make_params((4, 64))
        got = ref.dequantize_hif4(
            params["scale_factor"],
            params["scale_lv2"],
            params["scale_lv3"],
            params["sign"],
            params["mant"],
            (4, 64),
        )
        self.assertEqual(got.dtype, torch.float32)
        self.assertEqual(tuple(got.shape), (4, 64))

    def test_params_wrapper(self):
        shape = (2, 128)
        params = self._make_params(shape)
        direct = ref.dequantize_hif4(
            params["scale_factor"],
            params["scale_lv2"],
            params["scale_lv3"],
            params["sign"],
            params["mant"],
            shape,
        )
        wrapper = ref.dequantize_hif4_params(params, shape)
        self.assertTrue(torch.equal(direct, wrapper))
        # Extra keys are ignored; missing keys raise; non-mapping raises.
        extra = dict(params, junk=torch.zeros(1))
        self.assertTrue(
            torch.equal(ref.dequantize_hif4_params(extra, shape), direct)
        )
        del extra["mant"]
        with self.assertRaises(KeyError):
            ref.dequantize_hif4_params(extra, shape)
        with self.assertRaises(TypeError):
            ref.dequantize_hif4_params(list(params.values()), shape)

    def test_error_paths(self):
        params = self._make_params((2, 64))
        with self.assertRaises(ValueError):  # last dim not divisible by 64
            ref.dequantize_hif4(
                params["scale_factor"], params["scale_lv2"],
                params["scale_lv3"], params["sign"], params["mant"],
                (2, 100),
            )
        bad_sign = params["sign"].to(torch.int32)  # non-floating tensor
        with self.assertRaises(TypeError):
            ref.dequantize_hif4(
                params["scale_factor"], params["scale_lv2"],
                params["scale_lv3"], bad_sign, params["mant"], (2, 64),
            )
        # Pass mant (shape ...8,2,4) where scale_lv2 (shape ...8,1,1) is expected.
        with self.assertRaises(ValueError):  # wrong param shape
            ref.dequantize_hif4(
                params["scale_factor"], params["mant"],
                params["scale_lv3"], params["sign"], params["mant"], (2, 64),
            )


# ===========================================================================
# reference_ops: Linear
# ===========================================================================


class TestReferenceLinear(unittest.TestCase):
    def test_2d_matches_manual_matmul(self):
        x = torch.randn(5, 8)
        w = torch.randn(3, 8)
        got = ref.linear_output(x, w)
        manual = torch.zeros(5, 3)
        for m in range(5):
            for n in range(3):
                manual[m, n] = torch.dot(x[m], w[n])
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.allclose(got, manual, rtol=1e-5, atol=1e-7))

    def test_batched_matches_einsum(self):
        x = torch.randn(2, 3, 8)
        w = torch.randn(4, 8)
        got = ref.linear_output(x, w)
        expected = torch.einsum("bmk,nk->bmn", x, w)
        self.assertEqual(tuple(got.shape), (2, 3, 4))
        self.assertTrue(torch.allclose(got, expected, rtol=1e-5, atol=1e-7))

    def test_nvfp4_wrapper_consistency(self):
        from tools.synthetic_data import make_nvfp4_pair

        xq, xs = make_nvfp4_pair("normal", (4, 64), seed=101)
        wq, ws = make_nvfp4_pair("normal", (3, 64), seed=102)
        direct = ref.linear_output(
            ref.dequantize_nvfp4(xq, xs), ref.dequantize_nvfp4(wq, ws)
        )
        wrapped = ref.linear_output_nvfp4(xq, xs, wq, ws)
        self.assertTrue(torch.equal(direct, wrapped))

    def test_error_paths(self):
        with self.assertRaises(ValueError):  # inner dim mismatch
            ref.linear_output(torch.randn(3, 8), torch.randn(4, 7))
        with self.assertRaises(ValueError):  # 1-D input
            ref.linear_output(torch.randn(8), torch.randn(4, 8))
        with self.assertRaises(TypeError):
            ref.linear_output(torch.randn(3, 8), "w")


# ===========================================================================
# reference_ops: Attention (MHA / MQA / GQA, causal)
# ===========================================================================


class TestReferenceAttention(unittest.TestCase):
    def _case(self, kv_num_heads, causal, seq_q=7, seq_k=9, head_dim=5):
        q_num_heads = 4
        q = torch.randn(seq_q, q_num_heads * head_dim) * 0.7
        k = torch.randn(seq_k, kv_num_heads * head_dim) * 0.7
        v = torch.randn(seq_k, kv_num_heads * head_dim) * 0.7
        return q, k, v, q_num_heads, kv_num_heads, head_dim, causal

    def test_matches_independent_per_head_formulation(self):
        for kv_num_heads, causal in ((4, False), (2, False), (1, False),
                                     (4, True), (2, True), (1, True)):
            args = self._case(kv_num_heads, causal)
            got = ref.attention_output(*args[:-1], causal=args[-1])
            expected = _simple_attention(*args)
            self.assertEqual(got.dtype, torch.float32)
            self.assertEqual(
                tuple(got.shape),
                (args[0].shape[-2], args[3] * args[5]),
            )
            self.assertTrue(
                torch.allclose(got, expected, rtol=1e-4, atol=1e-5),
                f"mismatch kv={kv_num_heads} causal={causal}",
            )

    def test_causal_mask_structure(self):
        """Causal attention ignores future keys: query row 0 sees only key 0."""
        q, k, v, qh, kvh, hd, _ = self._case(2, False, seq_q=5, seq_k=5)
        out = ref.attention_output(q, k, v, qh, kvh, hd, causal=True)
        # Row 0 attends only to key 0, so each Q head reads its KV head's v[0]
        # (broadcast contiguously over the rep = qh // kvh repeated groups).
        rep = qh // kvh
        expected = (
            v[0].to(torch.float32)
            .unflatten(-1, (kvh, hd))
            .repeat_interleave(rep, dim=0)
            .reshape(-1)
        )
        self.assertTrue(
            torch.allclose(out[0], expected, rtol=1e-5, atol=1e-6),
            "row 0 must attend only to key 0",
        )
        # Non-causal row 0 must NOT collapse to the broadcast v[0] when seq_k > 1.
        out_plain = ref.attention_output(q, k, v, qh, kvh, hd, causal=False)
        self.assertFalse(
            torch.allclose(out_plain[0], expected, rtol=1e-2, atol=1e-3)
        )

    def test_batch_prefix(self):
        qh, kvh, hd = 2, 1, 4
        q = torch.randn(3, 6, qh * hd)
        k = torch.randn(3, 8, kvh * hd)
        v = torch.randn(3, 8, kvh * hd)
        got = ref.attention_output(q, k, v, qh, kvh, hd)
        self.assertEqual(tuple(got.shape), (3, 6, qh * hd))
        for b in range(3):
            single = ref.attention_output(q[b], k[b], v[b], qh, kvh, hd)
            self.assertTrue(torch.allclose(got[b], single, rtol=1e-6, atol=1e-7))

    def test_nvfp4_wrapper_consistency(self):
        from tools.synthetic_data import make_nvfp4_pair

        qq, qs = make_nvfp4_pair("normal", (6, 128), seed=201)
        kq, ks = make_nvfp4_pair("normal", (6, 64), seed=202)
        vq, vs = make_nvfp4_pair("normal", (6, 64), seed=203)
        direct = ref.attention_output(
            ref.dequantize_nvfp4(qq, qs),
            ref.dequantize_nvfp4(kq, ks),
            ref.dequantize_nvfp4(vq, vs),
            2, 1, 64,
        )
        wrapped = ref.attention_output_nvfp4(
            qq, qs, kq, ks, vq, vs, 2, 1, 64
        )
        self.assertTrue(torch.equal(direct, wrapped))

    def test_error_paths(self):
        q, k, v, qh, kvh, hd, _ = self._case(2, False)
        with self.assertRaises(ValueError):  # qh not divisible by kvh
            ref.attention_output(q, k, v, 4, 3, hd)
        with self.assertRaises(ValueError):  # causal needs seq_k >= seq_q
            ref.attention_output(
                q[:, :4], k[:, :4], v[:, :4], qh, kvh, hd, causal=True
            )
        with self.assertRaises(ValueError):  # q hidden mismatch
            ref.attention_output(q[:, :-1], k, v, qh, kvh, hd)
        with self.assertRaises(ValueError):  # k/v seq mismatch
            ref.attention_output(q, k[:, 1:], v, qh, kvh, hd)
        with self.assertRaises(ValueError):  # batch prefix mismatch
            qb = torch.randn(2, 5, qh * hd)
            kb = torch.randn(3, 7, kvh * hd)
            vb = torch.randn(3, 7, kvh * hd)
            ref.attention_output(qb, kb, vb, qh, kvh, hd)
        with self.assertRaises(TypeError):
            ref.attention_output(q.tolist(), k, v, qh, kvh, hd)


# ===========================================================================
# reference_ops: scalar_mse
# ===========================================================================


class TestReferenceScalarMse(unittest.TestCase):
    def test_known_value(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 3.0, 3.0])
        mse = ref.scalar_mse(pred, target)
        self.assertIsInstance(mse, float)
        # Computed in FP32: 0.33333334, not exactly 1/3.
        self.assertAlmostEqual(mse, 1.0 / 3.0, delta=1e-6)

    def test_zero(self):
        x = torch.randn(4, 8)
        self.assertEqual(ref.scalar_mse(x, x.clone()), 0.0)

    def test_error_paths(self):
        with self.assertRaises(ValueError):
            ref.scalar_mse(torch.randn(2, 3), torch.randn(3, 2))
        with self.assertRaises(TypeError):
            ref.scalar_mse([1.0, 2.0], torch.randn(2))


# ===========================================================================
# synthetic_data: determinism
# ===========================================================================


class TestSyntheticDeterminism(unittest.TestCase):
    def test_make_tensor_reproducible(self):
        for dist in syn.DISTS:
            a = syn.make_tensor(dist, (4, 64), seed=42)
            b = syn.make_tensor(dist, (4, 64), seed=42)
            self.assertTrue(torch.equal(a, b), f"dist={dist}")
            # Different seeds differ.
            c = syn.make_tensor(dist, (4, 64), seed=43)
            self.assertFalse(torch.equal(a, c), f"dist={dist}")

    def test_make_nvfp4_pair_reproducible(self):
        for dist in syn.DISTS:
            a = syn.make_nvfp4_pair(dist, (4, 64), seed=7)
            b = syn.make_nvfp4_pair(dist, (4, 64), seed=7)
            self.assertTrue(torch.equal(a[0], b[0]), f"quant dist={dist}")
            self.assertTrue(torch.equal(a[1], b[1]), f"scale dist={dist}")

    def test_group_builders_reproducible(self):
        for dist in syn.DISTS:
            g1 = syn.make_linear_group(seed=11, dist=dist, n_calib=2, n_test=2)
            g2 = syn.make_linear_group(seed=11, dist=dist, n_calib=2, n_test=2)
            for key in ("weight",):
                self.assertTrue(torch.equal(g1[key][0], g2[key][0]), key)
            for i, (p1, p2) in enumerate(zip(
                g1["calib_activation_list"], g2["calib_activation_list"]
            )):
                self.assertTrue(torch.equal(p1[0], p2[0]), f"calib[{i}]")
                self.assertTrue(torch.equal(p1[1], p2[1]), f"calib[{i}]")
            a1 = syn.make_attention_group(seed=13, dist=dist, n_calib=1, n_test=1)
            a2 = syn.make_attention_group(seed=13, dist=dist, n_calib=1, n_test=1)
            for role in ("q", "k", "v"):
                self.assertTrue(
                    torch.equal(a1["test"][0][role][0], a2["test"][0][role][0]),
                    f"{role} dist={dist}",
                )


# ===========================================================================
# synthetic_data: NVFP4 carrier / scale legality
# ===========================================================================


class TestSyntheticLegality(unittest.TestCase):
    def _assert_legal_pair(self, pair, tag):
        quant, scale = pair
        self.assertIsInstance(quant, torch.Tensor, tag)
        self.assertIsInstance(scale, torch.Tensor, tag)
        self.assertEqual(quant.dtype, torch.bfloat16, tag)
        self.assertEqual(scale.dtype, torch.bfloat16, tag)
        channels = int(quant.shape[-1])
        self.assertEqual(channels % NVFP4_BLOCK, 0, tag)
        self.assertEqual(
            tuple(scale.shape), tuple(quant.shape[:-1]) + (channels // 16,),
            tag,
        )
        for v in quant.unique().tolist():
            self.assertIn(v, syn.NVFP4_CARRIERS, f"{tag}: carrier {v}")
        self.assertTrue(torch.isfinite(scale).all(), tag)
        self.assertTrue((scale > 0).all(), f"{tag}: scales must be positive")

    def test_all_distributions_and_shapes(self):
        for dist in syn.DISTS:
            for shape in ((8, 64), (3, 4, 128), (16,)):
                pair = syn.make_nvfp4_pair(dist, shape, seed=17)
                self._assert_legal_pair(pair, f"{dist}/{shape}")

    def test_carrier_set_constant(self):
        self.assertEqual(tuple(syn.NVFP4_CARRIERS),
                         (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
                          0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
        self.assertEqual(syn.CARRIER_MAX, 6.0)
        self.assertEqual(syn.NVFP4_BLOCK, 16)
        self.assertEqual(syn.HIF4_BLOCK, 64)

    def test_reconstruction_error_bounded_by_block_scale(self):
        """Nearest-carrier rounding error is below the block scale."""
        for dist in syn.DISTS:
            for seed in (20, 21, 22):
                original = syn.make_tensor(dist, (4, 64), seed)
                quant, scale = syn.quantize_nvfp4(original)
                dq = ref.dequantize_nvfp4(quant, scale).to(torch.float64)
                err = (dq - original.double()).abs()
                block_scale = scale.double().unsqueeze(-1)
                bound = (err.unflatten(-1, (-1, 16)) / block_scale).max()
                self.assertLessEqual(
                    float(bound), 2.0 + 1e-6,
                    f"dist={dist} seed={seed}: error exceeds block scale",
                )

    def test_quantize_does_not_mutate_input(self):
        original = syn.make_tensor("normal", (4, 64), seed=31)
        snapshot = original.clone()
        syn.quantize_nvfp4(original)
        self.assertTrue(torch.equal(original, snapshot))


# ===========================================================================
# synthetic_data: distribution properties and builder layout
# ===========================================================================


class TestSyntheticDistributions(unittest.TestCase):
    def test_sparse_zero_fraction(self):
        x = syn.make_tensor("sparse", (4, 64), seed=7, sparsity=0.8)
        frac = float((x == 0).double().mean())
        self.assertGreater(frac, 0.5)
        self.assertLess(frac, 1.0)
        dense = syn.make_tensor("sparse", (4, 64), seed=7, sparsity=0.0)
        self.assertLess(float((dense == 0).double().mean()), 0.01)

    def test_heavy_tail_outliers(self):
        x = syn.make_tensor("heavy_tail", (64, 64), seed=3)
        self.assertTrue(torch.isfinite(x).all())
        self.assertGreater(float(x.abs().max()), 5.0)

    def test_channel_outliers(self):
        x = syn.make_tensor(
            "channel_outlier", (4, 64), seed=5, n_outliers=2, outlier_scale=10.0
        )
        self.assertGreater(float(x.abs().max()), 5.0)

    def test_mixed_block_magnitude_spread(self):
        x = syn.make_tensor("mixed_block", (4, 64), seed=9)
        self.assertTrue(torch.isfinite(x).all())
        blocks = x.unflatten(-1, (-1, 16)).abs().amax(dim=-1)
        spread = float(blocks.max() / max(float(blocks.min()), 1e-12))
        self.assertGreater(spread, 1.5, "block magnitudes should vary")

    def test_normal_stats(self):
        x = syn.make_tensor("normal", (256, 64), seed=33)
        self.assertLess(abs(float(x.mean())), 0.1)
        self.assertAlmostEqual(float(x.std()), 1.0, delta=0.1)

    def test_all_dists_finite(self):
        for dist in syn.DISTS:
            x = syn.make_tensor(dist, (8, 64), seed=29)
            self.assertTrue(torch.isfinite(x).all(), dist)

    def test_unknown_dist_and_bad_args(self):
        with self.assertRaises(ValueError):
            syn.make_tensor("gaussian", (4, 64), seed=1)
        with self.assertRaises(ValueError):
            syn.make_tensor("sparse", (4, 64), seed=1, sparsity=1.5)
        with self.assertRaises(ValueError):
            syn.make_tensor("mixed_block", (4, 10), seed=1)  # 10 % 16 != 0


class TestSyntheticBuilders(unittest.TestCase):
    def test_linear_group_layout(self):
        group = syn.make_linear_group(
            seed=11, dist="channel_outlier", out_features=32, in_features=64,
            seq_len=8, n_calib=2, n_test=3, n_outliers=2,
        )
        self.assertEqual(
            sorted(group), ["calib_activation_list", "key",
                            "test_activation_list", "weight"]
        )
        self.assertEqual(group["key"], "linear")
        wq, ws = group["weight"]
        self.assertEqual(tuple(wq.shape), (32, 64))
        self.assertEqual(tuple(ws.shape), (32, 4))
        self.assertEqual(len(group["calib_activation_list"]), 2)
        self.assertEqual(len(group["test_activation_list"]), 3)
        for i, pair in enumerate(group["calib_activation_list"]):
            self.assertEqual(tuple(pair[0].shape), (8, 64), f"calib[{i}]")
            self.assertEqual(tuple(pair[1].shape), (8, 4), f"calib[{i}]")
        with self.assertRaises(ValueError):  # in_features % 64 != 0
            syn.make_linear_group(seed=1, in_features=32)

    def test_attention_group_layouts(self):
        for kv, kind in ((4, "mha"), (2, "gqa"), (1, "mqa")):
            group = syn.make_attention_group(
                seed=13, dist="mixed_block", q_num_heads=4, kv_num_heads=kv,
                head_dim=64, seq_len=8, n_calib=1, n_test=2,
            )
            self.assertEqual(group["key"], "attn")
            self.assertEqual(group["attn_type"], "gqa")
            self.assertEqual(group["q_num_heads"], 4)
            self.assertEqual(group["kv_num_heads"], kv)
            self.assertEqual(group["head_dim"], 64)
            sample = group["test"][0]
            self.assertEqual(tuple(sample["q"][0].shape), (8, 256))
            self.assertEqual(tuple(sample["q"][1].shape), (8, 16))
            self.assertEqual(tuple(sample["k"][0].shape), (8, kv * 64))
            self.assertEqual(tuple(sample["k"][1].shape), (8, kv * 4))
            self.assertEqual(tuple(sample["v"][0].shape), (8, kv * 64))
            self.assertEqual(len(group["calib"]), 1)
            self.assertEqual(len(group["test"]), 2)

    def test_attention_group_validations(self):
        with self.assertRaises(ValueError):  # head_dim % 64 != 0
            syn.make_attention_group(seed=1, head_dim=32)
        with self.assertRaises(ValueError):  # qh % kvh != 0
            syn.make_attention_group(seed=1, q_num_heads=3, kv_num_heads=2)


class TestEvaluatorGuards(unittest.TestCase):
    def test_state_legality(self):
        self.assertEqual(evaluator._state_legality_errors(None), [])
        self.assertEqual(
            evaluator._state_legality_errors({"scale": torch.ones(4)}), []
        )
        self.assertTrue(
            evaluator._state_legality_errors(torch.ones(1, dtype=torch.float64))
        )
        self.assertTrue(evaluator._state_legality_errors({1: "bad key"}))
        self.assertTrue(evaluator._state_legality_errors(float("nan")))

    def test_aggregate_rejects_case_mismatch(self):
        case = {
            "scenario": "linear",
            "group": 0,
            "test": 0,
            "dist": "normal",
            "attn_type": None,
            "shape": {},
            "mse": 1.0,
        }
        with self.assertRaises(ValueError):
            evaluator._aggregate({"cases": [case]}, {"cases": []}, {})
        with self.assertRaises(ValueError):
            evaluator._aggregate(
                {"cases": [case]}, {"cases": [dict(case, test=1)]}, {}
            )

    def test_suite_id_changes_with_config(self):
        config = {"seed": 0, "public_mini_sample": False}
        self.assertEqual(
            evaluator._suite_id(config), evaluator._suite_id(dict(config))
        )
        self.assertNotEqual(
            evaluator._suite_id(config),
            evaluator._suite_id({"seed": 1, "public_mini_sample": False}),
        )


# ===========================================================================
# solution.py: interface and output-format legality
# ===========================================================================


class TestSolutionInterfaces(unittest.TestCase):
    def test_all_six_apis_exist_and_callable(self):
        for name in SOLUTION_APIS:
            self.assertTrue(callable(getattr(solution, name, None)), name)

    def test_weight_calibration_output_contract(self):
        group = syn.make_linear_group(
            seed=1, dist="normal", out_features=16, in_features=64,
            seq_len=8, n_calib=2, n_test=1,
        )
        wq, ws = group["weight"]
        result = solution.hif4_calibration_and_quantize_weight(
            wq, ws, group["calib_activation_list"]
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(
            sorted(result), ["activation_state", "weight_params"]
        )
        assert_hif4_params_legal(self, result["weight_params"], wq.shape, "weight")
        assert_state_legal(self, result["activation_state"], "activation_state")

    def test_activation_output_contract(self):
        group = syn.make_linear_group(seed=2, n_calib=1, n_test=1)
        tq, ts = group["test_activation_list"][0]
        params = solution.hif4_dynamic_quantize_activation(tq, ts, None)
        assert_hif4_params_legal(self, params, tq.shape, "activation")
        # 1-D input is also supported (shape contract still holds).
        flat = tq[0]
        flat_scale = ts[0]
        params1d = solution.hif4_dynamic_quantize_activation(flat, flat_scale, None)
        assert_hif4_params_legal(self, params1d, flat.shape, "activation-1d")

    def test_attention_calibration_output_contract(self):
        group = syn.make_attention_group(seed=3, n_calib=2, n_test=1)
        result = solution.hif4_calibration_attention(
            group["calib"], 4, 2, 64
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(sorted(result), ["k_state", "q_state", "v_state"])
        for role in ("q", "k", "v"):
            assert_state_legal(self, result[f"{role}_state"], f"{role}_state")

    def test_qkv_output_contract(self):
        for kv in (4, 2, 1):  # MHA, GQA, MQA
            group = syn.make_attention_group(
                seed=4, dist="normal", q_num_heads=4, kv_num_heads=kv,
                head_dim=64, seq_len=8, n_calib=1, n_test=1,
            )
            cal = solution.hif4_calibration_attention(
                group["calib"], 4, kv, 64
            )
            sample = group["test"][0]
            q_params = solution.hif4_dynamic_quantize_q(
                sample["q"][0], sample["q"][1], 4, 64, cal["q_state"]
            )
            k_params = solution.hif4_dynamic_quantize_k(
                sample["k"][0], sample["k"][1], kv, 64, cal["k_state"]
            )
            v_params = solution.hif4_dynamic_quantize_v(
                sample["v"][0], sample["v"][1], kv, 64, cal["v_state"]
            )
            assert_hif4_params_legal(self, q_params, sample["q"][0].shape, "q")
            assert_hif4_params_legal(self, k_params, sample["k"][0].shape, "k")
            assert_hif4_params_legal(self, v_params, sample["v"][0].shape, "v")


# ===========================================================================
# solution.py: no input mutation, deterministic repeated calls
# ===========================================================================


class TestSolutionStability(unittest.TestCase):
    def _linear_case(self):
        group = syn.make_linear_group(
            seed=5, dist="normal", out_features=16, in_features=64,
            seq_len=8, n_calib=2, n_test=1,
        )
        return group

    def _attention_case(self, kv=2):
        return syn.make_attention_group(
            seed=6, dist="normal", q_num_heads=4, kv_num_heads=kv,
            head_dim=64, seq_len=8, n_calib=2, n_test=1,
        )

    @staticmethod
    def _params_equal(a, b):
        return all(torch.equal(a[k], b[k]) for k in a)

    def test_weight_calibration_no_mutation_and_deterministic(self):
        group = self._linear_case()
        wq, ws = group["weight"]
        wq0, ws0 = wq.clone(), ws.clone()
        cal0 = [_clone_pair(p) for p in group["calib_activation_list"]]
        r1 = solution.hif4_calibration_and_quantize_weight(
            wq, ws, group["calib_activation_list"]
        )
        r2 = solution.hif4_calibration_and_quantize_weight(
            wq, ws, group["calib_activation_list"]
        )
        self.assertTrue(torch.equal(wq, wq0) and torch.equal(ws, ws0))
        for i, (p, p0) in enumerate(
            zip(group["calib_activation_list"], cal0)
        ):
            self.assertTrue(torch.equal(p[0], p0[0]), f"calib[{i}] quant")
            self.assertTrue(torch.equal(p[1], p0[1]), f"calib[{i}] scale")
        self.assertTrue(
            self._params_equal(r1["weight_params"], r2["weight_params"])
        )
        assert_state_equal(self, r1["activation_state"], r2["activation_state"])

    def test_activation_no_mutation_and_deterministic(self):
        group = self._linear_case()
        tq, ts = group["test_activation_list"][0]
        tq0, ts0 = tq.clone(), ts.clone()
        p1 = solution.hif4_dynamic_quantize_activation(tq, ts, None)
        p2 = solution.hif4_dynamic_quantize_activation(tq, ts, None)
        self.assertTrue(torch.equal(tq, tq0) and torch.equal(ts, ts0))
        self.assertTrue(self._params_equal(p1, p2))

    def test_attention_calibration_no_mutation_and_deterministic(self):
        group = self._attention_case()
        cal0 = _clone_calib_qkv_list(group["calib"])
        r1 = solution.hif4_calibration_attention(group["calib"], 4, 2, 64)
        r2 = solution.hif4_calibration_attention(group["calib"], 4, 2, 64)
        for i, (s, s0) in enumerate(zip(group["calib"], cal0)):
            for role in ("q", "k", "v"):
                self.assertTrue(torch.equal(s[role][0], s0[role][0]),
                                f"calib[{i}].{role}")
                self.assertTrue(torch.equal(s[role][1], s0[role][1]),
                                f"calib[{i}].{role}")
        for role in ("q", "k", "v"):
            assert_state_equal(
                test=self,
                left=r1[f"{role}_state"],
                right=r2[f"{role}_state"],
            )

    def test_qkv_no_mutation_and_deterministic(self):
        group = self._attention_case(kv=2)
        cal = solution.hif4_calibration_attention(group["calib"], 4, 2, 64)
        sample = group["test"][0]
        snapshots = {role: _clone_pair(sample[role]) for role in ("q", "k", "v")}
        q1 = solution.hif4_dynamic_quantize_q(
            sample["q"][0], sample["q"][1], 4, 64, cal["q_state"]
        )
        k1 = solution.hif4_dynamic_quantize_k(
            sample["k"][0], sample["k"][1], 2, 64, cal["k_state"]
        )
        v1 = solution.hif4_dynamic_quantize_v(
            sample["v"][0], sample["v"][1], 2, 64, cal["v_state"]
        )
        for role in ("q", "k", "v"):
            self.assertTrue(
                torch.equal(sample[role][0], snapshots[role][0]), role
            )
            self.assertTrue(
                torch.equal(sample[role][1], snapshots[role][1]), role
            )
        q2 = solution.hif4_dynamic_quantize_q(
            sample["q"][0], sample["q"][1], 4, 64, cal["q_state"]
        )
        k2 = solution.hif4_dynamic_quantize_k(
            sample["k"][0], sample["k"][1], 2, 64, cal["k_state"]
        )
        v2 = solution.hif4_dynamic_quantize_v(
            sample["v"][0], sample["v"][1], 2, 64, cal["v_state"]
        )
        self.assertTrue(self._params_equal(q1, q2))
        self.assertTrue(self._params_equal(k1, k2))
        self.assertTrue(self._params_equal(v1, v2))


# ===========================================================================
# solution.py: end-to-end MSE sanity against reference NVFP4 outputs
# ===========================================================================


class TestSolutionEndToEnd(unittest.TestCase):
    """Quantization must not blow up the output: error well below signal power.

    Tight bound (ratio <= 0.1) for the well-behaved normal distribution where
    the candidate measures ~0.01-0.02; a loose sanity bound (ratio <= 4.0)
    applies to every distribution, including mixed-block-scale stress cases.
    """

    def test_linear_normal_distribution(self):
        for seed in (7, 8):
            group = syn.make_linear_group(
                seed=seed, dist="normal", out_features=16, in_features=64,
                seq_len=8, n_calib=1, n_test=1,
            )
            wq, ws = group["weight"]
            result = solution.hif4_calibration_and_quantize_weight(
                wq, ws, group["calib_activation_list"]
            )
            w_hif = ref.dequantize_hif4_params(
                result["weight_params"], wq.shape
            )
            tq, ts = group["test_activation_list"][0]
            act = solution.hif4_dynamic_quantize_activation(
                tq, ts, result["activation_state"]
            )
            x_hif = ref.dequantize_hif4_params(act, tq.shape)
            out_ref = ref.linear_output_nvfp4(tq, ts, wq, ws)
            out_hif = ref.linear_output(x_hif, w_hif)
            self.assertTrue(torch.isfinite(out_hif).all())
            var = float(out_ref.double().pow(2).mean())
            mse = ref.scalar_mse(out_hif, out_ref)
            self.assertLess(mse, 0.1 * max(var, 1e-9), f"seed={seed}")

    def test_linear_all_distributions_sanity(self):
        for dist in syn.DISTS:
            group = syn.make_linear_group(
                seed=9, dist=dist, out_features=16, in_features=64,
                seq_len=8, n_calib=1, n_test=1,
            )
            wq, ws = group["weight"]
            result = solution.hif4_calibration_and_quantize_weight(
                wq, ws, group["calib_activation_list"]
            )
            w_hif = ref.dequantize_hif4_params(result["weight_params"], wq.shape)
            tq, ts = group["test_activation_list"][0]
            act = solution.hif4_dynamic_quantize_activation(
                tq, ts, result["activation_state"]
            )
            x_hif = ref.dequantize_hif4_params(act, tq.shape)
            out_ref = ref.linear_output_nvfp4(tq, ts, wq, ws)
            out_hif = ref.linear_output(x_hif, w_hif)
            var = float(out_ref.double().pow(2).mean())
            mse = ref.scalar_mse(out_hif, out_ref)
            self.assertLess(mse, 4.0 * max(var, 1e-9), f"dist={dist}")

    def test_attention_normal_distribution(self):
        for kv, name in ((4, "MHA"), (2, "GQA"), (1, "MQA")):
            for seed in (10, 11):
                group = syn.make_attention_group(
                    seed=seed, dist="normal", q_num_heads=4, kv_num_heads=kv,
                    head_dim=64, seq_len=8, n_calib=1, n_test=1,
                )
                cal = solution.hif4_calibration_attention(
                    group["calib"], 4, kv, 64
                )
                sample = group["test"][0]
                q = ref.dequantize_hif4_params(
                    solution.hif4_dynamic_quantize_q(
                        sample["q"][0], sample["q"][1], 4, 64, cal["q_state"]
                    ),
                    sample["q"][0].shape,
                )
                k = ref.dequantize_hif4_params(
                    solution.hif4_dynamic_quantize_k(
                        sample["k"][0], sample["k"][1], kv, 64, cal["k_state"]
                    ),
                    sample["k"][0].shape,
                )
                v = ref.dequantize_hif4_params(
                    solution.hif4_dynamic_quantize_v(
                        sample["v"][0], sample["v"][1], kv, 64, cal["v_state"]
                    ),
                    sample["v"][0].shape,
                )
                out_ref = ref.attention_output_nvfp4(
                    sample["q"][0], sample["q"][1],
                    sample["k"][0], sample["k"][1],
                    sample["v"][0], sample["v"][1],
                    4, kv, 64,
                )
                out_hif = ref.attention_output(q, k, v, 4, kv, 64)
                self.assertTrue(torch.isfinite(out_hif).all())
                var = float(out_ref.double().pow(2).mean())
                mse = ref.scalar_mse(out_hif, out_ref)
                self.assertLess(
                    mse, 0.1 * max(var, 1e-9), f"{name} seed={seed}"
                )

    def test_attention_all_distributions_sanity(self):
        for dist in syn.DISTS:
            group = syn.make_attention_group(
                seed=12, dist=dist, q_num_heads=4, kv_num_heads=2,
                head_dim=64, seq_len=8, n_calib=1, n_test=1,
            )
            cal = solution.hif4_calibration_attention(
                group["calib"], 4, 2, 64
            )
            sample = group["test"][0]
            q = ref.dequantize_hif4_params(
                solution.hif4_dynamic_quantize_q(
                    sample["q"][0], sample["q"][1], 4, 64, cal["q_state"]
                ),
                sample["q"][0].shape,
            )
            k = ref.dequantize_hif4_params(
                solution.hif4_dynamic_quantize_k(
                    sample["k"][0], sample["k"][1], 2, 64, cal["k_state"]
                ),
                sample["k"][0].shape,
            )
            v = ref.dequantize_hif4_params(
                solution.hif4_dynamic_quantize_v(
                    sample["v"][0], sample["v"][1], 2, 64, cal["v_state"]
                ),
                sample["v"][0].shape,
            )
            out_ref = ref.attention_output_nvfp4(
                sample["q"][0], sample["q"][1],
                sample["k"][0], sample["k"][1],
                sample["v"][0], sample["v"][1],
                4, 2, 64,
            )
            out_hif = ref.attention_output(q, k, v, 4, 2, 64)
            var = float(out_ref.double().pow(2).mean())
            mse = ref.scalar_mse(out_hif, out_ref)
            self.assertLess(mse, 4.0 * max(var, 1e-9), f"dist={dist}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
