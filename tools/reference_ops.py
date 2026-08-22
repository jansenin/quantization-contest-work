"""Reference numerical ops for the NVFP4-to-HiF4 contest (CPU / FP32).

These helpers define the *reference* computations that the contest scoring
compares against, following the contract in ``statements_from_docx.txt`` and
``docs/problem-contract.md``:

* ``dequantize_nvfp4``   -- NVFP4 carrier + per-16-block scale -> BF16, with the
  exact op sequence of the statement's reference helper
  (``quant.unflatten(-1, (-1, 16)) * scale.unsqueeze(-1)`` then flatten and
  cast to ``bfloat16``).
* ``dequantize_hif4``    -- HiF4 five-tensor params + logical shape -> FP32,
  reconstructing ``sign * mant * scale_lv2 * scale_lv3 * scale_factor`` over
  the intra-block ``(8, 2, 4)`` layout, reshaped to the logical shape.
* ``linear_output``      -- ``X @ W^T`` in FP32 (dequantized inputs).
* ``attention_output``   -- ``softmax(Q K^T / sqrt(head_dim)) V`` in FP32 with
  contiguous-head MHA/MQA/GQA mapping and optional causal masking.
* ``scalar_mse``         -- scalar mean squared error as a Python float.

Semantics used here (local-evaluation assumptions, see
``docs/problem-contract.md`` section 4): inputs are dequantized to FP32 before
MatMul/Attention, and the final compute is carried out in FP32 on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

__all__ = [
    "dequantize_nvfp4",
    "dequantize_hif4",
    "dequantize_hif4_params",
    "linear_output",
    "linear_output_nvfp4",
    "attention_output",
    "attention_output_nvfp4",
    "scalar_mse",
]

# Trailing block layout of the five HiF4 parameters. For a logical shape
# (*prefix, C) with C % 64 == 0, every parameter has shape
# (*prefix, C // 64, *trailing); see example/self_check.py.
_HIF4_LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("scale_factor", (1, 1, 1)),
    ("scale_lv2", (8, 1, 1)),
    ("scale_lv3", (8, 2, 1)),
    ("sign", (8, 2, 4)),
    ("mant", (8, 2, 4)),
)

_HIF4_BLOCK_SIZE = 64
_NVFP4_BLOCK_SIZE = 16


def _as_int_shape(shape: Sequence[int], name: str) -> tuple[int, ...]:
    """Normalize a shape to a tuple of positive ints (defensive)."""
    result = tuple(int(s) for s in shape)
    if not result:
        raise ValueError(f"{name} must be a non-empty shape, got {tuple(shape)}")
    if any(s <= 0 for s in result):
        raise ValueError(
            f"{name} must contain only positive sizes, got {tuple(shape)}"
        )
    return result


def dequantize_nvfp4(
    quant: torch.Tensor,
    scale: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an NVFP4 carrier + per-block scale (BF16 reference semantics).

    Replicates the statement's reference helper: one scale per ``blk_size``
    values, multiplied element-wise after unflattening, then flattened back and
    cast to ``dtype`` (default ``torch.bfloat16``).

    Args:
        quant:
            NVFP4 value carrier, shape ``(..., C)`` with ``C % blk_size == 0``.
        scale:
            NVFP4 block scale, shape ``(..., C // blk_size)``.
        blk_size:
            NVFP4 block size (16 per the contest contract).
        dtype:
            Output dtype; ``torch.bfloat16`` matches the provided semantics.

    Returns:
        Tensor of shape ``quant.shape`` in ``dtype``.
    """
    if not isinstance(quant, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("quant and scale must be torch.Tensor")
    if quant.ndim < 1:
        raise ValueError(
            f"quant must have at least one dimension, got shape {tuple(quant.shape)}"
        )
    if int(blk_size) <= 0:
        raise ValueError(f"blk_size must be positive, got {blk_size!r}")

    channels = int(quant.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )
    expected_scale = tuple(quant.shape[:-1]) + (channels // blk_size,)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} != expected {expected_scale}"
        )

    x = quant.unflatten(-1, (-1, blk_size))
    x = x * scale.unsqueeze(-1)
    return x.flatten(-2, -1).to(dtype)


def dequantize_hif4(
    scale_factor: torch.Tensor,
    scale_lv2: torch.Tensor,
    scale_lv3: torch.Tensor,
    sign: torch.Tensor,
    mant: torch.Tensor,
    logical_shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize the five HiF4 parameters back to a tensor of ``logical_shape``.

    The dequantization relation is ``x_hat = sign * mant * scale_lv2 *
    scale_lv3 * scale_factor``, broadcast over the intra-block ``(8, 2, 4)``
    layout and reshaped to the logical shape (matching ``self_check.py``).

    Args:
        scale_factor:
            Shape ``(*prefix, C // 64, 1, 1, 1)`` (E6M2 per-64-block scale).
        scale_lv2:
            Shape ``(*prefix, C // 64, 8, 1, 1)``.
        scale_lv3:
            Shape ``(*prefix, C // 64, 8, 2, 1)``.
        sign:
            Shape ``(*prefix, C // 64, 8, 2, 4)``.
        mant:
            Shape ``(*prefix, C // 64, 8, 2, 4)``.
        logical_shape:
            Original tensor shape ``(*prefix, C)`` with ``C % 64 == 0``.
        dtype:
            Output dtype (default FP32).

    Returns:
        Tensor of shape ``logical_shape``.
    """
    shape = _as_int_shape(logical_shape, "logical_shape")
    channels = shape[-1]
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by "
            f"HiF4 block size {_HIF4_BLOCK_SIZE}"
        )
    prefix = shape[:-1] + (channels // _HIF4_BLOCK_SIZE,)

    tensors = {
        "scale_factor": scale_factor,
        "scale_lv2": scale_lv2,
        "scale_lv3": scale_lv3,
        "sign": sign,
        "mant": mant,
    }
    for name, trailing in _HIF4_LAYOUT:
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got {type(value).__name__}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        expected = prefix + trailing
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} != expected {expected}"
            )

    value = (
        sign.to(torch.float32)
        * mant.to(torch.float32)
        * scale_lv2.to(torch.float32)
        * scale_lv3.to(torch.float32)
        * scale_factor.to(torch.float32)
    )
    return value.to(dtype).reshape(shape)


def dequantize_hif4_params(
    params: Mapping[str, torch.Tensor],
    logical_shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convenience wrapper around :func:`dequantize_hif4` for a params dict.

    ``params`` must contain the five keys ``scale_factor``, ``scale_lv2``,
    ``scale_lv3``, ``sign``, ``mant`` (extra keys are ignored).
    """
    if not isinstance(params, Mapping):
        raise TypeError(
            "params must be a mapping of the five HiF4 tensors, "
            f"got {type(params).__name__}"
        )
    missing = [name for name, _ in _HIF4_LAYOUT if name not in params]
    if missing:
        raise KeyError(f"params is missing HiF4 parameters: {sorted(missing)}")
    return dequantize_hif4(
        params["scale_factor"],
        params["scale_lv2"],
        params["scale_lv3"],
        params["sign"],
        params["mant"],
        logical_shape,
        dtype=dtype,
    )


def linear_output(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Reference Linear output ``X @ W^T`` computed in FP32.

    Args:
        x:
            Dequantized activation, shape ``(..., M, K)``.
        w:
            Dequantized weight, shape ``(..., N, K)``.

    Returns:
        FP32 tensor of shape ``(..., M, N)``.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(w, torch.Tensor):
        raise TypeError("x and w must be torch.Tensor")
    if x.ndim < 2 or w.ndim < 2:
        raise ValueError(
            f"x and w must have at least two dimensions, got "
            f"{tuple(x.shape)} and {tuple(w.shape)}"
        )
    if x.shape[-1] != w.shape[-1]:
        raise ValueError(
            f"inner dimension mismatch: x {tuple(x.shape)} vs w {tuple(w.shape)}"
        )
    return torch.matmul(
        x.to(torch.float32),
        w.to(torch.float32).transpose(-1, -2),
    )


def linear_output_nvfp4(
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    w_quant: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference Linear output from NVFP4 pairs: dequantize, then ``X @ W^T``."""
    return linear_output(
        dequantize_nvfp4(x_quant, x_scale),
        dequantize_nvfp4(w_quant, w_scale),
    )


def attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    causal: bool = False,
) -> torch.Tensor:
    """Reference scaled dot-product Attention, computed in FP32.

    Computes ``softmax(Q K^T / sqrt(head_dim)) V`` with heads laid out
    contiguously in the hidden dimension (``[..., seq, num_heads * head_dim]``).
    Supports MHA (``q_num_heads == kv_num_heads``), MQA (``kv_num_heads == 1``)
    and GQA (``q_num_heads % kv_num_heads == 0``): the GQA mapping is
    contiguous, i.e. Q head ``h = j * rep + r`` (``rep = q_num_heads //
    kv_num_heads``) reads the K/V head ``j``.

    Args:
        q:
            Dequantized query, shape ``(..., seq_q, q_num_heads * head_dim)``.
        k:
            Dequantized key, shape ``(..., seq_k, kv_num_heads * head_dim)``.
        v:
            Dequantized value, shape ``(..., seq_k, kv_num_heads * head_dim)``.
        q_num_heads:
            Number of query heads.
        kv_num_heads:
            Number of key/value heads (must divide ``q_num_heads``).
        head_dim:
            Dimension of each head.
        causal:
            If True, query position ``i`` attends only to key positions
            ``j <= i`` (requires ``seq_k >= seq_q``).

    Returns:
        FP32 tensor of shape ``(..., seq_q, q_num_heads * head_dim)``.
    """
    if not isinstance(q_num_heads, int) or q_num_heads <= 0:
        raise ValueError(f"q_num_heads must be a positive int, got {q_num_heads!r}")
    if not isinstance(kv_num_heads, int) or kv_num_heads <= 0:
        raise ValueError(
            f"kv_num_heads must be a positive int, got {kv_num_heads!r}"
        )
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive int, got {head_dim!r}")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError(
            f"q_num_heads {q_num_heads} must be divisible by "
            f"kv_num_heads {kv_num_heads}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim < 2:
            raise ValueError(
                f"{name} must have shape [..., seq, hidden], got {tuple(tensor.shape)}"
            )
    kv_hidden = kv_num_heads * head_dim
    if q.shape[-1] != q_num_heads * head_dim:
        raise ValueError(
            f"q last dim {q.shape[-1]} != q_num_heads * head_dim "
            f"{q_num_heads * head_dim}"
        )
    if k.shape[-1] != kv_hidden or v.shape[-1] != kv_hidden:
        raise ValueError(
            f"k/v last dim ({k.shape[-1]}, {v.shape[-1]}) != "
            f"kv_num_heads * head_dim {kv_hidden}"
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"k and v sequence lengths differ: {k.shape[-2]} vs {v.shape[-2]}"
        )
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("batch prefix of q, k and v must match")

    seq_q = int(q.shape[-2])
    seq_k = int(k.shape[-2])
    if causal and seq_k < seq_q:
        raise ValueError(
            f"causal attention requires seq_k ({seq_k}) >= seq_q ({seq_q})"
        )

    # (..., seq_q, q_num_heads, head_dim) -> (..., seq_q, kv, rep, head_dim)
    rep = q_num_heads // kv_num_heads
    batch = tuple(q.shape[:-2])
    q = (
        q.to(torch.float32)
        .unflatten(-1, (q_num_heads, head_dim))
        .reshape(*batch, seq_q, kv_num_heads, rep, head_dim)
    )
    q = q * (head_dim ** -0.5)
    k = k.to(torch.float32).unflatten(-1, (kv_num_heads, head_dim))
    v = v.to(torch.float32).unflatten(-1, (kv_num_heads, head_dim))

    # scores: (..., seq_q, kv, rep, seq_k)
    scores = torch.einsum("...sire,...tie->...sirt", q, k)
    if causal:
        mask = torch.arange(seq_k, device=scores.device)[None, :].gt(
            torch.arange(seq_q, device=scores.device)[:, None]
        )
        # Align the (seq_q, seq_k) mask with the (..., seq_q, kv, rep, seq_k)
        # scores layout so the broadcast does not expand the rep dimension.
        mask = mask.view(seq_q, 1, 1, seq_k)
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    # out: (..., seq_q, kv, rep, head_dim) -> (..., seq_q, q_num_heads*head_dim)
    out = torch.einsum("...sirt,...tie->...sire", probs, v)
    return out.reshape(*batch, seq_q, q_num_heads * head_dim)


def attention_output_nvfp4(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    causal: bool = False,
) -> torch.Tensor:
    """Reference Attention output from NVFP4 pairs: dequantize, then attend."""
    return attention_output(
        dequantize_nvfp4(q_quant, q_scale),
        dequantize_nvfp4(k_quant, k_scale),
        dequantize_nvfp4(v_quant, v_scale),
        q_num_heads,
        kv_num_heads,
        head_dim,
        causal=causal,
    )


def scalar_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Scalar mean squared error ``mean((pred - target)^2)`` in FP32.

    Args:
        pred:
            Predicted tensor (e.g. HiF4 output).
        target:
            Reference tensor (e.g. NVFP4 reference output); same shape.

    Returns:
        Python float; inputs are cast to FP32 before reduction.
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch.Tensor")
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    diff = pred.to(torch.float32) - target.to(torch.float32)
    return float(torch.mean(diff * diff).item())
