#!/usr/bin/env python3
"""Deterministic local NVFP4 (E2M1 carrier + per-16 E4M3FN scale) quantizer.

Produces contest-format NVFP4 pairs ``[quant, scale]`` from arbitrary
floating-point tensors whose last dimension is divisible by ``block_size``:

* ``quant``: BF16 tensor (on the input's device) of the same shape as the
  input whose values are exactly the decoded NVFP4 E2M1 carriers
  ``{-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}``.
* ``scale``: BF16 tensor (on the input's device) of shape
  ``value.shape[:-1] + (C // block_size,)`` holding one exact positive finite
  E4M3FN value per block of ``block_size`` (16 per the contest contract)
  consecutive values.

The scale recipe follows the public-sample fingerprint
(``docs/public-nvfp4-fingerprint.md``): per-16 scales are exact positive
E4M3FN values derived from the block maximum ``max_abs``.  Every E2M1 carrier
and E4M3FN scale is exact in BF16, and a carrier*scale product needs at most
6 significant bits (carrier significands are odd integers up to 3, scale
significands odd integers up to 15, so the product significand is an odd
integer up to 45, i.e. <= 6 bits; BF16 has an 8-bit significand).  Every
``quant * scale`` product is therefore exactly representable in BF16: the
reference ``dequantize_nvfp4`` adds no rounding on top of these outputs.

Scale modes (``scale_mode``)
----------------------------
Let ``t = max_abs / 6`` be the smallest scale that represents the whole block
without clipping (carrier magnitude ``6`` is the E2M1 maximum).

* ``"ceil"``:      the smallest E4M3FN value ``>= t``.  This is the
  ``scale == ceil_E4M3(max_abs / 6)`` identity observed on the public sample.
* ``"nearest"``:   the E4M3FN value closest to ``t``; exact ties (``t``
  exactly midway between two adjacent grid values) round up to the larger
  value.  A nearest scale can be *below* ``t`` when ``t`` is closer to the
  lower grid value -- such blocks clip carriers at magnitude 6.  The tie
  convention only biases the exact-tie case toward the larger scale; it does
  not guarantee clipping never happens.
* ``"stochastic"``: seeded, per-block independent choice: if ``t`` is exactly
  an E4M3FN value the block deterministically gets that value; otherwise for
  adjacent grid values ``a <= t <= b`` the block draws one uniform ``u`` in
  ``[0, 1)`` and takes ``b`` with probability ``(t - a) / (b - a)`` and ``a``
  with probability ``(b - t) / (b - a)``.

Stochastic draws
----------------
Draws never touch global RNG state.  Block ``i`` (flattened row-major block
index, see below) draws from a counter-based SplitMix64-style mixer:

    u_i = mix64(base ^ mix64(i)) / 2**63,
    base = mix64(FNV-1a(seed | tensor_identity)).

``tensor_identity`` is the explicit ``tensor_id`` argument when given, else
the empty string.  The default draw sequence therefore depends on the seed
alone: it is reproducible across processes and tensor objects.  Consequence:
under the default identity, every tensor quantized with the same seed shares
the same draw sequence, so scale choices are correlated across tensors.
Dataset callers should pass a distinct stable ``tensor_id`` per logical
tensor to decorrelate them.

Because ``u_i`` depends only on ``(seed, tensor_id, i)``, results are
invariant to how the work is partitioned: quantizing the whole tensor in one
call gives bitwise-identical outputs to processing arbitrary chunks in any
order, provided each chunk uses the same ``tensor_id`` and the correct
``block_offset`` (the flattened block index of its first block).

Bounded chunked processing
--------------------------
The tensor is read in row-major chunks of ``chunk_blocks`` blocks (default
``DEFAULT_CHUNK_BLOCKS = 65536``, i.e. 1,048,576 elements per chunk with the
contest block size 16).  Only one chunk at a time is materialized in float64,
so the peak temporary memory is conservatively about 1.1 KB per block
(~70 MB at the default chunk size; measured peak including allocator warm-up
is ~130 MB), plus the preallocated BF16 outputs and the caller's input
tensor, independent of the total tensor size.  Finiteness is validated per
chunk (no whole-input boolean temporary), and a non-finite element anywhere
raises ``ValueError`` before any output can escape (outputs are function
locals, so a late raise never leaks a partial result).  Non-contiguous
inputs incur one reshape copy in the input dtype.  Every per-block
computation is local to its chunk, so the output is bit-identical for every
``chunk_blocks`` value and for all scale modes.

Robust edge behavior
--------------------
* All-zero block: scale ``E4M3_MIN = 2**-9`` (the smallest positive E4M3FN
  value) and all-zero carriers in every mode; dequantization is exactly zero
  regardless of the chosen scale.
* Overflow: a block with ``max_abs > 6 * E4M3_MAX = 2688`` saturates: the
  scale is ``E4M3_MAX = 448`` and carriers clip at magnitude ``6``, so the
  reconstruction saturates at ``+-2688`` and is always finite.  No mode
  raises on overflow; the error is bounded, never NaN/Inf.
* Signed zeros: a negative nonzero source value whose magnitude rounds to the
  zero carrier keeps its sign as ``-0.0`` (the public sample contains such
  carriers); exact source zeros (``+0.0`` and ``-0.0``) are normalized to
  ``+0.0``.
* Non-finite input (NaN/Inf) raises ``ValueError``.
* Carrier quantization is deterministic in every mode: the nearest legal E2M1
  magnitude with exact ties rounded away from zero in magnitude
  (``0.25 -> 0.5, 0.75 -> 1.0, 1.25 -> 1.5, 1.75 -> 2.0, 2.5 -> 3.0,
  3.5 -> 4.0, 5.0 -> 6.0``) and magnitudes above ``6`` clipped to ``6``.

Precision
---------
All internal arithmetic is float64 (input is cast from any floating dtype
without loss, since bf16/fp16/fp32 are subsets of fp64).  The E4M3 and E2M1
grids are exact dyadic rationals, so every comparison, tie, and probability
is bitwise deterministic across platforms.  Outputs are BF16; both grids are
exact in BF16.
"""

from __future__ import annotations

from typing import Any

import torch

from tools.fingerprint_nvfp4 import NVFP4_CARRIERS, positive_e4m3fn_values
from tools.reference_ops import dequantize_nvfp4  # re-exported for convenience

__all__ = [
    "E2M1_MAGNITUDES",
    "E2M1_MAX",
    "E4M3_POSITIVE_VALUES",
    "E4M3_MIN",
    "E4M3_MAX",
    "SCALE_MODES",
    "DEFAULT_CHUNK_BLOCKS",
    "quantize_nvfp4",
    "dequantize_nvfp4",
]

#: NVFP4 (E2M1) carrier magnitudes including zero, ascending.
E2M1_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: All 126 positive finite E4M3FN values, ascending (reused from the
#: fingerprint tool: single source of truth, no duplicate grid definition).
E4M3_POSITIVE_VALUES: tuple[float, ...] = positive_e4m3fn_values()

E4M3_MIN: float = E4M3_POSITIVE_VALUES[0]  # 2**-9
E4M3_MAX: float = E4M3_POSITIVE_VALUES[-1]  # 448.0
E2M1_MAX: float = E2M1_MAGNITUDES[-1]  # 6.0

SCALE_MODES: tuple[str, ...] = ("ceil", "nearest", "stochastic")

#: Reconstruction saturation magnitude (E2M1_MAX * E4M3_MAX).
SATURATION: float = E2M1_MAX * E4M3_MAX  # 2688.0

#: Default number of 16-value blocks processed per internal chunk.  Each chunk
#: materializes conservatively ~1.1 KB of temporaries per block (~70 MB at
#: this default; measured peak ~130 MB including allocator warm-up),
#: independent of the total tensor size.
DEFAULT_CHUNK_BLOCKS: int = 65536

_E4M3_GRID = torch.tensor(E4M3_POSITIVE_VALUES, dtype=torch.float64)
_E2M1_GRID = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float64)

# SplitMix64 works over unsigned 64-bit words; torch int64 is two's
# complement, so we work modulo 2**63 (all values stay non-negative and ``>>``
# behaves like a logical shift).  The canonical constants are reduced mod 2**63
# at import time.
_MASK63 = (1 << 63) - 1
_MIX_C1 = 0xBF58476D1CE4E5B9 & _MASK63
_MIX_C2 = 0x94D049BB133111EB & _MASK63

_FNV1A_OFFSET = 0xCBF29CE484222325
_FNV1A_PRIME = 0x100000001B3

#: Largest signed int64 value; stochastic counter-based draws use int64
#: indices in ``[block_offset, block_offset + n_blocks)``.
_INT64_MAX = 2**63 - 1


def _fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash over ``data`` (deterministic across processes)."""
    value = _FNV1A_OFFSET
    for byte in data:
        value ^= byte
        value = (value * _FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def _tensor_key(seed: int, tensor_id: str | int | None) -> int:
    """Deterministic 63-bit key for (seed, tensor identity).

    The default identity (``tensor_id is None``) is the empty string, so the
    key is a pure function of the seed: reproducible across processes and
    independent of the tensor object.  No ``id()`` or Python ``hash()`` is
    used anywhere in the draw path.
    """
    identity = "" if tensor_id is None else str(tensor_id)
    raw = _fnv1a64(f"{seed}|{identity}".encode("utf-8"))
    return raw & _MASK63


def _mix64(state: torch.Tensor) -> torch.Tensor:
    """SplitMix64-style integer mix on non-negative int64 tensors."""
    state = state & _MASK63
    state = (state ^ (state >> 30)) * _MIX_C1 & _MASK63
    state = (state ^ (state >> 27)) * _MIX_C2 & _MASK63
    return state ^ (state >> 31)


def _stochastic_uniform(
    count: int,
    key: int,
    block_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-block uniform draws in [0, 1), counter-based on (key, block index).

    Draw ``i`` (``0 <= i < count``) is a pure function of ``key`` and the
    global block index ``block_offset + i``, so chunking and processing order
    cannot change the result.
    """
    indices = torch.arange(
        block_offset, block_offset + count, dtype=torch.int64, device=device
    )
    base = torch.tensor(key, dtype=torch.int64, device=device)
    mixed = _mix64((base ^ _mix64(indices)) & _MASK63)
    return mixed.to(torch.float64) / float(1 << 63)


def _select_scale(
    target: torch.Tensor,
    mode: str,
    uniform: torch.Tensor | None,
) -> torch.Tensor:
    """Map per-block ``target = max_abs / 6`` to an exact positive E4M3FN scale.

    ``target`` is a float64 tensor; the result is a float64 tensor of the same
    shape holding grid values.  ``uniform`` is required only for
    ``"stochastic"``.
    """
    grid = _E4M3_GRID.to(target.device)
    n = grid.numel()
    if mode == "ceil":
        index = torch.searchsorted(grid, target, right=False)
        return grid[index.clamp(max=n - 1)]

    index_right = torch.searchsorted(grid, target, right=True)
    lower_index = (index_right - 1).clamp(min=0, max=n - 1)
    upper_index = index_right.clamp(min=0, max=n - 1)
    lower = grid[lower_index]
    upper = grid[upper_index]

    if mode == "nearest":
        # Ties (upper - target == target - lower) round up to the larger
        # scale.  At the extremes lower == upper == the clamped endpoint, so
        # this also handles target <= E4M3_MIN and target >= E4M3_MAX.
        take_upper = (upper - target) <= (target - lower)
        return torch.where(take_upper, upper, lower)

    # Stochastic: P(b) = (t - a)/(b - a).  The interval is degenerate at the
    # grid ends (lower == upper), where the masked division keeps the
    # probability explicit: span <= 0 forces take_upper to False, so the
    # endpoint is selected without ever producing a NaN.
    span = upper - lower
    safe_span = torch.where(span > 0.0, span, torch.ones_like(span))
    probability = (target - lower) / safe_span
    take_upper = (uniform < probability) & (span > 0.0)
    return torch.where(take_upper, upper, lower)


def _carrier_magnitude(ratio: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude with exact ties rounded away from zero.

    ``ratio`` is a non-negative float64 tensor.  Magnitudes above 6 clip to 6.
    """
    grid = _E2M1_GRID.to(ratio.device)
    n = grid.numel()
    index_right = torch.searchsorted(grid, ratio, right=True)
    lower_index = (index_right - 1).clamp(min=0, max=n - 1)
    upper_index = index_right.clamp(min=0, max=n - 1)
    lower = grid[lower_index]
    upper = grid[upper_index]
    # Ties (e.g. ratio 0.25 between 0 and 0.5) round away from zero.
    take_upper = (upper - ratio) <= (ratio - lower)
    return torch.where(take_upper, upper, lower)


def quantize_nvfp4(
    value: torch.Tensor,
    scale_mode: str = "ceil",
    seed: int = 0,
    tensor_id: str | int | None = None,
    block_offset: int = 0,
    block_size: int = 16,
    chunk_blocks: int = DEFAULT_CHUNK_BLOCKS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating tensor to contest-format NVFP4 (E2M1 + E4M3FN).

    Args:
        value:
            Floating-point tensor (float16/bfloat16/float32/float64) whose
            last dimension is divisible by ``block_size``.  All elements must
            be finite.
        scale_mode:
            One of ``"ceil"`` (default), ``"nearest"``, ``"stochastic"``.
        seed:
            Explicit seed for ``"stochastic"`` mode (ignored otherwise).
        tensor_id:
            Stable identity mixed into every stochastic draw; ``str`` or
            ``int``.  Defaults to the empty identity, so the draw sequence is
            a pure function of ``seed`` (reproducible across processes and
            tensor objects; same-seed tensors share a correlated draw
            sequence).  Dataset callers should pass a distinct ``tensor_id``
            per logical tensor.
        block_offset:
            Flattened (row-major) block index of this call's first block.
            Used with a shared ``tensor_id`` when a caller splits one logical
            tensor into chunks; keeps draws identical to a single full-tensor
            call regardless of chunk order.
        block_size:
            NVFP4 block size (16 per the contest contract).
        chunk_blocks:
            Number of blocks processed per internal chunk; results are
            bit-identical for every value (see "Bounded chunked processing").
            Default ``DEFAULT_CHUNK_BLOCKS``.

    Returns:
        ``(quant, scale)``: both BF16.  ``quant`` has ``value.shape`` and
        contains only decoded E2M1 carrier values; ``scale`` has shape
        ``value.shape[:-1] + (C // block_size,)`` and contains only positive
        finite E4M3FN values.

    Raises:
        TypeError: non-tensor, non-floating, or complex ``value``; bad
            ``seed``/``tensor_id``/``block_offset``/``chunk_blocks`` types
            (``bool`` is rejected for every one of them).
        ValueError: last dimension not divisible by ``block_size``; empty
            tensor; unknown ``scale_mode``; non-finite elements (detected per
            chunk; no partial output is returned); non-positive
            ``block_size``/``chunk_blocks``; negative ``block_offset``; in
            ``"stochastic"`` mode, an index range
            ``[block_offset, block_offset + n_blocks)`` beyond signed int64.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"value must be a torch.Tensor, got {type(value).__name__}")
    if not value.is_floating_point():
        raise TypeError(
            "value must be a floating-point tensor, got dtype "
            f"{value.dtype}"
        )
    if value.ndim < 1:
        raise ValueError(
            f"value must have at least one dimension, got shape {tuple(value.shape)}"
        )
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    channels = int(value.shape[-1])
    if channels % block_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {block_size}"
        )
    if value.numel() == 0:
        raise ValueError("value must be non-empty")
    if scale_mode not in SCALE_MODES:
        raise ValueError(
            f"unknown scale_mode {scale_mode!r}; choose from {SCALE_MODES}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if isinstance(block_offset, bool) or not isinstance(block_offset, int):
        raise TypeError(
            f"block_offset must be an int, got {type(block_offset).__name__}"
        )
    if block_offset < 0:
        raise ValueError(f"block_offset must be non-negative, got {block_offset}")
    if tensor_id is not None and (
        isinstance(tensor_id, bool) or not isinstance(tensor_id, (str, int))
    ):
        raise TypeError(
            f"tensor_id must be a str or int, got {type(tensor_id).__name__}"
        )
    if isinstance(chunk_blocks, bool) or not isinstance(chunk_blocks, int):
        raise TypeError(
            f"chunk_blocks must be an int, got {type(chunk_blocks).__name__}"
        )
    if chunk_blocks <= 0:
        raise ValueError(f"chunk_blocks must be positive, got {chunk_blocks}")

    device = value.device
    n_blocks = value.numel() // block_size
    if scale_mode == "stochastic":
        # The counter-based draw range is torch.arange(block_offset,
        # block_offset + n_blocks) on int64; both endpoints must fit.
        index_end = block_offset + n_blocks  # exclusive end
        if index_end > _INT64_MAX:
            raise ValueError(
                "stochastic mode requires block_offset + n_blocks <= "
                f"{_INT64_MAX} for its int64 index range, got "
                f"block_offset={block_offset}, n_blocks={n_blocks}"
            )
    # View when the input is contiguous; one input-dtype copy otherwise.
    blocks = value.reshape(-1, block_size)

    # Preallocate the BF16 outputs once; every chunk writes into a view.
    quant_out = torch.empty(value.shape, dtype=torch.bfloat16, device=device)
    scale_out = torch.empty(
        value.shape[:-1] + (channels // block_size,),
        dtype=torch.bfloat16,
        device=device,
    )
    quant_view = quant_out.reshape(-1, block_size)
    scale_view = scale_out.reshape(-1)

    key = _tensor_key(seed, tensor_id) if scale_mode == "stochastic" else None

    for start in range(0, n_blocks, chunk_blocks):
        stop = min(start + chunk_blocks, n_blocks)
        chunk = blocks[start:stop]  # (Nc, block_size) view in the input dtype
        x64 = chunk.to(torch.float64)
        # Per-chunk finiteness check: bounded boolean temporary, no
        # whole-input pass.  Raising here discards the local outputs, so no
        # partial result can ever be returned.
        if not torch.isfinite(x64).all():
            raise ValueError("value must contain only finite elements")
        max_abs = x64.abs().amax(dim=1)
        target = max_abs / 6.0  # exact float64 division, deterministic
        if scale_mode == "stochastic":
            uniform = _stochastic_uniform(
                stop - start, key, block_offset + start, device
            )
        else:
            uniform = None
        scale = _select_scale(target, scale_mode, uniform)

        ratio = (x64 / scale.unsqueeze(1)).abs()
        magnitude = _carrier_magnitude(ratio)

        # Signed-zero policy: exact source zeros (+0.0 / -0.0) normalize to
        # +0.0; negative nonzero source values whose magnitude rounds to the
        # zero carrier keep their sign as -0.0.
        negative = chunk < 0.0
        sign = torch.where(negative, -1.0, 1.0)
        carriers = magnitude * sign
        carriers = torch.where(
            negative & (magnitude == 0.0), -0.0, carriers
        )
        carriers = torch.where(chunk == 0.0, 0.0, carriers)

        quant_view[start:stop] = carriers.to(torch.bfloat16)
        scale_view[start:stop] = scale.to(torch.bfloat16)

    return quant_out, scale_out
