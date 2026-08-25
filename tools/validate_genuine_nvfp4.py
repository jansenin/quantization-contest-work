#!/usr/bin/env python3
"""Validate genuine ModelOpt NVFP4 checkpoints against the contest NVFP4 pair format.

ModelOpt (NVIDIA Model Optimizer) NVFP4 checkpoints store each Linear weight as
three tensors (verified against ``NVFP4/Qwen3-0.6B-FP4``):

- ``<name>.weight``: packed ``uint8`` ``[N, K/2]`` where each byte holds two
  E2M1 carriers: the low nibble is the carrier for the even K index and the
  high nibble is the carrier for the odd K index (K = the reduced/in-feature
  dimension, N = the output dimension);
- ``<name>.weight_scale``: one E4M3 block scale per 16 logical K values,
  shape ``[N, K/16]`` (stored as ``float8_e4m3fn`` in current checkpoints);
- ``<name>.weight_scale_2``: a scalar ``float32`` per-tensor global factor.

The contest format instead stores a self-contained BF16 pair: a BF16 carrier
tensor of shape ``(..., C)`` (``C % 16 == 0``) and a BF16 scale tensor of shape
``(..., C // 16)``.  This tool decodes the ModelOpt packing, folds the global
``weight_scale_2`` factor into the per-16 scale, and validates the result
against an optional BF16 parent checkpoint (the original unquantized weights),
reporting canonical / swapped / no-global hypotheses so that a genuine
ModelOpt checkpoint is positively discriminated from mis-decoded or
global-scale-free data.

The snapshot reader is memory-bounded: safetensors is imported lazily, tensor
headers are read without loading data, and each 2-D tensor group is processed
in row chunks of ``--chunk-rows``; a full model is never loaded.

Examples::

    python tools/validate_genuine_nvfp4.py \\
        --nvfp4-snapshot data/huggingface-cache/models--NVFP4--Qwen3-0.6B-FP4/snapshots/<rev>/ \\
        --bf16-snapshot data/huggingface-cache/models--Qwen--Qwen3-0.6B/snapshots/<rev>/ \\
        --json-output /tmp/nvfp4-report.json \\
        --markdown-output /tmp/nvfp4-report.md

    python tools/validate_genuine_nvfp4.py \\
        --nvfp4-snapshot model.safetensors --tensor v_proj --tensor down_proj
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

if __package__ in (None, ""):  # allow `python tools/validate_genuine_nvfp4.py` too
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


#: NVFP4 E2M1 value set indexed by nibble: indices 0-7 are positive, 8-15 are
#: the sign-encoded mirror (index 8 is NVFP4's negative zero).
E2M1_TABLE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
E2M1_TENSOR = torch.tensor(E2M1_TABLE, dtype=torch.float32)
E2M1_SET = frozenset(E2M1_TABLE)

#: Contest NVFP4 block size (one scale per 16 K values, per the contract).
NVFP4_BLOCK = 16

#: Default CLI tensor selection: q/v/o/down projections only.
DEFAULT_TENSOR_PATTERNS = (
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
)

REPORT_FORMAT = "genuine-nvfp4-validator-v1"
OOM_SCORE_ADJ = 500
MAX_CANONICAL_NMSE = 0.02
MIN_CANONICAL_CORRELATION = 0.99
_SAFETENSORS = None


def positive_e4m3fn_values() -> tuple[float, ...]:
    """Return all positive finite E4M3FN values, including subnormals."""
    values = {
        (2.0 ** -6) * (mantissa / 8.0)
        for mantissa in range(1, 8)
    }
    values.update(
        (2.0 ** (exponent - 7)) * (1.0 + mantissa / 8.0)
        for exponent in range(1, 16)
        for mantissa in range(8)
        if not (exponent == 15 and mantissa == 7)
    )
    return tuple(sorted(values))


E4M3_TENSOR = torch.tensor(positive_e4m3fn_values(), dtype=torch.float32)
E4M3_SET = frozenset(positive_e4m3fn_values())


def set_oom_score(score: int = OOM_SCORE_ADJ) -> bool:
    """Best-effort /proc/self/oom_score_adj write; returns whether it applied."""
    try:
        Path("/proc/self/oom_score_adj").write_text(f"{int(score)}\n")
        return True
    except (OSError, PermissionError, ValueError):
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _lazy_import_safetensors():
    """Import safetensors on first use so pure helpers work without it."""
    global _SAFETENSORS
    if _SAFETENSORS is None:
        try:
            import safetensors as module
        except ImportError:
            raise RuntimeError(
                "safetensors is required for snapshot validation; "
                "install it with `pip install safetensors`"
            ) from None
        _SAFETENSORS = module
    return _SAFETENSORS


# ---------------------------------------------------------------------------
# Pure decode / validation helpers (in-memory tensors, no I/O)
# ---------------------------------------------------------------------------


def decode_packed_nvfp4(packed: torch.Tensor, *, swapped: bool = False) -> torch.Tensor:
    """Decode packed uint8 E2M1 bytes to float carriers of shape (..., K).

    Each byte stores two E2M1 carriers: the low nibble is the even-K carrier
    and the high nibble is the odd-K carrier (the ModelOpt layout).  With
    ``swapped=True`` the two nibbles are exchanged, which produces the
    "swapped" hypothesis used for discrimination.

    The decode uses ``E2M1_TABLE``, so nibble 8 (NVFP4's sign-encoded zero)
    decodes to the table's positive ``0.0``; decoded carriers therefore never
    carry a negative-zero bit pattern.
    """
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise ValueError(
            f"packed NVFP4 weights must be a torch.uint8 tensor, got "
            f"{getattr(packed, 'dtype', type(packed))!s}"
        )
    if packed.ndim < 1:
        raise ValueError("packed NVFP4 weights must have at least one dimension")
    if packed.numel() == 0:
        raise ValueError("packed NVFP4 weights must not be empty")
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    if swapped:
        first, second = E2M1_TENSOR[high], E2M1_TENSOR[low]
    else:
        first, second = E2M1_TENSOR[low], E2M1_TENSOR[high]
    return torch.stack((first, second), dim=-1).reshape(
        packed.shape[:-1] + (packed.shape[-1] * 2,)
    )


def validate_group_inputs(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    *,
    name: str = "nvfp4_group",
) -> dict[str, Any]:
    """Validate one ModelOpt NVFP4 weight group and normalize its tensors.

    Raises :class:`ValueError` on invalid dtype/shape/scale layouts.  On
    success returns a dict with the decoded fp32 carriers, the per-16 scale in
    fp32, the scalar global factor, and shape/dtype metadata.

    ``weight_scale_2`` is optional: when absent (``None``) the global factor is
    taken to be ``1.0`` and ``scale2_present`` is ``False``.
    """
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise ValueError(
            f"{name}: packed NVFP4 weight must be a torch.uint8 tensor, got "
            f"{getattr(packed, 'dtype', type(packed))!s}"
        )
    if packed.ndim < 1:
        raise ValueError(f"{name}: packed NVFP4 weight must have at least one dimension")
    if packed.numel() == 0:
        raise ValueError(f"{name}: packed NVFP4 weight must not be empty")
    packed_bytes = packed.shape[-1]
    if packed_bytes % (NVFP4_BLOCK // 2) != 0:
        raise ValueError(
            f"{name}: packed last dimension {packed_bytes} is not a multiple of "
            f"{NVFP4_BLOCK // 2} (logical K must be a multiple of {NVFP4_BLOCK})"
        )
    k = packed_bytes * 2
    if not isinstance(weight_scale, torch.Tensor) or not weight_scale.is_floating_point():
        raise ValueError(
            f"{name}: per-16 weight scale must be a floating-point tensor, got "
            f"{getattr(weight_scale, 'dtype', type(weight_scale))!s}"
        )
    expected_scale_shape = packed.shape[:-1] + (packed_bytes // (NVFP4_BLOCK // 2),)
    if tuple(weight_scale.shape) != tuple(expected_scale_shape):
        raise ValueError(
            f"{name}: per-16 weight scale shape {tuple(weight_scale.shape)} != "
            f"expected {tuple(expected_scale_shape)} (one scale per {NVFP4_BLOCK} "
            f"K values)"
        )
    scale1 = weight_scale.detach().cpu().to(torch.float32)
    if not bool(torch.isfinite(scale1).all()):
        raise ValueError(f"{name}: per-16 weight scale must be finite")
    g = 1.0
    scale2_dtype = None
    scale2_present = False
    if weight_scale_2 is not None:
        if (
            not isinstance(weight_scale_2, torch.Tensor)
            or not weight_scale_2.is_floating_point()
        ):
            raise ValueError(
                f"{name}: weight_scale_2 must be a floating-point scalar tensor, "
                f"got {getattr(weight_scale_2, 'dtype', type(weight_scale_2))!s}"
            )
        if weight_scale_2.numel() != 1:
            raise ValueError(
                f"{name}: weight_scale_2 must be a scalar (numel 1), got "
                f"{tuple(weight_scale_2.shape)}"
            )
        g = float(weight_scale_2.detach().cpu().to(torch.float32).reshape(()))
        if not math.isfinite(g):
            raise ValueError(f"{name}: weight_scale_2 must be finite")
        scale2_dtype = str(weight_scale_2.dtype)
        scale2_present = True
    carriers = decode_packed_nvfp4(packed)
    return {
        "name": name,
        "carriers": carriers,
        "scale1": scale1,
        "scale1_dtype": str(weight_scale.dtype),
        "scale2": g,
        "scale2_dtype": scale2_dtype,
        "scale2_present": scale2_present,
        "k": k,
        "packed_shape": tuple(packed.shape),
        "logical_shape": tuple(carriers.shape),
        "scale_shape": tuple(weight_scale.shape),
    }


def _expand_per16(scale1: torch.Tensor) -> torch.Tensor:
    """Expand per-16 scales (..., K/16) to one value per K element."""
    return scale1.repeat_interleave(NVFP4_BLOCK, dim=-1)


def _modelopt_dequant_pieces(
    carriers: torch.Tensor, scale1: torch.Tensor, g: float
) -> torch.Tensor:
    """ModelOpt-style dequantization in fp32 math, rounded to BF16.

    Reference order: ``(carrier * per16_scale) * global_factor``, matching the
    packed-weight layout where the per-16 scale and the global factor are
    separate storage fields.
    """
    return ((carriers * _expand_per16(scale1)) * g).to(torch.bfloat16)


def build_contest_pair(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    *,
    name: str = "nvfp4_group",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the contest-format pair from a ModelOpt NVFP4 group.

    Returns ``(carrier_bf16, scale_bf16)`` where the carrier has shape
    ``(..., K)`` and the folded per-16 scale has shape ``(..., K/16)``.  The
    folded scale is ``bf16(weight_scale * weight_scale_2)`` computed in fp32,
    i.e. the scalar global factor is folded into the per-16 scale so the pair
    is self-contained exactly like a contest NVFP4 pair.
    """
    group = validate_group_inputs(packed, weight_scale, weight_scale_2, name=name)
    carrier_bf16 = group["carriers"].to(torch.bfloat16)
    scale_bf16 = (group["scale1"] * group["scale2"]).to(torch.bfloat16)
    return carrier_bf16, scale_bf16


def dequantize_modelopt(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    *,
    name: str = "nvfp4_group",
) -> torch.Tensor:
    """Dequantize a ModelOpt NVFP4 group to BF16 the way ModelOpt stores it."""
    group = validate_group_inputs(packed, weight_scale, weight_scale_2, name=name)
    return _modelopt_dequant_pieces(group["carriers"], group["scale1"], group["scale2"])


def dequantize_contest(carrier: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Contest-reference dequantization of a BF16 carrier/scale pair."""
    return (
        carrier.unflatten(-1, (-1, NVFP4_BLOCK))
        * scale.unsqueeze(-1)
    ).flatten(-2, -1).to(torch.bfloat16)


def validate_contest_pair(
    carrier: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str = "contest_pair",
) -> dict[str, Any]:
    """Check contest-format pair legality (values and layout).

    Raises :class:`ValueError` when the shapes cannot align with the contest
    block layout; value-level checks (carrier set membership, negative zero,
    scale finiteness) are reported in the returned dict.
    """
    if not isinstance(carrier, torch.Tensor) or carrier.dtype != torch.bfloat16:
        raise ValueError(
            f"{name}: carrier must be a torch.bfloat16 tensor, got "
            f"{getattr(carrier, 'dtype', type(carrier))!s}"
        )
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.bfloat16:
        raise ValueError(
            f"{name}: scale must be a torch.bfloat16 tensor, got "
            f"{getattr(scale, 'dtype', type(scale))!s}"
        )
    if carrier.ndim < 1 or carrier.shape[-1] % NVFP4_BLOCK != 0:
        raise ValueError(
            f"{name}: carrier last dimension must be a multiple of {NVFP4_BLOCK}"
        )
    expected_scale_shape = carrier.shape[:-1] + (carrier.shape[-1] // NVFP4_BLOCK,)
    if tuple(scale.shape) != tuple(expected_scale_shape):
        raise ValueError(
            f"{name}: scale shape {tuple(scale.shape)} != expected "
            f"{tuple(expected_scale_shape)}"
        )
    carriers = carrier.detach().cpu().to(torch.float32)
    legal = torch.isin(carriers.reshape(-1), E2M1_TENSOR)
    nonfinite = ~torch.isfinite(scale.detach().cpu().to(torch.float32))
    return {
        "carrier_legal": bool(legal.all().item()),
        "illegal_carrier_count": int((~legal).sum().item()),
        "negative_zero_count": int(
            (torch.signbit(carriers) & (carriers == 0.0)).sum().item()
        ),
        "scale_shape_ok": True,
        "nonfinite_scale_count": int(nonfinite.sum().item()),
        "carrier_elements": carriers.numel(),
        "scale_elements": scale.numel(),
    }


def error_stats(xhat: torch.Tensor, parent: torch.Tensor) -> dict[str, float]:
    """Accumulators for comparing a BF16 reconstruction against a BF16 parent."""
    x = xhat.detach().cpu().to(torch.float64)
    y = parent.detach().cpu().to(torch.float64)
    diff = x - y
    return {
        "n": float(diff.numel()),
        "sum_sq_err": float((diff * diff).sum().item()),
        "sum_abs_err": float(diff.abs().sum().item()),
        "max_abs_err": float(diff.abs().max().item()),
        "sum_x": float(x.sum().item()),
        "sum_y": float(y.sum().item()),
        "sum_x2": float((x * x).sum().item()),
        "sum_y2": float((y * y).sum().item()),
        "sum_xy": float((x * y).sum().item()),
    }


def merge_error_stats(
    first: dict[str, float] | None, second: dict[str, float] | None
) -> dict[str, float] | None:
    """Merge two :func:`error_stats` accumulators."""
    if first is None:
        return second
    if second is None:
        return first
    merged = dict(first)
    for key in ("sum_sq_err", "sum_abs_err", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy"):
        merged[key] += second[key]
    merged["n"] += second["n"]
    merged["max_abs_err"] = max(merged["max_abs_err"], second["max_abs_err"])
    return merged


def finalize_error_stats(stats: dict[str, float] | None) -> dict[str, Any] | None:
    """Convert merged :func:`error_stats` into reportable comparison metrics."""
    if stats is None or stats["n"] == 0:
        return None
    n = stats["n"]
    mean_x = stats["sum_x"] / n
    mean_y = stats["sum_y"] / n
    var_x = max(stats["sum_x2"] / n - mean_x * mean_x, 1e-30)
    var_y = max(stats["sum_y2"] / n - mean_y * mean_y, 1e-30)
    covariance = stats["sum_xy"] / n - mean_x * mean_y
    mse = stats["sum_sq_err"] / n
    return {
        "mse": mse,
        "normalized_mse": mse / var_y,
        "correlation": covariance / math.sqrt(var_x * var_y),
        "max_abs_error": stats["max_abs_err"],
        "mean_abs_error": stats["sum_abs_err"] / n,
        "parent_variance": var_y,
        "n": n,
    }


def compare_contest_pair(
    carrier: torch.Tensor,
    scale: torch.Tensor,
    parent: torch.Tensor,
) -> dict[str, Any]:
    """Dequantize a contest pair and compare it against a BF16 parent."""
    return finalize_error_stats(error_stats(dequantize_contest(carrier, scale), parent))


def _process_chunk(
    packed_chunk: torch.Tensor,
    scale_chunk: torch.Tensor,
    g_tensor: torch.Tensor | None,
    parent_chunk: torch.Tensor | None,
    *,
    name: str = "nvfp4_group",
) -> dict[str, Any]:
    """Compute all per-chunk accumulators for one row chunk of a group."""
    group = validate_group_inputs(packed_chunk, scale_chunk, g_tensor, name=name)
    carriers = group["carriers"]
    scale1 = group["scale1"]
    g = group["scale2"]

    xhat_canon = _modelopt_dequant_pieces(carriers, scale1, g)
    xhat_swap = _modelopt_dequant_pieces(
        decode_packed_nvfp4(packed_chunk, swapped=True), scale1, g
    )
    xhat_no_global = _modelopt_dequant_pieces(carriers, scale1, 1.0)

    carrier_bf16 = carriers.to(torch.bfloat16)
    scale_folded = (scale1 * g).to(torch.bfloat16)
    xhat_contest = dequantize_contest(carrier_bf16, scale_folded)

    chunk: dict[str, Any] = {
        "carrier_elements": carriers.numel(),
        "illegal_carrier_count": int(
            (~torch.isin(carriers.reshape(-1), E2M1_TENSOR)).sum().item()
        ),
        "negative_zero_count": int(
            (torch.signbit(carriers) & (carriers == 0.0)).sum().item()
        ),
        "nibble8_count": int(((packed_chunk & 0x0F) == 8).sum().item()) + int(
            ((packed_chunk >> 4) == 8).sum().item()
        ),
        "scale_elements": scale1.numel(),
        "e4m3_scale_count": int(torch.isin(scale1.reshape(-1), E4M3_TENSOR).sum().item()),
        "nonfinite_scale_count": int((~torch.isfinite(scale1)).sum().item()),
        "canonical_stats": None,
        "swapped_stats": None,
        "no_global_stats": None,
        "fold_elements": xhat_contest.numel(),
        "fold_exact_count": int((xhat_contest == xhat_canon).sum().item()),
        "fold_abs_sum": float(
            (xhat_contest.double() - xhat_canon.double()).abs().sum().item()
        ),
        "fold_max_abs": float(
            (xhat_contest.float() - xhat_canon.float()).abs().max().item()
        ),
        "sample_carriers": carrier_bf16.reshape(-1)[:8].tolist(),
        "sample_scales": scale_folded.reshape(-1)[:4].tolist(),
    }
    if parent_chunk is not None:
        chunk["canonical_stats"] = error_stats(xhat_canon, parent_chunk)
        chunk["swapped_stats"] = error_stats(xhat_swap, parent_chunk)
        chunk["no_global_stats"] = error_stats(xhat_no_global, parent_chunk)
    return chunk


def _new_totals() -> dict[str, Any]:
    return {
        "carrier_elements": 0,
        "illegal_carrier_count": 0,
        "negative_zero_count": 0,
        "nibble8_count": 0,
        "scale_elements": 0,
        "e4m3_scale_count": 0,
        "nonfinite_scale_count": 0,
        "canonical_stats": None,
        "swapped_stats": None,
        "no_global_stats": None,
        "fold_elements": 0,
        "fold_exact_count": 0,
        "fold_abs_sum": 0.0,
        "fold_max_abs": 0.0,
    }


def _merge_chunk(totals: dict[str, Any], chunk: dict[str, Any]) -> None:
    for key in (
        "carrier_elements", "illegal_carrier_count", "negative_zero_count",
        "nibble8_count", "scale_elements", "e4m3_scale_count",
        "nonfinite_scale_count", "fold_elements", "fold_exact_count",
    ):
        totals[key] += chunk[key]
    totals["fold_abs_sum"] += chunk["fold_abs_sum"]
    totals["fold_max_abs"] = max(totals["fold_max_abs"], chunk["fold_max_abs"])
    for mode in ("canonical", "swapped", "no_global"):
        totals[f"{mode}_stats"] = merge_error_stats(
            totals[f"{mode}_stats"], chunk[f"{mode}_stats"]
        )


def evaluate_group(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    parent: torch.Tensor | None = None,
    *,
    name: str = "nvfp4_group",
) -> dict[str, Any]:
    """Validate one in-memory ModelOpt NVFP4 group against an optional parent.

    Convenience wrapper over :func:`_process_chunk` for full tensors; the
    snapshot validator uses the same helpers per row chunk so results are
    identical regardless of chunking.
    """
    chunk = _process_chunk(packed, weight_scale, weight_scale_2, parent, name=name)
    totals = _new_totals()
    _merge_chunk(totals, chunk)
    g = 1.0
    scale2_dtype = None
    scale2_present = False
    if weight_scale_2 is not None:
        g = float(weight_scale_2.detach().cpu().to(torch.float32).reshape(()))
        scale2_dtype = str(weight_scale_2.dtype)
        scale2_present = True
    parent_available = parent is not None
    return _assemble_tensor_result(
        name=name,
        packed_shape=tuple(packed.shape),
        scale_shape=tuple(weight_scale.shape),
        scale1_dtype=str(weight_scale.dtype),
        scale2_value=g if scale2_present else None,
        scale2_present=scale2_present,
        scale2_dtype=scale2_dtype,
        totals=totals,
        parent_available=parent_available,
        parent_shape=tuple(parent.shape) if parent_available else None,
        parent_dtype=str(parent.dtype) if parent_available else None,
        sample_carriers=chunk["sample_carriers"],
        sample_scales=chunk["sample_scales"],
    )


def _tensor_role(name: str) -> str:
    for role in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
    ):
        if role in name:
            return role.split(".")[-1]
    return "other"


def _assemble_tensor_result(
    *,
    name: str,
    packed_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    scale1_dtype: str,
    scale2_value: float | None,
    scale2_present: bool,
    scale2_dtype: str | None,
    totals: dict[str, Any],
    parent_available: bool,
    parent_shape: tuple[int, ...] | None,
    parent_dtype: str | None,
    sample_carriers: list[float],
    sample_scales: list[float],
) -> dict[str, Any]:
    logical_shape = packed_shape[:-1] + (packed_shape[-1] * 2,)
    if parent_available:
        canonical = finalize_error_stats(totals["canonical_stats"])
        swapped = finalize_error_stats(totals["swapped_stats"])
        no_global = finalize_error_stats(totals["no_global_stats"])
        best_mode = "canonical"
        best_nmse = canonical["normalized_mse"]
        for mode, metrics in (("swapped", swapped), ("no_global", no_global)):
            if metrics["normalized_mse"] < best_nmse:
                best_mode, best_nmse = mode, metrics["normalized_mse"]
        canonical_fit = (
            best_mode == "canonical"
            and canonical["normalized_mse"] <= MAX_CANONICAL_NMSE
            and canonical["correlation"] >= MIN_CANONICAL_CORRELATION
        )
        metrics = {
            "parent_available": True,
            "parent_shape": list(parent_shape) if parent_shape else None,
            "parent_dtype": parent_dtype,
            "canonical": canonical,
            "swapped": swapped,
            "no_global": no_global,
            "best_mode": best_mode,
            "is_genuine": canonical_fit,
            "fit_thresholds": {
                "max_normalized_mse": MAX_CANONICAL_NMSE,
                "min_correlation": MIN_CANONICAL_CORRELATION,
            },
        }
    else:
        metrics = {
            "parent_available": False,
            "parent_shape": None,
            "parent_dtype": None,
            "canonical": None,
            "swapped": None,
            "no_global": None,
            "best_mode": None,
            "is_genuine": None,
            "fit_thresholds": {
                "max_normalized_mse": MAX_CANONICAL_NMSE,
                "min_correlation": MIN_CANONICAL_CORRELATION,
            },
        }
    legality = {
        "carrier_legal": totals["illegal_carrier_count"] == 0,
        "illegal_carrier_count": totals["illegal_carrier_count"],
        "carrier_elements": totals["carrier_elements"],
        "e4m3_scale_count": totals["e4m3_scale_count"],
        "e4m3_scale_fraction": (
            totals["e4m3_scale_count"] / totals["scale_elements"]
            if totals["scale_elements"]
            else None
        ),
        "nonfinite_scale_count": totals["nonfinite_scale_count"],
        "negative_zero_count": totals["negative_zero_count"],
        "nibble8_zero_count": totals["nibble8_count"],
    }
    return {
        "name": name,
        "role": _tensor_role(name),
        "status": "ok",
        "packed_shape": list(packed_shape),
        "logical_shape": list(logical_shape),
        "scale_shape": list(scale_shape),
        "dtypes": {
            "packed": "U8",
            "weight_scale": scale1_dtype,
            "weight_scale_2": scale2_dtype,
        },
        "global_scale_present": scale2_present,
        "global_scale": scale2_value,
        "legality": legality,
        "metrics": metrics,
        "contest_fold_agreement": {
            "exact_fraction": (
                totals["fold_exact_count"] / totals["fold_elements"]
                if totals["fold_elements"]
                else None
            ),
            "mean_abs_diff": (
                totals["fold_abs_sum"] / totals["fold_elements"]
                if totals["fold_elements"]
                else None
            ),
            "max_abs_diff": totals["fold_max_abs"],
            "elements": totals["fold_elements"],
        },
        "contest_pair": {
            "carrier_dtype": "bfloat16",
            "scale_dtype": "bfloat16",
            "carrier_shape": list(logical_shape),
            "scale_shape": list(scale_shape),
            "sample_carriers": sample_carriers,
            "sample_scales": sample_scales,
        },
    }


# ---------------------------------------------------------------------------
# Snapshot I/O (lazy safetensors, one tensor group at a time)
# ---------------------------------------------------------------------------


def resolve_snapshot(path: str | Path) -> dict[str, Any]:
    """Resolve a snapshot path to a single-file or indexed-shard spec.

    Accepts a ``.safetensors`` file, a ``model.safetensors.index.json`` file,
    or a directory containing either ``model.safetensors.index.json`` (shards)
    or ``model.safetensors``.
    """
    snapshot_path = Path(path)
    if snapshot_path.is_dir():
        index = snapshot_path / "model.safetensors.index.json"
        if index.is_file():
            return _resolve_index(index)
        single = snapshot_path / "model.safetensors"
        if single.is_file():
            return {
                "kind": "single",
                "root": snapshot_path,
                "index_path": None,
                "file": single,
                "files": [single],
            }
        raise ValueError(
            f"no model.safetensors or model.safetensors.index.json in "
            f"{snapshot_path}"
        )
    if snapshot_path.is_file():
        if snapshot_path.suffix == ".safetensors":
            return {
                "kind": "single",
                "root": snapshot_path.parent,
                "index_path": None,
                "file": snapshot_path,
                "files": [snapshot_path],
            }
        if snapshot_path.suffix == ".json" and "index" in snapshot_path.name:
            return _resolve_index(snapshot_path)
    raise ValueError(
        f"unsupported snapshot path {snapshot_path}: expected a .safetensors "
        f"file, a model.safetensors.index.json file, or a directory containing "
        f"one of them"
    )


def _resolve_index(index_path: Path) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path}: index has no weight_map object")
    files = {
        shard_name: (index_path.parent / shard_name).resolve()
        for shard_name in sorted({str(value) for value in weight_map.values()})
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError(f"{index_path}: missing shard files {missing}")
    return {
        "kind": "shards",
        "root": index_path.parent,
        "index_path": index_path,
        "file": None,
        "files": list(files.values()),
        "weight_map": weight_map,
    }


def read_inventory(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read tensor names/dtypes/shapes from safetensors headers without data."""
    safetensors = _lazy_import_safetensors()
    inventory: dict[str, dict[str, Any]] = {}
    for file_path in spec["files"]:
        with safetensors.safe_open(str(file_path), framework="pt") as handle:
            for name in handle.keys():
                if name == "__metadata__":
                    continue
                slice_handle = handle.get_slice(name)
                inventory[name] = {
                    "dtype": str(slice_handle.get_dtype()),
                    "shape": tuple(slice_handle.get_shape()),
                    "file": str(file_path),
                }
    return inventory


class _SnapshotReader:
    """Row-chunked tensor reader over a resolved snapshot spec.

    Keeps at most one safetensors handle open (per shard) so memory stays
    bounded by the chunk size rather than by the checkpoint size.
    """

    def __init__(self, spec: dict[str, Any], inventory: dict[str, dict[str, Any]]):
        self._spec = spec
        self._inventory = inventory
        self._handle = None
        self._handle_file = None

    def _open(self, name: str):
        file_path = self._inventory[name]["file"]
        if self._handle is not None and self._handle_file == file_path:
            return self._handle
        self.close()
        safetensors = _lazy_import_safetensors()
        self._handle = safetensors.safe_open(file_path, framework="pt")
        self._handle_file = file_path
        return self._handle

    def read_rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        shape = self._inventory[name]["shape"]
        if not shape:
            if start != 0 or stop != 1:
                raise ValueError(f"{name}: scalar tensor cannot be row-sliced")
            return self._open(name).get_tensor(name)
        handle = self._open(name)
        return handle.get_slice(name)[start:stop]

    def read_tensor(self, name: str) -> torch.Tensor:
        return self._open(name).get_tensor(name)

    def close(self) -> None:
        if self._handle is not None:
            closer = getattr(self._handle, "close", None)
            if closer is None:
                exit_method = getattr(self._handle, "__exit__", None)
                if exit_method is not None:
                    exit_method(None, None, None)
            else:
                closer()
            self._handle = None
            self._handle_file = None


def _scale1_name(name: str, inventory: dict[str, dict[str, Any]]) -> str:
    prefix = name[: -len(".weight")]
    for candidate in (prefix + ".weight_scale", prefix + ".weight_scale_1"):
        if candidate in inventory:
            return candidate
    raise ValueError(
        f"missing per-16 weight scale tensor for {name} (expected "
        f"{prefix + '.weight_scale'})"
    )


def _matches_any(name: str, selectors: Sequence[str]) -> bool:
    for selector in selectors:
        if selector == name or selector in name or fnmatch.fnmatch(name, selector):
            return True
    return False


def _validate_one_tensor(
    name: str,
    inventory: dict[str, dict[str, Any]],
    reader: _SnapshotReader,
    parent_inventory: dict[str, dict[str, Any]] | None,
    parent_reader: _SnapshotReader | None,
    chunk_rows: int,
) -> dict[str, Any]:
    packed_info = inventory[name]
    scale_name = _scale1_name(name, inventory)
    scale_info = inventory[scale_name]
    scale2_name = name[: -len(".weight")] + ".weight_scale_2"
    scale2_present = scale2_name in inventory
    g_tensor = reader.read_tensor(scale2_name) if scale2_present else None
    scale2_value = float(g_tensor.to(torch.float32).reshape(())) if g_tensor is not None else None
    scale2_dtype = str(g_tensor.dtype) if g_tensor is not None else None

    packed_shape = packed_info["shape"]
    parent_available = False
    parent_shape = None
    parent_dtype = None
    if parent_inventory is not None and name in parent_inventory:
        parent_shape = tuple(parent_inventory[name]["shape"])
        if parent_shape == packed_shape[:-1] + (packed_shape[-1] * 2,):
            parent_available = True
            parent_dtype = parent_inventory[name]["dtype"]

    totals = _new_totals()
    sample_carriers: list[float] | None = None
    sample_scales: list[float] | None = None
    rows = packed_shape[0] if len(packed_shape) == 2 else 1
    if len(packed_shape) == 2 and chunk_rows > 0 and rows > chunk_rows:
        for r0 in range(0, rows, chunk_rows):
            r1 = min(r0 + chunk_rows, rows)
            packed_chunk = reader.read_rows(name, r0, r1)
            scale_chunk = reader.read_rows(scale_name, r0, r1)
            parent_chunk = (
                parent_reader.read_rows(name, r0, r1) if parent_available else None
            )
            chunk = _process_chunk(packed_chunk, scale_chunk, g_tensor, parent_chunk, name=name)
            if sample_carriers is None:
                sample_carriers, sample_scales = chunk["sample_carriers"], chunk["sample_scales"]
            _merge_chunk(totals, chunk)
    else:
        packed_full = reader.read_tensor(name)
        scale_full = reader.read_tensor(scale_name)
        parent_full = parent_reader.read_tensor(name) if parent_available else None
        chunk = _process_chunk(packed_full, scale_full, g_tensor, parent_full, name=name)
        sample_carriers, sample_scales = chunk["sample_carriers"], chunk["sample_scales"]
        _merge_chunk(totals, chunk)

    return _assemble_tensor_result(
        name=name,
        packed_shape=packed_shape,
        scale_shape=scale_info["shape"],
        scale1_dtype=scale_info["dtype"],
        scale2_value=scale2_value,
        scale2_present=scale2_present,
        scale2_dtype=scale2_dtype,
        totals=totals,
        parent_available=parent_available,
        parent_shape=parent_shape,
        parent_dtype=parent_dtype,
        sample_carriers=sample_carriers or [],
        sample_scales=sample_scales or [],
    )


def _inventory_summary(inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_dtype: dict[str, int] = {}
    packed_weights = 0
    weight_scales = 0
    weight_scales_2 = 0
    packed_shapes: dict[str, int] = {}
    for name, info in inventory.items():
        dtype = info["dtype"]
        by_dtype[dtype] = by_dtype.get(dtype, 0) + 1
        if dtype == "U8" and name.endswith(".weight"):
            packed_weights += 1
            packed_shapes[str(list(info["shape"]))] = (
                packed_shapes.get(str(list(info["shape"])), 0) + 1
            )
        if name.endswith(".weight_scale"):
            weight_scales += 1
        if name.endswith(".weight_scale_2"):
            weight_scales_2 += 1
    return {
        "tensor_count": len(inventory),
        "by_dtype": dict(sorted(by_dtype.items())),
        "packed_weight_count": packed_weights,
        "weight_scale_count": weight_scales,
        "weight_scale_2_count": weight_scales_2,
        "packed_shape_histogram": dict(sorted(packed_shapes.items())),
        "example_tensor_names": sorted(inventory)[:5],
    }


def _producer_metadata(root: Path) -> dict[str, Any] | None:
    """Extract quantization-producer metadata from config.json / hf_quant_config.json."""
    producer: dict[str, Any] = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            producer["config.json"] = {"error": str(exc)}
        else:
            keys = (
                "model_type", "architectures", "quantization_config",
                "num_hidden_layers", "hidden_size", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "head_dim",
            )
            subset = {key: config[key] for key in keys if key in config}
            producer["config.json"] = subset
    for fname in ("hf_quant_config.json", "quant_config.json"):
        quant_path = root / fname
        if not quant_path.is_file():
            continue
        try:
            producer[fname] = json.loads(quant_path.read_text(encoding="utf-8"))
        except Exception as exc:
            producer[fname] = {"error": str(exc)}
    return producer or None


def _snapshot_provenance(spec: dict[str, Any]) -> dict[str, Any]:
    files = []
    total_size = 0
    for file_path in spec["files"]:
        stat = file_path.stat()
        total_size += stat.st_size
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": str(file_path),
                "size_bytes": stat.st_size,
                "sha256": digest.hexdigest(),
                "mtime_iso": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    return {
        "kind": spec["kind"],
        "root": str(spec["root"]),
        "files": files,
        "total_size_bytes": total_size,
    }


def validate_snapshot(
    nvfp4_snapshot: str | Path,
    bf16_snapshot: str | Path | None = None,
    tensor_selectors: Sequence[str] | None = None,
    *,
    chunk_rows: int = 256,
) -> dict[str, Any]:
    """Validate a ModelOpt NVFP4 snapshot (single file or indexed shards).

    Reads only tensor headers up front, then processes each selected packed
    weight group one row chunk at a time.  ``tensor_selectors`` may contain
    exact tensor names, substrings, or fnmatch globs; defaults to the q/v/o and
    down projections.
    """
    nvfp4_spec = resolve_snapshot(nvfp4_snapshot)
    inventory = read_inventory(nvfp4_spec)
    reader = _SnapshotReader(nvfp4_spec, inventory)

    parent_spec = None
    parent_inventory: dict[str, dict[str, Any]] = {}
    parent_reader = None
    if bf16_snapshot is not None:
        parent_spec = resolve_snapshot(bf16_snapshot)
        parent_inventory = read_inventory(parent_spec)
        parent_reader = _SnapshotReader(parent_spec, parent_inventory)

    selectors = list(tensor_selectors) if tensor_selectors else list(DEFAULT_TENSOR_PATTERNS)
    candidates = sorted(
        name
        for name, info in inventory.items()
        if info["dtype"] == "U8" and name.endswith(".weight")
    )
    selected = [name for name in candidates if _matches_any(name, selectors)]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        for name in selected:
            try:
                results.append(
                    _validate_one_tensor(
                        name, inventory, reader, parent_inventory, parent_reader, chunk_rows
                    )
                )
            except Exception as exc:
                errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        reader.close()
        if parent_reader is not None:
            parent_reader.close()

    summary = _build_summary(results, errors, selected)
    tool = {
        "name": "tools/validate_genuine_nvfp4.py",
        "report_format": REPORT_FORMAT,
        "torch": torch.__version__,
        "safetensors": getattr(_SAFETENSORS, "__version__", None),
        "oom_score_adj": OOM_SCORE_ADJ,
        "timestamp_utc": _now_iso(),
    }
    return {
        "format": REPORT_FORMAT,
        "provenance": {
            "nvfp4_snapshot": _snapshot_provenance(nvfp4_spec),
            "bf16_snapshot": _snapshot_provenance(parent_spec) if parent_spec else None,
            "producer": _producer_metadata(nvfp4_spec["root"]),
            "tool": tool,
        },
        "inventory": _inventory_summary(inventory),
        "tensors": results,
        "errors": errors,
        "summary": summary,
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _build_summary(
    results: list[dict[str, Any]],
    errors: list[dict[str, str]],
    selected: list[str],
) -> dict[str, Any]:
    validated = len(results)
    genuine = sum(1 for result in results if result["metrics"]["is_genuine"] is True)
    canonical_best = sum(
        1 for result in results if result["metrics"]["best_mode"] == "canonical"
    )
    swapped = sum(1 for result in results if result["metrics"]["best_mode"] == "swapped")
    no_global = sum(
        1 for result in results if result["metrics"]["best_mode"] == "no_global"
    )
    parent_missing = sum(
        1 for result in results if not result["metrics"]["parent_available"]
    )
    global_scale_absent = sum(
        1 for result in results if not result["global_scale_present"]
    )
    canonical_nmse = [
        result["metrics"]["canonical"]["normalized_mse"]
        for result in results
        if result["metrics"]["canonical"] is not None
    ]
    canonical_corr = [
        result["metrics"]["canonical"]["correlation"]
        for result in results
        if result["metrics"]["canonical"] is not None
    ]
    fold_exact = [
        result["contest_fold_agreement"]["exact_fraction"]
        for result in results
        if result["contest_fold_agreement"]["exact_fraction"] is not None
    ]

    if not selected and not errors:
        conclusion = (
            "No U8 packed `.weight` tensors matched the tensor selectors in the "
            "NVFP4 snapshot; nothing was validated."
        )
    elif validated == 0:
        conclusion = (
            f"All {len(selected)} selected tensors failed validation "
            f"(see `errors`)."
        )
    elif parent_missing == validated:
        conclusion = (
            "No BF16 parent tensors were available, so only structural validation "
            "(E2M1 packing legality, E4M3 scale grid, and contest-fold agreement) "
            "was performed; canonical/swapped/no-global discrimination requires "
            "--bf16-snapshot."
        )
    elif genuine == validated:
        conclusion = (
            f"All {validated} validated tensors have a strong canonical ModelOpt "
            "NVFP4 layout: packed E2M1 carriers (low nibble = even K), one E4M3 "
            "block scale per 16 K values, and the scalar weight_scale_2 global "
            "factor folded into the per-16 BF16 scale."
        )
    elif swapped == validated:
        conclusion = (
            f"All {validated} validated tensors fit the swapped nibble "
            "interpretation (odd K in the low nibble) better than the canonical "
            "one; the packing convention used by this checkpoint differs from "
            "the ModelOpt layout implemented here."
        )
    elif no_global == validated:
        conclusion = (
            f"All {validated} validated tensors fit best without the "
            "weight_scale_2 global factor; the checkpoint's per-16 scales may "
            "already be globally scaled."
        )
    elif canonical_best == validated:
        conclusion = (
            f"The canonical interpretation fit best for all {validated} validated "
            f"tensors, but only {genuine} passed the absolute fit requirements "
            f"(normalized MSE <= {MAX_CANONICAL_NMSE:g} and correlation >= "
            f"{MIN_CANONICAL_CORRELATION:g})."
        )
    else:
        conclusion = (
            f"Mixed discrimination results across {validated} tensors: "
            f"{canonical_best} canonical-best ({genuine} strong fits), "
            f"{swapped} swapped, {no_global} no-global, and {parent_missing} "
            "without matching parents."
        )
    if global_scale_absent and validated:
        conclusion += (
            f" {global_scale_absent} validated tensor(s) had no weight_scale_2 "
            "(treated as the unit global factor)."
        )

    return {
        "tensor_candidates": len(selected),
        "validated_tensor_count": validated,
        "tensor_errors": len(errors),
        "genuine_count": genuine,
        "canonical_best_count": canonical_best,
        "swapped_count": swapped,
        "no_global_count": no_global,
        "parent_missing_count": parent_missing,
        "global_scale_absent_count": global_scale_absent,
        "mean_normalized_mse_canonical": _mean(canonical_nmse),
        "mean_correlation_canonical": _mean(canonical_corr),
        "mean_fold_exact_fraction": _mean(fold_exact),
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Rendering and atomic output
# ---------------------------------------------------------------------------


def _percent(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.6f}%"


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def render_markdown(report: dict[str, Any]) -> str:
    """Deterministically render the validation report as Markdown."""
    provenance = report["provenance"]
    inventory = report["inventory"]
    summary = report["summary"]
    nvfp4 = provenance["nvfp4_snapshot"]
    bf16 = provenance["bf16_snapshot"]
    producer = provenance["producer"] or {}
    config_json = producer.get("config.json", {})
    tool = provenance["tool"]
    tensors = report["tensors"]

    lines = [
        "# Genuine NVFP4 Checkpoint Validation",
        "",
        "Validates a ModelOpt NVFP4 checkpoint against the contest NVFP4 pair "
        "format: packed E2M1 carriers (low nibble = even K), one E4M3 block "
        "scale per 16 K values, a scalar `weight_scale_2` global factor, and "
        "the resulting BF16 carrier + folded per-16 BF16 scale.",
        "",
        "## Provenance",
        "",
        f"- NVFP4 snapshot: `{nvfp4['root']}` ({nvfp4['kind']}, "
        f"{nvfp4['total_size_bytes']:,} bytes across {len(nvfp4['files'])} file(s))",
    ]
    if bf16 is not None:
        lines.append(
            f"- BF16 parent: `{bf16['root']}` ({bf16['kind']}, "
            f"{bf16['total_size_bytes']:,} bytes)"
        )
    else:
        lines.append("- BF16 parent: none (structural validation only)")
    if producer:
        summary_bits = []
        for key in ("model_type", "architectures", "quantization_config"):
            if key in config_json:
                summary_bits.append(f"{key}={json.dumps(config_json[key], sort_keys=True)}")
        if summary_bits:
            lines.append(f"- Producer config: {', '.join(summary_bits)}")
        if "hf_quant_config.json" in producer:
            lines.append(
                "- Producer hf_quant_config.json: "
                f"`{json.dumps(producer['hf_quant_config.json'], sort_keys=True)}`"
            )
    else:
        lines.append("- Producer config: none found next to the snapshot")
    lines.extend([
        f"- Tool: `{tool['name']}` (format {tool['report_format']}, torch "
        f"{tool['torch']}, safetensors {tool['safetensors'] or 'n/a'}, "
        f"oom_score_adj={tool['oom_score_adj']})",
        f"- Generated: {tool['timestamp_utc']} UTC",
        "",
        "## Inventory",
        "",
        f"- Tensors: {inventory['tensor_count']:,} total; "
        f"{inventory['packed_weight_count']:,} packed U8 weights; "
        f"{inventory['weight_scale_count']:,} per-16 E4M3 scales; "
        f"{inventory['weight_scale_2_count']:,} scalar global factors.",
        f"- Dtype histogram: {json.dumps(inventory['by_dtype'], sort_keys=True)}",
        "",
        "## Results",
        "",
        "| Tensor | Role | N | K | Scale dtype | g | Canonical norm-MSE | "
        "Swapped norm-MSE | No-global norm-MSE | Correlation | Fold exact | "
        "Best mode |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in tensors:
        metrics = result["metrics"]
        canonical = metrics["canonical"]
        lines.append(
            f"| `{result['name']}` | {result['role']} | "
            f"{result['logical_shape'][0]:,} | {result['logical_shape'][-1]:,} | "
            f"{result['dtypes']['weight_scale']} | "
            f"{'yes' if result['global_scale_present'] else 'no'} | "
            f"{_fmt(canonical['normalized_mse'] if canonical else None)} | "
            f"{_fmt(metrics['swapped']['normalized_mse'] if metrics['swapped'] else None)} | "
            f"{_fmt(metrics['no_global']['normalized_mse'] if metrics['no_global'] else None)} | "
            f"{_fmt(canonical['correlation'] if canonical else None)} | "
            f"{_percent(result['contest_fold_agreement']['exact_fraction'] or 0, 1)} | "
            f"{metrics['best_mode'] or 'n/a'} |"
        )
    if not tensors:
        lines.append("| _(no tensors selected)_ | | | | | | | | | | | |")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- Validated tensors: {summary['validated_tensor_count']} "
        f"(candidates: {summary['tensor_candidates']}, errors: "
        f"{summary['tensor_errors']})",
        f"- Strong canonical fit: {summary['genuine_count']}; swapped-fit: "
        f"{summary['swapped_count']}; no-global-fit: {summary['no_global_count']}.",
        f"- Mean canonical normalized MSE: "
        f"{_fmt(summary['mean_normalized_mse_canonical'])}; mean correlation: "
        f"{_fmt(summary['mean_correlation_canonical'])}; mean contest-fold exact "
        f"fraction: {_fmt(summary['mean_fold_exact_fraction'])}.",
        "",
        f"**Conclusion:** {summary['conclusion']}",
        "",
    ])
    if report["errors"]:
        lines.extend(["## Errors", ""])
        for error in report["errors"]:
            lines.append(f"- `{error['name']}`: {error['error']}")
        lines.append("")
    lines.extend([
        "## Method and caveats",
        "",
        "- ModelOpt packed weights are `uint8` tensors of shape `[N, K/2]`; each "
        "byte decodes with the E2M1 table "
        "`[0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]`, low nibble = even K, "
        "high nibble = odd K.",
        "- The per-16 scale tensor (`weight_scale`) must hold one E4M3 value per "
        "16 K values; the scalar `weight_scale_2` global factor is folded into "
        "the per-16 BF16 scale as `bf16(weight_scale * weight_scale_2)`.",
        "- Nibble 8 is sign-encoded zero. This converter follows ModelOpt's "
        "software lookup table and emits positive BF16 zero; hardware may retain "
        "a negative-zero sign bit, which is numerically immaterial here.",
        "- Without a BF16 parent, U8 carrier legality and E4M3 dtype membership "
        "are structural consistency checks, not proof of checkpoint provenance. "
        "Producer metadata is self-reported by files beside the checkpoint.",
        "- Canonical/swapped/no-global comparisons dequantize the ModelOpt "
        "storage in fp32 and round to BF16, then compare against the optional "
        "BF16 parent with MSE, normalized MSE (by parent variance), Pearson "
        "correlation, and max absolute error. `best_mode` is the lowest "
        "normalized MSE; `is_genuine` additionally requires canonical normalized "
        f"MSE <= {MAX_CANONICAL_NMSE} and correlation >= "
        f"{MIN_CANONICAL_CORRELATION}. This establishes a strong numerical fit, "
        "not cryptographic provenance.",
        "- `contest_fold_agreement` measures how closely the contest-format "
        "BF16 pair reproduces the ModelOpt BF16 dequantization after the global "
        "factor is folded into the per-16 scale.",
        "- Memory bound: tensor headers are read without loading data and each "
        "2-D group is processed in row chunks; a full model is never loaded. "
        "Single-file and indexed-shard snapshots are supported.",
        "",
    ])
    return "\n".join(lines)


def atomic_write(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_path, target)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nvfp4-snapshot",
        type=Path,
        required=True,
        help=(
            "ModelOpt NVFP4 snapshot: a model.safetensors file, a "
            "model.safetensors.index.json file, or a directory containing one"
        ),
    )
    parser.add_argument(
        "--bf16-snapshot",
        type=Path,
        default=None,
        help=(
            "optional BF16 parent snapshot (same file forms as --nvfp4-snapshot) "
            "used for canonical/swapped/no-global discrimination"
        ),
    )
    parser.add_argument(
        "--tensor",
        action="append",
        default=None,
        metavar="NAME_OR_PATTERN",
        help=(
            "tensor to validate: exact name, substring, or fnmatch glob; "
            "repeatable. Default: the q/v/o and down projections "
            f"({', '.join(DEFAULT_TENSOR_PATTERNS)})"
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
        help="row-chunk size for memory-bounded processing (default: 256)",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_oom_score(OOM_SCORE_ADJ)  # best effort; validator dies before the runner
    try:
        report = validate_snapshot(
            args.nvfp4_snapshot,
            args.bf16_snapshot,
            args.tensor,
            chunk_rows=args.chunk_rows,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        markdown = render_markdown(report)
        if args.json_output:
            atomic_write(
                args.json_output,
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )
        if args.markdown_output:
            atomic_write(args.markdown_output, markdown)
        if not args.json_output and not args.markdown_output:
            print(markdown, end="")
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: could not write report: {exc}", file=sys.stderr)
        return 2
    return _report_exit_code(report["summary"])


def _report_exit_code(summary: dict[str, Any]) -> int:
    """Return success only for clean structural checks and strong parent fits."""
    if not summary["validated_tensor_count"] or summary["tensor_errors"]:
        return 1
    parent_compared = (
        summary["validated_tensor_count"] - summary["parent_missing_count"]
    )
    if parent_compared and summary["genuine_count"] != parent_compared:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
