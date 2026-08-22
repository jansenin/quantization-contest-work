#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare a candidate ``solution.py`` against a baseline for NVFP4->HiF4.

The contest metric is reproduced locally on deterministic synthetic data
(plus an optional run over the public ``example/mini_sample``):

* Every test case (one test activation for Linear, one full Q/K/V test sample
  for Attention) is scored as ``100 * (baseline_mse - candidate_mse) /
  baseline_mse`` where both MSEs are computed against the same reference
  output derived from the *dequantized original NVFP4* data (FP32 compute,
  BF16 NVFP4 dequantization -- see ``tools/reference_ops.py``).
* The overall score is the arithmetic mean of the per-case metrics.
* The complete six-function API is exercised, including calibration (the
  weight/attention calibration functions are called and their wall time is
  measured separately from the online dynamic-quantization functions).
* Reference Linear output is ``X @ W^T``; reference Attention is the ordinary
  unmasked scaled-dot-product ``softmax(Q K^T / sqrt(head_dim)) V`` for
  contiguous-head GQA, MHA and MQA.

The baseline and the candidate are each specified either as a filesystem
``.py`` path (or a directory containing ``solution.py``) or as a Git ref whose
``solution.py`` is read with ``git show <ref>:solution.py``.  Each module is
loaded under a fresh random module name (never under the reusable name
``solution``) and both runs receive bit-identical fresh copies of the input
data, so no module can mutate data seen by the other.

Outputs:
* A detailed JSON record is always written to
  ``benchmarks/records/<variant>.json`` (schema fixed; MSE/metric fields are
   deterministic for a fixed seed and thread configuration -- the wall-clock
   ``*_s`` timing fields vary per run by nature).
* On success (status ``ok``) one compact row is appended to
  ``progress/results.jsonl`` unless ``--no-append`` is given.

Runs whose baseline MSE is zero or non-finite for any test case are rejected
with a clear status (no ``results.jsonl`` row is appended); the same applies to
candidate crashes, non-finite candidate MSE, or HiF4 parameters that violate
the format contract enforced locally by ``example/self_check.py``.

Only the Python standard library and torch are used; no Git mutations are
performed (only read-only ``git show`` / ``git rev-parse``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Local tool imports (same directory as this file)
# ---------------------------------------------------------------------------

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from reference_ops import (  # noqa: E402
    attention_output,
    attention_output_nvfp4,
    dequantize_hif4_params,
    linear_output,
    linear_output_nvfp4,
    scalar_mse,
)
from synthetic_data import DISTS, make_attention_group, make_linear_group  # noqa: E402

_REPO_ROOT = os.path.dirname(_TOOLS_DIR)

API_NAMES = {
    "weight": "hif4_calibration_and_quantize_weight",
    "activation": "hif4_dynamic_quantize_activation",
    "attention": "hif4_calibration_attention",
    "q": "hif4_dynamic_quantize_q",
    "k": "hif4_dynamic_quantize_k",
    "v": "hif4_dynamic_quantize_v",
}

_HIF4_LAYOUT = {
    "scale_factor": (1, 1, 1),
    "scale_lv2": (8, 1, 1),
    "scale_lv3": (8, 2, 1),
    "sign": (8, 2, 4),
    "mant": (8, 2, 4),
}

_STATE_DTYPES = {
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


# ===========================================================================
# Solution loading (path or Git ref), never under a reusable module name
# ===========================================================================

def _load_module_from_path(path: str, name: str) -> Any:
    """Import a module from an arbitrary file path under a fresh name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    module_dir = os.path.dirname(os.path.abspath(path))
    old_path = list(sys.path)
    sys.path.insert(0, module_dir)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        sys.path = old_path
    return module


def _git(args: list[str], repo: str, error_hint: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"{error_hint}: the 'git' executable is not available on PATH"
        ) from None


def _load_solution(spec: str, repo: str) -> dict[str, Any]:
    """Resolve and load a solution module from a path or a Git ref.

    Returns ``{"kind", "label", "commit", "module"}`` where ``label`` is the
    human-readable spec (path or ref) and ``commit`` is the resolved full
    commit hash when ``spec`` is a Git ref (``None`` otherwise).
    """
    candidates = [spec]
    if not os.path.isabs(spec):
        candidates.append(os.path.join(repo, spec))
    for candidate in candidates:
        if candidate.endswith(".py") and os.path.isfile(candidate):
            module = _load_module_from_path(
                os.path.abspath(candidate), f"_eval_{uuid.uuid4().hex[:12]}"
            )
            return {"kind": "path", "label": candidate, "commit": None, "module": module}
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, "solution.py")
        ):
            path = os.path.join(candidate, "solution.py")
            module = _load_module_from_path(
                os.path.abspath(path), f"_eval_{uuid.uuid4().hex[:12]}"
            )
            return {"kind": "path", "label": path, "commit": None, "module": module}

    show = _git(["show", f"{spec}:solution.py"], repo, f"cannot load solution spec {spec!r}")
    if show.returncode != 0:
        raise ValueError(
            f"solution spec {spec!r} is neither an existing .py file/directory "
            f"nor a Git ref (git show failed: {show.stderr.strip()[:200]})"
        )
    rev = _git(["rev-parse", spec], repo, f"cannot resolve Git ref {spec!r}")
    commit = rev.stdout.strip() if rev.returncode == 0 else None

    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="eval_solution_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(show.stdout)
        module = _load_module_from_path(tmp_path, f"_eval_{uuid.uuid4().hex[:12]}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return {"kind": "git", "label": spec, "commit": commit, "module": module}


def _get_funcs(module: Any) -> dict[str, Any]:
    funcs: dict[str, Any] = {}
    missing = [API_NAMES[role] for role in API_NAMES if not callable(getattr(module, API_NAMES[role], None))]
    if missing:
        raise ValueError(f"module is missing callable functions: {', '.join(missing)}")
    for role, name in API_NAMES.items():
        funcs[role] = getattr(module, name)
    return funcs


# ===========================================================================
# Data helpers
# ===========================================================================

def _clone_data(value: Any) -> Any:
    """Deep copy of the pure-data structures (tensors, lists, tuples, dicts)."""
    if type(value) is torch.Tensor:
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, list):
        return [_clone_data(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_data(v) for v in value)
    if isinstance(value, dict):
        return {key: _clone_data(v) for key, v in value.items()}
    return value


def _build_dataset(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Build synthetic groups for every requested dist/attention type.

    Returns ``(groups, counts)`` where ``groups`` holds the master (pristine)
    group structures and ``counts`` maps ``"linear"`` / ``"attention"`` to the
    number of groups.
    """
    dists = [d.strip() for d in args.dists.split(",") if d.strip()]
    for dist in dists:
        if dist not in DISTS:
            raise ValueError(f"unknown distribution {dist!r}; choose from {DISTS}")
    attn_types = [t.strip() for t in args.attn_types.split(",") if t.strip()]
    for attn_type in attn_types:
        if attn_type not in ("gqa", "mha", "mqa"):
            raise ValueError(
                f"unknown attention type {attn_type!r}; choose from gqa, mha, mqa"
            )

    groups: dict[str, list[dict[str, Any]]] = {"linear": [], "attention": []}
    counter = int(args.seed) * 7919

    for dist in dists:
        for _ in range(args.n_groups_per_dist):
            counter += 1
            groups["linear"].append(
                {
                    "dist": dist,
                    "data": make_linear_group(
                        seed=counter,
                        dist=dist,
                        out_features=args.out_features,
                        in_features=args.in_features,
                        seq_len=args.seq_len,
                        n_calib=args.n_calib,
                        n_test=args.n_test,
                    ),
                }
            )
        for attn_type in attn_types:
            if attn_type == "mha":
                qh, kvh = args.q_heads, args.q_heads
            elif attn_type == "mqa":
                qh, kvh = args.q_heads, 1
            else:  # gqa
                qh, kvh = args.q_heads, max(1, args.q_heads // 2)
            for _ in range(args.n_groups_per_dist):
                counter += 1
                groups["attention"].append(
                    {
                        "dist": dist,
                        "attn_type": attn_type,
                        "data": make_attention_group(
                            seed=counter,
                            dist=dist,
                            q_num_heads=qh,
                            kv_num_heads=kvh,
                            head_dim=args.head_dim,
                            seq_len=args.seq_len,
                            n_calib=args.n_calib,
                            n_test=args.n_test,
                        ),
                    }
                )

    counts = {"linear": len(groups["linear"]), "attention": len(groups["attention"])}
    return groups, counts


# ---------------------------------------------------------------------------
# Public mini-sample loading (port of example/self_check.py normalization)
# ---------------------------------------------------------------------------

def _normalize_nvfp4_pair(value: Any, tag: str) -> list[torch.Tensor]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{tag}: expected [quant_tensor, scale_tensor]")
    quant, scale = value
    if type(quant) is not torch.Tensor or type(scale) is not torch.Tensor:
        raise TypeError(f"{tag}: quant and scale must be plain torch.Tensor")
    if quant.ndim < 1:
        raise ValueError(f"{tag}: quant tensor must have at least one dimension")
    channels = int(quant.shape[-1])
    if channels % 16 != 0:
        raise ValueError(f"{tag}: last dimension must be divisible by 16")
    if tuple(scale.shape) != tuple(quant.shape[:-1]) + (channels // 16,):
        raise ValueError(
            f"{tag}: scale shape {tuple(scale.shape)} != expected "
            f"{tuple(quant.shape[:-1]) + (channels // 16,)}"
        )
    return [quant, scale]


def _normalize_linear_groups(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("linear.pt root must be a non-empty list")
    out: list[dict[str, Any]] = []
    for idx, raw_group in enumerate(raw):
        if not isinstance(raw_group, dict):
            raise TypeError(f"linear group {idx}: expected dict")
        required = ("weight", "calib_activation_list", "test_activation_list")
        missing = [name for name in required if name not in raw_group]
        if missing:
            raise KeyError(f"linear group {idx}: missing keys {missing}")
        weight = _normalize_nvfp4_pair(raw_group["weight"], f"linear group {idx}.weight")
        channels = int(weight[0].shape[-1])
        if not raw_group["calib_activation_list"] or not raw_group["test_activation_list"]:
            raise ValueError(f"linear group {idx}: calib/test must be non-empty lists")
        calib: list[list[torch.Tensor]] = []
        for i, item in enumerate(raw_group["calib_activation_list"]):
            pair = _normalize_nvfp4_pair(item, f"linear group {idx}.calib_activation_list[{i}]")
            if int(pair[0].shape[-1]) != channels:
                raise ValueError(
                    f"linear group {idx}.calib_activation_list[{i}]: channels "
                    f"{pair[0].shape[-1]} != weight channels {channels}"
                )
            calib.append(pair)
        tests: list[list[torch.Tensor]] = []
        for i, item in enumerate(raw_group["test_activation_list"]):
            pair = _normalize_nvfp4_pair(item, f"linear group {idx}.test_activation_list[{i}]")
            if int(pair[0].shape[-1]) != channels:
                raise ValueError(
                    f"linear group {idx}.test_activation_list[{i}]: channels "
                    f"{pair[0].shape[-1]} != weight channels {channels}"
                )
            tests.append(pair)
        out.append(
            {
                "weight": weight,
                "calib_activation_list": calib,
                "test_activation_list": tests,
            }
        )
    return out


def _normalize_attention_sample(
    sample: Any, tag: str, q_num_heads: int, kv_num_heads: int, head_dim: int
) -> dict[str, list[torch.Tensor]]:
    if not isinstance(sample, dict) or any(role not in sample for role in ("q", "k", "v")):
        raise ValueError(f"{tag}: expected dict with q/k/v")
    q = _normalize_nvfp4_pair(sample["q"], f"{tag}.q")
    k = _normalize_nvfp4_pair(sample["k"], f"{tag}.k")
    v = _normalize_nvfp4_pair(sample["v"], f"{tag}.v")
    if q[0].ndim != 2 or k[0].ndim != 2 or v[0].ndim != 2:
        raise ValueError(f"{tag}: Q/K/V quant tensors must be 2D [seq_len, hidden]")
    if int(q[0].shape[-1]) != int(q_num_heads) * int(head_dim):
        raise ValueError(f"{tag}.q: hidden size mismatch")
    if int(k[0].shape[-1]) != int(kv_num_heads) * int(head_dim):
        raise ValueError(f"{tag}.k: hidden size mismatch")
    if int(v[0].shape[-1]) != int(kv_num_heads) * int(head_dim):
        raise ValueError(f"{tag}.v: hidden size mismatch")
    if not (q[0].shape[0] == k[0].shape[0] == v[0].shape[0]):
        raise ValueError(f"{tag}: Q/K/V sequence lengths must match")
    return {"q": q, "k": k, "v": v}


def _normalize_attention_groups(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("attn.pt root must be a non-empty list")
    out: list[dict[str, Any]] = []
    for idx, raw_group in enumerate(raw):
        if not isinstance(raw_group, dict):
            raise TypeError(f"attention group {idx}: expected dict")
        required = ("q_num_heads", "kv_num_heads", "head_dim", "calib", "test")
        missing = [name for name in required if name not in raw_group]
        if missing:
            raise KeyError(f"attention group {idx}: missing keys {missing}")
        qh, kvh, hd = raw_group["q_num_heads"], raw_group["kv_num_heads"], raw_group["head_dim"]
        if not all(type(value) is int and value > 0 for value in (qh, kvh, hd)):
            raise ValueError(f"attention group {idx}: head params must be positive ints")
        if qh % kvh != 0:
            raise ValueError(f"attention group {idx}: q_num_heads must be divisible by kv_num_heads")
        if not raw_group["calib"] or not raw_group["test"]:
            raise ValueError(f"attention group {idx}: calib/test must be non-empty lists")
        calib = [
            _normalize_attention_sample(sample, f"attention group {idx}.calib[{i}]", qh, kvh, hd)
            for i, sample in enumerate(raw_group["calib"])
        ]
        tests = [
            _normalize_attention_sample(sample, f"attention group {idx}.test[{i}]", qh, kvh, hd)
            for i, sample in enumerate(raw_group["test"])
        ]
        attn_type = raw_group.get("attn_type")
        if attn_type not in ("gqa", "mha", "mqa"):
            attn_type = "mha" if qh == kvh else ("mqa" if kvh == 1 else "gqa")
        out.append(
            {
                "attn_type": attn_type,
                "q_num_heads": qh,
                "kv_num_heads": kvh,
                "head_dim": hd,
                "calib": calib,
                "test": tests,
            }
        )
    return out


def _add_mini_sample(groups: dict[str, list[dict[str, Any]]], datasets_dir: str) -> None:
    linear_path = os.path.join(datasets_dir, "linear.pt")
    attn_path = os.path.join(datasets_dir, "attn.pt")
    if not (os.path.isfile(linear_path) and os.path.isfile(attn_path)):
        raise ValueError(
            f"mini-sample directory {datasets_dir!r} must contain linear.pt and attn.pt"
        )
    linear_raw = torch.load(linear_path, weights_only=True, map_location="cpu")
    attn_raw = torch.load(attn_path, weights_only=True, map_location="cpu")
    for group in _normalize_linear_groups(linear_raw):
        groups["linear"].append({"dist": "public", "data": group})
    for group in _normalize_attention_groups(attn_raw):
        groups["attention"].append(
            {"dist": "public", "attn_type": group["attn_type"], "data": group}
        )


# ===========================================================================
# Reference outputs (computed once on the pristine data)
# ===========================================================================

def _compute_references(groups: dict[str, list[dict[str, Any]]]) -> tuple[dict[tuple, torch.Tensor], dict[tuple, float]]:
    refs: dict[tuple, torch.Tensor] = {}
    powers: dict[tuple, float] = {}
    for gi, entry in enumerate(groups["linear"]):
        lg = entry["data"]
        wq, ws = lg["weight"]
        for ti, (aq, as_) in enumerate(lg["test_activation_list"]):
            ref = linear_output_nvfp4(aq, as_, wq, ws)
            refs[("linear", gi, ti)] = ref
            powers[("linear", gi, ti)] = float((ref.to(torch.float32) ** 2).mean().item())
    for gi, entry in enumerate(groups["attention"]):
        ag = entry["data"]
        heads = (ag["q_num_heads"], ag["kv_num_heads"], ag["head_dim"])
        for ti, sample in enumerate(ag["test"]):
            ref = attention_output_nvfp4(
                sample["q"][0], sample["q"][1],
                sample["k"][0], sample["k"][1],
                sample["v"][0], sample["v"][1],
                *heads,
                causal=False,
            )
            refs[("attn", gi, ti)] = ref
            powers[("attn", gi, ti)] = float((ref.to(torch.float32) ** 2).mean().item())
    return refs, powers


# ===========================================================================
# HiF4 parameter legality (mirrors example/self_check.py validate_hif4_params)
# ===========================================================================

def _expected_hif4_shapes(shape: Any) -> dict[str, tuple[int, ...]]:
    shape = tuple(int(s) for s in shape)
    if not shape:
        raise ValueError("shape must have at least one dimension")
    channels = shape[-1]
    if channels % 64 != 0:
        raise ValueError(f"last dimension {channels} is not divisible by HiF4 block size 64")
    prefix = shape[:-1] + (channels // 64,)
    return {name: prefix + trailing for name, trailing in _HIF4_LAYOUT.items()}


def _hif4_legality_errors(params: Any, shape: Any) -> list[str]:
    """Return the list of format-contract violations (empty == legal)."""
    errors: list[str] = []
    if not isinstance(params, dict):
        return [f"expected dict of HiF4 params, got {type(params).__name__}"]
    try:
        expected = _expected_hif4_shapes(shape)
    except ValueError as exc:
        return [str(exc)]
    tensors: dict[str, torch.Tensor] = {}
    for name, expected_shape in expected.items():
        if name not in params:
            errors.append(f"missing parameter {name!r}")
            continue
        value = params[name]
        if not isinstance(value, torch.Tensor):
            errors.append(f"{name}: expected torch.Tensor, got {type(value).__name__}")
            continue
        if tuple(value.shape) != expected_shape:
            errors.append(f"{name}: shape {tuple(value.shape)} != expected {expected_shape}")
            continue
        if torch.is_complex(value):
            errors.append(f"{name}: complex tensor is not allowed")
            continue
        try:
            tensors[name] = value.detach().to(dtype=torch.float64, device="cpu")
        except Exception as exc:
            errors.append(f"{name}: conversion failed: {type(exc).__name__}: {exc}")
    if errors:
        return errors
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            errors.append(f"{name}: contains non-finite values")
    if errors:
        return errors

    scale_factor = tensors["scale_factor"]
    min_scale, max_scale = 2.0 ** -48, 49152.0
    if (scale_factor < min_scale).any():
        errors.append("scale_factor: values below 2^-48")
    if (scale_factor > max_scale).any():
        errors.append("scale_factor: values above 49152")
    if errors:
        return errors
    sf_clamped = scale_factor.clamp(min=2.0 ** -126)
    sf_exp = torch.floor(torch.log2(sf_clamped))
    sf_e6m2 = torch.round(scale_factor * (2.0 ** (2 - sf_exp))) * (2.0 ** (sf_exp - 2))
    if not torch.equal(scale_factor, sf_e6m2):
        errors.append("scale_factor: not all values are exact E6M2 values")

    for name, allowed in (("scale_lv2", (1.0, 2.0)), ("scale_lv3", (1.0, 2.0))):
        value = tensors[name]
        if not ((value == allowed[0]) | (value == allowed[1])).all():
            errors.append(f"{name}: values must be exactly {{1.0, 2.0}}")
    sign = tensors["sign"]
    if not ((sign == -1.0) | (sign == 0.0) | (sign == 1.0)).all():
        errors.append("sign: values must be exactly {-1.0, 0.0, 1.0}")
    mant = tensors["mant"]
    if (mant < 0.0).any():
        errors.append("mant: negative values are not allowed")
    if (mant > 1.75).any():
        errors.append("mant: values above 1.75 are not allowed")
    if not torch.equal(mant * 4.0, torch.round(mant * 4.0)):
        errors.append("mant: values must be exact multiples of 0.25 in [0, 1.75]")
    return errors


def _state_legality_errors(state: Any) -> list[str]:
    """Return calibration-state violations from the public checker contract."""
    errors: list[str] = []
    nodes = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 4096:
            return
        if depth > 8:
            errors.append("state nesting depth exceeds 8")
            return
        if type(value) is torch.Tensor:
            if value.device.type != "cpu" or value.layout is not torch.strided:
                errors.append("state tensors must be dense CPU tensors")
            if value.dtype not in _STATE_DTYPES:
                errors.append(f"state tensor dtype {value.dtype} is not allowed")
            if value.requires_grad:
                errors.append("state tensors must not require gradients")
            if value.is_floating_point() and not torch.isfinite(value).all():
                errors.append("state tensor contains non-finite values")
            return
        if value is None or type(value) in (bool, int):
            return
        if type(value) is float:
            if not math.isfinite(value):
                errors.append("state contains a non-finite float")
            return
        if type(value) is str:
            if len(value.encode("utf-8")) > 4096:
                errors.append("state string exceeds 4096 bytes")
            return
        if type(value) in (list, tuple):
            for item in value:
                visit(item, depth + 1)
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    errors.append("state dictionary keys must be strings")
                elif len(key.encode("utf-8")) > 4096:
                    errors.append("state dictionary key exceeds 4096 bytes")
                visit(item, depth + 1)
            return
        errors.append(f"unsupported state type {type(value).__name__}")

    visit(state, 0)
    if nodes > 4096:
        errors.append("state exceeds 4096 nodes")
    return errors


# ===========================================================================
# Module run: exercise all six API functions, measure wall time separately
# ===========================================================================

def _run_module(
    funcs: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    refs: dict[tuple, torch.Tensor],
    label: str,
) -> dict[str, Any]:
    """Run one full evaluation pass over the dataset.

    Returns per-case MSE (against the shared reference), per-group wall time
    for calibration vs online quantization, and legality summaries.  Raises on
    the first API contract violation (missing return keys, exceptions).
    """
    calib_s = 0.0
    online_s = 0.0
    scoring_s = 0.0
    cases: list[dict[str, Any]] = []
    legality_errors: list[str] = []
    legality_bad = False

    def note_errors(errors: list[str]) -> bool:
        """Record legality violations; returns True when the params are legal."""
        nonlocal legality_bad
        if errors:
            legality_bad = True
            legality_errors.extend(errors[:3])
            return False
        return True

    t_start = time.perf_counter()

    for gi, entry in enumerate(groups["linear"]):
        lg = entry["data"]
        dist = entry["dist"]
        data = _clone_data(lg)
        wq, ws = data["weight"]

        t = time.perf_counter()
        calib_result = funcs["weight"](wq, ws, data["calib_activation_list"])
        calib_s += time.perf_counter() - t
        expected_keys = {"weight_params", "activation_state"}
        if not isinstance(calib_result, dict) or set(calib_result) != expected_keys:
            raise RuntimeError(
                f"{label} linear group {gi}: calibration result keys must be "
                f"{sorted(expected_keys)}"
            )
        wparams = calib_result["weight_params"]
        state = calib_result["activation_state"]
        weight_legal = note_errors(_hif4_legality_errors(wparams, tuple(wq.shape)))
        state_legal = note_errors(_state_legality_errors(state))

        t = time.perf_counter()
        w_hif4 = dequantize_hif4_params(wparams, tuple(wq.shape))
        scoring_s += time.perf_counter() - t

        for ti, (aq, as_) in enumerate(data["test_activation_list"]):
            t = time.perf_counter()
            activation_params = funcs["activation"](aq, as_, _clone_data(state))
            online_s += time.perf_counter() - t
            act_legal = note_errors(_hif4_legality_errors(activation_params, tuple(aq.shape)))

            t = time.perf_counter()
            a_hif4 = dequantize_hif4_params(activation_params, tuple(aq.shape))
            out = linear_output(a_hif4, w_hif4)
            mse = scalar_mse(out, refs[("linear", gi, ti)])
            scoring_s += time.perf_counter() - t
            cases.append(
                {
                    "scenario": "linear",
                    "group": gi,
                    "test": ti,
                    "dist": dist,
                    "attn_type": None,
                    "shape": {"input": tuple(aq.shape), "output": tuple(out.shape)},
                    "mse": mse,
                    "legality_ok": weight_legal and state_legal and act_legal,
                }
            )

    for gi, entry in enumerate(groups["attention"]):
        ag = entry["data"]
        dist = entry["dist"]
        attn_type = entry["attn_type"]
        data = _clone_data(ag)
        heads = (data["q_num_heads"], data["kv_num_heads"], data["head_dim"])

        t = time.perf_counter()
        calib_result = funcs["attention"](data["calib"], *heads)
        calib_s += time.perf_counter() - t
        expected_keys = {"q_state", "k_state", "v_state"}
        if not isinstance(calib_result, dict) or set(calib_result) != expected_keys:
            raise RuntimeError(
                f"{label} attention group {gi}: calibration result keys must be "
                f"{sorted(expected_keys)}"
            )
        states = {role: calib_result[f"{role}_state"] for role in ("q", "k", "v")}
        for role, state in states.items():
            note_errors(_state_legality_errors(state))
        qh, kvh, hd = heads

        for ti, sample in enumerate(data["test"]):
            params: dict[str, Any] = {}
            sample_legal = True
            for role, func_name in (("q", "q"), ("k", "k"), ("v", "v")):
                quant_t, scale_t = sample[role]
                num_heads = qh if role == "q" else kvh
                t = time.perf_counter()
                params[role] = funcs[func_name](quant_t, scale_t, num_heads, hd, _clone_data(states[role]))
                online_s += time.perf_counter() - t
                sample_legal = note_errors(_hif4_legality_errors(params[role], tuple(quant_t.shape))) and sample_legal

            t = time.perf_counter()
            q_hif4 = dequantize_hif4_params(params["q"], tuple(sample["q"][0].shape))
            k_hif4 = dequantize_hif4_params(params["k"], tuple(sample["k"][0].shape))
            v_hif4 = dequantize_hif4_params(params["v"], tuple(sample["v"][0].shape))
            out = attention_output(q_hif4, k_hif4, v_hif4, qh, kvh, hd, causal=False)
            mse = scalar_mse(out, refs[("attn", gi, ti)])
            scoring_s += time.perf_counter() - t
            cases.append(
                {
                    "scenario": "attention",
                    "group": gi,
                    "test": ti,
                    "dist": dist,
                    "attn_type": attn_type,
                    "shape": {"input": tuple(sample["q"][0].shape), "output": tuple(out.shape)},
                    "mse": mse,
                    "legality_ok": sample_legal,
                }
            )

    wall_s = time.perf_counter() - t_start
    return {
        "cases": cases,
        "calibration_s": calib_s,
        "online_s": online_s,
        "api_total_s": calib_s + online_s,
        "scoring_s": scoring_s,
        "wall_s": wall_s,
        "legality_bad": legality_bad,
        "legality_errors": legality_errors[:5],
    }


# ===========================================================================
# Aggregation and status
# ===========================================================================

def _aggregate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    ref_powers: dict[tuple, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge per-case MSEs into the per-case metric records.

    Returns ``(case_records, invalid_reasons)``.  A case is invalid when the
    baseline MSE is zero or non-finite, or the candidate MSE is non-finite.
    """
    case_records: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    if len(baseline["cases"]) != len(candidate["cases"]):
        raise ValueError("baseline and candidate produced different case counts")
    for bc, cc in zip(baseline["cases"], candidate["cases"]):
        case_id = (bc["scenario"], bc["group"], bc["test"])
        candidate_id = (cc["scenario"], cc["group"], cc["test"])
        if case_id != candidate_id:
            raise ValueError(
                f"baseline/candidate case order mismatch: {case_id} != {candidate_id}"
            )
        key = ("linear" if bc["scenario"] == "linear" else "attn", bc["group"], bc["test"])
        ref_power = ref_powers[key]
        bm = float(bc["mse"])
        cm = float(cc["mse"])
        record = {
            "scenario": bc["scenario"],
            "group": bc["group"],
            "test": bc["test"],
            "dist": bc["dist"],
            "attn_type": bc.get("attn_type"),
            "shape": bc["shape"],
            "ref_power": ref_power,
            "baseline_mse": bm,
            "candidate_mse": cm,
            "improvement_percent": None,
            "valid": True,
        }
        reason = None
        if not math.isfinite(bm):
            reason = "baseline_mse_nonfinite"
        elif bm == 0.0:
            reason = "baseline_mse_zero"
        elif not math.isfinite(cm):
            reason = "candidate_mse_nonfinite"
        if reason is not None:
            record["valid"] = False
            record["invalid_reason"] = reason
            invalid_reasons.append(reason)
        else:
            record["improvement_percent"] = 100.0 * (bm - cm) / bm
        case_records.append(record)
    return case_records, invalid_reasons


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _round(value: float) -> float:
    value = float(value)
    return round(value, 10) if math.isfinite(value) else None


def _benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dists": [d.strip() for d in args.dists.split(",") if d.strip()],
        "attn_types": [t.strip() for t in args.attn_types.split(",") if t.strip()],
        "n_groups_per_dist": args.n_groups_per_dist,
        "out_features": args.out_features,
        "in_features": args.in_features,
        "seq_len": args.seq_len,
        "n_calib": args.n_calib,
        "n_test": args.n_test,
        "q_heads": args.q_heads,
        "head_dim": args.head_dim,
        "seed": args.seed,
        "public_mini_sample": bool(args.mini_sample),
    }


def _suite_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return "synthetic-v1-" + hashlib.sha256(payload).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a candidate solution.py against a baseline on the "
        "NVFP4-to-HiF4 task (mean per-case MSE improvement).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidate",
        default="solution.py",
        help="candidate solution: a .py path, a directory containing solution.py, "
        "or a Git ref whose solution.py is read with 'git show <ref>:solution.py'",
    )
    parser.add_argument(
        "--baseline",
        default="solution.py",
        help="baseline solution (same spec forms as --candidate)",
    )
    parser.add_argument("--variant", default="v-local", help="variant name used in records/results")
    parser.add_argument("--seed", type=int, default=0, help="deterministic data seed")
    parser.add_argument(
        "--dists",
        default=",".join(DISTS),
        help="comma-separated value distributions (normal, heavy_tail, sparse, "
        "channel_outlier, mixed_block)",
    )
    parser.add_argument("--attn-types", default="gqa,mha,mqa")
    parser.add_argument("--n-groups-per-dist", type=int, default=1)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--n-calib", type=int, default=2)
    parser.add_argument("--n-test", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--mini-sample",
        default=None,
        help="optional directory containing the public linear.pt/attn.pt "
        "(e.g. example/mini_sample)",
    )
    parser.add_argument(
        "--records-dir",
        default=None,
        help="directory for detailed JSON records (default: <repo>/benchmarks/records)",
    )
    parser.add_argument(
        "--results-jsonl",
        default=None,
        help="results.jsonl path (default: <repo>/progress/results.jsonl)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="do not append the compact row to progress/results.jsonl",
    )
    args = parser.parse_args(argv)

    wall_t0 = time.perf_counter()
    records_dir = args.records_dir or os.path.join(_REPO_ROOT, "benchmarks", "records")
    results_jsonl = args.results_jsonl or os.path.join(_REPO_ROOT, "progress", "results.jsonl")
    variant = args.variant.replace("/", "_")
    config = _benchmark_config(args)
    suite_id = _suite_id(config)

    # Initialized so the failure path can print a consistent summary.
    baseline_info = candidate_info = None
    baseline_res = candidate_res = None
    case_records: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    reference_s = 0.0
    status = "failed"
    failure = None
    mean_improvement = None
    wall_total_s = 0.0

    try:
        # ---- dataset + references -------------------------------------------------
        groups, counts = _build_dataset(args)
        if args.mini_sample:
            _add_mini_sample(groups, os.path.abspath(args.mini_sample))
        counts = {"linear": len(groups["linear"]), "attention": len(groups["attention"])}
        t_ref = time.perf_counter()
        refs, ref_powers = _compute_references(groups)
        reference_s = time.perf_counter() - t_ref

        # ---- load both modules up-front (fail fast on import errors) ------------
        baseline_info = _load_solution(args.baseline, _REPO_ROOT)
        candidate_info = _load_solution(args.candidate, _REPO_ROOT)
        baseline_funcs = _get_funcs(baseline_info["module"])
        candidate_funcs = _get_funcs(candidate_info["module"])

        # ---- run (order irrelevant: each module gets pristine cloned data) ------
        t0 = time.perf_counter()
        baseline_res = _run_module(baseline_funcs, groups, refs, "baseline")
        t1 = time.perf_counter()
        candidate_res = _run_module(candidate_funcs, groups, refs, "candidate")
        t2 = time.perf_counter()
        run_s = t2 - t0

        case_records, invalid_reasons = _aggregate(baseline_res, candidate_res, ref_powers)
        valid = [c for c in case_records if c["valid"]]

        status = "ok"
        failure = None
        if invalid_reasons:
            status = "baseline_mse_invalid" if "baseline_mse" in invalid_reasons[0] else "invalid"
            failure = f"{len(invalid_reasons)} invalid case(s): {sorted(set(invalid_reasons))}"
        elif not valid:
            status = "no_valid_cases"
            failure = "no valid test cases (baseline MSE rejected)"
        elif baseline_res["legality_bad"]:
            status = "invalid_output"
            failure = (
                f"baseline produced illegal HiF4 params: {baseline_res['legality_errors']}"
            )
        elif candidate_res["legality_bad"]:
            status = "invalid_output"
            failure = (
                f"candidate produced illegal HiF4 params: {candidate_res['legality_errors']}"
            )

        mean_improvement = (
            _round(_mean([c["improvement_percent"] for c in valid]))
            if status == "ok"
            else None
        )
        linear_improvement = _round(
            _mean([c["improvement_percent"] for c in valid if c["scenario"] == "linear"])
        )
        attention_improvement = _round(
            _mean([c["improvement_percent"] for c in valid if c["scenario"] == "attention"])
        )
        linear_count = sum(1 for c in valid if c["scenario"] == "linear")
        attention_count = sum(1 for c in valid if c["scenario"] == "attention")

        wall_total_s = time.perf_counter() - wall_t0

        # ---- detailed record ------------------------------------------------------
        now = datetime.now(timezone.utc)
        repo_head = None
        head_proc = _git(["rev-parse", "HEAD"], _REPO_ROOT, "cannot resolve repo HEAD")
        if head_proc.returncode == 0:
            repo_head = head_proc.stdout.strip()

        record = {
            "schema_version": 1,
            "variant": args.variant,
            "created_at": now.isoformat(),
            "status": status,
            "repo_root": _REPO_ROOT,
            "repo_head": repo_head,
            "seed": args.seed,
            "suite": suite_id,
            "baseline": {
                "spec": args.baseline,
                "kind": baseline_info["kind"],
                "label": baseline_info["label"],
                "commit": baseline_info["commit"],
            },
            "candidate": {
                "spec": args.candidate,
                "kind": candidate_info["kind"],
                "label": candidate_info["label"],
                "commit": candidate_info["commit"],
            },
            "config": config,
            "torch_num_threads": torch.get_num_threads(),
            "groups": {"linear": counts["linear"], "attention": counts["attention"]},
            "timing": {
                "reference_s": _round(reference_s),
                "baseline": {
                    "calibration_s": _round(baseline_res["calibration_s"]),
                    "online_s": _round(baseline_res["online_s"]),
                    "api_total_s": _round(baseline_res["api_total_s"]),
                    "scoring_s": _round(baseline_res["scoring_s"]),
                    "wall_s": _round(baseline_res["wall_s"]),
                },
                "candidate": {
                    "calibration_s": _round(candidate_res["calibration_s"]),
                    "online_s": _round(candidate_res["online_s"]),
                    "api_total_s": _round(candidate_res["api_total_s"]),
                    "scoring_s": _round(candidate_res["scoring_s"]),
                    "wall_s": _round(candidate_res["wall_s"]),
                },
                "run_s": _round(run_s),
                "wall_total_s": _round(wall_total_s),
            },
            "metrics": {
                "mean_improvement_percent": mean_improvement,
                "case_count": len(valid),
                "case_count_total": len(case_records),
                "invalid_case_count": len(case_records) - len(valid),
                "linear_mean_improvement_percent": linear_improvement,
                "attention_mean_improvement_percent": attention_improvement,
                "linear_case_count": linear_count,
                "attention_case_count": attention_count,
            },
            "legality": {
                "baseline": {
                    "violations": baseline_res["legality_bad"],
                    "first_errors": baseline_res["legality_errors"],
                },
                "candidate": {
                    "violations": candidate_res["legality_bad"],
                    "first_errors": candidate_res["legality_errors"],
                },
            },
            "cases": case_records,
        }
        if failure:
            record["failure"] = failure
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        record = {
            "schema_version": 1,
            "variant": args.variant,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}",
            "seed": args.seed,
            "suite": suite_id,
            "config": config,
            "baseline_spec": args.baseline,
            "candidate_spec": args.candidate,
            "metrics": None,
            "cases": [],
        }
        status = "failed"
        mean_improvement = None
        wall_total_s = time.perf_counter() - wall_t0
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

    # ---- persist --------------------------------------------------------------
    os.makedirs(records_dir, exist_ok=True)
    record_path = os.path.join(records_dir, f"{variant}.json")
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")

    appended = False
    if status == "ok" and not args.no_append:
        now = datetime.now(timezone.utc)
        row = {
            "variant": args.variant,
            "commit": candidate_info["commit"] or candidate_info["label"],
            "suite": suite_id,
            "mean_improvement_percent": mean_improvement,
            "case_count": len(valid),
            "runtime": _round(wall_total_s),
            "status": status,
            "timestamp": now.isoformat(),
            "ts": now.isoformat(),
        }
        os.makedirs(os.path.dirname(results_jsonl) or ".", exist_ok=True)
        with open(results_jsonl, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        appended = True

    # ---- console summary -------------------------------------------------------
    print("=" * 72)
    print(f"variant   : {args.variant}")
    if baseline_info is not None:
        print(f"baseline  : {baseline_info['kind']:5s} {args.baseline}")
    if candidate_info is not None:
        print(f"candidate : {candidate_info['kind']:5s} {args.candidate}")
    print(f"status    : {status}" + (f"  ({failure})" if failure else ""))
    print(f"case_count: {len(valid)} valid / {len(case_records)} total")
    if mean_improvement is not None:
        print(f"mean improvement      : {mean_improvement:>12.10f} %")
        print(f"  linear     ({linear_count}): {linear_improvement:.10f} %")
        print(f"  attention  ({attention_count}): {attention_improvement:.10f} %")
    else:
        print("mean improvement      : n/a")
    if status == "ok" and baseline_res is not None and candidate_res is not None:
        print(f"baseline api wall time: {baseline_res['api_total_s']:.4f} s "
              f"(calib {baseline_res['calibration_s']:.4f}, online {baseline_res['online_s']:.4f})")
        print(f"candidate api wall time: {candidate_res['api_total_s']:.4f} s "
              f"(calib {candidate_res['calibration_s']:.4f}, online {candidate_res['online_s']:.4f})")
        print(f"reference compute     : {reference_s:.4f} s")
    print(f"wall total            : {wall_total_s:.4f} s")
    print(f"record                : {record_path}")
    print(f"results.jsonl         : {results_jsonl} (appended: {appended})")
    print("=" * 72)

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
