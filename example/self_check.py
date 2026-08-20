#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contestant-side local HiF4 output-format self-check.

Checks the public six-function submission interface, calibration-state format,
and returned HiF4 parameter format using the provided local mini-sample data.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from typing import Any, Iterable

import torch


DEFAULT_DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mini_sample",
)

API_NAMES = {
    "weight": "hif4_calibration_and_quantize_weight",
    "activation": "hif4_dynamic_quantize_activation",
    "attention": "hif4_calibration_attention",
    "q": "hif4_dynamic_quantize_q",
    "k": "hif4_dynamic_quantize_k",
    "v": "hif4_dynamic_quantize_v",
}

FROZEN_STATE_ALLOWED_TENSOR_DTYPES = frozenset({
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
})
FROZEN_STATE_MAX_DEPTH = 8
FROZEN_STATE_MAX_NODES = 4096
FROZEN_STATE_MAX_STRING_BYTES = 4096


class CheckSummary:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed

    def add(self, ok: bool) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def merge(self, other: "CheckSummary") -> None:
        self.passed += other.passed
        self.failed += other.failed


# =============================================================================
# HiF4 parameter validation
# =============================================================================

def _expected_hif4_param_shapes(
    original_shape: Iterable[int],
) -> dict[str, tuple[int, ...]]:
    shape = tuple(int(s) for s in original_shape)
    if not shape:
        raise ValueError("original_shape must have at least one dimension")

    channels = int(shape[-1])
    if channels % 64 != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by HiF4 block size 64"
        )

    prefix = shape[:-1] + (channels // 64,)
    return {
        "scale_factor": prefix + (1, 1, 1),
        "scale_lv2": prefix + (8, 1, 1),
        "scale_lv3": prefix + (8, 2, 1),
        "sign": prefix + (8, 2, 4),
        "mant": prefix + (8, 2, 4),
    }


def validate_hif4_params(
    params: Any,
    original_shape: Iterable[int],
    tag: str,
) -> list[str]:
    errors: list[str] = []

    try:
        expected_shapes = _expected_hif4_param_shapes(original_shape)
    except Exception as exc:
        return [f"{tag}: {exc}"]

    if not isinstance(params, dict):
        return [f"{tag}: expected dict, got {type(params).__name__}"]

    tensors: dict[str, torch.Tensor] = {}
    for name, expected_shape in expected_shapes.items():
        if name not in params:
            errors.append(f"{tag}: missing parameter {name!r}")
            continue

        value = params[name]
        if not isinstance(value, torch.Tensor):
            errors.append(
                f"{tag}.{name}: expected torch.Tensor, got {type(value).__name__}"
            )
            continue

        if tuple(value.shape) != expected_shape:
            errors.append(
                f"{tag}.{name}: shape {tuple(value.shape)} != expected {expected_shape}"
            )
            continue

        if torch.is_complex(value):
            errors.append(f"{tag}.{name}: complex tensor is not allowed")
            continue

        try:
            tensors[name] = value.detach().to(dtype=torch.float64, device="cpu")
        except Exception as exc:
            errors.append(
                f"{tag}.{name}: failed to convert tensor for validation: "
                f"{type(exc).__name__}: {exc}"
            )

    if errors:
        return errors

    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            count = int((~torch.isfinite(value)).sum().item())
            errors.append(f"{tag}.{name}: contains {count} non-finite values")

    if errors:
        return errors

    scale_factor = tensors["scale_factor"]
    scale_lv2 = tensors["scale_lv2"]
    scale_lv3 = tensors["scale_lv3"]
    sign = tensors["sign"]
    mant = tensors["mant"]

    try:
        dequant = sign * mant * scale_lv2 * scale_lv3 * scale_factor
        dequant = dequant.reshape(tuple(int(s) for s in original_shape))
    except Exception as exc:
        return [
            f"{tag}: HiF4 parameter broadcasting/reshape failed: "
            f"{type(exc).__name__}: {exc}"
        ]

    if not torch.isfinite(dequant).all():
        count = int((~torch.isfinite(dequant)).sum().item())
        errors.append(f"{tag}: dequantized tensor contains {count} non-finite values")
        return errors

    expected_numel = math.prod(tuple(int(s) for s in original_shape))
    if int(dequant.numel()) != int(expected_numel):
        errors.append(
            f"{tag}: dequantized numel {dequant.numel()} != expected {expected_numel}"
        )

    # scale_factor must be an exact legal E6M2 value in the public HiF4 range.
    min_scale = 2.0 ** (-48)
    max_scale = 49152.0

    below = scale_factor < min_scale
    if below.any():
        errors.append(
            f"{tag}.scale_factor: {int(below.sum().item())} values below 2^-48"
        )

    above = scale_factor > max_scale
    if above.any():
        errors.append(
            f"{tag}.scale_factor: {int(above.sum().item())} values above 49152"
        )

    sf_clamped = scale_factor.clamp(min=2.0 ** (-126))
    sf_exp = torch.floor(torch.log2(sf_clamped))
    sf_e6m2 = (
        torch.round(scale_factor * (2.0 ** (2 - sf_exp)))
        * (2.0 ** (sf_exp - 2))
    )
    e6m2_ok = scale_factor == sf_e6m2
    if not e6m2_ok.all():
        errors.append(
            f"{tag}.scale_factor: {int((~e6m2_ok).sum().item())} values "
            "are not exact E6M2 values"
        )

    lv2_ok = (scale_lv2 == 1.0) | (scale_lv2 == 2.0)
    if not lv2_ok.all():
        invalid = scale_lv2[~lv2_ok].unique().tolist()[:5]
        errors.append(
            f"{tag}.scale_lv2: values must be exactly {{1.0, 2.0}}; "
            f"examples={invalid}"
        )

    lv3_ok = (scale_lv3 == 1.0) | (scale_lv3 == 2.0)
    if not lv3_ok.all():
        invalid = scale_lv3[~lv3_ok].unique().tolist()[:5]
        errors.append(
            f"{tag}.scale_lv3: values must be exactly {{1.0, 2.0}}; "
            f"examples={invalid}"
        )

    sign_ok = (sign == -1.0) | (sign == 0.0) | (sign == 1.0)
    if not sign_ok.all():
        invalid = sign[~sign_ok].unique().tolist()[:5]
        errors.append(
            f"{tag}.sign: values must be exactly {{-1.0, 0.0, 1.0}}; "
            f"examples={invalid}"
        )

    if (mant < 0.0).any():
        errors.append(f"{tag}.mant: negative values are not allowed")
    if (mant > 1.75).any():
        errors.append(f"{tag}.mant: values above 1.75 are not allowed")

    mant_scaled = mant * 4.0
    mant_ok = mant_scaled == torch.round(mant_scaled)
    if not mant_ok.all():
        invalid = mant[~mant_ok].unique().tolist()[:5]
        errors.append(
            f"{tag}.mant: values must be exact multiples of 0.25 in [0, 1.75]; "
            f"examples={invalid}"
        )

    return errors


# =============================================================================
# Calibration-state validation
# =============================================================================

def validate_frozen_state(value: Any, tag: str) -> list[str]:
    """Validate the public pure-data state contract."""

    errors: list[str] = []
    node_count = 0

    def visit(v: Any, path: str, depth: int) -> None:
        nonlocal node_count

        node_count += 1
        if node_count > FROZEN_STATE_MAX_NODES:
            errors.append(
                f"{tag}: state contains more than {FROZEN_STATE_MAX_NODES} nodes"
            )
            return

        if depth > FROZEN_STATE_MAX_DEPTH:
            errors.append(
                f"{path}: nesting depth exceeds {FROZEN_STATE_MAX_DEPTH}"
            )
            return

        if type(v) is torch.Tensor:
            if v.device.type != "cpu":
                errors.append(f"{path}: tensor must be on CPU")
            if v.layout is not torch.strided:
                errors.append(f"{path}: tensor must use dense strided layout")
            if v.dtype not in FROZEN_STATE_ALLOWED_TENSOR_DTYPES:
                errors.append(f"{path}: tensor dtype {v.dtype} is not allowed")
            if v.requires_grad:
                errors.append(f"{path}: tensor requires_grad must be False")
            if torch.is_complex(v):
                errors.append(f"{path}: complex tensor is not allowed")
            if v.is_floating_point():
                try:
                    if not torch.isfinite(v.detach()).all():
                        errors.append(f"{path}: tensor contains non-finite values")
                except Exception:
                    errors.append(f"{path}: tensor finiteness check failed")
            return

        if v is None or type(v) is bool or type(v) is int:
            return

        if type(v) is float:
            if not math.isfinite(v):
                errors.append(f"{path}: float must be finite")
            return

        if type(v) is str:
            if len(v.encode("utf-8")) > FROZEN_STATE_MAX_STRING_BYTES:
                errors.append(
                    f"{path}: string exceeds {FROZEN_STATE_MAX_STRING_BYTES} UTF-8 bytes"
                )
            return

        if type(v) is list or type(v) is tuple:
            for i, nested in enumerate(v):
                visit(nested, f"{path}[{i}]", depth + 1)
            return

        if type(v) is dict:
            for key, nested in v.items():
                if type(key) is not str:
                    errors.append(
                        f"{path}: dict keys must be str, got {type(key).__name__}"
                    )
                    continue
                if len(key.encode("utf-8")) > FROZEN_STATE_MAX_STRING_BYTES:
                    errors.append(
                        f"{path}: dict key exceeds "
                        f"{FROZEN_STATE_MAX_STRING_BYTES} UTF-8 bytes"
                    )
                visit(nested, f"{path}.{key}", depth + 1)
            return

        errors.append(f"{path}: unsupported state type {type(v).__name__}")

    visit(value, tag, 0)
    return errors


def clone_frozen_state(value: Any) -> Any:
    """Create the fresh state copy used for each local online-call check."""

    if type(value) is torch.Tensor:
        return value.detach().to(device="cpu").contiguous().clone()
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is list:
        return [clone_frozen_state(v) for v in value]
    if type(value) is tuple:
        return tuple(clone_frozen_state(v) for v in value)
    if type(value) is dict:
        return {key: clone_frozen_state(v) for key, v in value.items()}
    raise TypeError(f"unsupported state type {type(value).__name__}")


def validate_linear_calibration_result(
    result: Any,
    weight_shape: Iterable[int],
    tag: str,
) -> list[str]:
    if not isinstance(result, dict):
        return [f"{tag}: expected dict, got {type(result).__name__}"]

    errors: list[str] = []
    allowed = {"weight_params", "activation_state"}
    unknown = sorted(set(result) - allowed)
    if unknown:
        errors.append(f"{tag}: unsupported keys {unknown}")

    if "weight_params" not in result:
        errors.append(f"{tag}: missing 'weight_params'")
    else:
        errors.extend(
            validate_hif4_params(
                result["weight_params"],
                weight_shape,
                f"{tag}.weight_params",
            )
        )

    if "activation_state" not in result:
        errors.append(f"{tag}: missing 'activation_state'")
    else:
        errors.extend(
            validate_frozen_state(
                result["activation_state"],
                f"{tag}.activation_state",
            )
        )

    return errors


def validate_attention_calibration_result(
    result: Any,
    tag: str,
) -> list[str]:
    if not isinstance(result, dict):
        return [f"{tag}: expected dict, got {type(result).__name__}"]

    errors: list[str] = []
    allowed = {"q_state", "k_state", "v_state"}
    unknown = sorted(set(result) - allowed)
    if unknown:
        errors.append(f"{tag}: unsupported keys {unknown}")

    for role in ("q", "k", "v"):
        key = f"{role}_state"
        if key not in result:
            errors.append(f"{tag}: missing {key!r}")
        else:
            errors.extend(validate_frozen_state(result[key], f"{tag}.{key}"))

    return errors


# =============================================================================
# Public mini-sample dataset validation
# =============================================================================

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

    expected_scale_shape = tuple(quant.shape[:-1]) + (channels // 16,)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"{tag}: scale shape {tuple(scale.shape)} != expected {expected_scale_shape}"
        )

    return [quant, scale]


def _normalize_linear_group(raw_group: Any, group_idx: int) -> dict[str, Any]:
    if not isinstance(raw_group, dict):
        raise TypeError(f"linear group {group_idx}: expected dict")

    required = ("weight", "calib_activation_list", "test_activation_list")
    missing = [name for name in required if name not in raw_group]
    if missing:
        raise KeyError(f"linear group {group_idx}: missing keys {missing}")

    weight = _normalize_nvfp4_pair(
        raw_group["weight"],
        f"linear group {group_idx}.weight",
    )
    channels = int(weight[0].shape[-1])

    calib_raw = raw_group["calib_activation_list"]
    test_raw = raw_group["test_activation_list"]
    if not isinstance(calib_raw, list):
        raise TypeError(f"linear group {group_idx}: calib_activation_list must be list")
    if not isinstance(test_raw, list):
        raise TypeError(f"linear group {group_idx}: test_activation_list must be list")
    if not calib_raw:
        raise ValueError(f"linear group {group_idx}: calibration list must not be empty")
    if not test_raw:
        raise ValueError(f"linear group {group_idx}: test list must not be empty")

    calib: list[list[torch.Tensor]] = []
    for i, item in enumerate(calib_raw):
        pair = _normalize_nvfp4_pair(
            item,
            f"linear group {group_idx}.calib_activation_list[{i}]",
        )
        if int(pair[0].shape[-1]) != channels:
            raise ValueError(
                f"linear group {group_idx}.calib_activation_list[{i}]: "
                f"channels {pair[0].shape[-1]} != weight channels {channels}"
            )
        calib.append(pair)

    tests: list[list[torch.Tensor]] = []
    for i, item in enumerate(test_raw):
        pair = _normalize_nvfp4_pair(
            item,
            f"linear group {group_idx}.test_activation_list[{i}]",
        )
        if int(pair[0].shape[-1]) != channels:
            raise ValueError(
                f"linear group {group_idx}.test_activation_list[{i}]: "
                f"channels {pair[0].shape[-1]} != weight channels {channels}"
            )
        tests.append(pair)

    return {
        "weight_quant": weight[0],
        "weight_scale": weight[1],
        "calib_activation_list": calib,
        "test_activation_list": tests,
    }


def _normalize_linear_dataset(raw_data: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_data, list):
        raise TypeError("linear.pt root must be a list")
    if not raw_data:
        raise ValueError("linear.pt must contain at least one group")
    return [
        _normalize_linear_group(group, idx)
        for idx, group in enumerate(raw_data)
    ]


def _normalize_attention_sample(
    sample: Any,
    tag: str,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, list[torch.Tensor]]:
    if not isinstance(sample, dict):
        raise TypeError(f"{tag}: expected dict with q/k/v")

    missing = [name for name in ("q", "k", "v") if name not in sample]
    if missing:
        raise KeyError(f"{tag}: missing keys {missing}")

    q = _normalize_nvfp4_pair(sample["q"], f"{tag}.q")
    k = _normalize_nvfp4_pair(sample["k"], f"{tag}.k")
    v = _normalize_nvfp4_pair(sample["v"], f"{tag}.v")

    if q[0].ndim != 2 or k[0].ndim != 2 or v[0].ndim != 2:
        raise ValueError(f"{tag}: Q/K/V quant tensors must be 2D [seq_len, hidden]")

    expected_q_hidden = int(q_num_heads) * int(head_dim)
    expected_kv_hidden = int(kv_num_heads) * int(head_dim)

    if int(q[0].shape[-1]) != expected_q_hidden:
        raise ValueError(
            f"{tag}.q: hidden size {q[0].shape[-1]} != "
            f"q_num_heads*head_dim {expected_q_hidden}"
        )
    if int(k[0].shape[-1]) != expected_kv_hidden:
        raise ValueError(
            f"{tag}.k: hidden size {k[0].shape[-1]} != "
            f"kv_num_heads*head_dim {expected_kv_hidden}"
        )
    if int(v[0].shape[-1]) != expected_kv_hidden:
        raise ValueError(
            f"{tag}.v: hidden size {v[0].shape[-1]} != "
            f"kv_num_heads*head_dim {expected_kv_hidden}"
        )

    if not (q[0].shape[0] == k[0].shape[0] == v[0].shape[0]):
        raise ValueError(f"{tag}: Q/K/V sequence lengths must match")

    return {"q": q, "k": k, "v": v}


def _normalize_attention_group(raw_group: Any, group_idx: int) -> dict[str, Any]:
    if not isinstance(raw_group, dict):
        raise TypeError(f"attention group {group_idx}: expected dict")

    required = (
        "q_num_heads",
        "kv_num_heads",
        "head_dim",
        "calib",
        "test",
    )
    missing = [name for name in required if name not in raw_group]
    if missing:
        raise KeyError(f"attention group {group_idx}: missing keys {missing}")

    q_num_heads = raw_group["q_num_heads"]
    kv_num_heads = raw_group["kv_num_heads"]
    head_dim = raw_group["head_dim"]

    if not all(
        type(value) is int and value > 0
        for value in (q_num_heads, kv_num_heads, head_dim)
    ):
        raise ValueError(
            f"attention group {group_idx}: q_num_heads, kv_num_heads and "
            "head_dim must be positive integers"
        )

    if q_num_heads % kv_num_heads != 0:
        raise ValueError(
            f"attention group {group_idx}: q_num_heads must be divisible by kv_num_heads"
        )

    calib_raw = raw_group["calib"]
    test_raw = raw_group["test"]
    if not isinstance(calib_raw, list):
        raise TypeError(f"attention group {group_idx}: calib must be list")
    if not isinstance(test_raw, list):
        raise TypeError(f"attention group {group_idx}: test must be list")
    if not calib_raw:
        raise ValueError(f"attention group {group_idx}: calibration list must not be empty")
    if not test_raw:
        raise ValueError(f"attention group {group_idx}: test list must not be empty")

    calib = [
        _normalize_attention_sample(
            sample,
            f"attention group {group_idx}.calib[{i}]",
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        for i, sample in enumerate(calib_raw)
    ]

    tests = [
        _normalize_attention_sample(
            sample,
            f"attention group {group_idx}.test[{i}]",
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        for i, sample in enumerate(test_raw)
    ]

    return {
        "q_num_heads": int(q_num_heads),
        "kv_num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "calib": calib,
        "test": tests,
    }


def _normalize_attention_dataset(raw_data: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_data, list):
        raise TypeError("attn.pt root must be a list")
    if not raw_data:
        raise ValueError("attn.pt must contain at least one group")
    return [
        _normalize_attention_group(group, idx)
        for idx, group in enumerate(raw_data)
    ]


# =============================================================================
# Submission checks
# =============================================================================

def _format_errors(errors: Iterable[str]) -> str:
    return "\n      ".join(str(error) for error in errors)


def _safe_exception_text(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def check_linear_group(
    weight_func: Any,
    activation_func: Any,
    group: dict[str, Any],
    group_idx: int,
) -> CheckSummary:
    summary = CheckSummary()

    try:
        calibration_result = weight_func(
            group["weight_quant"],
            group["weight_scale"],
            group["calib_activation_list"],
        )
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        summary.add(False)
        print(
            f"[Linear][Group {group_idx}] calibration: FAILED "
            f"({_safe_exception_text(exc)})"
        )
        return summary

    errors = validate_linear_calibration_result(
        calibration_result,
        tuple(group["weight_quant"].shape),
        f"[Linear][Group {group_idx}] calibration",
    )
    if errors:
        summary.add(False)
        print(
            f"[Linear][Group {group_idx}] calibration: FAILED\n"
            f"      {_format_errors(errors)}"
        )
        return summary

    summary.add(True)
    print(f"[Linear][Group {group_idx}] calibration: PASSED")

    state = calibration_result["activation_state"]
    for test_idx, (activation_quant, activation_scale) in enumerate(
        group["test_activation_list"]
    ):
        try:
            fresh_state = clone_frozen_state(state)
            result = activation_func(
                activation_quant,
                activation_scale,
                fresh_state,
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            summary.add(False)
            print(
                f"[Linear][Group {group_idx}][Test {test_idx}] activation: FAILED "
                f"({_safe_exception_text(exc)})"
            )
            continue

        errors = validate_hif4_params(
            result,
            tuple(activation_quant.shape),
            f"[Linear][Group {group_idx}][Test {test_idx}] activation",
        )
        ok = not errors
        summary.add(ok)
        if ok:
            print(
                f"[Linear][Group {group_idx}][Test {test_idx}] activation: PASSED"
            )
        else:
            print(
                f"[Linear][Group {group_idx}][Test {test_idx}] activation: FAILED\n"
                f"      {_format_errors(errors)}"
            )

    return summary


def check_attention_group(
    calibration_func: Any,
    q_func: Any,
    k_func: Any,
    v_func: Any,
    group: dict[str, Any],
    group_idx: int,
) -> CheckSummary:
    summary = CheckSummary()

    q_num_heads = group["q_num_heads"]
    kv_num_heads = group["kv_num_heads"]
    head_dim = group["head_dim"]

    try:
        calibration_result = calibration_func(
            group["calib"],
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        summary.add(False)
        print(
            f"[Attention][Group {group_idx}] calibration: FAILED "
            f"({_safe_exception_text(exc)})"
        )
        return summary

    errors = validate_attention_calibration_result(
        calibration_result,
        f"[Attention][Group {group_idx}] calibration",
    )
    if errors:
        summary.add(False)
        print(
            f"[Attention][Group {group_idx}] calibration: FAILED\n"
            f"      {_format_errors(errors)}"
        )
        return summary

    summary.add(True)
    print(f"[Attention][Group {group_idx}] calibration: PASSED")

    states = {
        "q": calibration_result["q_state"],
        "k": calibration_result["k_state"],
        "v": calibration_result["v_state"],
    }
    funcs = {"q": q_func, "k": k_func, "v": v_func}

    for test_idx, sample in enumerate(group["test"]):
        for role in ("q", "k", "v"):
            quant_tensor, scale_tensor = sample[role]
            num_heads = q_num_heads if role == "q" else kv_num_heads

            try:
                fresh_state = clone_frozen_state(states[role])
                result = funcs[role](
                    quant_tensor,
                    scale_tensor,
                    num_heads,
                    head_dim,
                    fresh_state,
                )
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                summary.add(False)
                print(
                    f"[Attention][Group {group_idx}][Test {test_idx}] "
                    f"{role.upper()}: FAILED ({_safe_exception_text(exc)})"
                )
                continue

            errors = validate_hif4_params(
                result,
                tuple(quant_tensor.shape),
                f"[Attention][Group {group_idx}][Test {test_idx}] {role.upper()}",
            )
            ok = not errors
            summary.add(ok)
            if ok:
                print(
                    f"[Attention][Group {group_idx}][Test {test_idx}] "
                    f"{role.upper()}: PASSED"
                )
            else:
                print(
                    f"[Attention][Group {group_idx}][Test {test_idx}] "
                    f"{role.upper()}: FAILED\n      {_format_errors(errors)}"
                )

    return summary


# =============================================================================
# Loading and CLI
# =============================================================================

def _load_solution_module(solution_dir: str):
    old_path = list(sys.path)
    solution_dir = os.path.abspath(solution_dir)
    sys.path.insert(0, solution_dir)
    sys.modules.pop("solution", None)
    try:
        return importlib.import_module("solution")
    finally:
        sys.path = old_path


def _load_dataset(path: str) -> Any:
    return torch.load(path, weights_only=True, map_location="cpu")


def self_check(solution_dir: str, datasets_dir: str = DEFAULT_DATASETS_DIR) -> bool:
    solution_dir = os.path.abspath(solution_dir)
    datasets_dir = os.path.abspath(datasets_dir)

    solution_path = os.path.join(solution_dir, "solution.py")
    if not os.path.isfile(solution_path):
        print("Error: solution.py was not found in --solution_dir")
        return False

    try:
        module = _load_solution_module(solution_dir)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        print(f"Error: failed to import solution.py ({_safe_exception_text(exc)})")
        return False

    funcs: dict[str, Any] = {}
    interface_errors: list[str] = []
    for role, name in API_NAMES.items():
        func = getattr(module, name, None)
        if not callable(func):
            interface_errors.append(f"solution.{name} is missing or not callable")
        else:
            funcs[role] = func

    if interface_errors:
        for error in interface_errors:
            print(f"Error: {error}")
        return False

    print("Interface check: PASSED (6/6 functions found)")

    linear_path = os.path.join(datasets_dir, "linear.pt")
    attn_path = os.path.join(datasets_dir, "attn.pt")
    if not os.path.isfile(linear_path) or not os.path.isfile(attn_path):
        print("Error: --datasets_dir must contain linear.pt and attn.pt")
        return False

    try:
        linear_data = _normalize_linear_dataset(_load_dataset(linear_path))
        attention_data = _normalize_attention_dataset(_load_dataset(attn_path))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        print(
            "Error: failed to load or validate the local mini-sample dataset "
            f"({_safe_exception_text(exc)})"
        )
        return False

    overall = CheckSummary()

    print(f"\n{'=' * 24} Linear {'=' * 24}")
    for group_idx, group in enumerate(linear_data):
        overall.merge(
            check_linear_group(
                funcs["weight"],
                funcs["activation"],
                group,
                group_idx,
            )
        )

    print(f"\n{'=' * 22} Attention {'=' * 22}")
    for group_idx, group in enumerate(attention_data):
        overall.merge(
            check_attention_group(
                funcs["attention"],
                funcs["q"],
                funcs["k"],
                funcs["v"],
                group,
                group_idx,
            )
        )

    print(f"\n{'=' * 24} Summary {'=' * 24}")
    print(f"Passed checks: {overall.passed}/{overall.total}")
    print(f"Failed checks: {overall.failed}/{overall.total}")

    success = overall.total > 0 and overall.failed == 0
    if success:
        print("ALL OUTPUT-FORMAT CHECKS PASSED")
    else:
        print("SOME OUTPUT-FORMAT CHECKS FAILED")
    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Contestant-side local HiF4 output-format self-check"
    )
    parser.add_argument(
        "--solution_dir",
        required=True,
        help="Directory containing solution.py",
    )
    parser.add_argument(
        "--datasets_dir",
        default=DEFAULT_DATASETS_DIR,
        help="Directory containing the provided mini_sample linear.pt and attn.pt",
    )
    args = parser.parse_args()

    return 0 if self_check(args.solution_dir, args.datasets_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
