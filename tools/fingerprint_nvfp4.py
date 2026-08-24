#!/usr/bin/env python3
"""Fingerprint NVFP4 carriers and source scales in contest-shaped datasets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import torch


NVFP4_CARRIERS = (
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)
ROLE_ORDER = ("weight", "activation", "q", "k", "v")


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


E4M3_VALUES = positive_e4m3fn_values()
E4M3_SET = frozenset(E4M3_VALUES)
E4M3_TENSOR = torch.tensor(E4M3_VALUES, dtype=torch.float32)


def _iter_dataset_pairs(
    linear_data: Any,
    attention_data: Any,
) -> Iterable[tuple[str, str, torch.Tensor, torch.Tensor]]:
    for group_index, group in enumerate(linear_data):
        yield f"linear[{group_index}].weight", "weight", *group["weight"]
        for split, key in (
            ("calib", "calib_activation_list"),
            ("test", "test_activation_list"),
        ):
            for sample_index, pair in enumerate(group[key]):
                yield (
                    f"linear[{group_index}].{split}[{sample_index}]",
                    "activation",
                    *pair,
                )

    for group_index, group in enumerate(attention_data):
        for split in ("calib", "test"):
            for sample_index, sample in enumerate(group[split]):
                for role in ("q", "k", "v"):
                    yield (
                        f"attention[{group_index}].{split}[{sample_index}].{role}",
                        role,
                        *sample[role],
                    )


def _is_power_of_two(value: float) -> bool:
    if not math.isfinite(value) or value <= 0.0:
        return False
    mantissa, _ = math.frexp(value)
    return mantissa == 0.5


def _is_power_of_two_fraction(value: Fraction) -> bool:
    numerator = value.numerator
    denominator = value.denominator
    return (
        numerator > 0
        and numerator & (numerator - 1) == 0
        and denominator & (denominator - 1) == 0
    )


def _valid_global_factors(scales: Iterable[float]) -> tuple[Fraction, ...]:
    """Find exact g such that every scale/g is a positive E4M3FN value."""
    scale_values = tuple(sorted({Fraction.from_float(float(x)) for x in scales}))
    if not scale_values:
        return ()
    e4_values = tuple(Fraction.from_float(x) for x in E4M3_VALUES)
    e4_set = frozenset(e4_values)
    first = scale_values[0]
    candidates = {
        first / e4
        for e4 in e4_values
        if all(scale / (first / e4) in e4_set for scale in scale_values)
    }
    return tuple(sorted(candidates))


def _e4_mantissa_index(value: float) -> int:
    fraction, exponent = math.frexp(value)
    normalized = fraction * 2.0
    if value < 2.0 ** -6:
        return int(round(value / (2.0 ** -9)))
    del exponent
    return int(round((normalized - 1.0) * 8.0))


def analyze_pair(
    name: str,
    role: str,
    quant: torch.Tensor,
    scale: torch.Tensor,
    chunk_blocks: int = 262_144,
) -> dict[str, Any]:
    """Analyze one decoded-carrier/per-16-scale NVFP4 pair."""
    if quant.ndim < 1 or quant.shape[-1] % 16:
        raise ValueError(f"{name}: carrier last dimension must be divisible by 16")
    expected_scale_shape = quant.shape[:-1] + (quant.shape[-1] // 16,)
    if scale.shape != expected_scale_shape:
        raise ValueError(
            f"{name}: scale shape {tuple(scale.shape)} != {tuple(expected_scale_shape)}"
        )

    quant = quant.detach().cpu()
    scale = scale.detach().cpu()
    blocks = quant.reshape(-1, 16)
    scales = scale.reshape(-1)
    unique_scales, scale_counts = torch.unique(scales, return_counts=True)
    scale_histogram = {
        float(value): int(count)
        for value, count in zip(unique_scales.tolist(), scale_counts.tolist())
    }

    carrier_counts = {
        value: int((quant == value).sum().item())
        for value in NVFP4_CARRIERS
    }
    legal_carriers = sum(carrier_counts.values())
    negative_zero_count = int(
        (quant.contiguous().view(torch.int16) == -32768).sum().item()
    )

    e4_scale_count = sum(
        count for value, count in scale_histogram.items() if value in E4M3_SET
    )
    power_of_two_scale_count = sum(
        count for value, count in scale_histogram.items() if _is_power_of_two(value)
    )
    e4_mantissa_counts: Counter[int] = Counter()
    for value, count in scale_histogram.items():
        if value in E4M3_SET:
            e4_mantissa_counts[_e4_mantissa_index(value)] += count

    max_carrier_counts: Counter[float] = Counter()
    fixed_point_matches = 0
    informative_fixed_point_matches = 0
    informative_blocks = 0
    exact_product_count = 0
    for start in range(0, blocks.shape[0], chunk_blocks):
        stop = min(start + chunk_blocks, blocks.shape[0])
        block = blocks[start:stop]
        block_scale = scales[start:stop]
        max_carrier = block.abs().amax(dim=-1).to(torch.float32)
        max_carrier_counts.update(max_carrier.tolist())

        target = max_carrier * block_scale.to(torch.float32) / 6.0
        indices = torch.searchsorted(E4M3_TENSOR, target).clamp_max(
            len(E4M3_VALUES) - 1
        )
        expected = E4M3_TENSOR[indices]
        matches = expected == block_scale.to(torch.float32)
        fixed_point_matches += int(matches.sum().item())
        informative = max_carrier < 6.0
        informative_blocks += int(informative.sum().item())
        informative_fixed_point_matches += int((matches & informative).sum().item())

        product_fp32 = (
            block.to(torch.float32)
            * block_scale.to(torch.float32).unsqueeze(-1)
        )
        product_bf16 = (
            block * block_scale.unsqueeze(-1)
        ).to(torch.float32)
        exact_product_count += int((product_fp32 == product_bf16).sum().item())

    valid_factors = _valid_global_factors(scale_histogram)
    return {
        "name": name,
        "role": role,
        "carrier_elements": quant.numel(),
        "scale_elements": scale.numel(),
        "legal_carrier_count": legal_carriers,
        "carrier_counts": {str(value): count for value, count in carrier_counts.items()},
        "zero_count": carrier_counts[0.0],
        "negative_zero_count": negative_zero_count,
        "unique_scale_count": len(scale_histogram),
        "scale_min": float(scales.min().item()),
        "scale_max": float(scales.max().item()),
        "e4m3_scale_count": e4_scale_count,
        "power_of_two_scale_count": power_of_two_scale_count,
        "e4m3_mantissa_counts": {
            str(index): e4_mantissa_counts[index]
            for index in sorted(e4_mantissa_counts)
        },
        "max_carrier_counts": {
            str(value): count for value, count in sorted(max_carrier_counts.items())
        },
        "fixed_point_matches": fixed_point_matches,
        "informative_blocks": informative_blocks,
        "informative_fixed_point_matches": informative_fixed_point_matches,
        "exact_bf16_product_count": exact_product_count,
        "valid_global_factor_count": len(valid_factors),
        "valid_global_factors_are_powers_of_two": all(
            _is_power_of_two_fraction(value) for value in valid_factors
        ),
        "unit_global_factor_valid": Fraction(1, 1) in valid_factors,
    }


def _new_aggregate(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "tensor_count": 0,
        "carrier_elements": 0,
        "scale_elements": 0,
        "legal_carrier_count": 0,
        "carrier_counts": Counter(),
        "zero_count": 0,
        "negative_zero_count": 0,
        "e4m3_scale_count": 0,
        "power_of_two_scale_count": 0,
        "e4m3_mantissa_counts": Counter(),
        "max_carrier_counts": Counter(),
        "fixed_point_matches": 0,
        "informative_blocks": 0,
        "informative_fixed_point_matches": 0,
        "exact_bf16_product_count": 0,
        "scale_min": math.inf,
        "scale_max": 0.0,
    }


def _add_to_aggregate(aggregate: dict[str, Any], pair: dict[str, Any]) -> None:
    aggregate["tensor_count"] += 1
    for key in (
        "carrier_elements", "scale_elements", "legal_carrier_count", "zero_count",
        "negative_zero_count", "e4m3_scale_count", "power_of_two_scale_count",
        "fixed_point_matches", "informative_blocks",
        "informative_fixed_point_matches", "exact_bf16_product_count",
    ):
        aggregate[key] += pair[key]
    for key in ("carrier_counts", "e4m3_mantissa_counts", "max_carrier_counts"):
        aggregate[key].update(pair[key])
    aggregate["scale_min"] = min(aggregate["scale_min"], pair["scale_min"])
    aggregate["scale_max"] = max(aggregate["scale_max"], pair["scale_max"])


def fingerprint_datasets(linear_data: Any, attention_data: Any) -> dict[str, Any]:
    pairs = []
    roles: dict[str, dict[str, Any]] = {}
    for name, role, quant, scale in _iter_dataset_pairs(linear_data, attention_data):
        pair = analyze_pair(name, role, quant, scale)
        pairs.append(pair)
        aggregate = roles.setdefault(role, _new_aggregate(role))
        _add_to_aggregate(aggregate, pair)

    serializable_roles = {}
    for role in ROLE_ORDER:
        if role not in roles:
            continue
        aggregate = roles[role]
        for key in ("carrier_counts", "e4m3_mantissa_counts", "max_carrier_counts"):
            aggregate[key] = dict(sorted(aggregate[key].items(), key=lambda item: float(item[0])))
        serializable_roles[role] = aggregate

    totals = _new_aggregate("all")
    for pair in pairs:
        _add_to_aggregate(totals, pair)
    for key in ("carrier_counts", "e4m3_mantissa_counts", "max_carrier_counts"):
        totals[key] = dict(sorted(totals[key].items(), key=lambda item: float(item[0])))

    return {
        "format": "nvfp4-source-fingerprint-v1",
        "e4m3_positive_value_count": len(E4M3_VALUES),
        "roles": serializable_roles,
        "totals": totals,
        "pairs": pairs,
    }


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.6f}%"


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    pairs = report["pairs"]
    lines = [
        "# Public NVFP4 Source Fingerprint",
        "",
        "Generated by `tools/fingerprint_nvfp4.py` from the ignored public mini-sample.",
        "The original unquantized tensors are unavailable, so this identifies exact",
        "properties of the supplied NVFP4 representation and ranks compatible source",
        "quantizers; it cannot uniquely recover the original quantization algorithm.",
        "",
        "## Exact observations",
        "",
        f"- Analyzed {totals['carrier_elements']:,} carriers and {totals['scale_elements']:,} per-16 scales across {len(pairs)} tensors.",
        f"- Legal E2M1 carriers: {_percent(totals['legal_carrier_count'], totals['carrier_elements'])}.",
        f"- Exact positive finite E4M3FN scales: {_percent(totals['e4m3_scale_count'], totals['scale_elements'])} ({report['e4m3_positive_value_count']} possible positive values).",
        f"- Exact BF16 carrier-times-scale products: {_percent(totals['exact_bf16_product_count'], totals['carrier_elements'])}. The reference BF16 multiplication introduces no additional rounding on this sample.",
        f"- `scale == ceil_E4M3(max_stored_abs / 6)`: {_percent(totals['fixed_point_matches'], totals['scale_elements'])}.",
        f"- The same fixed-point identity on informative blocks whose maximum carrier is below 6: {_percent(totals['informative_fixed_point_matches'], totals['informative_blocks'])} ({totals['informative_blocks']:,} blocks).",
        f"- Every tensor accepts global factor `g = 1` in `scale = g * E4M3`; all exact compatible factors found are powers of two: {all(pair['unit_global_factor_valid'] and pair['valid_global_factors_are_powers_of_two'] for pair in pairs)}.",
        "",
        "The fixed-point identity is an exact observation about the stored blocks, but it",
        "does not by itself identify historical ceiling rounding: it is tautological when",
        "a block contains carrier 6, and unsaturated subnormal blocks with maximum",
        "carrier 3 or 4 can also arise under nearest scale rounding (see",
        "non-identifiability below). The exact E4M3 scale grid is a separate observation",
        "that constrains the recipe without fixing its rounding mode.",
        "",
        "## Role summary",
        "",
        "| Role | Tensors | Carriers | Scales | E4M3 scales | Power-of-two scales | Zero carriers | Max carrier 6 | Scale range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role, values in report["roles"].items():
        max_six = int(values["max_carrier_counts"].get("6.0", 0))
        lines.append(
            f"| {role} | {values['tensor_count']} | {values['carrier_elements']:,} | "
            f"{values['scale_elements']:,} | {_percent(values['e4m3_scale_count'], values['scale_elements'])} | "
            f"{_percent(values['power_of_two_scale_count'], values['scale_elements'])} | "
            f"{_percent(values['zero_count'], values['carrier_elements'])} | "
            f"{_percent(max_six, values['scale_elements'])} | "
            f"{values['scale_min']:.8g} to {values['scale_max']:.8g} |"
        )

    lines.extend([
        "",
        "## Carrier magnitude frequencies",
        "",
        "| Role | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for role, values in report["roles"].items():
        signed = values["carrier_counts"]
        magnitude_counts = {0.0: int(signed.get("0.0", 0))}
        for magnitude in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
            magnitude_counts[magnitude] = int(signed.get(str(magnitude), 0)) + int(
                signed.get(str(-magnitude), 0)
            )
        cells = " | ".join(
            _percent(magnitude_counts[value], values["carrier_elements"])
            for value in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        )
        lines.append(f"| {role} | {cells} |")

    lines.extend([
        "",
        "## Design rationale",
        "",
        "The measured properties constrain the plausible source recipe without fixing",
        "it. Every one of the "
        f"{totals['scale_elements']:,} scales is an exact E4M3FN value, so a recipe",
        "that computes each per-16 scale on the E4M3 grid is compatible with this",
        "sample, while the current synthetic generator's general",
        "`BF16(max_abs / 6)` scale rule is not: arbitrary BF16 scales are not",
        "restricted to the E4M3 grid. Upward rounding of the block range to a positive",
        "E4M3 scale followed by E2M1 carrier rounding remains one compatible design.",
        "",
        "## Non-identifiability",
        "",
        "Historical ceiling rounding is not identified by the stored blocks. The",
        "`ceil_E4M3(max_abs / 6)` identity is tautological whenever a block contains",
        "carrier 6, and in the E4M3FN subnormal range the grid is coarse enough that",
        "nearest scale rounding also reproduces it. Explicit nearest-rounding",
        "counterexample, with `delta = 2^-9` (the E4M3FN subnormal step): an original",
        "block maximum of `9.5 * delta` rounds to nearest scale `2 * delta`, the",
        "normalized maximum `9.5 / 2 = 4.75` rounds to E2M1 carrier 4, and the stored",
        "block then satisfies",
        "`ceil_E4M3((4 * 2 * delta) / 6) = ceil_E4M3(4 * delta / 3) = 2 * delta = scale`",
        "exactly, with no ceiling rounding anywhere. Unsaturated subnormal blocks with",
        "maximum carrier 3 arise the same way, e.g. original maximum in",
        "`(3 * delta, 3.5 * delta)` with nearest scale `delta`.",
        "",
        "A nontrivial ModelOpt-style per-tensor global scale is not evidenced here.",
        "`g = 1` explains every scale, while multiplication by a power of two is",
        "algebraically indistinguishable where the resulting values remain in range.",
        "Carrier tie-breaking, source clipping before the observed block maximum, the",
        "original floating-point value distribution, and the source scale rounding mode",
        "cannot be recovered from the quantized tensors alone.",
        "",
        "These conclusions cover one public Linear group and one public Attention group;",
        "hidden groups could have been produced differently.",
        "",
    ])
    return "\n".join(lines)


def _load(path: Path) -> Any:
    return torch.load(path, weights_only=True, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("example/mini_sample"),
        help="directory containing linear.pt and attn.pt",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    report = fingerprint_datasets(
        _load(args.datasets_dir / "linear.pt"),
        _load(args.datasets_dir / "attn.pt"),
    )
    markdown = render_markdown(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
