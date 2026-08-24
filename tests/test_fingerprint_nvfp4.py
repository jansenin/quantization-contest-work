"""Tests for the NVFP4 source-fingerprint analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import fingerprint_nvfp4 as fingerprint  # noqa: E402


def _synthetic_report() -> dict:
    """Minimal in-memory report dict for exercising the Markdown renderer.

    No tensors and no dataset files are involved; the numbers are arbitrary
    except that every percentage is exact (denominator divides numerator).
    """
    role = {
        "tensor_count": 2,
        "carrier_elements": 160,
        "scale_elements": 10,
        "e4m3_scale_count": 10,
        "power_of_two_scale_count": 5,
        "zero_count": 16,
        "max_carrier_counts": {"0.0": 0, "4.0": 2, "6.0": 8},
        "scale_min": 0.001953125,
        "scale_max": 0.5,
        "carrier_counts": {"0.0": 16, "4.0": 8, "6.0": 16},
    }
    totals = {
        "carrier_elements": 320,
        "scale_elements": 20,
        "legal_carrier_count": 320,
        "e4m3_scale_count": 20,
        "power_of_two_scale_count": 10,
        "zero_count": 32,
        "exact_bf16_product_count": 320,
        "fixed_point_matches": 20,
        "informative_blocks": 2,
        "informative_fixed_point_matches": 2,
        "scale_min": 0.001953125,
        "scale_max": 0.5,
        "carrier_counts": {"0.0": 32, "4.0": 16, "6.0": 32},
        "max_carrier_counts": {"0.0": 0, "4.0": 4, "6.0": 16},
    }
    return {
        "format": "nvfp4-source-fingerprint-v1",
        "e4m3_positive_value_count": 126,
        "roles": {"weight": dict(role), "activation": dict(role)},
        "totals": totals,
        "pairs": [
            {"unit_global_factor_valid": True, "valid_global_factors_are_powers_of_two": True}
            for _ in range(4)
        ],
    }


class FingerprintTests(unittest.TestCase):
    def test_positive_e4m3fn_grid(self) -> None:
        values = fingerprint.positive_e4m3fn_values()
        self.assertEqual(len(values), 126)
        self.assertEqual(values[0], 2.0 ** -9)
        self.assertEqual(values[-1], 448.0)
        self.assertIn(1.0, values)
        self.assertNotIn(480.0, values)
        self.assertTrue(fingerprint._is_power_of_two(2.0 ** -9))
        self.assertFalse(fingerprint._is_power_of_two(1.5))

    def test_analyze_pair_recognizes_e4m3_ceiling_signature(self) -> None:
        quant = torch.tensor(
            [[6.0] + [0.0] * 15 + [4.0] + [0.0] * 15],
            dtype=torch.bfloat16,
        )
        scale = torch.tensor([[1.0, 2.0 ** -8]], dtype=torch.bfloat16)

        result = fingerprint.analyze_pair("sample", "weight", quant, scale)

        self.assertEqual(result["legal_carrier_count"], 32)
        self.assertEqual(result["e4m3_scale_count"], 2)
        self.assertEqual(result["fixed_point_matches"], 2)
        self.assertEqual(result["informative_blocks"], 1)
        self.assertEqual(result["informative_fixed_point_matches"], 1)
        self.assertEqual(result["exact_bf16_product_count"], 32)
        self.assertTrue(result["unit_global_factor_valid"])

    def test_analyze_pair_rejects_arbitrary_bf16_scale(self) -> None:
        quant = torch.tensor([[6.0] + [0.0] * 15], dtype=torch.bfloat16)
        scale = torch.tensor([[1.296875]], dtype=torch.bfloat16)

        result = fingerprint.analyze_pair("sample", "activation", quant, scale)

        self.assertEqual(result["e4m3_scale_count"], 0)
        self.assertEqual(result["fixed_point_matches"], 0)
        self.assertFalse(result["unit_global_factor_valid"])

    def test_dataset_roles_are_aggregated(self) -> None:
        pair = [
            torch.tensor([[6.0] + [0.0] * 15], dtype=torch.bfloat16),
            torch.tensor([[1.0]], dtype=torch.bfloat16),
        ]
        linear = [{
            "weight": pair,
            "calib_activation_list": [pair],
            "test_activation_list": [pair],
        }]
        attention = [{
            "calib": [{role: pair for role in ("q", "k", "v")}],
            "test": [{role: pair for role in ("q", "k", "v")}],
        }]

        report = fingerprint.fingerprint_datasets(linear, attention)

        self.assertEqual(report["totals"]["tensor_count"], 9)
        self.assertEqual(report["roles"]["weight"]["tensor_count"], 1)
        self.assertEqual(report["roles"]["activation"]["tensor_count"], 2)
        self.assertEqual(report["roles"]["q"]["tensor_count"], 2)

    def test_render_markdown_distinguishes_observation_from_nonidentifiability(
        self,
    ) -> None:
        md = fingerprint.render_markdown(_synthetic_report())

        # Corrected language: observations do not identify historical ceiling
        # rounding, and the nearest-rounding counterexample is documented.
        self.assertIn("## Design rationale", md)
        self.assertIn("## Non-identifiability", md)
        self.assertIn("does not by itself identify historical ceiling rounding", md)
        self.assertIn("`delta = 2^-9`", md)
        self.assertIn("9.5 * delta", md)
        self.assertIn("rounds to E2M1 carrier 4", md)
        self.assertIn("source scale rounding mode", md)

        # Exact measured statistics still appear unchanged.
        self.assertIn("Analyzed 320 carriers and 20 per-16 scales across 4 tensors.", md)
        self.assertIn("Legal E2M1 carriers: 100.000000%.", md)
        self.assertIn("`scale == ceil_E4M3(max_stored_abs / 6)`: 100.000000%.", md)

        # The old overclaiming template text must be gone.
        self.assertNotIn("## Interpretation", md)
        self.assertNotIn("Its useful evidence", md)
        self.assertNotIn("strongly compatible with a per-16 recipe", md)
        self.assertNotIn("rounds the block range upward", md)


if __name__ == "__main__":
    unittest.main()
