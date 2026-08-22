"""Role-gated NVFP4-to-HiF4 conversion with local E6M2 scale search."""

from typing import Any

import torch


def _dequantize_nvfp4(
    quant: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Expand one NVFP4 scale per 16 values."""
    return (
        quant.to(torch.float32)
        .unflatten(-1, (-1, 16))
        .mul(scale.to(torch.float32).unsqueeze(-1))
        .flatten(-2, -1)
    )


def _ceil_e6m2(value: torch.Tensor) -> torch.Tensor:
    """Round positive values upward to a legal E6M2 scale."""
    minimum = 2.0**-48
    value = value.clamp(min=minimum, max=49152.0)
    exponent = torch.floor(torch.log2(value))
    step = torch.pow(2.0, exponent - 2.0)
    return (torch.ceil(value / step) * step).clamp(max=49152.0)


def _offset_e6m2(value: torch.Tensor, offset: int) -> torch.Tensor:
    """Move an exact E6M2 value by a signed number of legal lattice ticks."""
    exponent = torch.floor(torch.log2(value))
    step = torch.pow(2.0, exponent - 2.0)
    significand = torch.round(value / step).to(torch.int64)
    index = ((exponent.to(torch.int64) + 48) * 4 + significand - 4 + offset).clamp(
        0, 254
    )
    new_exponent = torch.div(index, 4, rounding_mode="floor") - 48
    new_significand = index.remainder(4) + 4
    return new_significand.to(torch.float32) * torch.pow(
        2.0, new_exponent.to(torch.float32) - 2.0
    )


def _quantize_hif4(
    value: torch.Tensor, search_neighbors: bool
) -> dict[str, torch.Tensor]:
    """Search adjacent E6M2 scales and exactly select each local hierarchy."""
    if value.shape[-1] % 64 != 0:
        raise ValueError("the last dimension must be divisible by 64")

    x = value.to(torch.float32).unflatten(-1, (-1, 8, 2, 4))
    magnitude = x.abs()

    # A HiF4 value can reach 1.75 * 2 * 2 = 7 times its block scale.
    anchor = _ceil_e6m2(
        magnitude.amax(dim=(-3, -2, -1), keepdim=True) / 7.0
    )

    def evaluate_scale(scale_factor: torch.Tensor):
        def evaluate_lv2(lv2: float):
            base = scale_factor * lv2
            mant1 = torch.clamp(
                torch.round(magnitude / base * 4.0) / 4.0, 0, 1.75
            )
            mant2 = torch.clamp(
                torch.round(magnitude / (base * 2.0) * 4.0) / 4.0,
                0,
                1.75,
            )
            error1 = ((magnitude - mant1 * base) ** 2).sum(dim=-1, keepdim=True)
            error2 = ((magnitude - mant2 * base * 2.0) ** 2).sum(
                dim=-1, keepdim=True
            )
            use_two = error2 < error1
            lv3 = torch.where(use_two, 2.0, 1.0)
            mant = torch.where(use_two, mant2, mant1)
            error = torch.where(use_two, error2, error1).sum(
                dim=(-2, -1), keepdim=True
            )
            return error, lv3, mant

        error1, lv3_for_1, mant_for_1 = evaluate_lv2(1.0)
        error2, lv3_for_2, mant_for_2 = evaluate_lv2(2.0)
        use_lv2_two = error2 < error1
        group_error = torch.where(use_lv2_two, error2, error1)
        total_error = group_error.sum(dim=(-3, -2, -1), keepdim=True)
        scale_lv2 = torch.where(use_lv2_two, 2.0, 1.0)
        scale_lv3 = torch.where(use_lv2_two, lv3_for_2, lv3_for_1)
        mant = torch.where(use_lv2_two, mant_for_2, mant_for_1)
        return total_error, scale_lv2, scale_lv3, mant

    best_error = best_scale = best_lv2 = best_lv3 = best_mant = None
    offsets = (0, -1, 1) if search_neighbors else (0,)
    for offset in offsets:
        candidate_scale = _offset_e6m2(anchor, offset)
        error, lv2, lv3, mant = evaluate_scale(candidate_scale)
        if best_error is None:
            best_error, best_scale = error, candidate_scale
            best_lv2, best_lv3, best_mant = lv2, lv3, mant
            continue
        use_candidate = error < best_error
        best_error = torch.where(use_candidate, error, best_error)
        best_scale = torch.where(use_candidate, candidate_scale, best_scale)
        best_lv2 = torch.where(use_candidate, lv2, best_lv2)
        best_lv3 = torch.where(use_candidate, lv3, best_lv3)
        best_mant = torch.where(use_candidate, mant, best_mant)

    return {
        "scale_factor": best_scale,
        "scale_lv2": best_lv2,
        "scale_lv3": best_lv3,
        "sign": torch.sign(x),
        "mant": best_mant,
    }


def _convert(
    quant: torch.Tensor, scale: torch.Tensor, search_neighbors: bool = True
) -> dict[str, torch.Tensor]:
    return _quantize_hif4(
        _dequantize_nvfp4(quant, scale), search_neighbors=search_neighbors
    )


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    del calib_activation_list
    return {
        "weight_params": _convert(
            weight_quant, weight_scale, search_neighbors=False
        ),
        "activation_state": None,
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    del activation_state
    return _convert(activation_quant, activation_scale)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    del calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    del q_num_heads, head_dim, q_state
    return _convert(q_quant, q_scale)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, k_state
    return _convert(k_quant, k_scale)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, v_state
    return _convert(v_quant, v_scale)
