"""NVFP4-to-HiF4 conversion using the reference BF16 source semantics."""

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
        .to(torch.bfloat16)
        .to(torch.float32)
    )


def _ceil_e6m2(value: torch.Tensor) -> torch.Tensor:
    """Round positive values upward to a legal E6M2 scale."""
    minimum = 2.0**-48
    value = value.clamp(min=minimum, max=49152.0)
    exponent = torch.floor(torch.log2(value))
    step = torch.pow(2.0, exponent - 2.0)
    return (torch.ceil(value / step) * step).clamp(max=49152.0)


def _quantize_hif4(value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize each 64-value block and greedily select its hierarchy."""
    if value.shape[-1] % 64 != 0:
        raise ValueError("the last dimension must be divisible by 64")

    x = value.to(torch.float32).unflatten(-1, (-1, 8, 2, 4))
    magnitude = x.abs()

    # A HiF4 value can reach 1.75 * 2 * 2 = 7 times its block scale.
    scale_factor = _ceil_e6m2(
        magnitude.amax(dim=(-3, -2, -1), keepdim=True) / 7.0
    )

    def evaluate_lv2(lv2: float):
        base = scale_factor * lv2
        mant1 = torch.clamp(torch.round(magnitude / base * 4.0) / 4.0, 0, 1.75)
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

    scale_lv2 = torch.where(use_lv2_two, 2.0, 1.0)
    scale_lv3 = torch.where(use_lv2_two, lv3_for_2, lv3_for_1)
    mant = torch.where(use_lv2_two, mant_for_2, mant_for_1)

    return {
        "scale_factor": scale_factor,
        "scale_lv2": scale_lv2,
        "scale_lv3": scale_lv3,
        "sign": torch.sign(x),
        "mant": mant,
    }


def _convert(quant: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    return _quantize_hif4(_dequantize_nvfp4(quant, scale))


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    del calib_activation_list
    return {
        "weight_params": _convert(weight_quant, weight_scale),
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
