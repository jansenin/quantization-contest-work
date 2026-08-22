"""Deterministic in-memory generator of legal synthetic NVFP4 data.

Produces NVFP4 ``[quant, scale]`` pairs from seeded FP32 tensors and compact
Linear / Attention calibration+test case dicts that mirror the public
``example/mini_sample`` layout consumed by ``example/self_check.py``:

- ``quant``: BF16 CPU tensor whose values are exactly the NVFP4 (E2M1) carrier
  set ``{-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}``.
- ``scale``: BF16 CPU tensor with one scale per 16 values, shape
  ``tensor.shape[:-1] + (tensor.shape[-1] // 16,)``.
- Linear groups: ``{"key", "weight", "calib_activation_list",
  "test_activation_list"}``; Attention groups: ``{"key", "attn_type",
  "q_num_heads", "kv_num_heads", "head_dim", "calib", "test"}``.
- All hidden sizes are divisible by 64 (HiF4 block size).

Covered value distributions: normal, heavy-tail (Student-t df=3), sparse,
channel-outlier, and mixed-block-scale (per-16-block magnitude spread).

Deterministic: identical seed (and arguments) reproduces identical tensors,
independent of call order. No file I/O, no dependencies beyond ``torch``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Exact NVFP4 (E2M1) carrier values, largest magnitude first.
NVFP4_CARRIERS: tuple[float, ...] = (
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)

CARRIER_MAX: float = 6.0
NVFP4_BLOCK: int = 16  # one scale per 16 values
HIF4_BLOCK: int = 64  # HiF4 block size; all hidden dims are divisible by it

#: Supported distribution names for ``make_tensor`` / ``make_nvfp4_pair``.
DISTS: tuple[str, ...] = (
    "normal",
    "heavy_tail",
    "sparse",
    "channel_outlier",
    "mixed_block",
)

_CARRIER_T = torch.tensor(NVFP4_CARRIERS, dtype=torch.float32)


def _mix_seed(seed: int, tag: int) -> int:
    """Deterministic integer mix so every generated tensor gets its own seed."""
    return (int(seed) * 2654435761 + int(tag) * 40503) & 0xFFFFFFFF


def _check_shape(shape: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(s) for s in shape)
    if len(out) < 1 or any(s <= 0 for s in out):
        raise ValueError(f"shape must contain only positive ints, got {shape!r}")
    return out


# ---------------------------------------------------------------------------
# Seeded FP32 distribution tensors
# ---------------------------------------------------------------------------

def _normal(shape: tuple[int, ...], seed: int, std: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * std


def _heavy_tail(
    shape: tuple[int, ...],
    seed: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """Student-t with 3 degrees of freedom: z / sqrt(u/3), u ~ chi2(3)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(shape, generator=g)
    u = torch.randn(shape, generator=g) ** 2
    u = u + torch.randn(shape, generator=g) ** 2
    u = u + torch.randn(shape, generator=g) ** 2
    return z * torch.sqrt(3.0 / u.clamp_min(1e-9)) * scale


def _sparse(
    shape: tuple[int, ...],
    seed: int,
    sparsity: float = 0.8,
    std: float = 1.0,
) -> torch.Tensor:
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * std
    mask = torch.rand(shape, generator=g) < sparsity
    return x.masked_fill(mask, 0.0)


def _channel_outlier(
    shape: tuple[int, ...],
    seed: int,
    n_outliers: int = 4,
    outlier_scale: float = 10.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Normal base with a few widely spaced channels scaled up."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * std
    channels = int(shape[-1])
    count = min(int(n_outliers), channels)
    if count > 0:
        idx = torch.linspace(0, channels - 1, count, dtype=torch.long)
        x[..., idx] *= outlier_scale
    return x


def _mixed_block(
    shape: tuple[int, ...],
    seed: int,
    mag_min: int = -4,
    mag_max: int = 4,
    std: float = 1.0,
) -> torch.Tensor:
    """Normal base whose magnitude differs per 16-value block (2^e, e~U[mag_min, mag_max])."""
    channels = int(shape[-1])
    if channels % NVFP4_BLOCK != 0:
        raise ValueError(f"last dim {channels} not divisible by {NVFP4_BLOCK}")
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * std
    n_blocks = channels // NVFP4_BLOCK
    exponents = torch.randint(int(mag_min), int(mag_max) + 1, (n_blocks,), generator=g)
    block_scale = torch.pow(2.0, exponents.to(torch.float32)).unsqueeze(-1)
    return x.unflatten(-1, (-1, NVFP4_BLOCK)).mul(block_scale).flatten(-2, -1)


_DIST_FUNCS = {
    "normal": _normal,
    "heavy_tail": _heavy_tail,
    "sparse": _sparse,
    "channel_outlier": _channel_outlier,
    "mixed_block": _mixed_block,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_tensor(
    dist: str,
    shape: Sequence[int],
    seed: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Draw a deterministic FP32 tensor from a seeded value distribution.

    Args:
        dist: one of ``DISTS`` (normal, heavy_tail, sparse, channel_outlier,
            mixed_block).
        shape: tensor shape (last dim may need 16-divisibility for
            ``mixed_block``).
        seed: integer seed; identical seed reproduces identical tensors.
        kwargs: distribution-specific options (``std``, ``sparsity``,
            ``n_outliers``, ``outlier_scale``, ``mag_min``, ``mag_max``).
    """
    if dist not in _DIST_FUNCS:
        raise ValueError(f"unknown distribution {dist!r}; choose from {DISTS}")
    return _DIST_FUNCS[dist](_check_shape(shape), _mix_seed(seed, 0), **kwargs)


def quantize_nvfp4(
    tensor: torch.Tensor,
    block_size: int = NVFP4_BLOCK,
) -> list[torch.Tensor]:
    """Nearest-carrier NVFP4 quantization with one BF16 scale per block_size.

    Args:
        tensor: FP32 (or convertible) tensor; last dim divisible by
            ``block_size``.
        block_size: NVFP4 block size (16 per the contest contract).

    Returns:
        ``[quant, scale]``: ``quant`` (BF16, exact carrier values, same shape
        as input) and ``scale`` (BF16, shape ``tensor.shape[:-1] +
        (tensor.shape[-1] // block_size,)``).
    """
    t = tensor.detach().to(dtype=torch.float32)
    channels = int(t.shape[-1])
    if channels % block_size != 0:
        raise ValueError(
            f"last dimension {channels} not divisible by NVFP4 block size {block_size}"
        )

    blocks = t.unflatten(-1, (-1, block_size))
    # Scale covers each block's max magnitude at the largest carrier (6.0).
    scale = blocks.abs().amax(dim=-1) / CARRIER_MAX
    scale = scale.clamp_min(torch.finfo(torch.bfloat16).tiny)
    scale = scale.to(dtype=torch.bfloat16)

    scaled = blocks / scale.to(dtype=torch.float32).unsqueeze(-1)
    distances = (scaled.unsqueeze(-1) - _CARRIER_T).abs()
    indices = distances.argmin(dim=-1)
    quant = _CARRIER_T[indices].to(dtype=torch.bfloat16)

    return [quant.flatten(-2, -1), scale]


def make_nvfp4_pair(
    dist: str,
    shape: Sequence[int],
    seed: int,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Shorthand: ``make_tensor`` followed by ``quantize_nvfp4``.

    Returns a legal NVFP4 ``[quant, scale]`` pair (both BF16, CPU).
    """
    return quantize_nvfp4(make_tensor(dist, shape, seed, **kwargs))


def make_linear_group(
    seed: int,
    dist: str = "normal",
    out_features: int = 64,
    in_features: int = 64,
    seq_len: int = 16,
    n_calib: int = 3,
    n_test: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build one Linear group in the ``mini_sample`` layout.

    Weight is ``[out_features, in_features]``; activations are
    ``[seq_len, in_features]``. All pairs are legal NVFP4; ``in_features``
    must be divisible by 64 (HiF4 block size).
    """
    seed = int(seed)
    if int(in_features) % HIF4_BLOCK != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by HiF4 block size {HIF4_BLOCK}"
        )

    weight = make_nvfp4_pair(
        dist, (int(out_features), int(in_features)), _mix_seed(seed, 1), **kwargs
    )
    calib = [
        make_nvfp4_pair(
            dist, (int(seq_len), int(in_features)), _mix_seed(seed, 100 + i), **kwargs
        )
        for i in range(int(n_calib))
    ]
    tests = [
        make_nvfp4_pair(
            dist, (int(seq_len), int(in_features)), _mix_seed(seed, 200 + i), **kwargs
        )
        for i in range(int(n_test))
    ]
    return {
        "key": "linear",
        "weight": weight,
        "calib_activation_list": calib,
        "test_activation_list": tests,
    }


def make_attention_group(
    seed: int,
    dist: str = "normal",
    q_num_heads: int = 4,
    kv_num_heads: int = 2,
    head_dim: int = 64,
    seq_len: int = 16,
    n_calib: int = 3,
    n_test: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build one GQA Attention group in the ``mini_sample`` layout.

    Q/K/V samples are ``{"q": [q_quant, q_scale], "k": ..., "v": ...}`` with
    Q hidden = ``q_num_heads * head_dim`` and K/V hidden =
    ``kv_num_heads * head_dim``. ``head_dim`` must be divisible by 64.
    """
    seed = int(seed)
    qh, kvh, hd = int(q_num_heads), int(kv_num_heads), int(head_dim)
    if hd % HIF4_BLOCK != 0:
        raise ValueError(
            f"head_dim {hd} not divisible by HiF4 block size {HIF4_BLOCK}"
        )
    if qh % kvh != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")

    def sample(tag: int) -> dict[str, list[torch.Tensor]]:
        q = make_nvfp4_pair(
            dist, (int(seq_len), qh * hd), _mix_seed(seed, tag), **kwargs
        )
        k = make_nvfp4_pair(
            dist, (int(seq_len), kvh * hd), _mix_seed(seed, tag + 10), **kwargs
        )
        v = make_nvfp4_pair(
            dist, (int(seq_len), kvh * hd), _mix_seed(seed, tag + 20), **kwargs
        )
        return {"q": q, "k": k, "v": v}

    calib = [sample(300 + i) for i in range(int(n_calib))]
    tests = [sample(400 + i) for i in range(int(n_test))]
    return {
        "key": "attn",
        "attn_type": "gqa",
        "q_num_heads": qh,
        "kv_num_heads": kvh,
        "head_dim": hd,
        "calib": calib,
        "test": tests,
    }


# ---------------------------------------------------------------------------
# Smoke test (no file I/O)
# ---------------------------------------------------------------------------

def _assert_legal_pair(pair: list[torch.Tensor], tag: str) -> None:
    quant, scale = pair
    assert type(quant) is torch.Tensor and type(scale) is torch.Tensor, tag
    assert quant.dtype == torch.bfloat16 and scale.dtype == torch.bfloat16, tag
    channels = int(quant.shape[-1])
    assert channels % NVFP4_BLOCK == 0, tag
    assert tuple(scale.shape) == tuple(quant.shape[:-1]) + (channels // NVFP4_BLOCK,)
    assert all(v in NVFP4_CARRIERS for v in quant.unique().tolist()), tag
    assert torch.isfinite(scale).all(), tag


if __name__ == "__main__":
    for dist in DISTS:
        pair = make_nvfp4_pair(dist, (8, 64), seed=7)
        _assert_legal_pair(pair, f"pair[{dist}]")

    lin = make_linear_group(seed=11, dist="channel_outlier", n_outliers=2)
    _assert_legal_pair(lin["weight"], "linear.weight")
    assert lin["weight"][1].shape == (64, 4)
    for i, act in enumerate(lin["calib_activation_list"]):
        _assert_legal_pair(act, f"linear.calib[{i}]")
    assert len(lin["test_activation_list"]) == 3

    att = make_attention_group(seed=13, dist="mixed_block")
    for i, sample in enumerate(att["calib"]):
        for role in ("q", "k", "v"):
            _assert_legal_pair(sample[role], f"attn.calib[{i}].{role}")
    assert att["test"][0]["q"][1].shape == (16, 16)  # 4 heads * 64 // 16
    assert att["test"][0]["k"][1].shape == (16, 8)  # 2 heads * 64 // 16

    # Determinism: same seed reproduces identical tensors.
    a = make_nvfp4_pair("normal", (4, 64), seed=42)
    b = make_nvfp4_pair("normal", (4, 64), seed=42)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    print("synthetic_data smoke test: PASSED")
    print(f"  carriers: {NVFP4_CARRIERS}")
    print(f"  linear group keys: {sorted(lin)}")
    print(f"  attention group keys: {sorted(att)}")
