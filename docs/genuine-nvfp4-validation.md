# Genuine ModelOpt NVFP4 Validation

This report validates the packed-weight interpretation used by genuine ModelOpt
NVFP4 checkpoints and the conversion from that storage format to the contest's
decoded-carrier plus per-16-scale API. It does not imply that the organizer used
ModelOpt to generate hidden inputs.

## Storage Contract

Authoritative ModelOpt 0.33/0.46 source and the vLLM ModelOpt loader agree on
the following layout:

- Packed weight: U8 `[N, K/2]`.
- Low nibble: even K index; high nibble: odd K index.
- Nibble lookup: `[0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]`.
- `weight_scale`: E4M3FN `[N, K/16]`, one scale per 16 logical K values.
- `weight_scale_2`: scalar F32 global factor.
- Dequantization: `E2M1(nibble) * weight_scale * weight_scale_2`.

The corresponding contest pair is:

```text
carrier = BF16(E2M1(unpack(weight)))
scale   = BF16(weight_scale * weight_scale_2)
```

Nibble 8 is sign-encoded zero. The validator follows ModelOpt's software table
and emits positive BF16 zero; retaining a negative-zero sign bit would be
numerically immaterial for the contest reference multiplication.

## Qwen Validation

The decisive comparison used:

- NVFP4 checkpoint: `NVFP4/Qwen3-0.6B-FP4`, revision
  `c035dfb93aaac621bc73a473d10d773526a031e8`, self-reported producer ModelOpt
  0.33.0.
- NVFP4 safetensors SHA-256:
  `f93cda479e431262458e2ba04cc4259be6b6caadbb53a2f2b0813c045e8e8d55`.
- BF16 parent: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- BF16 safetensors SHA-256:
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

Inventory: 954 tensors total, including 196 packed U8 weights, 196 E4M3 block
scales, and 196 F32 global factors. Every one of the 196 packed weights was
compared against its matching BF16 parent tensor.

| Interpretation | Mean normalized MSE | Mean correlation |
|---|---:|---:|
| Canonical low-even nibble + global factor | 0.009003 | 0.995498 |
| Swapped nibble order | 1.999817 | -0.000064 |
| Canonical nibbles, global factor omitted | 39,835,963 | 0.995500 |

All 196 tensors pass the validator's absolute numerical-fit gate: canonical
normalized MSE at most 0.02 and correlation at least 0.99. Per-tensor canonical
normalized MSE lies in `[0.008869, 0.009118]`; swapped order is approximately
uncorrelated. This resolves the earlier naive-decoder anomaly: raw bytes are not
decoded carriers, each byte contains two K-adjacent nibbles, and the global
factor is essential.

## Contest Fold

Folding the two ModelOpt scales into one BF16 contest scale is close but not
bit-identical to ModelOpt's separate-factor BF16 result. Across all 440,401,920
decoded Qwen weight values:

- Exact BF16 agreement: 87.6824%.
- Mean absolute difference: `2.066e-5`.
- Maximum absolute difference: `0.0078125`.

The discrepancy is expected double-rounding: the contest representation first
rounds `weight_scale * weight_scale_2` to BF16, then multiplies by the BF16
carrier. ModelOpt retains the normalized E4M3 scale and F32 global factor until
dequantization. Therefore a genuine checkpoint can be converted to a legal
contest pair, but the fold is not an exact representation of every ModelOpt
dequantized value.

## Nemotron Validation

The independent NVIDIA checkpoint
`nvidia/Nemotron-3-Embed-1B-NVFP4`, revision
`4138e5572e8d49b7e69d7ed1257506571ab145fc`, self-reports ModelOpt 0.45.0.
Its safetensors SHA-256 is
`f2753954c89055eb679a45b7dfea27a3e05c04ecbdb1f4e6c086180fe8c32bc7`.

Its inventory independently contains 112 packed U8 weights, 112 E4M3 block
scales, and 112 scalar F32 global factors. All 112 groups have the expected
shapes and can be folded into contest pairs. No matching BF16 parent was
available locally, so this is only a structural consistency check; U8 carrier
legality and E4M3 dtype membership alone do not establish provenance or nibble
order.

## Reproduction

After downloading the `genuine` profile, run:

```bash
.venv-data/bin/python tools/validate_genuine_nvfp4.py \
  --nvfp4-snapshot data/huggingface-cache/models--NVFP4--Qwen3-0.6B-FP4/snapshots/<revision> \
  --bf16-snapshot data/huggingface-cache/models--Qwen--Qwen3-0.6B/snapshots/<revision> \
  --tensor "*.weight" \
  --json-output /tmp/qwen-genuine.json \
  --markdown-output /tmp/qwen-genuine.md
```

The validator reads safetensors in row chunks, supports indexed shards, records
file SHA-256 hashes, and never loads a complete model. Producer metadata is
self-reported by checkpoint configuration; the parent comparison is the
independent numerical evidence.
