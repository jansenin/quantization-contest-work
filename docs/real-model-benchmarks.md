# Real-model benchmarks

This report records results from raw BF16 model captures. The evaluator derives
contest-format NVFP4 inputs at evaluation time, then compares each tagged
variant with `solution/v000-baseline`. Percentages are means of per-case
`100 * (MSE_v000 - MSE_variant) / MSE_v000`; they are not official scores
because the organizer's standard conversion is unavailable.

## Qwen3-0.6B

Dataset `b7925ee95f17f32b` was captured from pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca` with Transformers 4.57.6 and one
CPU thread. It contains layers 0, 14, and 27: five Linear roles per layer and
one post-RoPE Attention group per layer. Each group has five calibration and
five test samples with lengths `10,128,512,1024,1024` in each split.

- Raw capture size: 458 MB.
- Capture (`/usr/bin/time -v`): 18m25s, 2.43 GiB peak RSS, no swap.
- Evaluation (`/usr/bin/time -v`): 18 groups, 3 source modes, 8 candidates,
  2,160 valid candidate cases, 13m03s, 1.16 GiB peak RSS.
- Failures or invalid outputs: none.

Reproduce the evaluation with:

```bash
python3 tools/evaluate_real.py \
  --dataset b7925ee95f17f32b \
  --baseline solution/v000-baseline \
  --candidate solution/v001-bf16-target \
  --candidate solution/v002-e6m2-neighbors \
  --candidate solution/v003-role-gated \
  --candidate solution/v004-calibration-weighted \
  --candidate solution/v005-activation-weighted \
  --candidate solution/v006-qk-weighted \
  --candidate solution/v007-akv-wide-search \
  --candidate solution/v008-gated-hadamard \
  --modes ceil,nearest,stochastic --threads 1
```

### Results by source mode

Each cell is based on 90 cases: 75 Linear and 15 Attention.

| Variant | Ceil | Nearest | Stochastic | Pooled |
|---|---:|---:|---:|---:|
| v001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v002 | **13.6530** | **12.9129** | **11.9974** | **12.8544** |
| v003 | 7.7060 | 7.1625 | 7.0354 | 7.3013 |
| v004 | 10.8153 | 9.8599 | 9.8901 | 10.1884 |
| v005 | 11.1783 | 10.2573 | 10.3187 | 10.5848 |
| v006 | 12.1898 | 11.2785 | 11.0724 | 11.5136 |
| v007 | 13.2201 | 11.6108 | 11.2147 | 12.0152 |
| v008 | 13.1508 | 11.9717 | 11.2443 | 12.1223 |

v002 wins all three source modes on this model. This is materially different
from the single public Linear group, where v002's weight-scale search regressed
badly. The result is evidence that the public group is not representative
enough to select that policy, not proof that v002 generalizes to hidden data.

v001 is exactly v000 here because the locally generated E2M1-carrier times
E4M3-scale products are already exactly representable in BF16; changing the
internal target from FP32 to BF16 therefore changes nothing.

### Linear and Attention

| Variant | Mode | Linear | Attention |
|---|---|---:|---:|
| v002 | ceil | 14.0877 | 11.4796 |
| v002 | nearest | 13.7976 | 8.4896 |
| v002 | stochastic | 13.2612 | 5.6782 |
| v006 | ceil | 11.1181 | 17.5485 |
| v006 | nearest | 10.6108 | 14.6169 |
| v006 | stochastic | 11.2468 | 10.2002 |
| v007 | ceil | 12.2823 | 17.9091 |
| v007 | nearest | 11.4512 | 12.4088 |
| v007 | stochastic | 11.5399 | 9.5888 |
| v008 | ceil | 12.1991 | 17.9091 |
| v008 | nearest | 11.8843 | 12.4088 |
| v008 | stochastic | 11.5754 | 9.5888 |

The source-scale mode affects Attention more strongly than Linear in this
capture. The wider Q/K/V scale searches help most under ceil and progressively
less under nearest and stochastic source generation.

### v008 by role

Values are pooled across the three selected layers.

| Role | Ceil | Nearest | Stochastic | Pooled |
|---|---:|---:|---:|---:|
| q_proj | 12.1250 | 13.7956 | 13.1043 | 13.0083 |
| gate_proj | 14.0343 | 11.9754 | 11.4538 | 12.4878 |
| up_proj | 8.0108 | 10.8083 | 9.0032 | 9.2741 |
| down_proj | 18.0732 | 13.3196 | 14.4198 | 15.2709 |
| o_proj | 8.7521 | 9.5223 | 9.8961 | 9.3902 |
| Attention | 17.9091 | 12.4088 | 9.5888 | 13.3022 |

Layer 27 is the strongest contributor. Its `down_proj` averages 29.49% pooled
and reaches 37.28% in ceil mode.

### v008 gate behavior

Only 25 of 270 v008 records differ from v007, all at layer 27:

- `gate_proj` changes in all modes and improves substantially.
- `up_proj` changes in ceil and stochastic modes and regresses substantially;
  nearest remains identical to v007.
- Every other group is identical between v007 and v008.

The net v008-minus-v007 change is -0.069 percentage points in ceil, +0.361 in
nearest, and +0.030 in stochastic. The current gate therefore does not reliably
distinguish a rotation-friendly `gate_proj` from a rotation-hostile `up_proj`
that receives the same input activation. The new negative tail introduced by
v008 is entirely from layer-27 `up_proj`.

These findings should be combined with additional architectures before changing
the submission candidate. One small Qwen model is realistic evidence, but it is
still one model family and only three sampled layers.
