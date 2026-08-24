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

## DeepSeek-R1-Distill-Qwen-1.5B

Dataset `834a78afb85e5d5a` was captured from pinned revision
`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` with Transformers 4.57.6 and one
CPU thread. It contains Qwen2 layers 0, 14, and 27 with 12 query heads, 2 KV
heads, and head dimension 128. The same five Linear roles and full sample-length
pattern as the Qwen capture were used.

- Capture: 18 groups, 36m18s, 4.22 GiB peak RSS, no swap.
- Evaluation: 2,160 valid candidate cases, 36m22s, 1.92 GiB peak RSS, no swap.
- Failures or invalid outputs: none.

| Variant | Ceil | Nearest | Stochastic | Pooled |
|---|---:|---:|---:|---:|
| v001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v002 | 9.7527 | 11.0750 | 10.5792 | 10.4690 |
| v003 | 5.7999 | 7.5558 | 6.9990 | 6.7849 |
| v004 | 7.9212 | 9.5718 | 9.0168 | 8.8366 |
| v005 | 8.7742 | 10.0967 | 9.7117 | 9.5275 |
| v006 | 9.7508 | 10.8573 | 10.2235 | 10.2772 |
| v007 | **11.1093** | **12.0375** | **11.6586** | **11.6018** |
| v008 | **11.1093** | **12.0375** | **11.6586** | **11.6018** |

v007 wins all three source modes. v008 is bit-identical because its H64 gate
never fires on this capture. The improvement is mostly Attention: pooled over
the modes, v002 scores 10.48% on Linear and 10.41% on Attention, while v007
scores 10.31% on Linear and 18.05% on Attention. Layer-0 Attention remains the
main negative tail for every variant; v002 reaches -51.99% on one case, and
v007 reduces but does not eliminate that regression.

## SmolLM2-1.7B

Dataset `097430592fe2f4d6` was captured from pinned revision
`31b70e2e869a7173562077fd711b654946d38674` with Transformers 4.57.6 and one
CPU thread. It contains Llama layers 0, 12, and 23 with 32 query heads, 32 KV
heads, and head dimension 64, adding ordinary multi-head Attention coverage.

- Capture: 18 groups, 1h06m35s, 4.49 GiB peak RSS, no swap.
- Evaluation: 2,160 valid candidate cases, 19m02s, 2.14 GiB peak RSS, no swap.
- Evaluation used six CPU threads; failures or invalid outputs: none.

| Variant | Ceil | Nearest | Stochastic | Pooled |
|---|---:|---:|---:|---:|
| v001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v002 | **8.9888** | 10.0799 | 10.3240 | **9.7976** |
| v003 | 5.4887 | 5.6978 | 6.3514 | 5.8460 |
| v004 | 7.8373 | 7.9335 | 8.6729 | 8.1479 |
| v005 | 8.2610 | 8.5177 | 9.1771 | 8.6519 |
| v006 | 8.9287 | 9.4613 | 9.5628 | 9.3176 |
| v007 | -1.9670 | **11.5605** | **11.8617** | 7.1517 |
| v008 | -1.9670 | **11.5605** | **11.8617** | 7.1517 |

The ceil result exposes a severe role-specific failure hidden by the other
modes: layer-23 `o_proj` under v007/v008 is negative on all five cases and
reaches -951.48% on the shortest sample. This single group-mode drives the
overall ceil result below zero. v002 is less aggressive and wins pooled on this
model, while v007/v008 win nearest and stochastic. As on DeepSeek, v008 is
bit-identical to v007 because the H64 gate never fires.

## Cross-model comparison

All three datasets contain the same number of groups and cases, so the following
means weight each captured model equally. Qwen and DeepSeek were evaluated with
one thread; SmolLM2 used six. A fixed six-thread run is expected to be
repeatable, but parallel reduction order can differ from one-thread arithmetic,
so thread count remains part of each run's provenance.

| Variant | Ceil | Nearest | Stochastic | Pooled |
|---|---:|---:|---:|---:|
| v001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v002 | **10.7982** | 11.3560 | 10.9669 | **11.0403** |
| v003 | 6.3316 | 6.8054 | 6.7952 | 6.6441 |
| v004 | 8.8579 | 9.1217 | 9.1933 | 9.0576 |
| v005 | 9.4045 | 9.6239 | 9.7358 | 9.5881 |
| v006 | 10.2898 | 10.5323 | 10.2862 | 10.3694 |
| v007 | 7.4541 | 11.7362 | 11.5783 | 10.2562 |
| v008 | 7.4310 | **11.8565** | **11.5882** | 10.2919 |

The pooled order is v002, v006, v008, v007, v005, v004, v003, v001. That does
not identify an official winner because the source-scale recipe and organizer
baseline are unknown, but it changes the interpretation of the Qwen-only run:

- v002 is the most robust Linear policy in this three-model sample and wins the
  pooled mean despite losing to v007/v008 on DeepSeek.
- v007/v008 are consistently strongest on Attention, with a cross-model pooled
  Attention improvement of 16.27%, versus 14.62% for v006 and 9.93% for v002.
- The wide activation search in v007 can produce catastrophic architecture- and
  mode-specific Linear tails. SmolLM2 layer-23 `o_proj` is the current clearest
  counterexample.
- v008's H64 gate changes only 25 of 810 records across the three captures, all
  on Qwen layer 27. It never fires on DeepSeek or SmolLM2, so its measured gain
  over v007 is narrow and Qwen-specific.
- v001 remains exactly v000 on every locally generated source mode and model.

These captures contain only three layers and five test prompts per model.
Individual tails are actionable counterexamples, while small differences in
global means should not be treated as high-confidence estimates of hidden-test
ranking.

## Qwen3.5-2B

Dataset `4405c7fadc8bef16` was captured from pinned revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` with Transformers 5.2.0 and six
CPU threads. Qwen3.5 is a hybrid architecture: only layers 3, 7, 11, 15, 19,
and 23 use ordinary softmax Attention. The capture samples full-attention layers
3, 11, and 23, with 8 query heads, 2 KV heads, and head dimension 256. Its 15
Linear and 3 post-RoPE Attention groups use the same full sample-length pattern
as the earlier captures.

- Capture: 18 groups, 10m38s, 5.19 GiB peak RSS, no swap.
- Evaluation: 2,160 valid candidate cases, 14m45s, 1.83 GiB peak RSS, no swap.
- Evaluation used six CPU threads; failures or invalid outputs: none.

| Variant | Ceil | Nearest | Stochastic | Pooled |
|---|---:|---:|---:|---:|
| v001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v002 | **14.3068** | **15.1556** | **15.2742** | **14.9122** |
| v003 | 8.4980 | 7.7710 | 7.8144 | 8.0278 |
| v004 | 11.3696 | 10.5763 | 10.4725 | 10.8061 |
| v005 | 12.1138 | 11.2166 | 11.1679 | 11.4994 |
| v006 | 12.5854 | 11.6798 | 11.6296 | 11.9649 |
| v007 | 13.9685 | 13.7080 | 13.0060 | 13.5609 |
| v008 | 13.9685 | 13.7080 | 13.0060 | 13.5609 |

v002 wins all three source modes. Pooled over modes, it scores 14.21% on
Linear and 18.44% on Attention. v007/v008 are weaker on Linear at 10.67% but
substantially stronger on Attention at 28.01%. Their Attention improvement
increases with depth: 23.29% at layer 3, 26.10% at layer 11, and 34.65% at
layer 23.

v008 is bit-identical to v007 on all 270 Qwen3.5 records because its H64 gate
never fires. The largest negative tail is Attention under stochastic source
generation: v002-v005 reach -41.72% on one layer-11 case; v007/v008 have only
two negative cases, -18.51% at layer 3 and -14.58% at layer 11. All values are
finite and legal.

## Four-model comparison

The following table adds Qwen3.5 to the equal-case macro average. Qwen3 and
DeepSeek used one evaluation thread; SmolLM2 and Qwen3.5 used six. The latter
also uses Transformers 5.2.0 rather than 4.57.6, so these settings remain part
of each run's provenance.

| Variant | Ceil | Nearest | Stochastic | Pooled |
|---|---:|---:|---:|---:|
| v001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v002 | **11.6753** | 12.3059 | 12.0437 | **12.0083** |
| v003 | 6.8732 | 7.0468 | 7.0500 | 6.9900 |
| v004 | 9.4858 | 9.4853 | 9.5131 | 9.4947 |
| v005 | 10.0818 | 10.0221 | 10.0939 | 10.0659 |
| v006 | 10.8637 | 10.8192 | 10.6221 | 10.7683 |
| v007 | 9.0827 | 12.2292 | 11.9352 | 11.0824 |
| v008 | 9.0654 | **12.3194** | **11.9427** | 11.1092 |

The pooled order is now v002, v008, v007, v006, v005, v004, v003, v001.
Across four models, v002 remains the strongest Linear policy at 12.00% pooled,
while v007/v008 remain the strongest Attention policy at 19.20%. v008 differs
from v007 in only 25 of 1,080 records, all from Qwen3-0.6B layer 27; its gate
never fires on DeepSeek, SmolLM2, or Qwen3.5.
