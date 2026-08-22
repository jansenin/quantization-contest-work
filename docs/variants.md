# Solution Variants

All percentages are local arithmetic means over the named deterministic suite,
relative to `solution/v000-baseline`. They are not official contest scores.

| Tag | Method | Suite | Cases | Linear | Attention | Overall | API time | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| `solution/v000-baseline` | `ceil(max/7)` E6M2 scale, exact local hierarchy selection, no calibration | `synthetic-v1-b2af837a7a44` | 40 | 0.00% | 0.00% | 0.00% | 0.047 s | 22/22 legal |
| `solution/v001-bf16-target` | Baseline with BF16-rounded NVFP4 target | `synthetic-v1-b2af837a7a44` | 40 | -2.51% | -0.38% | -0.91% | 0.045 s | rejected; 22/22 legal |
| `solution/v002-e6m2-neighbors` | Search anchor and adjacent E6M2 ticks for every tensor | `synthetic-v1-b2af837a7a44` | 40 | +12.93% | +8.40% | +9.53% | 0.144 s | experimental; public-only Linear -23.55%; 22/22 legal |
| `solution/v003-role-gated` | Baseline weight; adjacent-scale search for activation/Q/K/V | `synthetic-v1-b2af837a7a44` | 40 | +10.20% | +8.40% | +8.85% | 0.146 s | superseded; 22/22 legal |
| `solution/v004-calibration-weighted` | v003 plus fourth-root calibration-energy weighting for the fixed weight hierarchy | `synthetic-v1-b2af837a7a44` | 40 | +10.71% | +8.40% | +8.98% | 0.153 s | superseded; 22/22 legal |
| `solution/v005-activation-weighted` | v004 plus weight-column-energy weighting for dynamic activation hierarchy selection | `synthetic-v1-b2af837a7a44` | 40 | +12.92% | +8.40% | +9.53% | 0.144 s | superseded; 22/22 legal |
| `solution/v006-qk-weighted` | v005 plus counterpart-energy weighting for Q/K hierarchy selection | `synthetic-v1-b2af837a7a44` | 40 | +12.92% | +15.19% | +14.62% | 0.145 s | runtime-safe fallback; 22/22 legal |
| `solution/v007-akv-wide-search` | v006 plus E6M2 offsets `+2,+3` for activation/K/V, retaining adjacent Q search | `synthetic-v1-b2af837a7a44` | 40 | +14.97% | +17.16% | +16.61% | 0.209 s | current quality leader; 22/22 legal |

The extended v002 record includes the ten public cases. Public-only results are
-23.55% Linear and +19.41% Attention (-2.07% mean), so v002 is not the default
despite its broad synthetic improvement.

The v003 public-only split is -0.13% Linear and +19.41% Attention (+9.64% mean).
Dropping weight search removes v002's large public Linear regression and reduces
the extended-suite candidate API time from about 3.8 s to 2.3 s.

The v004 public-only split is +11.99% Linear and +19.41% Attention (+15.70%
mean). Its conservative weighting also improves canonical synthetic Linear by
0.51 percentage points over v003. More aggressive calibration exponents gained
more on the single public group but regressed heavy-tail and sparse synthetic
cases, so they were not promoted.

The v005 public-only split is +13.52% Linear and +19.41% Attention (+16.46%
mean). A broader synthetic run also improved overall Linear quality; the main
residual warning was a small regression on normally distributed Linear cases.

The v006 public-only split is +13.52% Linear and +20.13% Attention (+16.83%
mean). A separate 90-case synthetic Attention run improved by 4.68 percentage
points over v004, although mixed-block Attention remained the weakest family.

The v007 public-only split is +19.23% Linear and +22.90% Attention (+21.07%
mean). Across three broader synthetic seeds it improved over v006 by 2.65
percentage points on average. Widening Q as well gave negligible quality gain but
reduced the projected five-minute runtime margin; v007 therefore widens only
activation, K, and V. A public-shape 50+50 projection used about 81% of the local
budget, so v006 remains the conservative runtime fallback for a slower judge CPU.
