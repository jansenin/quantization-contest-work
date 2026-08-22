# Solution Variants

All percentages are local arithmetic means over the named deterministic suite,
relative to `solution/v000-baseline`. They are not official contest scores.

| Tag | Method | Suite | Cases | Linear | Attention | Overall | API time | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| `solution/v000-baseline` | `ceil(max/7)` E6M2 scale, exact local hierarchy selection, no calibration | `synthetic-v1-b2af837a7a44` | 40 | 0.00% | 0.00% | 0.00% | 0.047 s | 22/22 legal |
| `solution/v001-bf16-target` | Baseline with BF16-rounded NVFP4 target | `synthetic-v1-b2af837a7a44` | 40 | -2.51% | -0.38% | -0.91% | 0.045 s | rejected; 22/22 legal |
| `solution/v002-e6m2-neighbors` | Search anchor and adjacent E6M2 ticks for every tensor | `synthetic-v1-b2af837a7a44` | 40 | +12.93% | +8.40% | +9.53% | 0.144 s | experimental; public-only Linear -23.55%; 22/22 legal |

The extended v002 record includes the ten public cases. Public-only results are
-23.55% Linear and +19.41% Attention (-2.07% mean), so v002 is not the default
despite its broad synthetic improvement.
