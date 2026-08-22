# Solution Variants

All percentages are local arithmetic means over the named deterministic suite,
relative to `solution/v000-baseline`. They are not official contest scores.

| Tag | Method | Suite | Cases | Linear | Attention | Overall | API time | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| `solution/v000-baseline` | `ceil(max/7)` E6M2 scale, exact local hierarchy selection, no calibration | `synthetic-v1-b2af837a7a44` | 40 | 0.00% | 0.00% | 0.00% | 0.047 s | 22/22 legal |
| `solution/v001-bf16-target` | Baseline with BF16-rounded NVFP4 target | `synthetic-v1-b2af837a7a44` | 40 | -2.51% | -0.38% | -0.91% | 0.045 s | rejected; 22/22 legal |
