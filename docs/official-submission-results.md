# Official submission measurements

These are user-reported measurements from the contest platform. They are kept
separate from local synthetic and real-model benchmarks because the organizer's
standard conversion and hidden cases are unavailable locally.

## Variant results

| Variant | Score | Runtime |
|---|---:|---:|
| `solution/v000-baseline` | 1240 | 105 s, 92 s |
| `solution/v001-bf16-target` | 1240 | 111 s |
| `solution/v002-e6m2-neighbors` | 5025 | 113 s |
| `solution/v003-role-gated` | 3600 | 111 s |
| `solution/v004-calibration-weighted` | 4340 | 127 s |
| `solution/v005-activation-weighted` | 4600 | 111 s |
| `solution/v006-qk-weighted` | 4750 | 123 s |
| `solution/v007-akv-wide-search` | 5276 | 98 s |
| `solution/v008-gated-hadamard` | **5326** | 120 s |

The two baseline runs show at least 13 seconds of platform-level runtime
variation. The reported rough range was 92-120 seconds, although the explicit
v004 measurement is 127 seconds; the table preserves the individual values.
All variants are comfortably below the stated five-minute limit.

## Interpretation

- v001 exactly matches v000 in score, consistent with the BF16 target being a
  no-op when decoded NVFP4 carrier-scale products are already BF16-exact.
- v008 is the measured quality leader, 50 score points above v007. This is a
  small margin compared with their approximately 4,000-point gain over v000.
- Runtime does not increase monotonically with algorithmic complexity. The
  common tensor traversal, hierarchy construction, and memory traffic likely
  dominate, while the added scale candidates are vectorized. The v008 rotation
  is gated and therefore does not run on every Linear group.
- Runtime differences this small should not be attributed to one algorithm
  without repeated measurements, as demonstrated by the two baseline runs.

These measurements do not expose per-case, Linear/Attention, or hidden-group
breakdowns, so they should be combined with the disaggregated local evidence in
`docs/real-model-benchmarks.md` when selecting experiments.
