# Linear Experiment Ideas

These are discussion hypotheses for `Y = X W^T`, where:

```text
X: [m, k]       activations for m tokens
W: [n, k]       n output-channel weight rows
Y: [m, n]
```

They require evaluation on realistic data before promotion.

## 1. Sort or permute contraction channels

### Proposal

Choose a permutation `P: [k, k]` that groups weight channels by magnitude or
outlier statistics. Apply the same channel permutation to activations:

```text
X' = X P        [m, k]
W' = W P        [n, k]
X' W'^T = X P P^T W^T = X W^T
```

With the contest's row-major activation convention this permutes columns of both
`X` and `W`. It would permute rows only if activations were represented as
`[k, m]` instead.

The goal is to control which source channels share each 64-value HiF4 block,
placing compatible scales together and possibly isolating outlier channels.

### Existing related evidence

Matched permutations of complete NVFP4 16-value blocks were already tested. The
best tested policy improved public cases but slightly regressed a broad synthetic
pool, especially channel-outlier cases. Permutations within an existing 64-value
block were effectively inert because each source 16-block aligns with two
independent HiF4 level-2 groups.

The new sorting proposal may differ in its statistic and grouping policy, but it
belongs to the same transformation family and must start from the existing
permutation artifacts rather than duplicate the experiment.

### Questions

- Sort by per-column maximum, RMS, kurtosis, calibration importance, or a joint
  weight/activation statistic?
- Sort individual channels or preserve complete NVFP4 16-value blocks?
- Does breaking source 16-block adjacency discard useful scale structure?
- Should large channels be clustered together or distributed between HiF4
  blocks?
- Can one global permutation serve all weight rows, given that row-wise outlier
  locations differ?
- Does the cost of permuting every online activation fit the runtime limit?

## 2. Error-feedback or compensated rounding

### Proposal

When quantizing a weight row, keep an accumulated signed residual. If previous
values were mostly rounded in one direction, bias a later ambiguous value toward
the opposite direction. Weight residuals by calibration activation importance.

This resembles error diffusion or noise-shaping:

```text
residual <- residual + (original - quantized)
use residual when choosing a later legal quantized value
```

### Mathematical warning

For one row `w: [k]`, error `delta_w = w_hat - w: [k]`, and calibration matrix
`X_cal: [m_cal, k]`, the exact weight-only output error is:

```text
(1 / m_cal) ||X_cal delta_w||_2^2
= delta_w^T H delta_w
H = X_cal^T X_cal / m_cal: [k, k]
```

Making `sum(delta_w)` close to zero minimizes only the projection onto an
all-ones activation direction. It does not generally make `X_cal delta_w` small.
The stronger version of the idea should propagate residual in activation-output
space, or use covariance-aware compensation as in OBQ/GPTQ, rather than merely
cancel scalar weight errors.

HiF4 values within a block are coupled through the base and hierarchical scales,
so value-by-value error feedback cannot be added without specifying when scales
are fixed and which neighboring mantissa choices remain legal.

### Investigations

- Compare scalar residual cancellation with calibration-weighted cancellation.
- Test a low-rank sketch of `X_cal` as a cheaper residual space.
- Process channels in importance order rather than storage order.
- Constrain choices to the two nearest legal mantissas after scales are fixed.
- Measure whether cancellation survives simultaneous activation quantization.

## 3. Broader scale brute force

### Proposal

For a fixed legal E6M2 `scale_factor`, the best level-2, level-3, signs, and
mantissas under separable weighted SSE are already selected exactly by a small
finite search. Broaden the search over the main E6M2 scale while pruning scales
that are obviously too small or too large.

There are 255 checker-legal E6M2 scales in total, but only a local magnitude range
is relevant for a given block. Current dynamic activation search tests offsets:

```text
0, -1, +1, +2, +3
```

around `ceil_E6M2(max_abs / 7)`. Current weight conversion deliberately uses only
offset `0` because neighboring raw weighted-SSE choices regressed the public
Linear output.

### Pruning and sampling distinctions

- Scale-range pruning can safely eliminate candidates using representable range
  and lower bounds, if those bounds are proved for the chosen objective.
- Subsampling values inside one 64-value block is unlikely to save much because
  the block contains only 64 values and can change the selected scale incorrectly.
- Sampling weight rows or blocks can cheaply learn a global offset policy, but it
  cannot identify the best scale for every unsampled block.
- Sampling calibration tokens can reduce the cost of an output-aware objective
  because calibration may contain many rows.
- Random subsets must use fixed seeds for comparison, even if the submitted
  algorithm itself could use a fixed-seed stochastic search.

### Investigation required

Determine why neighboring-scale weight search improved synthetic Linear but
regressed the public group. Record, by role and dataset:

- Frequency of each selected scale offset.
- Raw tensor MSE and weighted tensor SSE changes.
- Weight-only output MSE with exact activations.
- Activation-only output MSE with exact weights.
- Joint output MSE with both operands quantized.
- Clipping frequency and selected level-2/level-3 patterns.
- Results by weight row, channel importance, and block magnitude profile.

This diagnosis should precede another broad weight search. Candidate explanations
include calibration covariance ignored by diagonal weights, interaction between
weight and activation errors, correlated rounding errors, and distribution shift
between synthetic and public tensors.

## 4. Runtime constraints

The five-minute limit covers approximately 50 weight calibrations and 250 dynamic
activation conversions, plus the Attention APIs. Public weights can contain more
than 16 million values each. A method that performs many full passes over every
weight block can consume the budget even if one small benchmark appears fast.

Every search proposal should report complexity in passes over tensor elements and
measure a public-shape 50-group projection. Calibration can be more expensive than
one dynamic call because it runs once per group, but it is not outside the stated
five-minute budget.
