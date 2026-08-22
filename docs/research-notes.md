# Research Notes

This file separates proved structure from empirical guidance. Detailed experiment
results belong in `benchmarks/records/`; this file records conclusions that should
survive individual variants.

## Exact HiF4 block facts

- A 64-value block has shape `(8, 2, 4)`. For a fixed legal E6M2 scale factor
  and a separable weighted squared-error objective, the best level-2, level-3,
  sign, and mantissa values are found exactly by independent finite choices.
- For each 8-value group, evaluate both level-2 values. For each of its two
  4-value children, evaluate both level-3 values and round each magnitude to the
  nearest mantissa in `{0, 0.25, ..., 1.75}`. Select the lowest summed cost.
- There are 255 legal E6M2 scale factors in the checker range. Enumerating all of
  them with the exact hierarchy optimization is globally exact for separable
  weighted SSE, but too expensive for the full workload without pruning.
- Cost as a function of scale is piecewise quadratic and can be non-unimodal.
  Therefore greedy scale movement has no general optimality guarantee.
- `ceil_E6M2(max_abs / 7)` is only a range-preserving heuristic. A better scale
  may be smaller (intentional clipping) or larger (better hierarchy alignment).

Empirical public-sample measurements found roughly 15-17% lower raw tensor MSE
from broader scale search. This is not the contest output metric, but it makes a
small legal-scale neighborhood search the safest first algorithmic variant.

## Linear output error

For `Y = X W^T`, weight-only error is exactly

```text
mean_tokens ||X delta_w||^2 = delta_w^T (X^T X / T) delta_w
```

for each weight row. Consequences:

- Full activation second moment is the exact weight-only objective.
- Per-channel diagonal weighting is exact for every error vector only when the
  activation second-moment matrix is diagonal. It remains a useful inexpensive
  proxy, not a theorem about the true objective.
- Independent 64-block optimization is exact only when the second-moment matrix
  is block diagonal in the same partition.
- Quantizing both activation and weight introduces an interaction term and cross
  terms. Separate operand MSEs do not add to exact output MSE in general.

Any invertible contraction-channel transform preserves the unquantized product:

```text
X' = X P
W' = W P^(-T)
X' W'^T = X W^T
```

Diagonal SmoothQuant scaling, matched permutations, and orthogonal rotations are
special cases. They preserve the ideal product, not the error produced by a
particular quantizer. Their value must be measured after quantization.

The public Linear sample has a stable, extreme activation-channel energy profile:
one channel holds about 46% of activation energy and the top 32 hold about 78%.
Calibration/test channel-power correlation is about 0.99997. This supports testing
calibration weighting and conservative paired channel balancing, while synthetic
shift cases are needed to guard against overfitting this single group.

## Attention invariances

For ordinary row-wise softmax attention, paired per-head transforms preserve
logits exactly:

```text
Q' = Q B
K' = K B^(-T)
```

The same transform must be used by every Q head sharing a KV head. A common
transform across all heads is safe even if the hidden GQA pairing convention is
not known. Diagonal reciprocal scaling and matched coordinate permutations are
low-cost special cases.

Adding the same vector to every key position in one KV head changes each query's
logits by a row-constant value, so softmax probabilities and outputs are unchanged.
This remains true with ordinary causal or padding masks. Large translations can
still cause floating-point cancellation or overflow in a non-stabilized softmax.

V has no analogous generic transform because attention outputs are weighted sums
of V directly. Q/K transforms must be scored jointly through Attention; V should
default to reconstruction-focused scale search.

The public Attention sample has stable but nonuniform per-head-dimension energy.
Calibration/test power correlations are approximately 0.9996 for Q, 0.9995 for K,
and 0.996 for V. This motivates reciprocal Q/K balancing after the safer local
scale-search variant is established.

## Experiment order

1. Local E6M2 neighborhood search with exact fixed-scale hierarchy selection.
2. Calibration-derived diagonal weighting for Linear and counterpart-energy
   weighting for Q/K.
3. Conservative paired diagonal transforms for Linear and Q/K.
4. Key centering and matched permutations, gated by broad synthetic results.
5. Full covariance, rotations, or evolutionary policy search only if simpler
   methods leave runtime and quality headroom.

The unknown official standard baseline prevents local reproduction of the contest
score. Local numbers compare tagged variants against `solution/v000-baseline` and
must always be reported as local mean per-case improvement.
