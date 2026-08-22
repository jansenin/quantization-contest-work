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

Only larger scales won when the local search was extended beyond adjacent E6M2
ticks in the tested suites: adding offset `-2` was bitwise inert, while offsets
`+2` and `+3` were selected by roughly 8-16% of blocks. Offsets beyond `+3` were
also inert. Widening activation, K, and V while retaining adjacent Q search gave
nearly the same quality as widening every dynamic role, with materially lower
runtime. This is empirical pruning, not a guarantee for arbitrary tensors.

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

The conservative calibration experiment weights fixed-scale hierarchy decisions
by the fourth root of mean-normalized channel second moments, clamped to
`[0.1, 10]`. It improved canonical synthetic Linear by 0.51 percentage points and
the five public Linear cases by 12.11 points relative to the unweighted v003
variant. Larger exponents favored the public outlier profile more strongly but
lost quality on synthetic heavy-tail and sparse cases.

Dynamic activation hierarchy decisions can use the fixed weight's column energy
as a diagonal output-error proxy. A square-root mapping clamped to `[0.1, 10]`
improved canonical Linear by 2.21 percentage points and the five public Linear
cases by 1.53 points over v004. The calibration state is only one FP32 value per
input channel, and no value transform is applied.

Paired SmoothQuant-style Linear scaling produced large gains on the single public
group but regressed broader multi-seed synthetic Linear suites, especially sparse
and channel-outlier cases. The exact product invariance alone therefore does not
justify enabling a transform; v005 retains the original values.

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

Reciprocal RMS balancing improved the public Attention group but catastrophically
regressed synthetic mixed-block Attention, so it was rejected. A safer use of the
same calibration information is to leave Q/K unchanged and weight hierarchy costs:
Q errors by mapped K energy and K errors by pooled Q energy. A fourth-root mapping
clamped to `[0.25, 4]` improved canonical Attention by 6.79 percentage points and
a separate 90-case synthetic Attention suite by 4.68 points over v004.

Dynamic key centering is exactly invariant before quantization, but mean, median,
midpoint, and blended centers all lost to no centering over a 450-case synthetic
Attention study. Centering enlarged residual block ranges and produced severe
mixed-block outliers despite improving the single public group, so it is disabled.
Block-local and narrower Q/K importance mappings were also seed-unstable and did
not safely improve on the global fourth-root weighting.

## Literature map

The closest format reference is Luo et al., [HiFloat4 Format for Language Model
Inference](https://arxiv.org/abs/2602.11287). Its direct-cast algorithm uses the
same 64-value E6M2/E1_8/E1_16/S1P2 hierarchy and a `max_abs / 7` scale anchor.
Its HiGPTQ results also confirm that output-aware reconstruction is an intended
way to improve over direct casting, rather than merely reducing tensor MSE.

The most relevant method families are:

- GPTQ/OBQ ([GPTQ](https://arxiv.org/abs/2210.17323)) make the full activation
  covariance the Linear weight-reconstruction objective. The exact objective is
  applicable here, but a full inverse or sequential compensation is too costly
  and prone to calibration overfit for the small per-group samples.
- [AWQ](https://arxiv.org/abs/2306.00978) and
  [SmoothQuant](https://arxiv.org/abs/2211.10438) motivate activation-aware
  channel scaling. Their exact paired transform is representable through the
  calibration state, but our multi-seed experiments rejected it because public
  gains did not survive sparse and mixed synthetic distributions.
- [QuaRot](https://arxiv.org/abs/2404.00456),
  [DuQuant](https://arxiv.org/abs/2406.01721), and NVIDIA's
  [NVFP4 pretraining study](https://arxiv.org/abs/2509.25149) motivate matched
  permutations and block-local Hadamard rotations. These transforms preserve the
  ideal contraction exactly when applied to both operands. Whether they improve
  this particular hierarchy after quantization is still an empirical question.
- KIVI ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) and
  SageAttention ([arXiv:2410.02367](https://arxiv.org/abs/2410.02367)) support
  treating Q, K, and V differently and optimizing Q/K for logit sensitivity.
  This is consistent with v006's counterpart-energy weighting and v007's
  role-specific scale-search widths.
- [SOAR](https://arxiv.org/abs/2605.12245) derives closed-form scale updates for
  fixed FP4 codes. For fixed HiF4 hierarchy coefficients the corresponding exact
  update is `s* = sum(w*x*c) / sum(w*c^2)`, followed by E6M2 floor/ceil testing.

SOAR-style coordinate refinement was tested from v007 and rejected. The fitted
`s*` stayed within about 5% of the anchor while adjacent E6M2 values are 14-25%
apart. It changed no synthetic scales and only about 0.02% of public activation
scales; appending it to v007 was bit-for-bit inert. Replacing discrete search by
the update dropped canonical improvement from 16.61% to about 1.11%. The scale
cost is non-unimodal because moving the scale can change the hierarchy, so the
closed-form fixed-code optimum cannot replace direct offset evaluation.

The literature therefore changes the experiment priority, not the proven search
structure: test exact matched permutations and block-local rotations next; defer
full covariance, learned affine transforms, and gradient-based calibration until
they have a strict CPU-cost justification.

## Experiment order

1. Local E6M2 neighborhood search with exact fixed-scale hierarchy selection.
2. Calibration-derived diagonal weighting for Linear and counterpart-energy
   weighting for Q/K.
3. Exact matched 16-channel-block permutations, gated by broad synthetic results.
4. Block-local Hadamard rotations for Linear and Q/K, with V unchanged.
5. Full covariance or constrained automated policy search only if simpler methods
   leave runtime and quality headroom. Do not retry rejected diagonal transforms,
   key centering, or fixed-code analytic scale refinement without new evidence.

The unknown official standard baseline prevents local reproduction of the contest
score. Local numbers compare tagged variants against `solution/v000-baseline` and
must always be reported as local mean per-case improvement.
