# Problem Contract — NVFP4-to-HiF4 Conversion (2026 Huawei Prelim)

Sources of truth: `statements_from_docx.txt` (contest statement; figures are missing
from the docx text extraction), `example/self_check.py` (contestant-side local
format oracle), `example/solution/solution.py` (interface template),
`example/环境说明.md` (local env + library list), and the public mini-sample data
`example/mini_sample/{linear,attn}.pt`.

Sections: (1) confirmed contract facts, (2) facts enforced **only** by the local
checker, (3) missing/ambiguous evaluator details, (4) explicit local-evaluation
assumptions, (5) empirical observations from the public mini-sample.

---

## 1. Confirmed contract facts

### 1.1 Task definition
- Convert NVFP4 (source format S) to HiF4 (target format T). Given MatMul inputs
  A, B (or Q, K, V for Attention) in format S, apply f_(S→T), dequantize both to
  FP32, and require the MSE of the result (MatMul / Attention) before vs. after
  conversion to be small.
- Objective per test case: minimize
  `MSE(output_HIF4, output_NVFP4_reference)`, where the reference output is
  computed from the **dequantized original NVFP4** data:
  - Linear: `X_HIF4 @ W_HIF4^T` vs. `X_NVFP4 @ W_NVFP4^T`
  - Attention: `Attention(Q_HIF4, K_HIF4, V_HIF4)` vs. `Attention(Q_NVFP4, K_NVFP4, V_NVFP4)`
- Calibration data may be used to precompute quantization states/parameters;
  calibration is **offline-only** and does not directly contribute to the score.

### 1.2 Submission and packaging
- A `solution.py` implementing **exactly 6 public functions** (names below) must
  be packaged **directly into `solution.zip`**; additional auxiliary code files
  are allowed. File read/write operations are prohibited in submitted code.
- No restrictions on other data structures, helper functions, or internal
  implementations. No restrictions on using the calibration-state mechanism.

### 1.3 Public API (Python, in `solution.py`)
| Function | Inputs | Returns |
|---|---|---|
| `hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list)` | weight NVFP4 carrier + scale; list of calib activation NVFP4 pairs | `{"weight_params": HiF4Params, "activation_state": state}` |
| `hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state)` | online activation NVFP4 pair + state | `HiF4Params` (logical shape must match the activation) |
| `hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)` | list of calib Q/K/V samples + head params | `{"q_state": …, "k_state": …, "v_state": …}` |
| `hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state)` | Q NVFP4 pair, heads, dim, state | `HiF4Params` |
| `hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state)` | K NVFP4 pair, heads, dim, state | `HiF4Params` |
| `hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state)` | V NVFP4 pair, heads, dim, state | `HiF4Params` |

- NVFP4 input contract: `quant` shape `(..., C)` with `C % 16 == 0`;
  `scale` shape `(..., C // 16)` (one scale per 16 values, block size 16).
  `dequantize_nvfp4` reference helper: `x = quant.unflatten(-1, (-1,16));
  x = x * scale.unsqueeze(-1); x.flatten(-2,-1).to(torch.bfloat16)`.
- Attention sample structure: dict `{"q": (q_quant, q_scale), "k": (k_quant, k_scale),
  "v": (v_quant, v_scale)}`; Q/K/V 2-D `[seq_len, hidden]`; Q hidden = `q_num_heads * head_dim`,
  K/V hidden = `kv_num_heads * head_dim`; seq_len equal across Q/K/V.
- Any redundant state may be set to `None`.
- Note: `dequantize_nvfp4` is listed in the API section but is **not** one of the 6
  required callables; the checker does not require it. Include it (or an equivalent)
  only as a helper if the algorithm needs it.

### 1.4 HiF4 parameter format (`HiF4Params`)
Returned as a dict with the 5 tensors below. For original input shape
`(*prefix, C)` with `C % 64 == 0` (HiF4 block size = 64):

| Key | Shape | Value constraints (contract-stated) |
|---|---|---|
| `scale_factor` | `(*prefix, C//64, 1, 1, 1)` | E6M2 format scale (one per 64-value block) |
| `scale_lv2` | `(*prefix, C//64, 8, 1, 1)` | ∈ {1, 2} (per 8-value group) |
| `scale_lv3` | `(*prefix, C//64, 8, 2, 1)` | ∈ {1, 2} (per 4-value sub-group) |
| `sign` | `(*prefix, C//64, 8, 2, 4)` | ∈ {-1, 0, 1} (per value) |
| `mant` | `(*prefix, C//64, 8, 2, 4)` | ∈ {0, 0.25, …, 1.75} (per value) |

- Dequantization relation: `x_hat = sign * mant * scale_lv3 * scale_lv2 * scale_factor`
  (broadcasts over the `(8,2,4)` block layout), reshaped to the original shape.
- This is the "three-level hierarchy" (E6M2 + E1_8 + E1_16) named in the statement;
  the statement figures defining the block layout are missing from the extracted text.

### 1.5 Calibration state format (`activation_state`, `q_state`, `k_state`, `v_state`)
- Allowed content: `None`, `bool`, `int`, finite `float`, `str`, CPU `torch.Tensor`,
  `list`, `tuple`, `dict` with **string keys**.
- State tensor dtypes: `bool, int8, int16, int32, int64, float16, bfloat16, float32`.
- State tensors: no NaN, no Inf, no complex, no gradient information.
- Maximum nesting depth: 8. Total node count ≤ 4096.
- States are generated at calibration and passed (read-only) to the corresponding
  online function; they may store calibration stats, scales, fixed weights, etc.

### 1.6 Evaluation flow and scoring
1. Per Linear group: `hif4_calibration_and_quantize_weight(weight, calib list)` →
   `weight_params` + `activation_state`; per test activation:
   `hif4_dynamic_quantize_activation(test_act, scale, activation_state)`.
2. Per Attention group: `hif4_calibration_attention(calib qkv, heads, dim)` →
   3 states; per test sample: `hif4_dynamic_quantize_q/k/v`.
3. Participant outputs are dequantized and Linear/Attention outputs computed;
   the same is done with the **standard HiF4 baseline** on the same test data.
4. Per test case: `MSE_STD` (baseline vs. NVFP4 reference) and `MSE_PLAYER`
   (player vs. same reference); `Score = (MSE_STD − MSE_PLAYER) / MSE_STD`.
5. Final score = sum of the per-test-case scores (MSE improvement percentages)
   across all Linear + Attention test cases. Worse-than-baseline MSE on any test
   case yields a negative score proportional to the degradation.
6. Failure ⇒ whole submission invalid: any timeout, runtime exception, missing
   output, invalid HiF4 parameters (any test case), or invalid calibration state.

### 1.7 Dataset scale and runtime (preliminary round, single stage)
- Linear: **50 groups**; each = 1 weight + 5 calib activations + 5 test activations.
- Attention: **50 groups**; each = Q, K, V categories, each with 5 calib + 5 test.
- Runtime: overall limit of **5 minutes** per submission; no per-test-case limit.
- Judge platform: **Kunpeng 920B** (CPU), Python-only interfaces; third-party
  libraries per the runtime environment list (`example/环境说明.md`, includes
  `torch==2.13.0`, `numpy==2.5.1`, `triton==3.7.1`, no explicit GPU guarantee).
- Tie-break: earlier submission time. Timeout/compile/runtime errors/invalid
  outputs ⇒ invalid submission, no score.

---

## 2. Facts enforced only by the local checker (`self_check.py`)

The statement text does not state these; they come from the checker implementation
(the local oracle). Treat them as *necessary* conditions locally; the remote
evaluator may enforce an equivalent or stricter set.

- **Result dict keys are strict**: Linear calibration result may contain only
  `{"weight_params", "activation_state"}`; Attention calibration result only
  `{"q_state", "k_state", "v_state"}` — unknown keys are errors.
  (For `HiF4Params` itself, extra keys are ignored, not rejected.)
- **HiF4 param value checks** (after conversion to CPU float64; non-finite ⇒ error):
  - `scale_factor` must be an **exact E6M2 value** in **[2^-48, 49152]**, verified by
    reconstruction `sf == round(sf * 2^(2−exp)) * 2^(exp−2)` with
    `exp = floor(log2(clamp(sf, min=2^-126)))` (i.e., 2 explicit mantissa bits + implicit 1).
  - `scale_lv2`, `scale_lv3` exactly 1.0 or 2.0; `sign` exactly −1.0/0.0/1.0;
    `mant` exact multiples of 0.25 in [0, 1.75] (checked as `mant*4 == round(mant*4)`).
  - Dequant product `sign*mant*scale_lv2*scale_lv3*scale_factor` must be finite,
    reshape to the original shape must succeed, and numel must match the input.
- **Input-shape preconditions the checker asserts on data**: last dim % 16 == 0 for
  NVFP4 pairs; last dim % 64 == 0 for HiF4 param shapes; Attention: positive head
  params, `q_num_heads % kv_num_heads == 0`, Q/K/V same seq_len.
- **State checks beyond the statement text**:
  - Strings (values and dict keys) ≤ 4096 UTF-8 bytes.
  - State tensors must be CPU, dense strided layout, `requires_grad == False`.
  - Exact type matching (`type(v) is list/tuple/dict/torch.Tensor`); subclasses fail.
  - Python `int` is unbounded (no range check); `float` must be finite; container
    depth counted from root (children of a depth-8 container are rejected); node
    count is per-state and capped at 4096.
- **State isolation**: the checker deep-clones the state (`detach().cpu().contiguous()`,
  recursively) before **every** online call. Online functions must therefore treat
  the state as immutable read-only data and must not rely on cross-call mutation
  or object identity.
- The checker does **not** verify MSE, runtime, or semantic quality of the
  quantization — it only validates interface, state format, and output format.

---

## 3. Missing or ambiguous evaluator details

- **Reference-output precision**: statement says "dequantized to FP32 before
  MatMul", but the provided `dequantize_nvfp4` casts to **bfloat16**. Unknown
  whether the NVFP4 reference (and the participant's dequantized output) are
  evaluated in FP32 or BF16, and at what precision the MatMul/Attention runs.
- **Standard HiF4 baseline algorithm**: not provided to contestants (template
  states this explicitly). Details (block-wise search, clipping, hierarchy
  selection policy, E6M2 rounding direction) are unknown and affect `MSE_STD`.
- **Attention formula details**: softmax scaling by `sqrt(head_dim)` or not,
  causal masking, GQA head-broadcasting convention, and compute precision are
  unspecified in the extracted text.
- **MSE definition**: per-test-case MSE over the full output tensor (mean over
  elements) is assumed; normalization, reduction order, and whether Linear and
  Attention are weighted equally in the final sum are not stated.
- **Scoring edges**: interplay of the per-case negative-score penalty with the
  total-sum formula; whether `MSE_STD == 0` is handled; whether the final reported
  score can go negative overall.
- **Runtime measurement**: 5-minute limit is per submission; wall-clock vs. CPU
  time, thread counts, and whether the whole pipeline (50+50 groups, calib + test)
  runs in one process are unspecified.
- **Hidden test-set specifics**: group count 50 is stated, but hidden shapes
  (seq_len, hidden sizes, head counts, `attn_type`), value distributions, and
  whether all groups are GQA are unknown. HiF4 figure content (block structure
  and "calculation method" figures in the statement) is missing from the docx
  extraction.
- **Platform**: "Kunpeng 920B" is stated; whether evaluation runs CPU-only (likely)
  and exact library versions are unconfirmed beyond the local list.
- **`attn_type`/`key` metadata**: present in the public sample group dicts; not
  part of the API and ignored by the checker — whether hidden data carries other
  metadata is unknown.

---

## 4. Explicit local-evaluation assumptions

Used to make the contract testable locally; each may differ from the remote judge.

1. `example/self_check.py` (run as `python self_check.py --solution_dir <dir>
   --datasets_dir example/mini_sample`) is the format/interface oracle: passing it
   is necessary, not sufficient, for the score.
2. Reference and player outputs are computed at **FP32** from **BF16-dequantized**
   values (following the statement's "dequantized to FP32" wording; BF16 cast as in
   `dequantize_nvfp4`).
3. Attention is standard `softmax(Q K^T / sqrt(head_dim)) V` in FP32, no mask,
   with K/V heads broadcast to Q heads for GQA.
4. Per-test-case MSE = mean squared error over all elements of the output tensor,
   against the NVFP4 reference. The local dashboard reports the arithmetic mean
   of per-case improvements; this differs from the stated official sum only by a
   constant factor when every run uses the same case suite.
5. The 5-minute budget on the local machine is a loose proxy for Kunpeng 920B CPU
   performance; local wall-clock targets should leave large margin (e.g., ≤ 1–2 min).
6. Hidden data follows the public sample's structure: BF16 carrier + BF16 scale,
   NVFP4 block 16, last dims divisible by 64, 5 calib + 5 test per group, 50 groups
   per scenario, seq_len/hidden sizes in the same magnitude as the sample.
7. Return CPU tensors throughout (the checker tolerates non-CPU tensors in
   `HiF4Params` but the judge platform is CPU-only, and state must be CPU).

---

## 5. Empirical observations (public mini-sample only — not proof of hidden data)

Measured on `example/mini_sample/*.pt` (1 group each; 58–59 MB per file):

- `linear.pt`: group keys `{key='linear', weight, calib_activation_list(5),
  test_activation_list(5)}`. Weight carrier `[8192, 2048]` BF16, weight scale
  `[8192, 128]` BF16. Activations `[10, 2048]` BF16, scales `[10, 128]` BF16.
- `attn.pt`: group keys `{key='attn', attn_type='gqa', q_num_heads=16,
  kv_num_heads=2, head_dim=256, calib(5), test(5)}`. Q `[10, 4096]` BF16 +
  scale `[10, 256]`; K/V `[10, 512]` BF16 + scale `[10, 32]`.
- NVFP4 carrier values are BF16 with multiples of 0.5 in {−6, …, 6} (consistent
  with NVFP4 E2M1 value set); scales are BF16 powers-of-two-ish fractions
  (e.g., 2^-9, 2^-8, 2^-7).
- All last dims (2048, 4096, 512) are divisible by 16 and 64; all scales have
  last dim `C//16`.
