# Idea: Realistic Open-Model Dataset

## Status

Proposed and considered the highest-value testing-infrastructure improvement.
Implementation and detailed design are delegated to the working session.

## Motivation

The current synthetic generator is useful for adversarial stress testing but is
not a realistic model of hidden transformer tensors. It omits real channel
correlations, layer-specific distributions, language-conditioned activations,
and the organizer's exact NVFP4 production pipeline. Passing synthetic tests is
therefore weak positive evidence, although failing them is useful negative
evidence about catastrophic behavior.

The public mini-sample contains only one Linear and one Attention group. It is
too small to select aggressive calibration policies reliably.

## Proposed dataset

Build contest-shaped groups from open transformer models:

1. Load open model weights.
2. Run representative text and capture inputs to selected Linear layers.
3. Capture Q, K, and V before Attention where feasible.
4. Split sequences into five calibration and five test samples per group.
5. Convert tensors to the contest's decoded NVFP4 carrier-plus-scale API format.
6. Evaluate tagged solutions with exactly the same reference and case accounting
   used by the existing evaluator.

Favor Chinese model families such as Qwen and DeepSeek-derived models because
they may better match the competition context, but include at least one distinct
architecture such as Llama or Mistral. The hidden model provenance is unknown;
model diversity reduces the risk of replacing synthetic overfit with Qwen-only
overfit.

Include different Linear roles rather than treating all weights as exchangeable:

- Attention Q/K/V projections.
- Attention output projections.
- MLP gate/up projections.
- MLP down projections.

## NVFP4 variants

Test both sources if feasible:

- Genuine NVFP4 checkpoints or tensors produced by an established NVIDIA
  NVFP4/ModelOpt-compatible pipeline.
- BF16/FP16 open models quantized locally to NVFP4.

The locally quantized path must document scale selection, scale data type,
rounding, clipping, and any global/per-tensor scale. The existing synthetic
`max_abs / 6` BF16-scale converter is legal for the contest API but is not known
to reproduce the organizer's pipeline.

## Important risks

- Open weights alone are insufficient; output-aware Linear testing requires
  representative activations for those exact weights.
- Download and disk cost may dominate with an approximately 1 MB/s connection.
- Loading a model may require more RAM than is available even if its files fit on
  disk.
- Capturing every layer and token can create much more data than the model itself;
  capture a selected, streamed subset.
- Different model libraries expose Q/K/V at different points, and fused Attention
  may require hooks or a small instrumentation patch.
- The official Attention operation and compute precision remain unspecified.
- Real-model data should supplement, not replace, synthetic edge cases.

## Suggested staged scope

Start with one small model and Linear only. Validate that hooks, shapes, five-plus-
five splits, NVFP4 conversion, and evaluator integration are correct before
downloading larger models or implementing Attention capture. Estimate download,
disk, RAM, and runtime before selecting model checkpoints.

## Success criteria

- Dataset creation is deterministic and records model revision and prompt corpus.
- Calibration and test samples are disjoint.
- Every tensor passes the official format checker assumptions.
- Tagged variants can be compared without regenerating different samples.
- Reports show results by model, layer role, shape, and quantization source.
- The dataset reveals whether conclusions based on synthetic families generalize
  to real transformer tensors.
