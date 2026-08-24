"""Streaming evaluator over raw-BF16 real-capture shards (NVFP4 -> HiF4).

This is the second real-model data milestone: it turns a captured dataset
(``data/real-captures/<dataset_id>/``, see ``tools/realdata/capture.py``) into
contest-style per-case MSE-improvement scores, streaming one group shard at a
time so peak memory stays bounded by a single group (plus the loaded solution
modules).

Pipeline per group
------------------
1. Load and validate exactly one shard (``shards.validate_linear_shard`` /
   ``shards.validate_attention_shard``); the group is dropped from memory after
   its cases are recorded.
2. Derive contest-format NVFP4 pairs *at evaluation time* from the raw BF16
   tensors with ``quantize_nvfp4`` under the requested source mode
   (``ceil`` / ``nearest`` / ``stochastic``) and seed.  Every logical tensor
   gets a stable, unique ``tensor_id`` (``<dataset_id>/<group>/...``), so
   stochastic draws are reproducible across processes and decorrelated between
   tensors.
3. Reference outputs are plain operations on the derived NVFP4 pairs
   (``linear_output_nvfp4`` / ``attention_output_nvfp4``).
4. For Linear, the weight calibration API is invoked exactly once per variant
   and the dynamic activation API exactly five times (one per test sample);
   for Attention, ``hif4_calibration_attention`` is invoked once per variant
   and the Q/K/V dynamic APIs once per test sample (five tests).
5. Per case the metric is ``100 * (baseline_mse - candidate_mse) /
   baseline_mse`` with robust zero/non-finite denominator handling
   (``improvement_percent`` is ``null`` and the case is invalid when the
   baseline MSE is zero/non-finite or the candidate MSE is non-finite).

Outputs and resume
------------------
* One deterministic, atomic JSON record per case under
  ``<output>/records/<record_key>.json`` (temp file + ``os.replace``).  The
  record key includes the evaluator tooling identity, dataset/mode/seed,
  resolved Git commits or path content hashes, group, and test index. A fixed
  configuration has exactly one file slot per case; code edits cannot reuse
  stale records. Reruns skip existing records (``--force`` recomputes), and
  new candidates only compute their missing slots.
* A per-(group, mode) baseline cache under ``<output>/baseline/<key>.json``
  lets later runs with additional candidates reuse baseline MSEs instead of
  re-running the baseline module.
* A run manifest under ``<output>/runs/<config_hash>.json`` summarizes the
  invocation (counts, per-mode metrics, failures).  One file per distinct
  configuration, overwritten atomically on rerun: retention stays bounded.
* No temporary files survive (atomic writes clean up on failure).

Only the Python standard library and torch are used; Git access is read-only
(``git show`` / ``git rev-parse`` via ``tools.evaluate._load_solution``).
Solution loading, data cloning, HiF4/state legality checks are reused from
``tools/evaluate.py`` so semantics stay identical to the synthetic evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]

from tools.evaluate import (  # noqa: E402
    _clone_data,
    _get_funcs,
    _hif4_legality_errors,
    _load_solution,
    _state_legality_errors,
)
from tools.nvfp4_quantize import quantize_nvfp4  # noqa: E402
from tools.realdata import shards  # noqa: E402
from tools.reference_ops import (  # noqa: E402
    attention_output,
    attention_output_nvfp4,
    dequantize_hif4_params,
    linear_output,
    linear_output_nvfp4,
    scalar_mse,
)

SCHEMA_VERSION = 1

#: Default resolution root for dataset ids passed to the CLI.
DEFAULT_CAPTURES_ROOT = Path("data/real-captures")

#: Default output root for records / baseline cache / run manifests.
DEFAULT_OUTPUT_DIR = Path("benchmarks/realdata")

#: Number of scored test samples per group (the capture contract fixes 5).
TESTS_PER_GROUP = shards.SAMPLES_PER_SPLIT

_GROUP_ID = "group_id"
_TEST_COUNT = shards.SAMPLES_PER_SPLIT


def _tooling_identity() -> str:
    """Fingerprint code that defines source derivation and scoring semantics."""
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        _REPO_ROOT / "tools" / "evaluate.py",
        _REPO_ROOT / "tools" / "nvfp4_quantize.py",
        _REPO_ROOT / "tools" / "reference_ops.py",
        _REPO_ROOT / "tools" / "realdata" / "shards.py",
    ):
        digest.update(path.read_bytes())
    digest.update(torch.__version__.encode("utf-8"))
    return digest.hexdigest()[:16]


TOOLING_IDENTITY = _tooling_identity()


class DatasetError(RuntimeError):
    """The dataset could not be resolved or is not evaluable."""


class GroupLoadError(RuntimeError):
    """A single group shard failed to load or validate (streaming error)."""

    def __init__(self, group_id: str, message: str) -> None:
        super().__init__(f"group {group_id}: {message}")
        self.group_id = group_id


class VariantError(RuntimeError):
    """A solution variant raised while evaluating one (group, mode)."""

    def __init__(self, label: str, message: str) -> None:
        super().__init__(f"variant {label}: {message}")
        self.label = label


# ---------------------------------------------------------------------------
# Hashing / keys
# ---------------------------------------------------------------------------


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_solution_info(info: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(info)
    if prepared.get("kind") == "path":
        path = Path(str(prepared.get("label"))).resolve()
        prepared["label"] = str(path)
        prepared["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        prepared["source_sha256"] = None
    return prepared


def solution_label(info: Mapping[str, Any]) -> str:
    """Stable, human-readable label for a loaded solution (kind + path/commit).

    For Git refs the resolved commit hash is preferred so a moving branch is
    treated as a different variant; paths are absolute.
    """
    if info["kind"] == "git":
        commit = info.get("commit") or info.get("label")
        return f"git:{commit}"
    path = Path(str(info.get("label"))).resolve()
    source_hash = info.get("source_sha256")
    if not source_hash:
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"path:{path}#{str(source_hash)[:16]}"


def baseline_cache_key(
    dataset_id: str, seed: int, baseline_label: str, group_id: str, mode: str
) -> str:
    payload = (
        f"{TOOLING_IDENTITY}|{dataset_id}|{seed}|{baseline_label}|{group_id}|{mode}"
    )
    return _sha256(payload)[:32]


def case_record_key(
    dataset_id: str,
    seed: int,
    baseline_label: str,
    candidate_label: str,
    group_id: str,
    mode: str,
    test_index: int,
    group_sha256: str = "",
) -> str:
    payload = (
        f"{TOOLING_IDENTITY}|{dataset_id}|{seed}|{baseline_label}|{candidate_label}|"
        f"{group_id}|{group_sha256}|{mode}|{test_index}"
    )
    return _sha256(payload)[:32]


def run_config_hash(
    dataset_spec: str,
    seed: int,
    modes: Sequence[str],
    baseline_label: str,
    candidate_labels: Sequence[str],
    filters: Sequence[Any] = (),
    limit: int | None = None,
    threads: int = 1,
) -> str:
    payload = "|".join(
        [
            TOOLING_IDENTITY,
            dataset_spec,
            str(seed),
            ",".join(modes),
            baseline_label,
            ",".join(candidate_labels),
            json.dumps(list(filters), sort_keys=True),
            str(limit),
            str(threads),
        ]
    )
    return _sha256(payload)[:16]


# ---------------------------------------------------------------------------
# Dataset resolution and streaming group iteration
# ---------------------------------------------------------------------------


def resolve_dataset(
    dataset_spec: str, captures_root: Path | str = DEFAULT_CAPTURES_ROOT
) -> tuple[Path, dict[str, Any]]:
    """Resolve ``dataset_spec`` (a directory path or a dataset id) to
    ``(dataset_dir, manifest)``.  A spec that is not an existing directory is
    treated as a dataset id under ``captures_root``."""
    root = Path(captures_root)
    direct = Path(dataset_spec).expanduser()
    if direct.is_dir():
        dataset_dir = direct
    else:
        candidate = root / dataset_spec
        if not candidate.is_dir():
            raise DatasetError(
                f"dataset spec {dataset_spec!r} is neither an existing "
                f"directory nor a dataset id under {root}"
            )
        dataset_dir = candidate
    manifest = shards.load_manifest(dataset_dir / "manifest.json")
    if manifest is None:
        raise DatasetError(f"no manifest.json found in {dataset_dir}")
    if manifest.get("status") != "complete":
        raise DatasetError(
            f"dataset {dataset_dir} status is {manifest.get('status')!r}; "
            "expected 'complete'"
        )
    if not manifest.get("groups"):
        raise DatasetError(f"manifest {dataset_dir} lists no groups")
    if not manifest.get("dataset_id"):
        raise DatasetError(f"manifest {dataset_dir} lacks a dataset_id")
    return dataset_dir, manifest


def parse_filters(values: Sequence[str] | None) -> list[list[tuple[str, str]]]:
    """Parse ``--group-filter`` values into predicates.

    Each filter value is a comma list of predicate tokens; the tokens inside
    one value are OR'd, and the resulting predicates are AND'd across values.
    Tokens: ``linear`` / ``attention`` (bare kind), ``kind:<kind>``,
    ``layer:<int>``, ``role:<name>``, ``id:<group id>``.  A keyed OR is written
    as ``role:q_proj,role:o_proj`` in one value.
    """
    if not values:
        return []
    predicates: list[list[tuple[str, str]]] = []
    for value in values:
        alternatives: list[tuple[str, str]] = []
        for raw in str(value).split(","):
            token = raw.strip()
            if not token:
                continue
            if token in ("linear", "attention"):
                alternatives.append(("kind", token))
                continue
            key, sep, val = token.partition(":")
            if not sep or not val:
                raise ValueError(
                    f"invalid group filter {token!r}; use kind:, layer:, role:, "
                    "id:, or a bare kind"
                )
            key = key.strip().lower()
            if key not in ("kind", "layer", "role", "id"):
                raise ValueError(
                    f"unknown group filter key {key!r} in {token!r}; "
                    "choose kind, layer, role, id"
                )
            if key == "layer":
                try:
                    int(val.strip())
                except ValueError:
                    raise ValueError(
                        f"invalid layer index {val.strip()!r} in {token!r}"
                    ) from None
            alternatives.append((key, val.strip()))
        if not alternatives:
            raise ValueError(f"empty group filter value {value!r}")
        predicates.append(alternatives)
    return predicates


def group_matches(entry: Mapping[str, Any], filters: Sequence[Sequence[tuple[str, str]]]) -> bool:
    metadata = entry.get("metadata") or {}
    for alternatives in filters:
        ok = False
        for key, value in alternatives:
            if key == "kind":
                ok = ok or entry.get("kind") == value
            elif key == "layer":
                ok = ok or str(metadata.get("layer_idx")) == str(value)
            elif key == "role":
                ok = ok or str(metadata.get("role")) == str(value)
            elif key == "id":
                ok = ok or entry.get("id") == value
        if not ok:
            return False
    return True


class _GroupIterator:
    """Iterator whose per-shard errors do not terminate later iteration."""

    def __init__(
        self,
        dataset_dir: Path | str,
        manifest: Mapping[str, Any],
        filters: Sequence[Sequence[tuple[str, str]]],
        limit: int | None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        entries = [
            entry
            for entry in manifest.get("groups", [])
            if group_matches(entry, filters)
        ]
        self.entries = sorted(
            entries,
            key=lambda entry: (
                int((entry.get("metadata") or {}).get("layer_idx", -1)),
                str(entry.get("id", "")),
            ),
        )
        if limit is not None:
            self.entries = self.entries[: int(limit)]
        self.index = 0

    def __iter__(self) -> "_GroupIterator":
        return self

    def __next__(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.index >= len(self.entries):
            raise StopIteration
        entry = self.entries[self.index]
        self.index += 1
        group_id = entry.get("id", "?")
        relative = entry.get("path")
        if not relative:
            raise GroupLoadError(group_id, "manifest entry lacks 'path'")
        target = self.dataset_dir / relative
        try:
            shard = shards.load_tensor(target)
        except Exception as error:
            raise GroupLoadError(
                group_id, f"cannot load shard {target}: {error}"
            ) from None
        try:
            kind = entry.get("kind")
            if kind == "linear":
                shards.validate_linear_shard(shard)
            elif kind == "attention":
                shards.validate_attention_shard(shard)
            else:
                raise ValueError(f"unknown group kind {kind!r}")
        except Exception as error:
            raise GroupLoadError(
                group_id, f"invalid shard {target}: {error}"
            ) from None
        return entry, shard


def iter_groups(
    dataset_dir: Path | str,
    manifest: Mapping[str, Any],
    filters: Sequence[Sequence[tuple[str, str]]] | None = None,
    limit: int | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(manifest_entry, validated_shard)`` one group at a time.

    Shards are loaded lazily inside ``next()``: a later shard does not touch
    memory until its turn, and a deleted/corrupt shard raises
    :class:`GroupLoadError` at its own iteration step.
    """
    return _GroupIterator(dataset_dir, manifest, filters or [], limit)


# ---------------------------------------------------------------------------
# NVFP4 derivation (source modes) and references
# ---------------------------------------------------------------------------


def _quantize(tensor: torch.Tensor, mode: str, seed: int, tensor_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_nvfp4(tensor, scale_mode=mode, seed=seed, tensor_id=tensor_id)


def derive_source(
    shard: Mapping[str, Any],
    dataset_id: str,
    group_id: str,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    """Convert a validated raw-BF16 shard into contest-format NVFP4 pairs.

    Every logical tensor gets a distinct stable ``tensor_id`` derived from the
    dataset/group (mode-independent), so stochastic draws are deterministic
    and decorrelated between tensors.
    """
    base = f"{dataset_id}/{group_id}"
    if shard["kind"] == "linear":
        weight = _quantize(shard["weight"], mode, seed, f"{base}/weight")
        calib = [
            _quantize(act, mode, seed, f"{base}/calib/{i}")
            for i, act in enumerate(shard["calib_activation_list"])
        ]
        test = [
            _quantize(act, mode, seed, f"{base}/test/{i}")
            for i, act in enumerate(shard["test_activation_list"])
        ]
        return {
            "kind": "linear",
            "weight": weight,
            "calib": calib,
            "test": test,
            "metadata": dict(shard.get("metadata") or {}),
        }
    calib: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
    test: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
    for i, sample in enumerate(shard["calib"]):
        calib.append(
            {
                role: _quantize(sample[role], mode, seed, f"{base}/calib/{i}/{role}")
                for role in ("q", "k", "v")
            }
        )
    for i, sample in enumerate(shard["test"]):
        test.append(
            {
                role: _quantize(sample[role], mode, seed, f"{base}/test/{i}/{role}")
                for role in ("q", "k", "v")
            }
        )
    return {
        "kind": "attention",
        "q_num_heads": int(shard["q_num_heads"]),
        "kv_num_heads": int(shard["kv_num_heads"]),
        "head_dim": int(shard["head_dim"]),
        "calib": calib,
        "test": test,
        "metadata": dict(shard.get("metadata") or {}),
    }


def compute_references(source: Mapping[str, Any]) -> list[torch.Tensor]:
    """Reference outputs are operations on the derived NVFP4 pairs."""
    if source["kind"] == "linear":
        wq, ws = source["weight"]
        return [
            linear_output_nvfp4(tq, ts, wq, ws) for tq, ts in source["test"]
        ]
    heads = (source["q_num_heads"], source["kv_num_heads"], source["head_dim"])
    refs: list[torch.Tensor] = []
    for sample in source["test"]:
        refs.append(
            attention_output_nvfp4(
                sample["q"][0], sample["q"][1],
                sample["k"][0], sample["k"][1],
                sample["v"][0], sample["v"][1],
                *heads,
                causal=False,
            )
        )
    return refs


# ---------------------------------------------------------------------------
# Variant runs (one group, one mode, one module)
# ---------------------------------------------------------------------------


def run_variant_group(
    funcs: Mapping[str, Callable[..., Any]],
    source: Mapping[str, Any],
    refs: Sequence[torch.Tensor],
    label: str,
) -> dict[str, Any]:
    """Run one variant over one (group, mode).

    Linear: ``hif4_calibration_and_quantize_weight`` once, then
    ``hif4_dynamic_quantize_activation`` once per test sample.
    Attention: ``hif4_calibration_attention`` once, then the Q/K/V dynamic
    APIs once per test sample.

    Returns per-case MSE/legality, wall times, and a ``raises``-style failure
    propagation (any API/legality raise becomes :class:`VariantError`).
    """
    calib_s = 0.0
    online_s = 0.0
    scoring_s = 0.0
    cases: list[dict[str, Any]] = []
    legality_bad = False
    legality_errors: list[str] = []

    def note_errors(errors: Sequence[str]) -> bool:
        nonlocal legality_bad
        if errors:
            legality_bad = True
            legality_errors.extend(errors[:3])
            return False
        return True

    def guard(operation: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except VariantError:
            raise
        except Exception as error:
            raise VariantError(
                label, f"{operation} failed: {type(error).__name__}: {error}"
            ) from error

    if source["kind"] == "linear":
        wq, ws = source["weight"]
        t0 = time.perf_counter()
        calib_result = guard(
            "weight calibration",
            lambda: funcs["weight"](wq, ws, source["calib"]),
        )
        calib_s += time.perf_counter() - t0
        expected = {"weight_params", "activation_state"}
        if not isinstance(calib_result, dict) or set(calib_result) != expected:
            raise VariantError(
                label,
                f"linear calibration result keys must be {sorted(expected)}, "
                f"got {sorted(calib_result) if isinstance(calib_result, dict) else type(calib_result).__name__}",
            )
        wparams = calib_result["weight_params"]
        state = calib_result["activation_state"]
        weight_legal = note_errors(guard(
            "weight legality",
            lambda: _hif4_legality_errors(wparams, tuple(wq.shape)),
        ))
        state_legal = note_errors(guard(
            "activation state legality", lambda: _state_legality_errors(state)
        ))

        t0 = time.perf_counter()
        w_hif4 = guard(
            "weight dequantization",
            lambda: dequantize_hif4_params(wparams, tuple(wq.shape)),
        )
        scoring_s += time.perf_counter() - t0

        for ti, (aq, as_) in enumerate(source["test"]):
            case_t0 = time.perf_counter()
            activation_params = guard(
                f"activation dynamic test {ti}",
                lambda: funcs["activation"](aq, as_, _clone_data(state)),
            )
            online_s += time.perf_counter() - case_t0
            act_legal = note_errors(guard(
                f"activation legality test {ti}",
                lambda: _hif4_legality_errors(activation_params, tuple(aq.shape)),
            ))
            t0 = time.perf_counter()
            mse = guard(
                f"linear scoring test {ti}",
                lambda: scalar_mse(
                    linear_output(
                        dequantize_hif4_params(activation_params, tuple(aq.shape)),
                        w_hif4,
                    ),
                    refs[ti],
                ),
            )
            scoring_s += time.perf_counter() - t0
            cases.append(
                {
                    "mse": mse,
                    "legality_ok": weight_legal and state_legal and act_legal,
                    "case_s": time.perf_counter() - case_t0,
                }
            )
    else:
        qh, kvh, hd = source["q_num_heads"], source["kv_num_heads"], source["head_dim"]
        heads = (qh, kvh, hd)
        t0 = time.perf_counter()
        calib_result = guard(
            "attention calibration",
            lambda: funcs["attention"](source["calib"], *heads),
        )
        calib_s += time.perf_counter() - t0
        expected = {"q_state", "k_state", "v_state"}
        if not isinstance(calib_result, dict) or set(calib_result) != expected:
            raise VariantError(
                label,
                f"attention calibration result keys must be {sorted(expected)}, "
                f"got {sorted(calib_result) if isinstance(calib_result, dict) else type(calib_result).__name__}",
            )
        states = {role: calib_result[f"{role}_state"] for role in ("q", "k", "v")}
        for role in ("q", "k", "v"):
            note_errors(guard(
                f"{role} state legality",
                lambda role=role: _state_legality_errors(states[role]),
            ))

        for ti, sample in enumerate(source["test"]):
            case_t0 = time.perf_counter()
            params: dict[str, Any] = {}
            sample_legal = True
            for role, func_name in (("q", "q"), ("k", "k"), ("v", "v")):
                quant_t, scale_t = sample[role]
                num_heads = qh if role == "q" else kvh
                t0 = time.perf_counter()
                params[role] = guard(
                    f"{role} dynamic test {ti}",
                    lambda role=role, func_name=func_name, quant_t=quant_t,
                    scale_t=scale_t, num_heads=num_heads: funcs[func_name](
                        quant_t, scale_t, num_heads, hd, _clone_data(states[role])
                    ),
                )
                online_s += time.perf_counter() - t0
                sample_legal = (
                    note_errors(guard(
                        f"{role} legality test {ti}",
                        lambda role=role, quant_t=quant_t: _hif4_legality_errors(
                            params[role], tuple(quant_t.shape)
                        ),
                    ))
                    and sample_legal
                )
            t0 = time.perf_counter()
            mse = guard(
                f"attention scoring test {ti}",
                lambda: scalar_mse(
                    attention_output(
                        dequantize_hif4_params(params["q"], tuple(sample["q"][0].shape)),
                        dequantize_hif4_params(params["k"], tuple(sample["k"][0].shape)),
                        dequantize_hif4_params(params["v"], tuple(sample["v"][0].shape)),
                        qh,
                        kvh,
                        hd,
                        causal=False,
                    ),
                    refs[ti],
                ),
            )
            scoring_s += time.perf_counter() - t0
            cases.append(
                {"mse": mse, "legality_ok": sample_legal, "case_s": time.perf_counter() - case_t0}
            )

    wall_s = calib_s + online_s + scoring_s
    return {
        "cases": cases,
        "calibration_s": calib_s,
        "online_s": online_s,
        "scoring_s": scoring_s,
        "wall_s": wall_s,
        "legality_bad": legality_bad,
        "legality_errors": legality_errors[:5],
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _round(value: float) -> float | None:
    value = float(value)
    return round(value, 10) if math.isfinite(value) else None


def _geometry(shard: Mapping[str, Any], test_index: int) -> dict[str, Any]:
    if shard["kind"] == "linear":
        weight = shard["weight"]
        activation = shard["test_activation_list"][test_index]
        return {
            "kind": "linear",
            "in_features": int(weight.shape[1]),
            "out_features": int(weight.shape[0]),
            "seq_len": int(activation.shape[0]),
            "weight_shape": list(weight.shape),
            "activation_shape": list(activation.shape),
            "output_shape": None,
            "calib_samples": len(shard["calib_activation_list"]),
            "test_samples": len(shard["test_activation_list"]),
        }
    sample = shard["test"][test_index]
    return {
        "kind": "attention",
        "q_num_heads": int(shard["q_num_heads"]),
        "kv_num_heads": int(shard["kv_num_heads"]),
        "head_dim": int(shard["head_dim"]),
        "seq_len": int(sample["q"].shape[0]),
        "q_hidden": int(sample["q"].shape[1]),
        "kv_hidden": int(sample["k"].shape[1]),
        "q_shape": list(sample["q"].shape),
        "k_shape": list(sample["k"].shape),
        "v_shape": list(sample["v"].shape),
        "output_shape": None,
        "calib_samples": len(shard["calib"]),
        "test_samples": len(shard["test"]),
    }


def _variant_meta(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": info.get("label"),
        "kind": info.get("kind"),
        "commit": info.get("commit"),
        "source_sha256": info.get("source_sha256"),
    }


def _model_meta(manifest: Mapping[str, Any]) -> dict[str, Any]:
    model = manifest.get("model") or {}
    keys = (
        "alias",
        "repo_id",
        "resolved_revision",
        "arch",
        "transformers_version",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    return {key: model.get(key) for key in keys}


def build_case_record(
    *,
    manifest: Mapping[str, Any],
    dataset_dir: Path,
    group_entry: Mapping[str, Any],
    shard: Mapping[str, Any],
    mode: str,
    seed: int,
    test_index: int,
    baseline_label: str,
    candidate_label: str,
    baseline_meta: Mapping[str, Any],
    candidate_meta: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    refs: Sequence[torch.Tensor],
    record_key: str,
    case_s: float,
    group_wall_s: float,
    threads: int,
    created_at: str,
) -> dict[str, Any]:
    """Assemble the disaggregated per-case JSON record."""
    bm = float(baseline_result["cases"][test_index]["mse"])
    cm = float(candidate_result["cases"][test_index]["mse"])
    b_legal = bool(baseline_result["cases"][test_index]["legality_ok"])
    c_legal = bool(candidate_result["cases"][test_index]["legality_ok"])

    reason: str | None = None
    if not b_legal:
        reason = "baseline_illegal"
    elif not c_legal:
        reason = "candidate_illegal"
    elif not math.isfinite(bm):
        reason = "baseline_mse_nonfinite"
    elif bm <= 0.0:
        reason = "baseline_mse_zero"
    elif not math.isfinite(cm):
        reason = "candidate_mse_nonfinite"
    improvement = None
    if reason is None:
        improvement = 100.0 * (bm - cm) / bm

    geometry = _geometry(shard, test_index)
    ref = refs[test_index]
    geometry["output_shape"] = list(ref.shape)
    geometry["ref_power"] = _round(float((ref.to(torch.float32) ** 2).mean().item()))

    metadata = shard.get("metadata") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "record_key": record_key,
        "created_at": created_at,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_dir": str(dataset_dir),
        "model": _model_meta(manifest),
        "group": group_entry.get("id"),
        "kind": shard["kind"],
        "layer": metadata.get("layer_idx"),
        "role": metadata.get("role"),
        "mode": mode,
        "seed": int(seed),
        "case": int(test_index),
        "geometry": geometry,
        "baseline": {
            **_variant_meta(baseline_meta),
            "mse": _round(bm),
            "legality_ok": b_legal,
            "calibration_s": _round(baseline_result["calibration_s"]),
            "online_s": _round(baseline_result["online_s"]),
            "scoring_s": _round(baseline_result["scoring_s"]),
        },
        "candidate": {
            **_variant_meta(candidate_meta),
            "mse": _round(cm),
            "legality_ok": c_legal,
            "calibration_s": _round(candidate_result["calibration_s"]),
            "online_s": _round(candidate_result["online_s"]),
            "scoring_s": _round(candidate_result["scoring_s"]),
        },
        "baseline_mse": _round(bm),
        "candidate_mse": _round(cm),
        "improvement_percent": _round(improvement) if improvement is not None else None,
        "valid": reason is None,
        "invalid_reason": reason,
        "threads": int(threads),
        "torch_num_threads": torch.get_num_threads(),
        "timing": {
            "case_s": _round(case_s),
            "group_wall_s": _round(group_wall_s),
            "baseline_wall_s": _round(baseline_result["wall_s"]),
            "candidate_wall_s": _round(candidate_result["wall_s"]),
        },
    }


# ---------------------------------------------------------------------------
# Baseline cache
# ---------------------------------------------------------------------------


def _baseline_cache_path(output_dir: Path, key: str) -> Path:
    return Path(output_dir) / "baseline" / f"{key}.json"


def _load_baseline_cache(
    output_dir: Path,
    key: str,
    group_sha256: str,
    force: bool,
) -> dict[str, Any] | None:
    if force:
        return None
    path = _baseline_cache_path(output_dir, key)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as stream:
            cache = json.load(stream)
        if not isinstance(cache, dict):
            return None
        if cache.get("record_key") != key:
            return None
        if cache.get("group_sha256") != group_sha256:
            return None
        if cache.get("status") != "ok":
            return None
        if len(cache.get("cases", [])) != _TEST_COUNT:
            return None
        return cache
    except (OSError, ValueError):
        return None


def _save_baseline_cache(
    output_dir: Path,
    key: str,
    cache: Mapping[str, Any],
) -> None:
    shards.atomic_write_json(_baseline_cache_path(output_dir, key), cache)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evaluate_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the streaming evaluation described by ``config``.

    Config keys (all optional unless noted):
      dataset_dir (required), manifest (required), modes, seed, baseline_spec,
      candidate_specs, output_dir, captures_root, filters, limit, force,
      threads, repo_root (git repo for path/ref solution resolution; defaults
      to the workspace root), dataset_spec (used only for the run-config hash).
    """
    dataset_dir = Path(config["dataset_dir"])
    manifest = config["manifest"]
    repo_root = Path(config.get("repo_root") or _REPO_ROOT)
    modes = list(config.get("modes") or shards.SOURCE_MODES)
    for mode in modes:
        if mode not in shards.SOURCE_MODES:
            raise ValueError(f"unknown source mode {mode!r}; choose from {shards.SOURCE_MODES}")
    seed = int(config.get("seed", 0))
    baseline_spec = config.get("baseline_spec", "solution/v000-baseline")
    candidate_specs = list(config.get("candidate_specs") or ["solution.py"])
    output_dir = Path(config.get("output_dir") or DEFAULT_OUTPUT_DIR)
    filters = config.get("filters") or []
    limit = config.get("limit")
    force = bool(config.get("force", False))
    threads = int(config.get("threads", 1))
    if threads < 1:
        raise ValueError("threads must be at least 1")
    torch.set_num_threads(threads)
    dataset_spec = config.get("dataset_spec") or str(dataset_dir)
    records_dir = output_dir / "records"
    runs_dir = output_dir / "runs"
    baseline_dir = output_dir / "baseline"

    created_at = _now()
    counts = {
        "groups_total": len(manifest.get("groups", [])),
        "groups_selected": 0,
        "groups_failed": 0,
        "cases_slots_total": 0,
        "cases_new": 0,
        "cases_skipped": 0,
        "cases_failed": 0,
        "cases_invalid": 0,
    }
    failed_groups: list[dict[str, Any]] = []
    per_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        per_mode[mode] = {
            "case_count": 0,
            "invalid_case_count": 0,
            "valid_count": 0,
            "sum_improvement": 0.0,
            "mean_improvement_percent": None,
        }

    def add_record_stats(mode: str, record: Mapping[str, Any]) -> None:
        per_mode[mode]["case_count"] += 1
        if not record.get("valid"):
            counts["cases_invalid"] += 1
            per_mode[mode]["invalid_case_count"] += 1
            return
        improvement = record.get("improvement_percent")
        if improvement is None:
            counts["cases_invalid"] += 1
            per_mode[mode]["invalid_case_count"] += 1
            return
        per_mode[mode]["valid_count"] += 1
        per_mode[mode]["sum_improvement"] += float(improvement)

    # ---- load all variants up-front (fail fast on import/API errors) --------
    baseline_info = _prepare_solution_info(
        _load_solution(baseline_spec, str(repo_root))
    )
    baseline_label = solution_label(baseline_info)
    baseline_funcs = _get_funcs(baseline_info["module"])
    candidates: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for spec in candidate_specs:
        info = _prepare_solution_info(_load_solution(spec, str(repo_root)))
        candidates.append((solution_label(info), info, _get_funcs(info["module"])))
    candidate_labels = [label for label, _, _ in candidates]
    run_hash = run_config_hash(
        dataset_spec,
        seed,
        modes,
        baseline_label,
        candidate_labels,
        filters=config.get("filters_raw", []),
        limit=limit,
        threads=threads,
    )

    # ---- stream one group at a time ----------------------------------------
    groups_iter = iter_groups(dataset_dir, manifest, filters, limit)
    while True:
        try:
            group_entry, shard = next(groups_iter)
        except StopIteration:
            break
        except GroupLoadError as error:
            counts["groups_failed"] += 1
            failed_slots = len(modes) * len(candidates) * _TEST_COUNT
            counts["cases_slots_total"] += failed_slots
            counts["cases_failed"] += failed_slots
            failed_groups.append(
                {"group": error.group_id, "error": str(error), "created_at": _now()}
            )
            print(f"ERROR: {error}", file=sys.stderr)
            continue
        counts["groups_selected"] += 1
        group_id = group_entry.get("id", "?")
        group_sha = group_entry.get("sha256", "")

        for mode in modes:
            counts["cases_slots_total"] += len(candidates) * _TEST_COUNT
            try:
                try:
                    source = derive_source(
                        shard, manifest["dataset_id"], group_id, mode, seed
                    )
                    refs = compute_references(source)
                except Exception as error:
                    raise GroupLoadError(
                        group_id,
                        f"{mode} source derivation/reference failed: "
                        f"{type(error).__name__}: {error}",
                    ) from error

                # -- baseline (cached per group+mode) -------------------------
                base_key = baseline_cache_key(
                    manifest["dataset_id"], seed, baseline_label, group_id, mode
                )
                baseline_result = None
                cache = _load_baseline_cache(output_dir, base_key, group_sha, force)
                if cache is not None:
                    baseline_result = {
                        "cases": cache["cases"],
                        "calibration_s": cache["calibration_s"],
                        "online_s": cache["online_s"],
                        "scoring_s": cache["scoring_s"],
                        "wall_s": cache["wall_s"],
                        "legality_bad": cache["legality_bad"],
                        "legality_errors": cache.get("legality_errors", []),
                    }
                else:
                    baseline_result = run_variant_group(
                        baseline_funcs, _clone_data(source), refs, baseline_label
                    )
                    if all(
                        math.isfinite(float(case["mse"]))
                        for case in baseline_result["cases"]
                    ):
                        _save_baseline_cache(
                            output_dir,
                            base_key,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "record_key": base_key,
                                "dataset_id": manifest["dataset_id"],
                                "group_id": group_id,
                                "mode": mode,
                                "seed": seed,
                                "group_sha256": group_sha,
                                "baseline": _variant_meta(baseline_info),
                                "cases": baseline_result["cases"],
                                "calibration_s": baseline_result["calibration_s"],
                                "online_s": baseline_result["online_s"],
                                "scoring_s": baseline_result["scoring_s"],
                                "wall_s": baseline_result["wall_s"],
                                "legality_bad": baseline_result["legality_bad"],
                                "legality_errors": baseline_result["legality_errors"],
                                "status": "ok",
                                "created_at": _now(),
                            },
                        )

                # -- candidates ------------------------------------------------
                for candidate_label, candidate_info, candidate_funcs in candidates:
                    record_keys = [
                        case_record_key(
                            manifest["dataset_id"], seed, baseline_label,
                            candidate_label, group_id, mode, ti,
                            group_sha,
                        )
                        for ti in range(_TEST_COUNT)
                    ]
                    paths = [records_dir / f"{key}.json" for key in record_keys]
                    if not force and all(path.is_file() for path in paths):
                        try:
                            existing_records = []
                            for key, path in zip(record_keys, paths):
                                with open(path, encoding="utf-8") as stream:
                                    record = json.load(stream)
                                if record.get("record_key") != key:
                                    raise ValueError(f"record key mismatch in {path}")
                                existing_records.append(record)
                        except (OSError, ValueError, TypeError):
                            existing_records = []
                        if existing_records:
                            counts["cases_skipped"] += _TEST_COUNT
                            for record in existing_records:
                                add_record_stats(mode, record)
                            continue
                    t_start = time.perf_counter()
                    try:
                        candidate_result = run_variant_group(
                            candidate_funcs,
                            _clone_data(source),
                            refs,
                            candidate_label,
                        )
                    except VariantError as error:
                        counts["cases_failed"] += _TEST_COUNT
                        failed_groups.append(
                            {
                                "group": group_id,
                                "mode": mode,
                                "candidate": candidate_label,
                                "error": str(error),
                                "created_at": _now(),
                            }
                        )
                        print(f"ERROR: {error}", file=sys.stderr)
                        continue
                    group_wall_s = time.perf_counter() - t_start
                    for ti in range(_TEST_COUNT):
                        record = build_case_record(
                            manifest=manifest,
                            dataset_dir=dataset_dir,
                            group_entry=group_entry,
                            shard=shard,
                            mode=mode,
                            seed=seed,
                            test_index=ti,
                            baseline_label=baseline_label,
                            candidate_label=candidate_label,
                            baseline_meta=baseline_info,
                            candidate_meta=candidate_info,
                            baseline_result=baseline_result,
                            candidate_result=candidate_result,
                            refs=refs,
                            record_key=record_keys[ti],
                            case_s=float(candidate_result["cases"][ti]["case_s"]),
                            group_wall_s=group_wall_s,
                            threads=threads,
                            created_at=created_at,
                        )
                        path = records_dir / f"{record_keys[ti]}.json"
                        shards.atomic_write_json(path, record)
                        counts["cases_new"] += 1
                        add_record_stats(mode, record)
            except VariantError as error:
                counts["groups_failed"] += 1
                counts["cases_failed"] += len(candidates) * _TEST_COUNT
                failed_groups.append(
                    {
                        "group": group_id,
                        "mode": mode,
                        "error": str(error),
                        "created_at": _now(),
                    }
                )
                print(f"ERROR: {error}", file=sys.stderr)
                continue
            except GroupLoadError as error:
                counts["groups_failed"] += 1
                counts["cases_failed"] += len(candidates) * _TEST_COUNT
                failed_groups.append(
                    {
                        "group": group_id,
                        "mode": mode,
                        "error": str(error),
                        "created_at": _now(),
                    }
                )
                print(f"ERROR: {error}", file=sys.stderr)
                continue

        # Release the group's tensors before loading the next shard.
        del shard

    status = "ok"
    if counts["groups_failed"] or counts["cases_failed"]:
        status = "partial"
    elif counts["cases_invalid"]:
        status = "invalid"
    # A variant crash after groups were processed is a hard failure; detect
    # nothing extra here (VariantError is per-group, DatasetError is fatal).

    for mode in modes:
        stats = per_mode[mode]
        valid_mean = None
        if stats["valid_count"]:
            valid_mean = stats["sum_improvement"] / stats["valid_count"]
        stats["valid_case_mean_improvement_percent"] = valid_mean
        if not stats["invalid_case_count"]:
            stats["mean_improvement_percent"] = valid_mean
        stats.pop("sum_improvement", None)
        stats.pop("valid_count", None)

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_config_hash": run_hash,
        "created_at": created_at,
        "updated_at": _now(),
        "status": status,
        "dataset": {
            "spec": dataset_spec,
            "id": manifest.get("dataset_id"),
            "dir": str(dataset_dir),
        },
        "model": _model_meta(manifest),
        "modes": modes,
        "seed": seed,
        "threads": threads,
        "torch_num_threads": torch.get_num_threads(),
        "tooling_identity": TOOLING_IDENTITY,
        "baseline": {
            "spec": baseline_spec,
            **_variant_meta(baseline_info),
            "label": baseline_label,
        },
        "candidates": [
            {"spec": spec, **_variant_meta(info), "label": label}
            for spec, (label, info, _) in zip(candidate_specs, candidates)
        ],
        "filters": [str(f) for f in config.get("filters_raw", [])],
        "limit": limit,
        "counts": counts,
        "per_mode": per_mode,
        "failed_groups": failed_groups,
        "output": {
            "records_dir": str(records_dir),
            "baseline_dir": str(baseline_dir),
            "runs_dir": str(runs_dir),
        },
    }
    run_path = runs_dir / f"{run_hash}.json"
    shards.atomic_write_json(run_path, run_manifest)
    return run_manifest


def summarize(run_manifest: Mapping[str, Any]) -> str:
    """Compact human-readable summary for the CLI."""
    counts = run_manifest["counts"]
    lines = [
        f"dataset   : {run_manifest['dataset']['id']} ({run_manifest['dataset']['spec']})",
        f"modes     : {','.join(run_manifest['modes'])} (seed {run_manifest['seed']})",
        f"baseline  : {run_manifest['baseline']['label']}",
    ]
    for candidate in run_manifest["candidates"]:
        lines.append(f"candidate : {candidate['label']}")
    lines.append(
        f"groups    : {counts['groups_selected']} selected / "
        f"{counts['groups_total']} total / {counts['groups_failed']} failed"
    )
    lines.append(
        f"cases     : {counts['cases_new']} new / {counts['cases_skipped']} skipped "
        f"/ {counts['cases_failed']} failed / {counts['cases_invalid']} invalid"
    )
    for mode, stats in run_manifest["per_mode"].items():
        mean = stats["mean_improvement_percent"]
        if mean is not None:
            lines.append(f"  {mode:10s}: mean improvement {mean:.10f} %")
        else:
            lines.append(f"  {mode:10s}: n/a")
    lines.append(f"status    : {run_manifest['status']}")
    return "\n".join(lines)
