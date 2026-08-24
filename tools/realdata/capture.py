"""Deterministic raw-BF16 capture of real-model activations.

This is the first real-model data milestone:

- a fixed 10-text corpus (``corpus.json``) tokenized deterministically to exact
  per-split sequence lengths (full: [10,128,512,1024,1024]; smoke:
  [8,16,24,32,40]) with exactly five calibration and five test samples;
- a content-addressed dataset directory ``<output_root>/real-captures/<id>``
  where the id is a SHA-256 prefix of a canonical capture configuration
  (model alias/repo/resolved revision, transformers version, corpus SHA-256,
  exact sequence lengths, selected layers/roles, seed, schema version);
- one atomic raw shard per selected Linear site (weight + 5 calib + 5 test
  BF16 activations) and one per selected Attention layer (flattened q/k/v),
  each finalized independently with the manifest updated after every shard so
  interrupted runs resume by verifying hashes and skipping complete groups;
- attention captured at the eager kernel boundary by monkeypatching
  ``transformers.models.qwen3.modeling_qwen3.eager_attention_forward`` with
  try/finally restoration (Qwen3 first; qwen2/llama plug in via the adapter
  registry).

``transformers`` and model weights are loaded lazily (only inside
``RealLoader.load_model``) and never at module import time.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence

import torch

from . import shards

SCHEMA_VERSION = 1
DATASET_ID_PREFIX_LEN = 16

#: Per-split sequence-length patterns (both splits share the same pattern).
FULL_LENGTHS: list[int] = [10, 128, 512, 1024, 1024]
SMOKE_LENGTHS: list[int] = [8, 16, 24, 32, 40]

DEFAULT_LINEAR_ROLES: tuple[str, ...] = (
    "q_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_MODEL_ALIAS = "qwen3-0.6b"
DEFAULT_OUTPUT_ROOT = Path("data")
DEFAULT_DOWNLOAD_STATE = Path("data/model-download-state.json")

#: Every capture process lowers its OOM-killer priority so it dies before the
#: orchestrating test runner if the machine runs out of memory.
OOM_SCORE_ADJ = 500


class CaptureError(RuntimeError):
    """The capture pipeline could not proceed."""


def set_oom_score(score: int = OOM_SCORE_ADJ) -> bool:
    """Best-effort /proc/self/oom_score_adj write; returns whether it applied."""
    try:
        Path("/proc/self/oom_score_adj").write_text(f"{int(score)}\n")
        return True
    except (OSError, PermissionError, ValueError):
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_corpus(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the fixed capture corpus (exactly 5 calib + 5 test)."""
    source = Path(path) if path else Path(__file__).with_name("corpus.json")
    with open(source, encoding="utf-8") as stream:
        corpus = json.load(stream)
    if not isinstance(corpus, dict) or corpus.get("schema_version") != SCHEMA_VERSION:
        raise CaptureError(f"corpus {source} has unsupported schema")
    for split in ("calib", "test"):
        texts = corpus.get(split)
        if not isinstance(texts, list) or len(texts) != 5:
            raise CaptureError(f"corpus {source} needs exactly 5 {split} texts")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise CaptureError(f"corpus {source} contains a non-text {split} entry")
    if len(set(corpus["calib"] + corpus["test"])) != 10:
        raise CaptureError(f"corpus {source} texts must all be distinct")
    return corpus


def corpus_sha256(corpus: Mapping[str, Any]) -> str:
    blob = json.dumps(corpus, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def tokenize_sample(tokenizer: Any, text: str, length: int) -> list[int]:
    """Deterministically repeat/truncate a text's token ids to ``length``."""
    if length < 1:
        raise CaptureError(f"sequence length must be positive, got {length}")
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = list(encoded["input_ids"])
    except Exception as error:  # pragma: no cover - defensive
        raise CaptureError(f"tokenization failed: {error}") from error
    if not ids:
        ids = [0]
    if len(ids) >= length:
        return ids[:length]
    quotient, remainder = divmod(length, len(ids))
    return ids * quotient + ids[:remainder]


def make_samples(
    tokenizer: Any, corpus: Mapping[str, Any], lengths: Sequence[int]
) -> list[dict[str, Any]]:
    """Build the ten ordered samples (calib 0..4 then test 5..9)."""
    if len(lengths) != 5:
        raise CaptureError(f"need exactly 5 sequence lengths, got {len(lengths)}")
    samples: list[dict[str, Any]] = []
    for split in ("calib", "test"):
        for index, text in enumerate(corpus[split]):
            ids = tokenize_sample(tokenizer, text, lengths[index])
            samples.append(
                {
                    "split": split,
                    "index": len(samples),
                    "prompt": text,
                    "length": int(lengths[index]),
                    "token_ids": ids,
                    "token_hash": shards.sample_token_hash(ids),
                    "prompt_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
    return samples


def _sample_manifest_entry(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if key != "token_ids"}


# ---------------------------------------------------------------------------
# Layer / role selection
# ---------------------------------------------------------------------------


def select_layers(num_layers: int, layers: Sequence[int] | None = None) -> list[int]:
    """Default to [0, midpoint, last]; otherwise validate a caller-supplied list."""
    num_layers = int(num_layers)
    if num_layers < 1:
        raise CaptureError(f"model has no layers (num_hidden_layers={num_layers})")
    if layers is None:
        layers = [0, num_layers // 2, num_layers - 1]
    selected: list[int] = []
    for raw in layers:
        index = int(raw)
        if not 0 <= index < num_layers:
            raise CaptureError(
                f"layer index {index} out of range for a {num_layers}-layer model"
            )
        if index not in selected:
            selected.append(index)
    return sorted(selected)


def normalize_linear_roles(
    roles: Sequence[str] | None,
) -> tuple[str, ...]:
    roles = tuple(roles) if roles else DEFAULT_LINEAR_ROLES
    if not roles or any(not isinstance(role, str) or not role for role in roles):
        raise CaptureError("linear roles must be a non-empty list of names")
    return roles


# ---------------------------------------------------------------------------
# Dataset id
# ---------------------------------------------------------------------------


def canonical_capture_config(
    meta: Mapping[str, Any],
    corpus_sha: str,
    lengths: Sequence[int],
    layers: Sequence[int],
    roles: Sequence[str],
    seed: int,
    threads: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "alias": meta.get("alias"),
            "repo_id": meta.get("repo_id"),
            "resolved_revision": meta.get("resolved_revision"),
            "arch": meta.get("arch"),
        },
        "transformers_version": meta.get("transformers_version"),
        "corpus_sha256": corpus_sha,
        "split_lengths": {
            "calib": [int(length) for length in lengths],
            "test": [int(length) for length in lengths],
        },
        "layers": [int(layer) for layer in layers],
        "linear_roles": list(roles),
        "seed": int(seed),
        "threads": int(threads),
    }


def build_dataset_id(canonical: Mapping[str, Any]) -> str:
    blob = json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:DATASET_ID_PREFIX_LEN]


# ---------------------------------------------------------------------------
# Model adapters (architecture dispatch)
# ---------------------------------------------------------------------------


class _BaseAdapter:
    """Structural contract shared by every architecture adapter.

    The model surface is stable: ``model.model.layers[i]`` holds a decoder
    layer whose ``self_attn`` (with ``layer_idx``) and ``mlp`` (with the linear
    roles) are the capture targets.
    """

    arch: str | None = None
    attention_attr = "self_attn"
    linear_parent_attr = "mlp"
    linear_roles: tuple[str, ...] = DEFAULT_LINEAR_ROLES

    def get_layer(self, model: Any, layer_idx: int) -> Any:
        return model.model.layers[int(layer_idx)]

    def get_attention_module(self, model: Any, layer_idx: int) -> Any:
        return getattr(self.get_layer(model, layer_idx), self.attention_attr)

    def get_linear_module(self, model: Any, layer_idx: int, role: str) -> Any:
        """Linear role routing: attention projections live on ``self_attn``
        (q/k/v/o), while the pointwise feed-forward roles live on ``mlp``."""
        layer = self.get_layer(model, layer_idx)
        if role in _ATTENTION_ROLES:
            return getattr(getattr(layer, self.attention_attr), role)
        return getattr(getattr(layer, self.linear_parent_attr), role)

    def attention_meta(self, attention_module: Any, config: Any) -> tuple[int, int, int]:
        q_heads = getattr(attention_module, "num_heads", None) or getattr(
            config, "num_attention_heads", None
        )
        kv_heads = getattr(attention_module, "num_key_value_heads", None) or getattr(
            config, "num_key_value_heads", None
        )
        head_dim = getattr(attention_module, "head_dim", None) or getattr(
            config, "head_dim", None
        )
        if head_dim is None and q_heads:
            head_dim = getattr(config, "hidden_size", 0) // q_heads
        if not (q_heads and kv_heads and head_dim):
            raise CaptureError(
                f"cannot resolve attention heads/head_dim for architecture {self.arch!r}"
            )
        return int(q_heads), int(kv_heads), int(head_dim)

    def validate(self, model: Any) -> None:
        """Cheap structural check that the loaded model matches the adapter."""
        try:
            for role in self.linear_roles:
                self.get_linear_module(model, 0, role)
            attention = self.get_attention_module(model, 0)
            self.attention_meta(attention, model.config)
        except CaptureError:
            raise
        except Exception as error:
            raise CaptureError(
                f"loaded model does not match {self.arch!r} adapter layout: {error}"
            ) from None

    def install_eager_monkeypatch(
        self, callback: Callable[..., None], pending_layers: Sequence[int]
    ) -> Callable[[], None]:
        """Wrap the eager attention kernel; returns a restore function."""
        raise NotImplementedError


class Qwen3Adapter(_BaseAdapter):
    """Real Qwen3 adapter: wraps ``eager_attention_forward`` in the modeling
    module so the wrapper observes query/key/value exactly at the eager kernel
    boundary (post q/k-norm, post RoPE, actual value)."""

    arch = "qwen3"
    modeling_module_name = "transformers.models.qwen3.modeling_qwen3"
    monkeypatch_attr = "eager_attention_forward"

    def install_eager_monkeypatch(
        self, callback: Callable[..., None], pending_layers: Sequence[int]
    ) -> Callable[[], None]:
        try:
            module = importlib.import_module(self.modeling_module_name)
        except Exception as error:
            raise CaptureError(
                f"cannot import {self.modeling_module_name}: {error}"
            ) from error
        original = getattr(module, self.monkeypatch_attr)
        pending = {int(layer) for layer in pending_layers}

        def wrapper(
            attention_module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout: float = 0.0,
            **kwargs,
        ):
            if int(attention_module.layer_idx) in pending:
                callback(attention_module, query, key, value)
            return original(
                attention_module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )

        setattr(module, self.monkeypatch_attr, wrapper)

        def restore() -> None:
            setattr(module, self.monkeypatch_attr, original)

        return restore


class FakeAdapter(_BaseAdapter):
    """Structural adapter for unit-test fake modules.  Fake attention modules
    invoke their ``_rd_attn_cb`` attribute at the same boundary contract as the
    real eager kernel, so no transformers monkeypatching is needed."""

    arch = "fake"

    def install_eager_monkeypatch(
        self, callback: Callable[..., None], pending_layers: Sequence[int]
    ) -> Callable[[], None]:
        return lambda: None


_REAL_ADAPTERS: dict[str, type[_BaseAdapter]] = {
    "qwen3": Qwen3Adapter,
    # qwen2 / llama plug in here: class Qwen2Adapter(_BaseAdapter) with
    # modeling_module_name = "transformers.models.qwen2.modeling_qwen2",
    # monkeypatch_attr = "eager_attention_forward" (same boundary contract).
}

#: Projection roles hosted by the attention module (self_attn).
_ATTENTION_ROLES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})


def get_adapter(arch: str | None, real: bool = True) -> _BaseAdapter:
    if not arch:
        raise CaptureError("model metadata lacks an architecture (model_type)")
    if not real:
        return FakeAdapter()
    try:
        adapter_type = _REAL_ADAPTERS[arch]
    except KeyError:
        raise NotImplementedError(
            f"no real capture adapter for architecture {arch!r}; "
            f"supported: {', '.join(sorted(_REAL_ADAPTERS))}"
        ) from None
    return adapter_type()


# ---------------------------------------------------------------------------
# Real model loader (lazy transformers import; no network)
# ---------------------------------------------------------------------------


def _mask_broken_torchvision() -> None:
    """Neutralize environments whose installed torch/torchvision pair crashes
    on import (the package registration raises before first use).  Masking
    ``torchvision`` before the first ``transformers`` import makes
    ``is_torchvision_available()`` False; Qwen3 text models need no vision.
    Must run before ``transformers`` is imported in this process."""
    if "torchvision" not in sys.modules:
        sys.modules["torchvision"] = None  # type: ignore[assignment]


def _transformers_version() -> str:
    return importlib_metadata.version("transformers")


def load_download_state(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CaptureError(
            f"download state file missing: {path} (run tools/download_models.py first)"
        )
    with open(path, encoding="utf-8") as stream:
        state = json.load(stream)
    if not isinstance(state, dict) or not isinstance(state.get("models"), dict):
        raise CaptureError(f"invalid download state in {path}")
    return state


class RealLoader:
    """Loads the pinned HF snapshot recorded by ``tools/download_models.py``.

    ``metadata()`` is cheap: it reads the download-state record and the
    snapshot ``config.json`` (no model load, no network).  ``load_model()``
    performs the lazy ``transformers`` import and the BF16 CPU model load and
    is called only when a capture actually needs to run.
    """

    def __init__(
        self,
        download_state_path: Path | str = DEFAULT_DOWNLOAD_STATE,
        threads: int = 1,
    ) -> None:
        self.download_state_path = Path(download_state_path)
        self.threads = int(threads)

    def metadata(self, config: Mapping[str, Any]) -> dict[str, Any]:
        alias = config.get("alias", DEFAULT_MODEL_ALIAS)
        state = load_download_state(self.download_state_path)
        try:
            record = state["models"][alias]
        except KeyError:
            raise CaptureError(
                f"no download record for model alias {alias!r} in {self.download_state_path}"
            ) from None
        if record.get("status") != "complete":
            raise CaptureError(
                f"model {alias} download status is {record.get('status')!r}; "
                "expected 'complete' (resolved, pinned snapshot)"
            )
        snapshot = Path(record["snapshot_path"])
        if not snapshot.is_dir():
            raise CaptureError(f"snapshot for {alias} missing: {snapshot}")
        config_path = snapshot / "config.json"
        if not config_path.is_file():
            raise CaptureError(f"snapshot for {alias} has no config.json: {snapshot}")
        with open(config_path, encoding="utf-8") as stream:
            model_config = json.load(stream)
        return {
            "real": True,
            "alias": alias,
            "repo_id": record.get("repo_id") or alias,
            "resolved_revision": record.get("resolved_revision") or snapshot.name,
            "snapshot_path": str(snapshot),
            "arch": model_config.get("model_type"),
            "num_layers": int(model_config.get("num_hidden_layers", 0)),
            "config": model_config,
            "transformers_version": _transformers_version(),
        }

    def load_model(self, config: Mapping[str, Any]) -> tuple[Any, Any]:
        meta = self.metadata(config)
        snapshot = Path(meta["snapshot_path"])
        _mask_broken_torchvision()
        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:  # pragma: no cover - already parallelized
            pass
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        model.eval()
        model.config.use_cache = False
        return model, tokenizer


# ---------------------------------------------------------------------------
# Capture session
# ---------------------------------------------------------------------------


def _as_sample_2d(value: Any) -> torch.Tensor:
    """Normalize a hook input ([seq, in] or [1, seq, in]) to BF16 [seq, in]."""
    tensor = value.detach()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise CaptureError(
            f"linear input must be [seq, in_features], got shape {tuple(tensor.shape)}"
        )
    return tensor.to(torch.bfloat16).contiguous()


def flatten_attention_tensor(
    value: torch.Tensor, heads: int, head_dim: int, seq_len: int
) -> torch.Tensor:
    """Flatten an eager-kernel [1, H, L, D] tensor to BF16 [L, H*D]."""
    tensor = value.detach()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        batch, seen_heads, seen_seq, seen_dim = tensor.shape
        if (seen_heads, seen_seq, seen_dim) != (heads, seq_len, head_dim):
            raise CaptureError(
                f"attention tensor shape {tuple(tensor.shape)} does not match "
                f"[1, {heads}, {seq_len}, {head_dim}]"
            )
        return tensor.transpose(1, 2).reshape(seq_len, heads * head_dim).to(
            torch.bfloat16
        ).contiguous()
    raise CaptureError(
        f"attention tensor must be [1, heads, seq, head_dim], got shape {tuple(tensor.shape)}"
    )


class _CaptureSession:
    """Holds capture state: immediate per-sample atomic temp saves, per-site
    single-call assertions, and per-group assembly."""

    def __init__(
        self,
        *,
        adapter: _BaseAdapter,
        model: Any,
        samples: Sequence[Mapping[str, Any]],
        output_dir: Path,
        pending_linear: Sequence[tuple[int, str]],
        pending_attn: Sequence[int],
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.samples = samples
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / ".tmp-capture"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.pending_linear = set(pending_linear)
        self.pending_attn = set(pending_attn)
        self.current = -1
        self._linear_calls: dict[tuple[tuple[int, str], int], int] = {}
        self._attn_calls: dict[tuple[int, int], int] = {}

    # -- temporary per-sample capture paths --------------------------------

    def linear_tmp(self, site: tuple[int, str], sample_index: int) -> Path:
        layer_idx, role = site
        return self.tmp_dir / f"{sample_index:02d}_{layer_idx}.{role}.pt"

    def attn_tmp(self, layer_idx: int, sample_index: int, kind: str) -> Path:
        return self.tmp_dir / f"{sample_index:02d}_attn_{layer_idx}.{kind}.pt"

    # -- hook / kernel callbacks -------------------------------------------

    def on_linear(self, site: tuple[int, str], value: Any) -> None:
        if site not in self.pending_linear:
            return
        if self.current < 0:
            raise CaptureError("linear hook fired outside a sample forward")
        calls = self._linear_calls.get((site, self.current), 0) + 1
        if calls > 1:
            raise CaptureError(
                f"linear hook fired more than once for site {site!r} in sample {self.current}"
            )
        self._linear_calls[(site, self.current)] = calls
        tensor = _as_sample_2d(value)
        sample = self.samples[self.current]
        if tensor.shape[0] != sample["length"]:
            raise CaptureError(
                f"linear input seq {tensor.shape[0]} != expected sample length "
                f"{sample['length']} for site {site!r}"
            )
        shards.atomic_save_torch(self.linear_tmp(site, self.current), tensor)

    def on_attention(self, attention_module: Any, query: Any, key: Any, value: Any) -> None:
        layer_idx = int(attention_module.layer_idx)
        if layer_idx not in self.pending_attn:
            return
        if self.current < 0:
            raise CaptureError("attention capture fired outside a sample forward")
        calls = self._attn_calls.get((layer_idx, self.current), 0) + 1
        if calls > 1:
            raise CaptureError(
                f"attention capture fired more than once for layer {layer_idx} "
                f"in sample {self.current}"
            )
        self._attn_calls[(layer_idx, self.current)] = calls
        sample = self.samples[self.current]
        q_heads, kv_heads, head_dim = self.adapter.attention_meta(
            attention_module, self.model.config
        )
        flattened = {
            kind: flatten_attention_tensor(
                tensor, heads, head_dim, sample["length"]
            )
            for kind, tensor, heads in (
                ("q", query, q_heads),
                ("k", key, kv_heads),
                ("v", value, kv_heads),
            )
        }
        for kind, tensor in flattened.items():
            shards.atomic_save_torch(
                self.attn_tmp(layer_idx, self.current, kind), tensor
            )

    def check_sample_counts(self, sample_index: int) -> None:
        for site in sorted(self.pending_linear):
            if self._linear_calls.get((site, sample_index)) != 1:
                raise CaptureError(
                    f"expected exactly one linear capture for site {site!r} in "
                    f"sample {sample_index}, got "
                    f"{self._linear_calls.get((site, sample_index), 0)}"
                )
        for layer_idx in sorted(self.pending_attn):
            if self._attn_calls.get((layer_idx, sample_index)) != 1:
                raise CaptureError(
                    f"expected exactly one attention capture for layer {layer_idx} "
                    f"in sample {sample_index}, got "
                    f"{self._attn_calls.get((layer_idx, sample_index), 0)}"
                )


def _install_capture(session: _CaptureSession) -> tuple[list[Any], Callable[[], None]]:
    handles = []
    for site in sorted(session.pending_linear):
        layer_idx, role = site
        module = session.adapter.get_linear_module(session.model, layer_idx, role)
        handles.append(
            module.register_forward_pre_hook(
                lambda _mod, args, _site=site: session.on_linear(_site, args[0])
            )
        )
    for layer_idx in sorted(session.pending_attn):
        attention = session.adapter.get_attention_module(session.model, layer_idx)
        attention._rd_attn_cb = session.on_attention  # fake-model boundary hook
    restore = session.adapter.install_eager_monkeypatch(
        session.on_attention, sorted(session.pending_attn)
    )
    return handles, restore


def _remove_capture(session: _CaptureSession, handles: Sequence[Any], restore: Callable[[], None]) -> None:
    for handle in handles:
        handle.remove()
    restore()
    for layer_idx in session.pending_attn:
        attention = session.adapter.get_attention_module(session.model, layer_idx)
        try:
            attention._rd_attn_cb = None
        except AttributeError:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Assembly and manifest bookkeeping
# ---------------------------------------------------------------------------


def _model_meta_dict(meta: Mapping[str, Any]) -> dict[str, Any]:
    config = meta.get("config") or {}
    return {
        "alias": meta.get("alias"),
        "repo_id": meta.get("repo_id"),
        "resolved_revision": meta.get("resolved_revision"),
        "snapshot_path": meta.get("snapshot_path"),
        "arch": meta.get("arch"),
        "transformers_version": meta.get("transformers_version"),
        "num_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "attn_implementation": "eager",
    }


def _new_manifest(
    meta: Mapping[str, Any],
    dataset_id: str,
    corpus: Mapping[str, Any],
    corpus_sha: str,
    samples: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    roles: Sequence[str],
    seed: int,
    threads: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "status": "in_progress",
        "created_at": _now(),
        "updated_at": _now(),
        "raw_bf16": shards.RAW_BF16_STORAGE,
        "source_modes": shards.SOURCE_MODES,
        "model": _model_meta_dict(meta),
        "corpus": {
            "schema_version": corpus.get("schema_version"),
            "sha256": corpus_sha,
            "tokenization": {"special_tokens": False},
        },
        "samples": [_sample_manifest_entry(sample) for sample in samples],
        "layers": {
            "selected": [int(layer) for layer in layers],
            "linear_roles": list(roles),
        },
        "groups": [],
        "seed": int(seed),
        "runtime": {
            "torch_version": torch.__version__,
            "threads": int(threads),
        },
    }


def _verify_group(group: Mapping[str, Any], output_dir: Path) -> None:
    relative = Path(group.get("path", ""))
    target = output_dir / relative
    if not target.is_file():
        raise shards.ResumeMismatchError(
            f"group {group.get('id')!r}: shard missing: {target}"
        )
    actual = shards.sha256_file(target)
    if actual != group.get("sha256"):
        raise shards.ResumeMismatchError(
            f"group {group.get('id')!r}: shard SHA-256 mismatch "
            f"(manifest {group.get('sha256')}, file {actual})"
        )


def _verify_manifest_groups(manifest: Mapping[str, Any], output_dir: Path) -> None:
    for group in manifest.get("groups", []):
        if group.get("status") == "complete":
            _verify_group(group, output_dir)


def _wipe_output(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _cleanup_tmp(tmp_dir: Path) -> None:
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _linear_group_metadata(
    session: _CaptureSession, site: tuple[int, str], weight: torch.Tensor, in_features: int
) -> dict[str, Any]:
    layer_idx, role = site
    module = session.adapter.get_linear_module(session.model, layer_idx, role)
    return {
        "layer_idx": layer_idx,
        "role": role,
        "in_features": in_features,
        "out_features": int(weight.shape[0]),
        "has_bias": bool(getattr(module, "bias", None) is not None),
        "sample_lengths": [int(sample["length"]) for sample in session.samples],
        "sample_token_hashes": [sample["token_hash"] for sample in session.samples],
    }


def _assemble_linear(
    session: _CaptureSession, site: tuple[int, str]
) -> tuple[Path, str, dict[str, Any]]:
    layer_idx, role = site
    module = session.adapter.get_linear_module(session.model, layer_idx, role)
    weight = module.weight.detach().to(torch.bfloat16).contiguous()
    tensors = [
        shards.load_tensor(session.linear_tmp(site, index))
        for index in range(len(session.samples))
    ]
    in_features = int(weight.shape[1])
    metadata = _linear_group_metadata(session, site, weight, in_features)
    shard = shards.build_linear_shard(
        metadata, weight, tensors[:5], tensors[5:]
    )
    shards.validate_linear_shard(shard)
    relative = Path("linear") / f"{layer_idx}.{role}.pt"
    digest = shards.atomic_save_torch(session.output_dir / relative, shard)
    for index in range(len(session.samples)):
        try:
            session.linear_tmp(site, index).unlink()
        except FileNotFoundError:
            pass
    return relative, digest, metadata


def _attention_group_metadata(
    session: _CaptureSession, layer_idx: int
) -> dict[str, Any]:
    return {
        "layer_idx": layer_idx,
        "arch": session.adapter.arch,
        "sample_lengths": [int(sample["length"]) for sample in session.samples],
        "sample_token_hashes": [sample["token_hash"] for sample in session.samples],
    }


def _assemble_attention(
    session: _CaptureSession, layer_idx: int
) -> tuple[Path, str, dict[str, Any]]:
    attention = session.adapter.get_attention_module(session.model, layer_idx)
    q_heads, kv_heads, head_dim = session.adapter.attention_meta(
        attention, session.model.config
    )
    calib: list[dict[str, torch.Tensor]] = []
    test: list[dict[str, torch.Tensor]] = []
    for index in range(len(session.samples)):
        sample = {
            kind: shards.load_tensor(session.attn_tmp(layer_idx, index, kind))
            for kind in ("q", "k", "v")
        }
        (calib if index < 5 else test).append(sample)
    metadata = _attention_group_metadata(session, layer_idx)
    shard = shards.build_attention_shard(
        metadata, q_heads, kv_heads, head_dim, calib, test
    )
    shards.validate_attention_shard(shard)
    relative = Path("attention") / f"{layer_idx}.self_attn.pt"
    digest = shards.atomic_save_torch(session.output_dir / relative, shard)
    for index in range(len(session.samples)):
        for kind in ("q", "k", "v"):
            try:
                session.attn_tmp(layer_idx, index, kind).unlink()
            except FileNotFoundError:
                pass
    return relative, digest, metadata


def _write_manifest_with_groups(
    manifest: dict[str, Any],
    manifest_path: Path,
    groups: Mapping[str, dict[str, Any]],
    status: str,
) -> None:
    manifest["groups"] = [groups[key] for key in sorted(groups)]
    manifest["status"] = status
    manifest["updated_at"] = _now()
    shards.write_manifest(manifest_path, manifest)


def _summary(
    dataset_id: str,
    output_dir: Path,
    manifest: Mapping[str, Any],
    reused: bool,
) -> dict[str, Any]:
    groups = manifest.get("groups", [])
    return {
        "dataset_id": dataset_id,
        "output_dir": str(output_dir),
        "status": manifest.get("status"),
        "reused": bool(reused),
        "samples": len(manifest.get("samples", [])),
        "groups": {
            "linear": sum(1 for group in groups if group.get("kind") == "linear"),
            "attention": sum(1 for group in groups if group.get("kind") == "attention"),
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_capture(
    config: Mapping[str, Any],
    loader: Any = None,
) -> dict[str, Any]:
    """Run (or resume) a capture.  ``loader`` exposes ``metadata(config)`` and
    ``load_model(config)``; defaults to :class:`RealLoader`."""
    if loader is None:
        loader = RealLoader(
            download_state_path=config.get("download_state_path", DEFAULT_DOWNLOAD_STATE),
            threads=int(config.get("threads", 1)),
        )
    output_root = Path(config.get("output_root", DEFAULT_OUTPUT_ROOT))
    smoke = bool(config.get("smoke", False))
    force = bool(config.get("force", False))
    seed = int(config.get("seed", 0))
    threads = int(config.get("threads", 1))
    if threads < 1:
        raise CaptureError(f"threads must be positive, got {threads}")
    lengths = SMOKE_LENGTHS if smoke else FULL_LENGTHS

    corpus = load_corpus()
    corpus_sha = corpus_sha256(corpus)

    meta = loader.metadata(config)
    adapter = get_adapter(meta.get("arch"), real=meta.get("real", True))
    num_layers = int(meta.get("num_layers") or 0)
    if num_layers < 1:
        raise CaptureError(
            f"metadata for {meta.get('alias')!r} lacks a positive num_hidden_layers"
        )
    layers = select_layers(num_layers, config.get("layers"))
    roles = normalize_linear_roles(config.get("linear_roles"))

    canonical = canonical_capture_config(
        meta, corpus_sha, lengths, layers, roles, seed, threads
    )
    dataset_id = build_dataset_id(canonical)
    output_dir = output_root / "real-captures" / dataset_id
    manifest_path = output_dir / "manifest.json"

    existing = shards.load_manifest(manifest_path)
    if existing is not None:
        if existing.get("dataset_id") != dataset_id:
            raise CaptureError(
                f"manifest dataset_id {existing.get('dataset_id')!r} does not match "
                f"derived {dataset_id!r}; remove {output_dir} to restart"
            )
        if existing.get("corpus", {}).get("sha256") != corpus_sha:
            raise CaptureError(
                f"manifest corpus SHA-256 changed for dataset {dataset_id}; "
                "the corpus file must stay fixed"
            )
        captured_torch = (existing.get("runtime") or {}).get("torch_version")
        if captured_torch != torch.__version__:
            raise shards.ResumeMismatchError(
                f"dataset {dataset_id} was captured with torch {captured_torch!r}, "
                f"but the current version is {torch.__version__!r}; use a separate "
                "output root rather than mixing runtime versions"
            )
        if existing.get("status") == "complete" and not force:
            _verify_manifest_groups(existing, output_dir)
            return _summary(dataset_id, output_dir, existing, reused=True)

    completed: dict[str, dict[str, Any]] = {}
    if existing is not None and not force:
        for group in existing.get("groups", []):
            if group.get("status") == "complete":
                try:
                    _verify_group(group, output_dir)
                except shards.ResumeMismatchError as error:
                    raise shards.ResumeMismatchError(
                        f"{error}; rerun with --force to discard and restart"
                    ) from None
                completed[group["id"]] = group
    else:
        _wipe_output(output_dir)

    pending_linear = [
        site
        for site in ((layer, role) for layer in layers for role in roles)
        if f"{site[0]}.{site[1]}" not in completed
    ]
    pending_attn = [
        layer for layer in layers if f"{layer}.self_attn" not in completed
    ]
    if not pending_linear and not pending_attn:
        manifest = existing or _new_manifest(
            meta, dataset_id, corpus, corpus_sha, [], layers, roles, seed, threads
        )
        _write_manifest_with_groups(manifest, manifest_path, completed, "complete")
        return _summary(dataset_id, output_dir, manifest, reused=False)

    torch.manual_seed(seed)
    model, tokenizer = loader.load_model(config)
    samples = make_samples(tokenizer, corpus, lengths)
    adapter.validate(model)

    session = _CaptureSession(
        adapter=adapter,
        model=model,
        samples=samples,
        output_dir=output_dir,
        pending_linear=pending_linear,
        pending_attn=pending_attn,
    )
    manifest = _new_manifest(
        meta, dataset_id, corpus, corpus_sha, samples, layers, roles, seed, threads
    )
    if existing is not None and existing.get("created_at"):
        manifest["created_at"] = existing["created_at"]
    manifest["groups"] = [completed[key] for key in sorted(completed)]
    shards.write_manifest(manifest_path, manifest)

    try:
        with torch.inference_mode():
            handles, restore = _install_capture(session)
            try:
                for sample in samples:
                    session.current = sample["index"]
                    input_ids = torch.tensor(
                        sample["token_ids"], dtype=torch.long
                    ).unsqueeze(0)
                    model(input_ids=input_ids, use_cache=False)
                    session.check_sample_counts(sample["index"])
            finally:
                _remove_capture(session, handles, restore)

        groups = dict(completed)
        for site in sorted(pending_linear):
            relative, digest, metadata = _assemble_linear(session, site)
            group_id = f"{site[0]}.{site[1]}"
            groups[group_id] = {
                "kind": "linear",
                "id": group_id,
                "path": str(relative),
                "sha256": digest,
                "metadata": metadata,
                "status": "complete",
            }
            _write_manifest_with_groups(manifest, manifest_path, groups, "in_progress")
        for layer_idx in sorted(pending_attn):
            relative, digest, metadata = _assemble_attention(session, layer_idx)
            group_id = f"{layer_idx}.self_attn"
            groups[group_id] = {
                "kind": "attention",
                "id": group_id,
                "path": str(relative),
                "sha256": digest,
                "metadata": metadata,
                "status": "complete",
            }
            _write_manifest_with_groups(manifest, manifest_path, groups, "in_progress")
        _cleanup_tmp(session.tmp_dir)
        _write_manifest_with_groups(manifest, manifest_path, groups, "complete")
    except BaseException:
        _cleanup_tmp(session.tmp_dir)
        raise

    return _summary(dataset_id, output_dir, manifest, reused=False)
