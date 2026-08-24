"""Unit tests for the real-model raw-BF16 capture pipeline.

Uses tiny fake tokenizer/model/modules plus a fake loader; ``transformers`` is
never imported and no model weights are loaded.  The subprocess smoke test
additionally sets ``/proc/self/oom_score_adj=500`` best-effort.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import capture_real
from tools.realdata import capture, shards
from tools.realdata.capture import (
    CaptureError,
    DEFAULT_LINEAR_ROLES,
    FULL_LENGTHS,
    SMOKE_LENGTHS,
    build_dataset_id,
    canonical_capture_config,
    flatten_attention_tensor,
    make_samples,
    run_capture,
    select_layers,
)
from tools.realdata.shards import ResumeMismatchError

FAKE_REVISION = "a" * 40
FAKE_TRANSFORMERS = "4.57.6-test"

#: The Qwen3.5-2B snapshot's hybrid layer pattern: 24 layers, full attention
#: on [3, 7, 11, 15, 19, 23], heads 8:2 with head_dim 256.
QWEN35_LAYER_TYPES = [
    layer_type
    for index in range(24)
    for layer_type in (
        ("full_attention",) if index % 4 == 3 else ("linear_attention",)
    )
]
QWEN35_FULL_ATTENTION = [
    index for index, layer_type in enumerate(QWEN35_LAYER_TYPES)
    if layer_type == "full_attention"
]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic tokenizer: id count and values derive from the text."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        count = 4 + digest[0] % 9  # 4..12 tokens
        ids = [(digest[index] * 0x9E3779B1 + index) % 4096 for index in range(count)]
        return {"input_ids": ids}


class FakeLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = torch.linspace(
            -0.5, 0.5, in_features * out_features, dtype=torch.bfloat16
        ).reshape(out_features, in_features)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, None)


class FakeMLP(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = FakeLinear(hidden, intermediate)
        self.up_proj = FakeLinear(hidden, intermediate)
        self.down_proj = FakeLinear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class FakeAttention(torch.nn.Module):
    """Mimics the real eager-kernel boundary: ``_rd_attn_cb(self, q, k, v)``
    receives q/k/v as [1, heads, seq, head_dim]."""

    def __init__(
        self,
        layer_idx: int,
        hidden: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = q_heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = FakeLinear(hidden, q_heads * head_dim)
        self.k_proj = FakeLinear(hidden, kv_heads * head_dim)
        self.v_proj = FakeLinear(hidden, kv_heads * head_dim)
        self.o_proj = FakeLinear(q_heads * head_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[0]
        q = self.q_proj(x).view(1, self.num_heads, seq, self.head_dim)
        k = self.k_proj(x).view(1, self.num_key_value_heads, seq, self.head_dim)
        v = self.v_proj(x).view(1, self.num_key_value_heads, seq, self.head_dim)
        callback = getattr(self, "_rd_attn_cb", None)
        if callback is not None:
            callback(self, q, k, v)
        return self.o_proj(q.reshape(seq, -1))


class DoubleFireAttention(FakeAttention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        return super().forward(x)


class DoubleFireMLP(FakeMLP):
    """Calls ``up_proj`` twice per forward so its pre-hook double-fires."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.up_proj(x)
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class FakeLinearAttention(torch.nn.Module):
    """Stand-in for Qwen3_5GatedDeltaNet: a linear-attention token mixer with
    no eager-kernel attention and no q/k/v/o projections."""

    def __init__(self, layer_idx: int, hidden: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.in_proj = FakeLinear(hidden, hidden)
        self.out_proj = FakeLinear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.in_proj(x))


class FakeDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden: int,
        intermediate: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        layer_type: str = "full_attention",
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = FakeAttention(layer_idx, hidden, q_heads, kv_heads, head_dim)
        else:
            self.linear_attn = FakeLinearAttention(layer_idx, hidden)
        self.mlp = FakeMLP(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "full_attention":
            x = self.self_attn(x)
        else:
            x = self.linear_attn(x)
        return self.mlp(x)


class _Layers:
    def __init__(self, layers):
        self.layers = layers


class FakeModel(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        hidden: int = 128,
        intermediate: int = 256,
        q_heads: int = 4,
        kv_heads: int = 2,
        head_dim: int = 32,
        layer_types: list[str] | None = None,
    ) -> None:
        super().__init__()
        if layer_types is not None:
            layer_types = list(layer_types)
            if len(layer_types) != num_layers:
                raise ValueError("fake layer_types length must equal num_layers")
        self.config = types.SimpleNamespace(
            model_type="qwen3_5" if layer_types is not None else "qwen3",
            num_hidden_layers=num_layers,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_attention_heads=q_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=4096,
            attention_bias=False,
            rms_norm_eps=1e-6,
        )
        if layer_types is not None:
            self.config.layer_types = layer_types
            self.config.text_config = types.SimpleNamespace(
                model_type="qwen3_5_text",
                num_hidden_layers=num_layers,
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_attention_heads=q_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
                layer_types=layer_types,
            )
        self.model = _Layers(
            [
                FakeDecoderLayer(
                    i,
                    hidden,
                    intermediate,
                    q_heads,
                    kv_heads,
                    head_dim,
                    layer_type=(
                        layer_types[i] if layer_types is not None else "full_attention"
                    ),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        seq = int(input_ids.shape[-1])
        generator = torch.Generator().manual_seed(0x5EED + seq)
        x = torch.randn(
            seq, self.config.hidden_size, dtype=torch.bfloat16, generator=generator
        )
        for layer in self.model.layers:
            x = layer(x)
        return types.SimpleNamespace(logits=x)


def replace_module(model: FakeModel, layer_idx: int, path: str, module) -> None:
    target = model.model.layers[layer_idx]
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], module)


class FakeLoader:
    """In-process stand-in for RealLoader; records load_model calls."""

    def __init__(self, model=None, tokenizer=None, meta_overrides=None) -> None:
        self.model = model if model is not None else FakeModel()
        self.tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
        self.meta_overrides = dict(meta_overrides or {})
        self.metadata_calls = 0
        self.load_calls: list[dict] = []

    def metadata(self, config: dict) -> dict:
        self.metadata_calls += 1
        model_config = self.model.config
        cfg = {
            "model_type": model_config.model_type,
            "num_hidden_layers": model_config.num_hidden_layers,
            "hidden_size": model_config.hidden_size,
            "intermediate_size": model_config.intermediate_size,
            "num_attention_heads": model_config.num_attention_heads,
            "num_key_value_heads": model_config.num_key_value_heads,
            "head_dim": model_config.head_dim,
            "vocab_size": model_config.vocab_size,
        }
        meta = {
            "real": False,
            "alias": config.get("alias", "fake-model"),
            "repo_id": "tests/fake-model",
            "resolved_revision": FAKE_REVISION,
            "snapshot_path": "(fake)",
            "arch": model_config.model_type,
            "num_layers": cfg["num_hidden_layers"],
            "config": cfg,
            "transformers_version": FAKE_TRANSFORMERS,
        }
        layer_types = getattr(model_config, "layer_types", None)
        if layer_types is not None:
            meta["layer_types"] = list(layer_types)
            meta["full_attention"] = [
                index
                for index, layer_type in enumerate(layer_types)
                if layer_type == "full_attention"
            ]
        meta.update(self.meta_overrides)
        return meta

    def load_model(self, config: dict):
        self.load_calls.append(dict(config))
        return self.model, self.tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class CorpusAndTokenizationTest(unittest.TestCase):
    def test_corpus_has_ten_distinct_texts_five_per_split(self) -> None:
        corpus = capture.load_corpus()
        self.assertEqual(len(corpus["calib"]), 5)
        self.assertEqual(len(corpus["test"]), 5)
        self.assertEqual(len(set(corpus["calib"] + corpus["test"])), 10)

    def test_tokenization_exact_lengths_full_and_smoke(self) -> None:
        tokenizer = FakeTokenizer()
        corpus = capture.load_corpus()
        for pattern in (FULL_LENGTHS, SMOKE_LENGTHS):
            samples = make_samples(tokenizer, corpus, pattern)
            self.assertEqual(len(samples), 10)
            for split_index, split in enumerate(("calib", "test")):
                for index in range(5):
                    sample = samples[split_index * 5 + index]
                    self.assertEqual(sample["split"], split)
                    self.assertEqual(sample["length"], pattern[index])
                    self.assertEqual(len(sample["token_ids"]), pattern[index])
                    self.assertEqual(sample["prompt"], corpus[split][index])

    def test_token_hashes_distinct_and_deterministic(self) -> None:
        tokenizer = FakeTokenizer()
        corpus = capture.load_corpus()
        samples = make_samples(tokenizer, corpus, SMOKE_LENGTHS)
        hashes = [sample["token_hash"] for sample in samples]
        self.assertEqual(len(set(hashes)), 10, "sample token hashes must be distinct")
        again = make_samples(tokenizer, corpus, SMOKE_LENGTHS)
        self.assertEqual(
            [sample["token_hash"] for sample in again], hashes, "tokenization must be deterministic"
        )
        # hash is a stable function of the ids
        self.assertEqual(
            hashes[0],
            shards.sample_token_hash(samples[0]["token_ids"]),
        )


class DatasetIdTest(unittest.TestCase):
    def _meta(self, **overrides) -> dict:
        meta = {
            "alias": "qwen3-0.6b",
            "repo_id": "Qwen/Qwen3-0.6B",
            "resolved_revision": "c" * 40,
            "arch": "qwen3",
            "transformers_version": "4.57.6",
        }
        meta.update(overrides)
        return meta

    def test_dataset_id_deterministic_and_sensitive(self) -> None:
        corpus_sha = capture.corpus_sha256(capture.load_corpus())
        kwargs = dict(
            meta=self._meta(),
            corpus_sha=corpus_sha,
            lengths=FULL_LENGTHS,
            layers=[0, 14, 27],
            roles=DEFAULT_LINEAR_ROLES,
            seed=0,
        )
        first = build_dataset_id(canonical_capture_config(**kwargs))
        canonical = canonical_capture_config(**kwargs)
        self.assertEqual(first, build_dataset_id(canonical))
        self.assertEqual(len(first), 16)
        sensitive = {
            "corpus_sha": "deadbeef",
            "lengths": SMOKE_LENGTHS,
            "layers": [0, 1],
            "roles": ("q_proj", "o_proj"),
            "seed": 7,
            "threads": 4,
            "meta": self._meta(resolved_revision="d" * 40),
            "meta_transformers": self._meta(transformers_version="4.57.5"),
        }
        seen = {first}
        for key, change in sensitive.items():
            current = dict(kwargs)
            if key == "meta":
                current["meta"] = change
            elif key == "meta_transformers":
                current["meta"] = change
            else:
                current[key] = change
            dataset_id = build_dataset_id(canonical_capture_config(**current))
            self.assertNotIn(dataset_id, seen, f"config change {key} must alter the id")
            seen.add(dataset_id)

    def test_select_layers_defaults_and_validation(self) -> None:
        self.assertEqual(select_layers(28), [0, 14, 27])
        self.assertEqual(select_layers(3), [0, 1, 2])
        self.assertEqual(select_layers(5, [4, 4, 0, 2]), [0, 2, 4])
        with self.assertRaises(CaptureError):
            select_layers(3, [0, 3])
        with self.assertRaises(CaptureError):
            select_layers(3, [-1])

    def test_select_layers_hybrid_default_is_full_attention_anchored(self) -> None:
        # The Qwen3.5-2B snapshot: 24 layers with full attention on
        # [3, 7, 11, 15, 19, 23]; conventional midpoint is 12, whose nearest
        # full-attention layer is 11 -> default [first, nearest, last] = [3, 11, 23].
        self.assertEqual(
            select_layers(24, layer_types=QWEN35_LAYER_TYPES),
            [3, 11, 23],
        )
        # A 12-layer hybrid with full attention on [0, 5, 11]: midpoint 6,
        # nearest full layer 5.
        self.assertEqual(
            select_layers(
                12,
                layer_types=["full_attention", "linear_attention", "linear_attention",
                             "linear_attention", "linear_attention", "full_attention",
                             "linear_attention", "linear_attention", "linear_attention",
                             "linear_attention", "linear_attention", "full_attention"],
            ),
            [0, 5, 11],
        )

    def test_select_layers_hybrid_default_all_full_or_all_linear(self) -> None:
        # Every layer full-attention: identical to the conventional default.
        self.assertEqual(
            select_layers(28, layer_types=["full_attention"] * 28),
            [0, 14, 27],
        )
        # No full-attention layer: fall back to the conventional default (MLP
        # roles are still capturable everywhere).
        self.assertEqual(
            select_layers(24, layer_types=["linear_attention"] * 24),
            [0, 12, 23],
        )

    def test_select_layers_explicit_list_ignores_layer_types(self) -> None:
        # Explicit layers may include linear-attention layers and are validated
        # against the layer count only.
        self.assertEqual(
            select_layers(24, layers=[1, 3, 5], layer_types=QWEN35_LAYER_TYPES),
            [1, 3, 5],
        )
        with self.assertRaises(CaptureError):
            select_layers(24, layers=[24], layer_types=QWEN35_LAYER_TYPES)


class AtomicStorageTest(unittest.TestCase):
    def test_atomic_json_and_torch_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "nested" / "manifest.json"
            shards.atomic_write_json(json_path, {"a": 1, "b": [2, 3]})
            self.assertEqual(json.loads(json_path.read_text()), {"a": 1, "b": [2, 3]})
            shards.atomic_write_json(json_path, {"a": 2})
            self.assertEqual(json.loads(json_path.read_text()), {"a": 2})

            tensor = torch.ones(4, 8, dtype=torch.bfloat16)
            tensor_path = root / "nested" / "g.pt"
            digest = shards.atomic_save_torch(tensor_path, tensor)
            self.assertEqual(digest, shards.sha256_file(tensor_path))
            loaded = shards.load_tensor(tensor_path)
            self.assertTrue(torch.equal(loaded, tensor))

            leftovers = [
                p for p in root.rglob(".tmp-*")
            ] + [p for p in root.rglob("*.tmp-*")]
            self.assertEqual(leftovers, [], "no temporary files may remain")

    def test_manifest_schema_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"schema_version": 999}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
                shards.load_manifest(path)


class RealAdapterAndLoaderTest(unittest.TestCase):
    def test_qwen3_eager_wrapper_captures_and_restores(self) -> None:
        calls = []

        def original(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            calls.append((module, query, key, value, attention_mask, scaling, dropout, kwargs))
            return "attention-output"

        modeling = types.SimpleNamespace(eager_attention_forward=original)
        callback_calls = []
        adapter = capture.Qwen3Adapter()
        with mock.patch.object(capture.importlib, "import_module", return_value=modeling):
            restore = adapter.install_eager_monkeypatch(
                lambda *args: callback_calls.append(args), [14]
            )
        attention = types.SimpleNamespace(layer_idx=14)
        q, k, v = object(), object(), object()
        result = modeling.eager_attention_forward(
            attention, q, k, v, "mask", 0.125, dropout=0.25, sliding_window=32
        )
        self.assertEqual(result, "attention-output")
        self.assertEqual(callback_calls, [(attention, q, k, v)])
        self.assertEqual(calls[0][4:], ("mask", 0.125, 0.25, {"sliding_window": 32}))
        restore()
        self.assertIs(modeling.eager_attention_forward, original)

    def test_qwen2_eager_wrapper_captures_and_restores(self) -> None:
        calls = []

        def original(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            calls.append((module, query, key, value, attention_mask, scaling, dropout, kwargs))
            return "qwen2-attention-output"

        modeling = types.SimpleNamespace(eager_attention_forward=original)
        callback_calls = []
        adapter = capture.Qwen2Adapter()
        self.assertEqual(adapter.arch, "qwen2")
        self.assertEqual(
            adapter.modeling_module_name, "transformers.models.qwen2.modeling_qwen2"
        )
        with mock.patch.object(capture.importlib, "import_module", return_value=modeling):
            restore = adapter.install_eager_monkeypatch(
                lambda *args: callback_calls.append(args), [7]
            )
        attention = types.SimpleNamespace(layer_idx=7)
        q, k, v = object(), object(), object()
        result = modeling.eager_attention_forward(
            attention, q, k, v, "mask", 0.125, dropout=0.0, sliding_window=None
        )
        self.assertEqual(result, "qwen2-attention-output")
        self.assertEqual(callback_calls, [(attention, q, k, v)])
        # qwen2 passes sliding_window through **kwargs; the wrapper must preserve it
        self.assertEqual(calls[0][4:], ("mask", 0.125, 0.0, {"sliding_window": None}))
        # layers outside the pending set are not captured
        skipped = types.SimpleNamespace(layer_idx=99)
        modeling.eager_attention_forward(skipped, q, k, v, None, 0.125)
        self.assertEqual(callback_calls, [(attention, q, k, v)])
        restore()
        self.assertIs(modeling.eager_attention_forward, original)

    def test_llama_eager_wrapper_captures_and_restores(self) -> None:
        calls = []

        def original(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            calls.append((module, query, key, value, attention_mask, scaling, dropout, kwargs))
            return "llama-attention-output"

        modeling = types.SimpleNamespace(eager_attention_forward=original)
        callback_calls = []
        adapter = capture.LlamaAdapter()
        self.assertEqual(adapter.arch, "llama")
        self.assertEqual(
            adapter.modeling_module_name, "transformers.models.llama.modeling_llama"
        )
        with mock.patch.object(capture.importlib, "import_module", return_value=modeling):
            restore = adapter.install_eager_monkeypatch(
                lambda *args: callback_calls.append(args), [14]
            )
        attention = types.SimpleNamespace(layer_idx=14)
        q, k, v = object(), object(), object()
        # llama does not pass sliding_window
        result = modeling.eager_attention_forward(attention, q, k, v, None, 0.0625)
        self.assertEqual(result, "llama-attention-output")
        self.assertEqual(callback_calls, [(attention, q, k, v)])
        self.assertEqual(calls[0][4:], (None, 0.0625, 0.0, {}))
        restore()
        self.assertIs(modeling.eager_attention_forward, original)

    def test_eager_wrapper_ignores_modules_without_layer_idx(self) -> None:
        # transformers 5.x modules (e.g. Qwen3_5VisionAttention) share the
        # module-level eager_attention_forward but carry no layer_idx; the
        # wrapper must pass them through untouched instead of crashing.
        calls = []

        def original(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            calls.append(module)
            return "attention-output"

        modeling = types.SimpleNamespace(eager_attention_forward=original)
        callback_calls = []
        adapter = capture.Qwen3Adapter()
        with mock.patch.object(capture.importlib, "import_module", return_value=modeling):
            restore = adapter.install_eager_monkeypatch(
                lambda *args: callback_calls.append(args), [3]
            )
        q, k, v = object(), object(), object()
        vision = types.SimpleNamespace()  # no layer_idx attribute
        result = modeling.eager_attention_forward(vision, q, k, v, None, 0.125)
        self.assertEqual(result, "attention-output")
        self.assertEqual(calls, [vision])
        self.assertEqual(callback_calls, [])
        restore()
        self.assertIs(modeling.eager_attention_forward, original)

    def test_attention_meta_falls_back_to_config_for_qwen2_llama_modules(self) -> None:
        # Real Qwen2Attention/LlamaAttention do not store num_heads; the base
        # adapter must resolve heads from the config and head_dim from the module.
        qwen2_module = types.SimpleNamespace(head_dim=128, num_key_value_groups=6)
        qwen2_config = types.SimpleNamespace(
            num_attention_heads=12, num_key_value_heads=2, hidden_size=1536
        )
        self.assertEqual(
            capture.Qwen2Adapter().attention_meta(qwen2_module, qwen2_config),
            (12, 2, 128),
        )
        llama_module = types.SimpleNamespace(head_dim=64)
        llama_config = types.SimpleNamespace(
            num_attention_heads=32, num_key_value_heads=32, hidden_size=2048
        )
        self.assertEqual(
            capture.LlamaAdapter().attention_meta(llama_module, llama_config),
            (32, 32, 64),
        )

    def test_qwen2_and_llama_metadata_and_adapter_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {}
            for alias, model_type, num_layers in (
                ("deepseek-r1-qwen-1.5b", "qwen2", 28),
                ("smollm2-1.7b", "llama", 24),
            ):
                snapshot = root / f"snapshot-{alias}"
                snapshot.mkdir()
                (snapshot / "config.json").write_text(
                    json.dumps(
                        {"model_type": model_type, "num_hidden_layers": num_layers}
                    ),
                    encoding="utf-8",
                )
                entries[alias] = {
                    "status": "complete",
                    "repo_id": f"tests/{alias}",
                    "resolved_revision": FAKE_REVISION,
                    "snapshot_path": str(snapshot),
                }
            state = root / "state.json"
            state.write_text(json.dumps({"models": entries}), encoding="utf-8")
            loader = capture.RealLoader(state)
            with mock.patch.object(
                capture.importlib_metadata, "version", return_value="4.57.6"
            ):
                qwen2_meta = loader.metadata({"alias": "deepseek-r1-qwen-1.5b"})
                llama_meta = loader.metadata({"alias": "smollm2-1.7b"})
            self.assertEqual(qwen2_meta["arch"], "qwen2")
            self.assertEqual(qwen2_meta["num_layers"], 28)
            self.assertEqual(llama_meta["arch"], "llama")
            self.assertEqual(llama_meta["num_layers"], 24)
            self.assertIsInstance(
                capture.get_adapter(qwen2_meta["arch"]), capture.Qwen2Adapter
            )
            self.assertIsInstance(
                capture.get_adapter(llama_meta["arch"]), capture.LlamaAdapter
            )
            self.assertIn("qwen2", capture._REAL_ADAPTERS)
            self.assertIn("llama", capture._REAL_ADAPTERS)
            with self.assertRaisesRegex(NotImplementedError, "supported: llama, qwen2, qwen3"):
                capture.get_adapter("unknown-arch")

    def test_real_loader_metadata_reads_pinned_snapshot_without_transformers_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps({"model_type": "qwen3", "num_hidden_layers": 28}),
                encoding="utf-8",
            )
            state = root / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen3-0.6b": {
                                "status": "complete",
                                "repo_id": "Qwen/Qwen3-0.6B",
                                "resolved_revision": FAKE_REVISION,
                                "snapshot_path": str(snapshot),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            loader = capture.RealLoader(state)
            with mock.patch.object(
                capture.importlib_metadata, "version", return_value="4.57.6"
            ):
                meta = loader.metadata({"alias": "qwen3-0.6b"})
            self.assertEqual(meta["arch"], "qwen3")
            self.assertEqual(meta["num_layers"], 28)
            self.assertEqual(meta["resolved_revision"], FAKE_REVISION)
            self.assertEqual(meta["transformers_version"], "4.57.6")

    def test_real_loader_metadata_reads_nested_text_config_for_qwen35(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "text_config": {
                            "model_type": "qwen3_5_text",
                            "num_hidden_layers": 24,
                            "layer_types": QWEN35_LAYER_TYPES,
                            "num_attention_heads": 8,
                            "num_key_value_heads": 2,
                            "head_dim": 256,
                            "hidden_size": 2048,
                        },
                    }
                ),
                encoding="utf-8",
            )
            state = root / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen3.5-2b": {
                                "status": "complete",
                                "repo_id": "Qwen/Qwen3.5-2B",
                                "resolved_revision": FAKE_REVISION,
                                "snapshot_path": str(snapshot),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            loader = capture.RealLoader(state)
            with mock.patch.object(
                capture.importlib_metadata, "version", return_value="5.2.0"
            ):
                meta = loader.metadata({"alias": "qwen3.5-2b"})
            self.assertEqual(meta["arch"], "qwen3_5")
            self.assertEqual(meta["num_layers"], 24)
            self.assertEqual(meta["layer_types"], QWEN35_LAYER_TYPES)
            self.assertEqual(meta["full_attention"], QWEN35_FULL_ATTENTION)
            self.assertEqual(
                meta["config"]["text_config"]["num_attention_heads"], 8
            )

    def test_real_loader_metadata_flat_config_is_unchanged(self) -> None:
        # A flat qwen3 snapshot (no text_config) must keep the legacy metadata.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps({"model_type": "qwen3", "num_hidden_layers": 28}),
                encoding="utf-8",
            )
            state = root / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen3-0.6b": {
                                "status": "complete",
                                "repo_id": "Qwen/Qwen3-0.6B",
                                "resolved_revision": FAKE_REVISION,
                                "snapshot_path": str(snapshot),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            loader = capture.RealLoader(state)
            with mock.patch.object(
                capture.importlib_metadata, "version", return_value="4.57.6"
            ):
                meta = loader.metadata({"alias": "qwen3-0.6b"})
            self.assertEqual(meta["arch"], "qwen3")
            self.assertEqual(meta["num_layers"], 28)
            self.assertEqual(meta["layer_types"], [])
            self.assertEqual(meta["full_attention"], [])

    def test_real_loader_uses_supported_model_kwargs_and_disables_cache(self) -> None:
        recorded = {}
        model = mock.Mock()
        model.config = types.SimpleNamespace(use_cache=True)

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                recorded["tokenizer"] = (path, kwargs)
                return "tokenizer"

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(path, **kwargs):
                recorded["model"] = (path, kwargs)
                return model

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer
        transformers.AutoModelForCausalLM = AutoModelForCausalLM
        loader = capture.RealLoader(threads=1)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            loader,
            "metadata",
            return_value={"snapshot_path": directory},
        ), mock.patch.dict(sys.modules, {"transformers": transformers}):
            loaded_model, tokenizer = loader.load_model({})
        self.assertIs(loaded_model, model)
        self.assertEqual(tokenizer, "tokenizer")
        kwargs = recorded["model"][1]
        self.assertNotIn("use_cache", kwargs)
        self.assertEqual(kwargs["dtype"], torch.bfloat16)
        self.assertEqual(kwargs["attn_implementation"], "eager")
        self.assertFalse(model.config.use_cache)


class Qwen3_5AdapterTest(unittest.TestCase):
    """Weight-free unit tests for the Qwen3.5 hybrid text adapter."""

    def _hybrid_model(self, num_layers: int = 8, layer_types: list[str] | None = None) -> FakeModel:
        pattern = layer_types or [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ][:num_layers]
        return FakeModel(num_layers=num_layers, layer_types=pattern)

    def test_adapter_registration_and_module_name(self) -> None:
        self.assertIn("qwen3_5", capture._REAL_ADAPTERS)
        adapter = capture.get_adapter("qwen3_5")
        self.assertIsInstance(adapter, capture.Qwen3_5Adapter)
        self.assertEqual(
            adapter.modeling_module_name,
            "transformers.models.qwen3_5.modeling_qwen3_5",
        )
        self.assertEqual(adapter.arch, "qwen3_5")

    def test_attention_meta_reads_nested_text_config(self) -> None:
        # The real Qwen3_5Attention stores head_dim but not num_heads; the
        # adapter must resolve 8:2 hd256 from the nested text_config.
        module = types.SimpleNamespace(head_dim=256, num_key_value_groups=4)
        config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=256,
                hidden_size=2048,
            )
        )
        self.assertEqual(
            capture.Qwen3_5Adapter().attention_meta(module, config), (8, 2, 256)
        )
        # head_dim may come from the module even when the config lacks it.
        module_only = types.SimpleNamespace(head_dim=128)
        config_no_head_dim = types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                num_attention_heads=8,
                num_key_value_heads=2,
                hidden_size=2048,
            )
        )
        self.assertEqual(
            capture.Qwen3_5Adapter().attention_meta(module_only, config_no_head_dim),
            (8, 2, 128),
        )

    def test_validate_does_not_assume_layer0_self_attn(self) -> None:
        model = self._hybrid_model()
        # layer 0 is linear-attention: no self_attn attribute at all.
        self.assertFalse(hasattr(model.model.layers[0], "self_attn"))
        capture.Qwen3_5Adapter().validate(model)  # must not raise

    def test_validate_rejects_missing_full_attention_layers(self) -> None:
        model = FakeModel(
            num_layers=4, layer_types=["linear_attention"] * 4
        )
        with self.assertRaisesRegex(CaptureError, "no full_attention layers"):
            capture.Qwen3_5Adapter().validate(model)

    def test_validate_rejects_layer_types_length_mismatch(self) -> None:
        model = FakeModel(num_layers=4, layer_types=["full_attention"] * 4)
        model.config.text_config.layer_types = ["full_attention"] * 3
        with self.assertRaisesRegex(CaptureError, "layer_types length"):
            capture.Qwen3_5Adapter().validate(model)

    def test_fake_adapter_validate_probes_first_full_layer(self) -> None:
        # The fake adapter used by weight-free capture runs must also validate
        # hybrid models whose layer 0 is a linear-attention layer.
        model = self._hybrid_model()
        capture.FakeAdapter().validate(model)

    def test_validate_fails_cleanly_on_wrong_layout(self) -> None:
        model = FakeModel(num_layers=2, layer_types=["full_attention"] * 2)
        # Sabotage the MLP surface on layer 0.
        del model.model.layers[0].mlp
        with self.assertRaisesRegex(CaptureError, "adapter layout"):
            capture.Qwen3_5Adapter().validate(model)


class ShardSchemaTest(unittest.TestCase):
    def _make_tensors(self, seq: int, channels: int) -> list[torch.Tensor]:
        return [
            torch.randn(seq, channels, dtype=torch.bfloat16).contiguous()
            for _ in range(5)
        ]

    def test_linear_shard_schema_and_validation(self) -> None:
        weight = torch.randn(256, 128, dtype=torch.bfloat16).contiguous()
        lengths = [8, 16, 24, 32, 40]
        metadata = {
            "layer_idx": 0,
            "role": "q_proj",
            "in_features": 128,
            "out_features": 256,
            "sample_lengths": lengths * 2,
        }
        shard = shards.build_linear_shard(
            metadata,
            weight,
            [
                torch.randn(length, 128, dtype=torch.bfloat16).contiguous()
                for length in lengths
            ],
            [
                torch.randn(length, 128, dtype=torch.bfloat16).contiguous()
                for length in lengths
            ],
        )
        shards.validate_linear_shard(shard)
        self.assertEqual(shard["kind"], "linear")

        variable = [
            torch.randn(length, 128, dtype=torch.bfloat16).contiguous()
            for length in lengths
        ]
        shards.validate_linear_shard(
            {**shard, "calib_activation_list": variable}
        )

        bad = {
            "kind": "linear",
            "schema_version": 1,
            "metadata": metadata,
            "weight": weight,
            "calib_activation_list": self._make_tensors(8, 128)[:4],
            "test_activation_list": self._make_tensors(8, 128),
        }
        with self.assertRaisesRegex(ValueError, "5 samples"):
            shards.validate_linear_shard(bad)

        bad_weight = torch.randn(256, 127, dtype=torch.bfloat16)  # not % 64
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            shards.validate_linear_shard(
                {**shard, "weight": bad_weight.contiguous()}
            )

        non_contiguous = torch.randn(128, 8, dtype=torch.bfloat16).t()  # [8,128], strided
        with self.assertRaisesRegex(ValueError, "contiguous"):
            shards.validate_linear_shard(
                {**shard, "calib_activation_list": [non_contiguous] * 5}
            )

        wrong_length = torch.randn(7, 128, dtype=torch.bfloat16).contiguous()
        with self.assertRaisesRegex(ValueError, "sample length"):
            shards.validate_linear_shard(
                {**shard, "calib_activation_list": [wrong_length] * 5}
            )

    def test_attention_flatten_and_shard_schema(self) -> None:
        heads, kv_heads, head_dim, seq = 4, 2, 32, 24
        raw = torch.randn(1, heads, seq, head_dim, dtype=torch.bfloat16)
        flat = flatten_attention_tensor(raw, heads, head_dim, seq)
        self.assertEqual(tuple(flat.shape), (seq, heads * head_dim))
        self.assertTrue(flat.is_contiguous())
        self.assertTrue(flat.dtype == torch.bfloat16)
        # flatten is a pure transpose of the heads/seq axes
        self.assertTrue(torch.equal(flat[3], raw[0, :, 3, :].reshape(-1)))

        with self.assertRaises(CaptureError):
            flatten_attention_tensor(raw, heads, head_dim, seq + 1)
        with self.assertRaises(CaptureError):
            flatten_attention_tensor(torch.randn(2, heads, seq, head_dim), heads, head_dim, seq)
        with self.assertRaises(CaptureError):
            flatten_attention_tensor(torch.randn(seq, heads * head_dim), heads, head_dim, seq)

        q = flatten_attention_tensor(
            torch.randn(1, heads, seq, head_dim, dtype=torch.bfloat16), heads, head_dim, seq
        )
        kv = flatten_attention_tensor(
            torch.randn(1, kv_heads, seq, head_dim, dtype=torch.bfloat16),
            kv_heads, head_dim, seq,
        )
        sample = {"q": q, "k": kv, "v": kv}
        metadata = {"layer_idx": 1, "sample_lengths": [seq] * 10}
        shard = shards.build_attention_shard(
            metadata, heads, kv_heads, head_dim, [sample] * 5, [sample] * 5
        )
        shards.validate_attention_shard(shard)
        self.assertEqual(shard["q_num_heads"], heads)
        self.assertEqual(shard["kv_num_heads"], kv_heads)

        missing_v = [{"q": q, "k": kv} for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "missing key"):
            shards.validate_attention_shard(
                {**shard, "test": missing_v}
            )
        bad_q = torch.randn(seq, kv_heads * head_dim, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "channels"):
            shards.validate_attention_shard(
                {**shard, "calib": [{"q": bad_q, "k": kv, "v": kv}] * 5}
            )


class CaptureRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _config(self, **overrides) -> dict:
        config = {
            "alias": "fake-model",
            "output_root": self.root,
            "smoke": True,
            "threads": 1,
            "seed": 0,
            "force": False,
            "layers": None,
            "linear_roles": None,
        }
        config.update(overrides)
        return config

    def _load_manifest(self, dataset_id: str) -> dict:
        path = self.root / "real-captures" / dataset_id / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_run_capture_end_to_end_with_fakes(self) -> None:
        loader = FakeLoader()
        summary = run_capture(self._config(), loader=loader)
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["reused"])
        self.assertEqual(summary["groups"], {"linear": 15, "attention": 3})
        self.assertEqual(summary["samples"], 10)
        self.assertEqual(len(loader.load_calls), 1)
        self.assertEqual(loader.metadata_calls, 1)

        output = Path(summary["output_dir"])
        self.assertEqual(len(list((output / "linear").glob("*.pt"))), 15)
        self.assertEqual(len(list((output / "attention").glob("*.pt"))), 3)
        self.assertFalse((output / ".tmp-capture").exists(), "temp files must be deleted")

        manifest = self._load_manifest(summary["dataset_id"])
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["dataset_id"], summary["dataset_id"])
        self.assertEqual(len(manifest["groups"]), 18)
        for group in manifest["groups"]:
            self.assertEqual(group["status"], "complete")
            self.assertEqual(
                group["sha256"], shards.sha256_file(output / group["path"]),
                f"manifest sha256 mismatch for {group['id']}",
            )

        linear = shards.load_tensor(output / "linear" / "0.q_proj.pt")
        self.assertEqual(linear["kind"], "linear")
        self.assertEqual(linear["metadata"]["role"], "q_proj")
        self.assertEqual(tuple(linear["weight"].shape), (128, 128))
        self.assertEqual(linear["weight"].dtype, torch.bfloat16)
        self.assertTrue(linear["weight"].is_contiguous())
        self.assertEqual(len(linear["calib_activation_list"]), 5)
        self.assertEqual(len(linear["test_activation_list"]), 5)
        self.assertEqual(tuple(linear["calib_activation_list"][0].shape), (8, 128))
        self.assertEqual(linear["metadata"]["sample_lengths"], SMOKE_LENGTHS * 2)

        attention = shards.load_tensor(output / "attention" / "1.self_attn.pt")
        self.assertEqual(attention["kind"], "attention")
        self.assertEqual(attention["q_num_heads"], 4)
        self.assertEqual(attention["kv_num_heads"], 2)
        self.assertEqual(attention["head_dim"], 32)
        self.assertEqual(tuple(attention["calib"][0]["q"].shape), (8, 128))
        self.assertEqual(tuple(attention["calib"][0]["k"].shape), (8, 64))
        self.assertEqual(tuple(attention["test"][4]["v"].shape), (40, 64))

    def test_manifest_contains_modes_prompts_and_hashes(self) -> None:
        summary = run_capture(self._config(), loader=FakeLoader())
        manifest = self._load_manifest(summary["dataset_id"])
        self.assertEqual(manifest["source_modes"], ["ceil", "nearest", "stochastic"])
        self.assertEqual(manifest["raw_bf16"]["dtype"], "bfloat16")
        self.assertTrue(manifest["raw_bf16"]["contiguous"])
        self.assertEqual(manifest["raw_bf16"]["device"], "cpu")
        self.assertEqual(manifest["model"]["alias"], "fake-model")
        self.assertEqual(manifest["model"]["arch"], "qwen3")
        self.assertEqual(manifest["model"]["resolved_revision"], FAKE_REVISION)
        self.assertEqual(manifest["model"]["transformers_version"], FAKE_TRANSFORMERS)
        self.assertEqual(manifest["layers"]["selected"], [0, 1, 2])
        self.assertEqual(manifest["layers"]["linear_roles"], list(DEFAULT_LINEAR_ROLES))
        self.assertEqual(manifest["seed"], 0)
        self.assertEqual(len(manifest["samples"]), 10)
        for index, sample in enumerate(manifest["samples"]):
            self.assertEqual(sample["index"], index)
            self.assertEqual(sample["length"], SMOKE_LENGTHS[index % 5])
            self.assertIn("prompt", sample)
            self.assertIn("prompt_hash", sample)
            self.assertIn("token_hash", sample)
            self.assertNotIn("token_ids", sample, "manifest must not store raw token ids")

    def test_run_capture_with_qwen2_and_llama_archs(self) -> None:
        for arch in ("qwen2", "llama"):
            with self.subTest(arch=arch):
                loader = FakeLoader(meta_overrides={"arch": arch})
                summary = run_capture(self._config(), loader=loader)
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["groups"], {"linear": 15, "attention": 3})
                manifest = self._load_manifest(summary["dataset_id"])
                self.assertEqual(manifest["model"]["arch"], arch)
                self.assertEqual(manifest["layers"]["selected"], [0, 1, 2])
                attention = shards.load_tensor(
                    Path(summary["output_dir"]) / "attention" / "1.self_attn.pt"
                )
                self.assertEqual(tuple(attention["calib"][0]["q"].shape), (8, 128))

    def test_run_capture_hybrid_qwen35_default_selection(self) -> None:
        model = FakeModel(num_layers=24, layer_types=QWEN35_LAYER_TYPES)
        loader = FakeLoader(model=model)
        summary = run_capture(self._config(), loader=loader)
        self.assertEqual(summary["status"], "complete")
        # Hybrid default selection: [3, 11, 23]; MLP roles on those layers plus
        # attention groups on the full-attention ones.
        self.assertEqual(summary["groups"], {"linear": 15, "attention": 3})
        output = Path(summary["output_dir"])
        manifest = self._load_manifest(summary["dataset_id"])
        self.assertEqual(manifest["model"]["arch"], "qwen3_5")
        self.assertEqual(manifest["layers"]["selected"], [3, 11, 23])
        self.assertEqual(manifest["layers"]["full_attention"], QWEN35_FULL_ATTENTION)
        # Only full-attention layers produce attention shards.
        self.assertEqual(len(list((output / "attention").glob("*.pt"))), 3)
        self.assertTrue((output / "attention" / "3.self_attn.pt").is_file())
        self.assertFalse((output / "attention" / "0.self_attn.pt").exists())
        # Linear sites exist on the selected layers (attention + MLP roles).
        self.assertTrue((output / "linear" / "3.q_proj.pt").is_file())
        self.assertTrue((output / "linear" / "11.gate_proj.pt").is_file())
        self.assertTrue((output / "linear" / "23.down_proj.pt").is_file())
        # Attention heads resolved through the nested text_config.
        attention = shards.load_tensor(output / "attention" / "11.self_attn.pt")
        self.assertEqual(attention["q_num_heads"], 4)
        self.assertEqual(attention["kv_num_heads"], 2)
        self.assertEqual(attention["head_dim"], 32)

    def test_run_capture_hybrid_explicit_linear_layers_keep_mlp_roles(self) -> None:
        model = FakeModel(num_layers=24, layer_types=QWEN35_LAYER_TYPES)
        config = self._config(
            layers=[0, 3],
            linear_roles=["gate_proj", "up_proj", "down_proj"],
        )
        summary = run_capture(config, loader=FakeLoader(model=model))
        self.assertEqual(summary["groups"], {"linear": 6, "attention": 1})
        output = Path(summary["output_dir"])
        # MLP roles on a linear-attention layer are captured.
        self.assertTrue((output / "linear" / "0.gate_proj.pt").is_file())
        # Attention on the full-attention layer only.
        self.assertTrue((output / "attention" / "3.self_attn.pt").is_file())
        self.assertFalse((output / "attention" / "0.self_attn.pt").exists())

    def test_run_capture_hybrid_filters_attention_roles_off_linear_layers(self) -> None:
        model = FakeModel(num_layers=24, layer_types=QWEN35_LAYER_TYPES)
        summary = run_capture(
            self._config(layers=[0, 3]),
            loader=FakeLoader(model=model),
        )
        # Layer 0 keeps only its MLP roles; layer 3 keeps all five roles.
        self.assertEqual(summary["groups"], {"linear": 8, "attention": 1})
        output = Path(summary["output_dir"])
        self.assertTrue((output / "linear" / "0.gate_proj.pt").is_file())
        self.assertFalse((output / "linear" / "0.q_proj.pt").exists())
        self.assertTrue((output / "linear" / "3.q_proj.pt").is_file())
        manifest = self._load_manifest(summary["dataset_id"])
        self.assertEqual(manifest["layers"]["selected"], [0, 3])

    def test_run_capture_hybrid_fails_clearly_when_selection_captures_nothing(self) -> None:
        model = FakeModel(num_layers=24, layer_types=QWEN35_LAYER_TYPES)
        config = self._config(
            layers=[0, 1],
            linear_roles=["q_proj", "o_proj"],
        )
        with self.assertRaisesRegex(CaptureError, "captures nothing"):
            run_capture(config, loader=FakeLoader(model=model))

    def test_manifest_full_attention_for_uniform_models(self) -> None:
        # Uniform architectures record every layer as full-attention.
        summary = run_capture(self._config(), loader=FakeLoader())
        manifest = self._load_manifest(summary["dataset_id"])
        self.assertEqual(manifest["layers"]["full_attention"], [0, 1, 2])
        self.assertEqual(manifest["layers"]["selected"], [0, 1, 2])

    def test_reuse_skips_model_load_when_complete(self) -> None:
        run_capture(self._config(), loader=FakeLoader())
        loader2 = FakeLoader()
        summary = run_capture(self._config(), loader=loader2)
        self.assertTrue(summary["reused"])
        self.assertEqual(loader2.load_calls, [], "complete capture must skip the model load")
        self.assertEqual(summary["groups"], {"linear": 15, "attention": 3})

    def test_resume_recaptures_only_missing_group(self) -> None:
        first = run_capture(self._config(), loader=FakeLoader())
        output = Path(first["output_dir"])
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        kept = manifest["groups"][:]
        surviving = [g for g in kept if g["id"] != "0.q_proj"]
        manifest["groups"] = surviving
        manifest["status"] = "in_progress"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        missing_shard = output / "linear" / "0.q_proj.pt"
        missing_shard.unlink()
        surviving_file = output / "linear" / "0.o_proj.pt"

        loader2 = FakeLoader()
        summary = run_capture(self._config(), loader=loader2)
        self.assertFalse(summary["reused"])
        self.assertEqual(summary["groups"], {"linear": 15, "attention": 3})
        self.assertEqual(len(loader2.load_calls), 1)
        self.assertTrue(missing_shard.is_file(), "missing group must be recaptured")
        new_manifest = self._load_manifest(first["dataset_id"])
        self.assertEqual(new_manifest["status"], "complete")
        # untouched groups keep their original file and digest
        original_sha = next(g for g in kept if g["id"] == "0.o_proj")["sha256"]
        self.assertEqual(shards.sha256_file(surviving_file), original_sha)

    def test_resume_detects_hash_mismatch(self) -> None:
        first = run_capture(self._config(), loader=FakeLoader())
        output = Path(first["output_dir"])
        (output / "linear" / "0.o_proj.pt").write_bytes(b"corrupted payload")
        with self.assertRaises(ResumeMismatchError):
            run_capture(self._config(), loader=FakeLoader())

        # same detection on an in-progress manifest (mid-resume corruption)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "in_progress"
        manifest["groups"] = [g for g in manifest["groups"] if g["id"] != "0.q_proj"]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "linear" / "1.gate_proj.pt").write_bytes(b"junk")
        with self.assertRaisesRegex(ResumeMismatchError, "--force"):
            run_capture(self._config(), loader=FakeLoader())

    def test_force_restarts_complete_capture(self) -> None:
        run_capture(self._config(), loader=FakeLoader())
        loader2 = FakeLoader()
        summary = run_capture(self._config(force=True), loader=loader2)
        self.assertFalse(summary["reused"])
        self.assertEqual(len(loader2.load_calls), 1)
        self.assertEqual(self._load_manifest(summary["dataset_id"])["status"], "complete")

    def test_resume_rejects_mixed_torch_versions(self) -> None:
        first = run_capture(self._config(), loader=FakeLoader())
        manifest_path = Path(first["output_dir"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime"]["torch_version"] = "different-torch"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(ResumeMismatchError, "mixing runtime versions"):
            run_capture(self._config(), loader=FakeLoader())

    def test_capture_count_assertion_failure_attention(self) -> None:
        model = FakeModel()
        replace_module(
            model, 1, "self_attn",
            DoubleFireAttention(1, 128, 4, 2, 32),
        )
        with self.assertRaisesRegex(CaptureError, "more than once"):
            run_capture(self._config(), loader=FakeLoader(model=model))

    def test_capture_count_assertion_failure_linear(self) -> None:
        model = FakeModel()
        replace_module(model, 1, "mlp", DoubleFireMLP(128, 256))
        with self.assertRaisesRegex(CaptureError, "more than once"):
            run_capture(self._config(), loader=FakeLoader(model=model))

    def test_missing_layer_validation(self) -> None:
        with self.assertRaises(CaptureError):
            run_capture(self._config(layers=[0, 3]), loader=FakeLoader())

    def test_cli_parser_defaults_and_helpers(self) -> None:
        args = capture_real.build_parser().parse_args([])
        self.assertEqual(args.model, "qwen3-0.6b")
        self.assertEqual(args.threads, 1)
        self.assertIsNone(args.layers)
        self.assertFalse(args.smoke)
        self.assertFalse(args.force)
        self.assertEqual(capture_real._parse_int_list("0,14,27"), [0, 14, 27])
        self.assertIsNone(capture_real._parse_int_list(None))
        self.assertEqual(
            capture_real._parse_name_list("q_proj,o_proj"), ["q_proj", "o_proj"]
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            capture_real._parse_int_list("0,x")
        # the --layers help must document the hybrid default
        layers_help = next(
            action.help
            for action in capture_real.build_parser()._actions
            if action.dest == "layers"
        )
        self.assertIn("hybrid", layers_help)
        self.assertIn("full-attention", layers_help)


class RealEagerKernelContractTest(unittest.TestCase):
    """Weight-free check of the adapter's core assumption: the eager attention
    kernels of qwen3/qwen2/llama and, when installed, qwen3_5 share one
    signature, so the shared wrapper applies to every registered architecture."""

    _HAVE_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

    @unittest.skipUnless(_HAVE_TRANSFORMERS, "transformers not installed")
    def test_real_eager_kernel_signatures_match_across_architectures(self) -> None:
        script = (
            "import sys; sys.modules['torchvision']=None;\n"
            "import inspect;\n"
            "from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as q3;\n"
            "from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward as q2;\n"
            "from transformers.models.llama.modeling_llama import eager_attention_forward as ll;\n"
            "sig = lambda fn: [(p.name, str(p.kind)) for p in inspect.signature(fn).parameters.values()];\n"
            "s3, s2, sl = sig(q3), sig(q2), sig(ll);\n"
            "assert s3 == s2 == sl, (s3, s2, sl);\n"
            "try:\n"
            " from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward as q35\n"
            "except ModuleNotFoundError:\n"
            " q35 = None\n"
            "if q35 is not None:\n"
            " assert s3 == sig(q35), (s3, sig(q35))\n"
            " print('QWEN35_EAGER_SIG_MATCH')\n"
            "print('EAGER_SIGS_MATCH')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("EAGER_SIGS_MATCH", result.stdout)


class SubprocessSmokeTest(unittest.TestCase):
    def test_subprocess_capture_with_oom_score(self) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(__file__)), "--subprocess-smoke"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("SUBPROCESS_SMOKE_OK", result.stdout)
        self.assertIn("OOM_ADJ_OK", result.stdout)


def _run_subprocess_smoke() -> int:
    import traceback

    oom_ok = capture.set_oom_score(500)
    try:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "alias": "fake-model",
                "output_root": Path(directory),
                "smoke": True,
                "threads": 1,
                "seed": 0,
                "force": False,
                "layers": None,
                "linear_roles": None,
            }
            summary = run_capture(config, loader=FakeLoader())
            assert summary["status"] == "complete", summary
            assert summary["groups"] == {"linear": 15, "attention": 3}, summary
            assert "transformers" not in sys.modules, (
                "subprocess must not import transformers"
            )
        print(f"OOM_ADJ_OK={oom_ok}")
        print("SUBPROCESS_SMOKE_OK")
        return 0
    except Exception:  # pragma: no cover - diagnostic path
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if "--subprocess-smoke" in sys.argv:
        raise SystemExit(_run_subprocess_smoke())
    unittest.main()
