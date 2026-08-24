"""Atomic raw-BF16 shard storage and manifest handling for real-model captures.

Every durable artifact (JSON manifest, torch shard, per-sample temporary
capture file) is written through a temp file in the same directory followed by
``os.replace``, so readers never observe a half-written file and interruption
never leaves a corrupt target path behind.

Shard layouts (both stored with ``torch.save``):

Linear group (``linear/<layer>.<role>.pt``)::

    {
      "kind": "linear",
      "schema_version": 1,
      "metadata": {layer_idx, role, in_features, out_features, ...},
      "weight": BF16 CPU contiguous [out_features, in_features],
      "calib_activation_list": [BF16 [seq, in_features]] x5,
      "test_activation_list":  [BF16 [seq, in_features]] x5,
    }

Attention group (``attention/<layer>.self_attn.pt``)::

    {
      "kind": "attention",
      "schema_version": 1,
      "metadata": {layer_idx, ...},
      "q_num_heads": int, "kv_num_heads": int, "head_dim": int,
      "calib": [{"q": BF16 [seq, q_heads*head_dim],
                 "k": BF16 [seq, kv_heads*head_dim],
                 "v": BF16 [seq, kv_heads*head_dim]}] x5,
      "test": [same] x5,
    }

Validators enforce: BF16, CPU, contiguous, exact 5+5 samples per split, and
channels divisible by 64 (the HiF4 block size).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

#: Schema version of manifests and shard payloads (bumped on breaking changes).
SCHEMA_VERSION = 1

#: Storage facts recorded in every manifest ("raw_bf16" section).
RAW_BF16_STORAGE: dict[str, Any] = {
    "dtype": "bfloat16",
    "device": "cpu",
    "contiguous": True,
    "file_format": "torch_save",
    "layout": "one_raw_shard_per_group",
}

#: Source modes the raw BF16 data can subsequently be converted to.
SOURCE_MODES: list[str] = ["ceil", "nearest", "stochastic"]

SAMPLES_PER_SPLIT = 5


class ResumeMismatchError(RuntimeError):
    """A previously finalized shard no longer matches the manifest."""


# ---------------------------------------------------------------------------
# Atomic primitives
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path | str, obj: Any) -> None:
    payload = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_save_torch(path: Path | str, obj: Any) -> str:
    """Atomically ``torch.save`` ``obj`` and return the SHA-256 of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=".pt", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        torch.save(obj, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return sha256_file(path)


def load_tensor(path: Path | str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sample_token_hash(token_ids: Sequence[int]) -> str:
    """Deterministic SHA-256 over a token-id sequence (4-byte big-endian)."""
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(4, "big"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def load_manifest(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError(f"manifest {path} is not a JSON object")
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"manifest {path} has unsupported schema_version "
            f"{obj.get('schema_version')!r}"
        )
    return obj


def write_manifest(path: Path | str, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(path, manifest)


# ---------------------------------------------------------------------------
# Shard construction and validation
# ---------------------------------------------------------------------------


def build_linear_shard(
    metadata: Mapping[str, Any],
    weight: torch.Tensor,
    calib_activation_list: Sequence[torch.Tensor],
    test_activation_list: Sequence[torch.Tensor],
) -> dict[str, Any]:
    return {
        "kind": "linear",
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(metadata),
        "weight": weight,
        "calib_activation_list": list(calib_activation_list),
        "test_activation_list": list(test_activation_list),
    }


def build_attention_shard(
    metadata: Mapping[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    calib: Sequence[Mapping[str, torch.Tensor]],
    test: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "kind": "attention",
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(metadata),
        "q_num_heads": int(q_num_heads),
        "kv_num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "calib": [dict(sample) for sample in calib],
        "test": [dict(sample) for sample in test],
    }


def _require_raw_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name}: expected a torch.Tensor, got {type(value).__name__}")
    if value.dtype != torch.bfloat16:
        raise ValueError(f"{name}: dtype {value.dtype} != bfloat16")
    if not value.is_cpu:
        raise ValueError(f"{name}: tensor is not on CPU")
    if not value.is_contiguous():
        raise ValueError(f"{name}: tensor is not contiguous")
    return value


def _require_activation(
    name: str, value: Any, channels: int, sample_length: int | None
) -> torch.Tensor:
    tensor = _require_raw_tensor(name, value)
    if tensor.ndim != 2:
        raise ValueError(f"{name}: expected [seq, channels], got shape {tuple(tensor.shape)}")
    if tensor.shape[1] != channels:
        raise ValueError(
            f"{name}: channels {tensor.shape[1]} != expected {channels}"
        )
    if sample_length is not None and tensor.shape[0] != sample_length:
        raise ValueError(
            f"{name}: seq {tensor.shape[0]} != expected sample length {sample_length}"
        )
    return tensor


def _check_channels_multiple_of_64(name: str, channels: int) -> None:
    if channels % 64 != 0:
        raise ValueError(f"{name}: channels {channels} not divisible by 64 (HiF4 block)")


def validate_linear_shard(shard: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(shard, dict) or shard.get("kind") != "linear":
        raise ValueError("shard is not a linear group")
    if not isinstance(shard.get("metadata"), dict):
        raise ValueError("linear shard metadata must be a dict")
    weight = _require_raw_tensor("weight", shard["weight"])
    if weight.ndim != 2:
        raise ValueError(f"weight: expected [out_features, in_features], got {tuple(weight.shape)}")
    out_features, in_features = (int(x) for x in weight.shape)
    _check_channels_multiple_of_64("weight.in_features", in_features)
    _check_channels_multiple_of_64("weight.out_features", out_features)
    sample_lengths = shard["metadata"].get("sample_lengths")
    for split, key in (
        ("calib", "calib_activation_list"),
        ("test", "test_activation_list"),
    ):
        samples = shard[key]
        if len(samples) != SAMPLES_PER_SPLIT:
            raise ValueError(f"{split}: expected {SAMPLES_PER_SPLIT} samples, got {len(samples)}")
        for index, activation in enumerate(samples):
            length = None
            if isinstance(sample_lengths, list) and index < len(sample_lengths):
                length = int(sample_lengths[index])
            _require_activation(
                f"{split}[{index}]", activation, in_features, length
            )
    return shard


def validate_attention_shard(shard: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(shard, dict) or shard.get("kind") != "attention":
        raise ValueError("shard is not an attention group")
    if not isinstance(shard.get("metadata"), dict):
        raise ValueError("attention shard metadata must be a dict")
    q_heads = int(shard["q_num_heads"])
    kv_heads = int(shard["kv_num_heads"])
    head_dim = int(shard["head_dim"])
    if q_heads < 1 or kv_heads < 1 or head_dim < 1:
        raise ValueError("attention head counts and head_dim must be positive")
    q_hidden = q_heads * head_dim
    kv_hidden = kv_heads * head_dim
    _check_channels_multiple_of_64("q_hidden", q_hidden)
    _check_channels_multiple_of_64("kv_hidden", kv_hidden)
    sample_lengths = shard["metadata"].get("sample_lengths")
    for split in ("calib", "test"):
        samples = shard[split]
        if len(samples) != SAMPLES_PER_SPLIT:
            raise ValueError(f"{split}: expected {SAMPLES_PER_SPLIT} samples, got {len(samples)}")
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"{split}[{index}]: sample must be a dict with q/k/v")
            for missing in ("q", "k", "v"):
                if missing not in sample:
                    raise ValueError(f"{split}[{index}]: missing key {missing!r}")
            length = None
            if isinstance(sample_lengths, list) and index < len(sample_lengths):
                length = int(sample_lengths[index])
            _require_activation(f"{split}[{index}].q", sample["q"], q_hidden, length)
            _require_activation(f"{split}[{index}].k", sample["k"], kv_hidden, length)
            _require_activation(f"{split}[{index}].v", sample["v"], kv_hidden, length)
    return shard
