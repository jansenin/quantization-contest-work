"""Real-model raw-BF16 capture dataset pipeline.

Submodules:

- ``shards``: atomic JSON/torch storage, manifest handling, shard validation.
- ``capture``: corpus, deterministic dataset ids, model adapters (Qwen3 first),
  lazy transformers loader, capture orchestration with resume.
- ``corpus.json``: the fixed 10-text calibration/test corpus.

Entry points: :func:`capture.run_capture` and the ``tools/capture_real.py`` CLI.
"""

from . import capture, shards  # noqa: F401
from .capture import (  # noqa: F401
    CaptureError,
    DEFAULT_LINEAR_ROLES,
    DEFAULT_MODEL_ALIAS,
    FULL_LENGTHS,
    SMOKE_LENGTHS,
    build_dataset_id,
    canonical_capture_config,
    corpus_sha256,
    flatten_attention_tensor,
    get_adapter,
    load_corpus,
    make_samples,
    normalize_linear_roles,
    run_capture,
    select_layers,
    set_oom_score,
    tokenize_sample,
)
from .shards import ResumeMismatchError  # noqa: F401

__all__ = [
    "CaptureError",
    "ResumeMismatchError",
    "DEFAULT_LINEAR_ROLES",
    "DEFAULT_MODEL_ALIAS",
    "FULL_LENGTHS",
    "SMOKE_LENGTHS",
    "build_dataset_id",
    "canonical_capture_config",
    "capture",
    "corpus_sha256",
    "flatten_attention_tensor",
    "get_adapter",
    "load_corpus",
    "make_samples",
    "normalize_linear_roles",
    "run_capture",
    "select_layers",
    "set_oom_score",
    "shards",
    "tokenize_sample",
]
