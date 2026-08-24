"""Real-model raw-BF16 capture dataset pipeline.

Submodules:

- ``shards``: atomic JSON/torch storage, manifest handling, shard validation.
- ``capture``: corpus, deterministic dataset ids, model adapters (Qwen3 first),
  lazy transformers loader, capture orchestration with resume.
- ``evaluate``: streaming one-group-at-a-time NVFP4->HiF4 evaluator over raw
  capture shards (source modes ceil/nearest/stochastic, per-case records,
  baseline cache, atomic resume).
- ``corpus.json``: the fixed 10-text calibration/test corpus.

Entry points: :func:`capture.run_capture`, :func:`evaluate.evaluate_dataset`
and the ``tools/capture_real.py`` / ``tools/evaluate_real.py`` CLIs.
"""

from . import capture, evaluate, shards  # noqa: F401
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
from .evaluate import (  # noqa: F401
    DatasetError,
    GroupLoadError,
    VariantError,
    build_case_record,
    case_record_key,
    compute_references,
    derive_source,
    evaluate_dataset,
    iter_groups,
    parse_filters,
    resolve_dataset,
)
from .shards import ResumeMismatchError  # noqa: F401

__all__ = [
    "CaptureError",
    "ResumeMismatchError",
    "DatasetError",
    "GroupLoadError",
    "VariantError",
    "DEFAULT_LINEAR_ROLES",
    "DEFAULT_MODEL_ALIAS",
    "FULL_LENGTHS",
    "SMOKE_LENGTHS",
    "build_case_record",
    "build_dataset_id",
    "canonical_capture_config",
    "capture",
    "case_record_key",
    "compute_references",
    "corpus_sha256",
    "derive_source",
    "evaluate",
    "evaluate_dataset",
    "flatten_attention_tensor",
    "get_adapter",
    "iter_groups",
    "load_corpus",
    "make_samples",
    "normalize_linear_roles",
    "parse_filters",
    "resolve_dataset",
    "run_capture",
    "select_layers",
    "set_oom_score",
    "shards",
    "tokenize_sample",
]
