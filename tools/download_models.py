"""Download capture-model snapshots with persistent resume and retries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
from typing import Any


DEFAULT_CATALOG = Path(__file__).with_name("model_downloads.json")
DEFAULT_CACHE = Path("data/huggingface-cache")
DEFAULT_STATE = Path("data/model-download-state.json")
STATE_VERSION = 1


class CatalogStateError(ValueError):
    """The persisted model identity no longer matches the catalog."""


def _handle_termination(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    if not isinstance(catalog.get("models"), dict):
        raise ValueError("catalog must contain a models object")
    if not isinstance(catalog.get("profiles"), dict):
        raise ValueError("catalog must contain a profiles object")
    return catalog


def select_models(
    catalog: dict[str, Any], profiles: list[str], names: list[str]
) -> list[str]:
    models = catalog["models"]
    selected: list[str] = []
    requested_profiles = profiles or ([] if names else ["laptop"])
    for profile in requested_profiles:
        if profile == "all":
            candidates = list(models)
        else:
            try:
                candidates = catalog["profiles"][profile]
            except KeyError as error:
                raise ValueError(f"unknown profile: {profile}") from error
        selected.extend(candidates)
    selected.extend(names)

    unknown = sorted(set(selected) - set(models))
    if unknown:
        raise ValueError(f"unknown model aliases: {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": STATE_VERSION, "models": {}}
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("version") != STATE_VERSION or not isinstance(
        state.get("models"), dict
    ):
        raise ValueError(f"unsupported download state in {path}")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _load_huggingface_hub() -> tuple[Any, Any, tuple[type[BaseException], ...]]:
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is missing; install requirements-data.txt"
        ) from error
    permanent = (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError)
    return HfApi, snapshot_download, permanent


def _print_catalog(catalog: dict[str, Any]) -> None:
    print("Profiles:")
    for name, models in catalog["profiles"].items():
        print(f"  {name:10s} {', '.join(models)}")
    print("  all        every model in the catalog")
    print("\nModels:")
    for alias, spec in catalog["models"].items():
        print(
            f"  {alias:27s} {spec['approx_gb']:6.2f} GB  "
            f"{spec['repo_id']}"
        )


def _print_status(path: Path, state: dict[str, Any]) -> None:
    print(f"state: {path.resolve()}")
    if not state["models"]:
        print("no model downloads recorded")
        return
    for alias, record in state["models"].items():
        revision = record.get("resolved_revision", "unresolved")
        print(
            f"{alias:27s} {record.get('status', 'unknown'):11s} "
            f"attempts={record.get('attempts', 0):<4d} revision={revision}"
        )


def _resolve_revision(
    alias: str,
    spec: dict[str, Any],
    record: dict[str, Any],
    api: Any,
    state_path: Path,
    state: dict[str, Any],
) -> str:
    repo_id = spec["repo_id"]
    requested = spec.get("revision", "main")
    has_identity = (
        record.get("repo_id") is not None
        or record.get("requested_revision") is not None
    )
    if has_identity:
        identity = (record.get("repo_id"), record.get("requested_revision"))
        if identity != (repo_id, requested):
            raise CatalogStateError(
                f"catalog entry {alias} changed since state was created; "
                f"remove its record from {state_path} to select a new revision"
            )
        if record.get("resolved_revision"):
            return str(record["resolved_revision"])

    print(f"[{alias}] resolving {repo_id}@{requested}", flush=True)
    info = api.model_info(repo_id=repo_id, revision=requested)
    if not info.sha:
        raise RuntimeError(f"Hub did not resolve a commit SHA for {repo_id}")
    record.update(
        {
            "repo_id": repo_id,
            "requested_revision": requested,
            "resolved_revision": info.sha,
            "status": "resolved",
            "updated_at": _timestamp(),
        }
    )
    save_state(state_path, state)
    return str(info.sha)


def _download_one(
    alias: str,
    spec: dict[str, Any],
    allow_patterns: list[str],
    cache_dir: Path,
    state_path: Path,
    state: dict[str, Any],
    api: Any,
    snapshot_download: Any,
    permanent_errors: tuple[type[BaseException], ...],
    workers: int,
    initial_delay: float,
    max_delay: float,
    max_retries: int,
) -> None:
    record = state["models"].setdefault(alias, {})
    snapshot_path = record.get("snapshot_path")
    if (
        record.get("status") == "complete"
        and snapshot_path
        and Path(snapshot_path).exists()
    ):
        print(f"[{alias}] already complete: {record['snapshot_path']}", flush=True)
        return

    delay = initial_delay
    failures = 0
    while True:
        record["status"] = "attempting"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["updated_at"] = _timestamp()
        save_state(state_path, state)
        try:
            revision = _resolve_revision(
                alias, spec, record, api, state_path, state
            )
            record["status"] = "downloading"
            record["updated_at"] = _timestamp()
            save_state(state_path, state)
            print(
                f"[{alias}] downloading {spec['repo_id']}@{revision} "
                f"(attempt {record['attempts']})",
                flush=True,
            )
            snapshot = snapshot_download(
                repo_id=spec["repo_id"],
                revision=revision,
                cache_dir=str(cache_dir),
                allow_patterns=allow_patterns,
                max_workers=workers,
            )
        except (CatalogStateError, *permanent_errors):
            record["status"] = "failed"
            record["updated_at"] = _timestamp()
            save_state(state_path, state)
            raise
        except (KeyboardInterrupt, SystemExit):
            record["status"] = "interrupted"
            record["updated_at"] = _timestamp()
            save_state(state_path, state)
            raise
        except Exception as error:
            failures += 1
            record["status"] = "retrying"
            record["last_error"] = f"{type(error).__name__}: {error}"
            record["updated_at"] = _timestamp()
            save_state(state_path, state)
            if max_retries and failures > max_retries:
                raise RuntimeError(
                    f"{alias} failed after {failures} retries"
                ) from error
            sleep_for = min(max_delay, delay) * random.uniform(0.9, 1.1)
            print(
                f"[{alias}] {record['last_error']}; retrying in "
                f"{sleep_for:.0f}s",
                flush=True,
            )
            time.sleep(sleep_for)
            delay = min(max_delay, max(initial_delay, delay * 2.0))
            continue

        record.update(
            {
                "status": "complete",
                "snapshot_path": str(Path(snapshot).resolve()),
                "updated_at": _timestamp(),
            }
        )
        record.pop("last_error", None)
        save_state(state_path, state)
        print(f"[{alias}] complete: {snapshot}", flush=True)
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--max-retry-delay", type=float, default=900.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="retries per model; 0 retries forever",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="show selection without networking"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.retry_delay <= 0 or args.max_retry_delay <= 0:
        raise ValueError("retry delays must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")

    catalog = load_catalog(args.catalog)
    if args.list:
        _print_catalog(catalog)
        return 0
    state = load_state(args.state_file)
    if args.status:
        _print_status(args.state_file, state)
        return 0

    selected = select_models(catalog, args.profile, args.model)
    total_gb = sum(float(catalog["models"][name]["approx_gb"]) for name in selected)
    print(f"selected: {', '.join(selected)}", flush=True)
    print(f"approximate model files: {total_gb:.2f} GB", flush=True)
    print(f"cache: {args.cache_dir.resolve()}", flush=True)
    print(f"state: {args.state_file.resolve()}", flush=True)
    if args.dry_run:
        return 0

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    HfApi, snapshot_download, permanent_errors = _load_huggingface_hub()
    api = HfApi()
    for alias in selected:
        _download_one(
            alias=alias,
            spec=catalog["models"][alias],
            allow_patterns=catalog["allow_patterns"],
            cache_dir=args.cache_dir,
            state_path=args.state_file,
            state=state,
            api=api,
            snapshot_download=snapshot_download,
            permanent_errors=permanent_errors,
            workers=args.workers,
            initial_delay=args.retry_delay,
            max_delay=args.max_retry_delay,
            max_retries=args.max_retries,
        )
    print("all selected model snapshots are complete", flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_termination)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("download interrupted; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(130)
