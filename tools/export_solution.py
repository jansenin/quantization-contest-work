"""Create a deterministic contest archive from a tagged Git revision."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


def _git(*args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def export_solution(ref: str, output: Path) -> tuple[str, str, str]:
    """Export ``solution.py`` from ``ref`` and return commit/source/archive hashes."""
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    source = _git("show", f"{ref}:solution.py")
    source_hash = hashlib.sha256(source).hexdigest()

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        info = zipfile.ZipInfo("solution.py", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(temporary_path, "w", compresslevel=9) as archive:
            archive.writestr(info, source)
        with zipfile.ZipFile(temporary_path, "r") as archive:
            if archive.namelist() != ["solution.py"]:
                raise RuntimeError("archive must contain only root-level solution.py")
            if archive.read("solution.py") != source:
                raise RuntimeError("archived source differs from Git source")
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)

    archive_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    return commit, source_hash, archive_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Git tag, commit, or branch")
    parser.add_argument("--output", type=Path, default=Path("solution.zip"))
    args = parser.parse_args()

    commit, source_hash, archive_hash = export_solution(args.ref, args.output)
    print(f"ref:            {args.ref}")
    print(f"commit:         {commit}")
    print(f"source sha256:  {source_hash}")
    print(f"archive sha256: {archive_hash}")
    print(f"output:         {args.output.resolve()}")


if __name__ == "__main__":
    main()
