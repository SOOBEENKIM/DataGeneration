"""Create the immutable inventory linking the legacy-v1 source snapshot to artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "audit" / "v1_artifact_manifest.json"
RUNTIME_ROOTS = ("data", "results", "logs")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_artifacts() -> list[Path]:
    paths: list[Path] = []
    for name in RUNTIME_ROOTS:
        base = ROOT / name
        if base.exists():
            paths.extend(path for path in base.rglob("*") if path.is_file())
    models = ROOT / "models"
    if models.exists():
        paths.extend(
            path
            for path in models.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        )
    return sorted(set(paths))


def main() -> None:
    entries = []
    for path in excluded_artifacts():
        stat = path.stat()
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": "legacy_v1_artifact_manifest.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": ".",
        "artifact_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "artifacts": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
