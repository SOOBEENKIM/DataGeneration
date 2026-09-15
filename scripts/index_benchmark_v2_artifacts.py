from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/benchmark_v2")
    args = parser.parse_args()
    root = Path(args.root)
    destination = root / "artifact_index.json"
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    destination.write_text(
        json.dumps(
            {
                "schema_version": "benchmark_v2_artifact_index_v1",
                "root": root.as_posix(),
                "artifact_count": len(records),
                "artifacts": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
