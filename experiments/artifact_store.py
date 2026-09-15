from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


REQUIRED = {
    "schema_version", "status", "git_sha", "git_dirty", "python_version",
    "dataset_hash", "config_hash", "sampling_plan_id", "model_seed",
    "sampling_seed", "bootstrap_seed", "sample_path",
}


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, value: Mapping[str, Any]) -> Path:
        target = self.root / name
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=f".{name}.")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    def complete(self, manifest: Mapping[str, Any]) -> Path:
        missing = REQUIRED - manifest.keys()
        sample = self.root / str(manifest.get("sample_path", ""))
        if missing or not sample.is_file():
            return self.write_json(
                "manifest.json",
                {**manifest, "status": "incomplete", "incomplete_reason": f"missing={sorted(missing)}, sample_exists={sample.is_file()}"},
            )
        return self.write_json("manifest.json", {**manifest, "status": "complete"})
