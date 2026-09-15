"""Reproducible raw-data acquisition for the CoFSeqGen-SAF family.

The command copies verified local sources or downloads a pinned public
snapshot into an ignored SAF-only runtime tree. It never preprocesses rows and
never reads a held-out model result. Every acquired file is hashed and made
read-only after an atomic move.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable
from urllib.request import Request, urlopen
import zipfile

import yaml


class AcquisitionError(RuntimeError):
    """Raised when a source cannot be acquired without changing its identity."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _check_expected(path: Path, expected: str | None) -> str:
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise AcquisitionError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )
    return observed


def _target_path(dataset_dir: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.name != name:
        raise AcquisitionError(f"target_name must be a plain file name: {name!r}")
    return dataset_dir / name


def _atomic_copy(source: Path, target: Path, expected: str | None) -> dict[str, Any]:
    if not source.is_file():
        raise AcquisitionError(f"source file is missing: {source}")
    source_sha = _check_expected(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target_sha = _check_expected(target, source_sha)
        return {"name": target.name, "bytes": target.stat().st_size, "sha256": target_sha}
    temporary = target.with_name(f".{target.name}.part")
    if temporary.exists():
        temporary.unlink()
    shutil.copyfile(source, temporary)
    _check_expected(temporary, source_sha)
    os.replace(temporary, target)
    target.chmod(0o444)
    return {"name": target.name, "bytes": target.stat().st_size, "sha256": source_sha}


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "CoFSeqGen-SAF-acquisition-v1"})
    with urlopen(request, timeout=120) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def acquire_local_dataset(
    dataset_id: str,
    spec: dict[str, Any],
    raw_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    dataset_dir = raw_root / dataset_id
    files = []
    for item in spec["files"]:
        source = Path(item["source_path"])
        if not source.is_absolute():
            source = repository_root / source
        files.append(
            _atomic_copy(
                source,
                _target_path(dataset_dir, item["target_name"]),
                item.get("expected_sha256"),
            )
        )
    return {"status": "ACQUIRED", "files": files}


def acquire_local_generator(
    dataset_id: str,
    spec: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    files = []
    for item in spec["files"]:
        source = repository_root / item["source_path"]
        digest = _check_expected(source, item.get("expected_sha256"))
        files.append(
            {
                "name": str(source.relative_to(repository_root)),
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
        )
    return {"status": "REGISTERED_LOCAL_GENERATOR", "files": files}


def acquire_git_snapshot(
    dataset_id: str,
    spec: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    dataset_dir = raw_root / dataset_id
    staging_root = raw_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{dataset_id}-", dir=staging_root) as tmp:
        checkout = Path(tmp) / "checkout"
        subprocess.run(
            ["git", "clone", "--no-checkout", spec["repository"], str(checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", spec["revision"]],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if revision != spec["revision"]:
            raise AcquisitionError(
                f"git revision mismatch: expected {spec['revision']}, observed {revision}"
            )
        files = []
        for item in spec["files"]:
            relative = Path(item["repository_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise AcquisitionError(f"unsafe repository path: {relative}")
            files.append(
                _atomic_copy(
                    checkout / relative,
                    _target_path(dataset_dir, item["target_name"]),
                    item.get("expected_sha256"),
                )
            )
    return {"status": "ACQUIRED", "revision": spec["revision"], "files": files}


def acquire_http_zip(
    dataset_id: str,
    spec: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    dataset_dir = raw_root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive = _target_path(dataset_dir, spec["archive_name"])
    if not archive.exists():
        temporary = archive.with_name(f".{archive.name}.part")
        if temporary.exists():
            temporary.unlink()
        _download(spec["source_url"], temporary)
        _check_expected(temporary, spec.get("archive_sha256"))
        os.replace(temporary, archive)
        archive.chmod(0o444)
    archive_sha = _check_expected(archive, spec.get("archive_sha256"))
    target = _target_path(dataset_dir, spec["target_name"])
    if not target.exists():
        temporary = target.with_name(f".{target.name}.part")
        with zipfile.ZipFile(archive) as bundle:
            members = {Path(name).name: name for name in bundle.namelist() if not name.endswith("/")}
            requested = Path(spec["archive_member"]).name
            if requested not in members:
                raise AcquisitionError(
                    f"archive member {requested!r} is absent; observed {sorted(members)}"
                )
            with bundle.open(members[requested]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        _check_expected(temporary, spec.get("expected_sha256"))
        os.replace(temporary, target)
        target.chmod(0o444)
    target_sha = _check_expected(target, spec.get("expected_sha256"))
    return {
        "status": "ACQUIRED",
        "files": [
            {"name": archive.name, "bytes": archive.stat().st_size, "sha256": archive_sha},
            {"name": target.name, "bytes": target.stat().st_size, "sha256": target_sha},
        ],
    }


def _materialize_kaggle_payload(
    payload: Path,
    required_name: str,
    staging: Path,
) -> Path:
    """Return the required CSV, extracting Kaggle's ZIP payload if needed.

    The Kaggle CLI can save a compressed response using the requested CSV file
    name rather than a ``.zip`` suffix. Detect the file format from its bytes;
    relying on the path suffix would silently register ZIP bytes as raw CSV.
    """

    if not zipfile.is_zipfile(payload):
        return payload

    extracted = staging / f".extracted-{required_name}"
    if extracted.exists():
        extracted.unlink()
    with zipfile.ZipFile(payload) as bundle:
        matches = [
            member
            for member in bundle.infolist()
            if not member.is_dir() and Path(member.filename).name == required_name
        ]
        if len(matches) != 1:
            observed = sorted(
                Path(member.filename).name
                for member in bundle.infolist()
                if not member.is_dir()
            )
            raise AcquisitionError(
                f"Kaggle archive for {required_name!r} must contain exactly one "
                f"matching member; observed {observed}"
            )
        with bundle.open(matches[0]) as source, extracted.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

    if zipfile.is_zipfile(extracted):
        raise AcquisitionError(f"nested archive returned for required CSV: {required_name}")
    return extracted


def acquire_kaggle_competition(
    dataset_id: str,
    spec: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    executable = shutil.which("kaggle")
    if executable is None:
        return {
            "status": "BLOCKED",
            "reason": "KAGGLE_CLI_OR_AUTHENTICATION_NOT_AVAILABLE",
            "required_files": spec["required_files"],
        }
    dataset_dir = raw_root / dataset_id
    staging_root = raw_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    files = []
    with tempfile.TemporaryDirectory(prefix=f"{dataset_id}-", dir=staging_root) as tmp:
        staging = Path(tmp)
        for name in spec["required_files"]:
            target = _target_path(dataset_dir, name)
            expected = spec.get("expected_sha256", {}).get(name)
            if target.is_file():
                if zipfile.is_zipfile(target):
                    raise AcquisitionError(
                        f"existing Kaggle target contains ZIP bytes, not CSV: {target}"
                    )
                observed = _check_expected(target, expected)
                files.append(
                    {
                        "name": target.name,
                        "bytes": target.stat().st_size,
                        "sha256": observed,
                    }
                )
                continue
            subprocess.run(
                [
                    executable,
                    "competitions",
                    "download",
                    "-c",
                    spec["competition"],
                    "-f",
                    name,
                    "-p",
                    str(staging),
                ],
                check=True,
            )
            candidates = [staging / name, staging / f"{name}.zip"]
            if candidates[0].is_file():
                payload = candidates[0]
            elif candidates[1].is_file():
                payload = candidates[1]
            else:
                raise AcquisitionError(f"Kaggle did not provide required file: {name}")
            source = _materialize_kaggle_payload(payload, name, staging)
            files.append(_atomic_copy(source, target, expected))
    return {"status": "ACQUIRED", "files": files}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def acquire(
    config_path: Path,
    repository_root: Path,
    dataset_ids: Iterable[str],
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "cof-seqgen-saf-acquisition-v1":
        raise AcquisitionError("unsupported acquisition config schema")
    raw_root = repository_root / config["raw_root"]
    manifest_root = repository_root / config["manifest_root"]
    requested = list(dataset_ids)
    unknown = sorted(set(requested) - set(config["datasets"]))
    if unknown:
        raise AcquisitionError(f"unknown datasets: {unknown}")
    results: dict[str, Any] = {}
    for dataset_id in requested:
        spec = config["datasets"][dataset_id]
        method = spec["method"]
        if method == "existing_local_verified":
            result = acquire_local_dataset(dataset_id, spec, raw_root, repository_root)
        elif method == "local_generator":
            result = acquire_local_generator(dataset_id, spec, repository_root)
        elif method == "git_snapshot":
            result = acquire_git_snapshot(dataset_id, spec, raw_root)
        elif method == "http_zip":
            result = acquire_http_zip(dataset_id, spec, raw_root)
        elif method == "kaggle_competition":
            result = acquire_kaggle_competition(dataset_id, spec, raw_root)
        else:
            raise AcquisitionError(f"unsupported acquisition method: {method}")
        result.update(
            {
                "dataset": dataset_id,
                "method": method,
                "source_url": spec.get("source_url") or spec.get("repository"),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        results[dataset_id] = result
        _write_json(manifest_root / f"{dataset_id}.json", result)
    all_results: dict[str, Any] = {}
    for dataset_id in config["datasets"]:
        manifest_path = manifest_root / f"{dataset_id}.json"
        if manifest_path.is_file():
            all_results[dataset_id] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
    summary = {
        "schema_version": config["schema_version"],
        "family": config["family"],
        "config_sha256": sha256_file(config_path),
        "datasets": all_results,
    }
    _write_json(manifest_root / "acquisition_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_v2/cof_seqgen_saf_acquisition.yaml"),
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    summary = acquire((root / args.config).resolve(), root, args.datasets)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
