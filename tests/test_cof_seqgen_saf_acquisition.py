from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest

from scripts.acquire_cof_seqgen_saf import (
    AcquisitionError,
    _materialize_kaggle_payload,
    acquire_kaggle_competition,
    acquire_local_dataset,
    sha256_file,
)


def test_local_acquisition_copies_only_verified_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\n1,2\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = {
        "files": [
            {
                "source_path": str(source),
                "target_name": "source.csv",
                "expected_sha256": expected,
            }
        ]
    }
    result = acquire_local_dataset("toy", spec, tmp_path / "raw", tmp_path)
    target = tmp_path / "raw" / "toy" / "source.csv"
    assert result["status"] == "ACQUIRED"
    assert sha256_file(target) == expected
    assert target.stat().st_mode & 0o222 == 0


def test_local_acquisition_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("not expected", encoding="utf-8")
    spec = {
        "files": [
            {
                "source_path": str(source),
                "target_name": "source.csv",
                "expected_sha256": "0" * 64,
            }
        ]
    }
    with pytest.raises(AcquisitionError, match="SHA-256 mismatch"):
        acquire_local_dataset("toy", spec, tmp_path / "raw", tmp_path)


def test_target_name_cannot_escape_dataset_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x", encoding="utf-8")
    spec = {
        "files": [
            {
                "source_path": str(source),
                "target_name": "../escape.csv",
                "expected_sha256": sha256_file(source),
            }
        ]
    }
    with pytest.raises(AcquisitionError, match="plain file name"):
        acquire_local_dataset("toy", spec, tmp_path / "raw", tmp_path)


def test_kaggle_zip_payload_is_detected_by_bytes_not_suffix(tmp_path: Path) -> None:
    payload = tmp_path / "transactions_train.csv"
    expected = b"entity_id,timestamp\na,2020-01-01\n"
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("transactions_train.csv", expected)

    materialized = _materialize_kaggle_payload(
        payload,
        "transactions_train.csv",
        tmp_path,
    )

    assert materialized != payload
    assert materialized.read_bytes() == expected
    assert not zipfile.is_zipfile(materialized)


def test_kaggle_plain_csv_payload_is_used_without_rewrite(tmp_path: Path) -> None:
    payload = tmp_path / "customers.csv"
    payload.write_bytes(b"customer_id\na\n")

    materialized = _materialize_kaggle_payload(payload, "customers.csv", tmp_path)

    assert materialized == payload


def test_kaggle_acquisition_reuses_hash_pinned_existing_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    target = raw_root / "hm" / "customers.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"customer_id\na\n")
    target.chmod(0o444)
    expected = sha256_file(target)
    monkeypatch.setattr("scripts.acquire_cof_seqgen_saf.shutil.which", lambda _: "/kaggle")
    spec = {
        "competition": "hm",
        "required_files": ["customers.csv"],
        "expected_sha256": {"customers.csv": expected},
    }

    result = acquire_kaggle_competition("hm", spec, raw_root)

    assert result["status"] == "ACQUIRED"
    assert result["files"] == [
        {"name": "customers.csv", "bytes": 14, "sha256": expected}
    ]
