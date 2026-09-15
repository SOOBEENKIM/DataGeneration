"""Append-only frozen-data materialization runner for external protocol v1.

Plan and dry-run are source/provenance checks only.  They never materialize
windows, fit transforms, write runtime data, import a model, or access CUDA.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from scripts.plan_external_sequence_protocol import build_external_protocol_plan


class ExternalMaterializationError(RuntimeError):
    """Raised when frozen materialization provenance or scope is invalid."""


ZERO_EXECUTION_COUNTS = {
    "data_write_calls": 0,
    "window_materialization_calls": 0,
    "transform_fit_calls": 0,
    "gpu_inventory_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "fidelity_calls": 0,
    "coherence_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "external_test_evaluation_calls": 0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ExternalMaterializationError("materialization config root must be a mapping")
    return value


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def audit_frozen_cof_compatibility(repo_root: Path) -> Mapping[str, Any]:
    """Statically prove external tensor shapes are constructor-derived.

    No frozen model module is imported or instantiated.  The audit reads the
    exact Git blobs named by the materialization config and verifies the
    adapter derives every relevant size from the training ``SequenceBatch``.
    """

    config_path = repo_root / "configs/benchmark_v2/external_frozen_materialization_v1.yaml"
    config = _load_config(config_path)
    contract = config["static_compatibility"]
    commit = str(contract["frozen_source_commit"])
    adapter = _git_blob(repo_root, commit, str(contract["adapter_path"]))
    if hashlib.sha256(adapter).hexdigest() != contract["adapter_sha256"]:
        raise ExternalMaterializationError("frozen CoF adapter hash mismatch")
    denoiser = _git_blob(repo_root, commit, str(contract["denoiser_path"]))
    adapter_text = adapter.decode("utf-8")
    denoiser_text = denoiser.decode("utf-8")
    required_adapter_tokens = (
        "bins = int(train.dt_bin.max()) + 1",
        "int(train.x_cat[..., i].max()) + 1",
        "train.x_num.shape[-1], bins, categories",
        "L_max=train.x_num.shape[1]",
    )
    required_denoiser_tokens = (
        "self.num_proj = nn.Linear(d_num, d_model)",
        "self.bin_emb = nn.Embedding(Bbins + 1, d_model)",
        "nn.Embedding(K + 1, d_model)",
        "self.pos_emb = nn.Embedding(L_max, d_model)",
    )
    missing = [
        token
        for token in required_adapter_tokens
        if token not in adapter_text
    ] + [token for token in required_denoiser_tokens if token not in denoiser_text]
    if missing:
        raise ExternalMaterializationError(
            f"frozen CoF dynamic-shape contract missing: {missing}"
        )
    datasets = {
        name: {
            "receiver_cardinality_upper_bound": int(cardinality),
            "compatible": True,
            "basis": "train.x_cat maximum dynamically constructs category embedding",
        }
        for name, cardinality in contract["receiver_cardinality_upper_bound"].items()
    }
    return {
        "status": "PASS",
        "frozen_source_commit": commit,
        "adapter_path": contract["adapter_path"],
        "adapter_sha256": hashlib.sha256(adapter).hexdigest(),
        "denoiser_path": contract["denoiser_path"],
        "denoiser_sha256": hashlib.sha256(denoiser).hexdigest(),
        "amount_input_width": int(contract["amount_input_width"]),
        "gap_bin_cardinality": int(contract["gap_bin_cardinality"]),
        "receiver_input_width": int(contract["receiver_input_width"]),
        "sequence_length": int(contract["sequence_length"]),
        "datasets": datasets,
        "architecture_change_required": False,
        "training_runner_created": False,
        "model_imports": 0,
        "model_calls": 0,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    """Convert numpy/scalar identities without permitting non-finite JSON."""

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if value == float("inf"):
            return "+inf"
        if value == float("-inf"):
            return "-inf"
        if value != value:
            raise ExternalMaterializationError("NaN is forbidden in materialization JSON")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        _json_ready(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _exclusive_bytes(path, payload)


def _sha256_tree_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tagged_identity(value: Any) -> Mapping[str, str]:
    if hasattr(value, "item"):
        value = value.item()
    return {"type": type(value).__name__, "value": str(value)}


def _batch_arrays(batch: Any) -> Mapping[str, Any]:
    import numpy as np

    entity_ids = np.asarray(batch.entity_ids)
    if entity_ids.dtype.kind == "O":
        entity_ids = entity_ids.astype(str)
    return {
        "x_num": np.ascontiguousarray(batch.x_num),
        "dt_bin": np.ascontiguousarray(batch.dt_bin),
        "x_cat": np.ascontiguousarray(batch.x_cat),
        "valid_mask": np.ascontiguousarray(batch.valid_mask),
        "y_entity": np.ascontiguousarray(batch.y_entity),
        "lengths": np.ascontiguousarray(batch.lengths),
        "entity_ids": np.ascontiguousarray(entity_ids),
    }


def _exclusive_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _split_items(prepared: Any) -> tuple[tuple[str, Any], ...]:
    return (
        ("train", prepared.train),
        ("validation", prepared.validation),
        ("internal_test", prepared.internal_test),
    )


def _validate_prepared_contract(prepared: Any) -> None:
    import numpy as np

    if prepared.transforms.fit_role != "train":
        raise ExternalMaterializationError("transform state was not fit on train")
    if any(int(value) != 0 for value in prepared.leakage_audit.values()):
        raise ExternalMaterializationError(
            f"entity/window/transaction leakage detected: {prepared.leakage_audit}"
        )
    entity_sets = {
        name: {window.entity_id for window in split.windows}
        for name, split in _split_items(prepared)
    }
    window_sets = {
        name: {
            (window.dataset, window.entity_id, int(window.ordinal))
            for window in split.windows
        }
        for name, split in _split_items(prepared)
    }
    transaction_sets = {
        name: {
            transaction_id
            for window in split.windows
            for transaction_id in window.transaction_ids
        }
        for name, split in _split_items(prepared)
    }
    pairs = (
        ("train", "validation"),
        ("train", "internal_test"),
        ("validation", "internal_test"),
    )
    independent = {
        "entity_overlap_count": sum(
            len(entity_sets[left] & entity_sets[right]) for left, right in pairs
        ),
        "window_overlap_count": sum(
            len(window_sets[left] & window_sets[right]) for left, right in pairs
        ),
        "transaction_overlap_count": sum(
            len(transaction_sets[left] & transaction_sets[right])
            for left, right in pairs
        ),
    }
    if any(independent.values()) or independent != prepared.leakage_audit:
        raise ExternalMaterializationError(
            f"independent leakage audit failed: {independent}"
        )
    train_valid = int(prepared.train.batch.valid_mask.sum())
    if prepared.transforms.fit_transaction_count != train_valid:
        raise ExternalMaterializationError("train fit-state row count mismatch")
    known_codes = set(prepared.transforms.receiver_to_code.values())
    allowed_codes = known_codes | {
        prepared.transforms.pad_code,
        prepared.transforms.unk_code,
    }
    if prepared.transforms.pad_code != 0 or prepared.transforms.unk_code != 1:
        raise ExternalMaterializationError("PAD/UNK code contract changed")
    for split_name, split in _split_items(prepared):
        batch = split.batch
        valid = batch.valid_mask
        codes = batch.x_cat[..., 0]
        if np.any(codes[~valid] != prepared.transforms.pad_code):
            raise ExternalMaterializationError(f"{split_name} padding is not PAD=0")
        if not set(np.unique(codes[valid]).tolist()).issubset(allowed_codes):
            raise ExternalMaterializationError(
                f"{split_name} contains a receiver outside train vocabulary/UNK"
            )
        for window, row_codes in zip(split.windows, codes, strict=True):
            receiver_column = (
                "RECEIVER_ACCOUNT_ID" if prepared.dataset == "amlsim" else "merchant"
            )
            expected = [
                prepared.transforms.receiver_to_code.get(
                    value, prepared.transforms.unk_code
                )
                for value in window.rows[receiver_column]
            ]
            if row_codes[: window.length].tolist() != expected:
                raise ExternalMaterializationError(
                    f"{split_name} unseen receiver is not encoded as UNK=1"
                )


def validate_prepared_materialization(prepared: Any) -> None:
    """Public fail-closed validation seam used before any artifact write."""

    _validate_prepared_contract(prepared)


def load_materialization_raw(
    path: Path,
    *,
    dataset: str,
    source_role: str,
) -> Any:
    """Load an authorized development CSV after role rejection.

    The adapter checks the Sparkov role and filename before calling pandas, so
    a public ``fraudTest`` source is rejected before CSV parsing.
    """

    from data.external_sequence_adapter import read_development_csv

    return read_development_csv(path, dataset=dataset, source_role=source_role)


def _transform_manifest(state: Any) -> Mapping[str, Any]:
    vocabulary = [
        {"identity": _tagged_identity(value), "code": int(code)}
        for value, code in sorted(
            state.receiver_to_code.items(), key=lambda item: int(item[1])
        )
    ]
    return {
        "schema_version": "external-train-transform-state-v1",
        "dataset": state.dataset,
        "fit_role": state.fit_role,
        "fit_transaction_count": int(state.fit_transaction_count),
        "fit_entity_count": int(state.fit_entity_count),
        "fit_transaction_ids_sha256": state.fit_transaction_ids_sha256,
        "fit_entity_ids_sha256": state.fit_entity_ids_sha256,
        "amount_log_mean": float(state.amount_log_mean),
        "amount_log_std": float(state.amount_log_std),
        "gap_edges": list(state.gap_edges),
        "gap_tau": list(state.gap_tau),
        "receiver_vocabulary": vocabulary,
        "pad_code": int(state.pad_code),
        "unk_code": int(state.unk_code),
        "state_sha256": state.state_sha256,
    }


def _split_manifest(prepared: Any) -> Mapping[str, Any]:
    assignment = prepared.split_assignment
    assignments = [
        {"entity": _tagged_identity(entity), "split": split}
        for entity, split in sorted(
            assignment.entity_to_split.items(), key=lambda item: str(item[0])
        )
    ]
    return {
        "schema_version": "external-entity-split-v1",
        "method": "group_stratified_by_retained_window_any_fraud_per_entity",
        "seed": int(assignment.seed),
        "assignments": assignments,
        "counts_by_label": assignment.counts_by_label,
    }


def _window_manifest(prepared: Any) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "external-nonoverlap-window-v1",
        "window_length": 32,
        "minimum_tail": 16,
        "stride": 32,
        "splits": {},
    }
    for split_name, split in _split_items(prepared):
        entries = []
        for window in split.windows:
            transaction_payload = [_tagged_identity(value) for value in window.transaction_ids]
            encoded = json.dumps(
                transaction_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            entries.append(
                {
                    "window_id": f"{prepared.dataset}:{window.entity_id}:{window.ordinal}",
                    "entity": _tagged_identity(window.entity_id),
                    "ordinal": int(window.ordinal),
                    "length": int(window.length),
                    "y": int(window.y_entity),
                    "transaction_count": len(window.transaction_ids),
                    "transaction_ids_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        result["splits"][split_name] = entries
    return result


def _summary(prepared: Any) -> Mapping[str, Any]:
    import numpy as np

    summaries: dict[str, Any] = {}
    for split_name, split in _split_items(prepared):
        batch = split.batch
        valid = batch.valid_mask
        lengths = batch.lengths.astype(int)
        receiver = batch.x_cat[..., 0]
        label_column = "IS_FRAUD" if prepared.dataset == "amlsim" else "is_fraud"
        transaction_fraud_count = sum(
            int(window.rows[label_column].astype(int).sum())
            for window in split.windows
        )
        valid_rows = int(valid.sum())
        summaries[split_name] = {
            "sequences": int(len(lengths)),
            "entities": int(len(set(str(value) for value in batch.entity_ids))),
            "valid_rows": valid_rows,
            "transaction_fraud_count": transaction_fraud_count,
            "transaction_fraud_prevalence": transaction_fraud_count / valid_rows,
            "positive_sequences": int(batch.y_entity.sum()),
            "positive_prevalence": float(batch.y_entity.mean()),
            "length_min": int(lengths.min()),
            "length_median": float(np.median(lengths)),
            "length_max": int(lengths.max()),
            "pad_count": int((~valid).sum()),
            "unk_count": int((receiver[valid] == prepared.transforms.unk_code).sum()),
        }
    return {
        "schema_version": "external-frozen-summary-v1",
        "dataset": prepared.dataset,
        "receiver_train_vocabulary_size": len(prepared.transforms.receiver_to_code),
        "receiver_vocabulary_cardinality": len(prepared.transforms.receiver_to_code) + 2,
        "pad_code": int(prepared.transforms.pad_code),
        "unk_code": int(prepared.transforms.unk_code),
        "splits": summaries,
    }


def write_frozen_dataset_attempt(
    *,
    prepared: Any,
    data_attempt: Path,
    artifact_attempt: Path,
    raw_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write one immutable materialization attempt; COMPLETE is written last."""

    if data_attempt.exists() or artifact_attempt.exists():
        raise ExternalMaterializationError("materialization attempt already exists")
    _validate_prepared_contract(prepared)
    data_attempt.parent.mkdir(parents=True, exist_ok=True)
    artifact_attempt.parent.mkdir(parents=True, exist_ok=True)
    data_attempt.mkdir()
    artifact_attempt.mkdir()
    _exclusive_json(
        artifact_attempt / "RUNNING.json",
        {
            "schema_version": "external-materialization-running-v1",
            "status": "RUNNING",
            "dataset": prepared.dataset,
            "created_at": _utc_now(),
        },
    )
    try:
        _exclusive_json(data_attempt / "raw_schema_manifest.json", dict(raw_manifest))
        _exclusive_json(data_attempt / "entity_split_manifest.json", _split_manifest(prepared))
        _exclusive_json(data_attempt / "window_manifest.json", _window_manifest(prepared))
        for split_name, split in _split_items(prepared):
            _exclusive_npz(data_attempt / f"{split_name}.npz", _batch_arrays(split.batch))
        transform_manifest = _transform_manifest(prepared.transforms)
        _exclusive_json(data_attempt / "train_transform_state.json", transform_manifest)
        _exclusive_json(data_attempt / "summary.json", _summary(prepared))
        _exclusive_json(
            data_attempt / "leakage_audit.json",
            {
                "schema_version": "external-leakage-audit-v1",
                "status": "PASS",
                **prepared.leakage_audit,
            },
        )
        split_hash = _sha256_file(data_attempt / "entity_split_manifest.json")
        transform_hash = _sha256_file(data_attempt / "train_transform_state.json")
        provenance_manifest = {
            "schema_version": "external-frozen-provenance-v1",
            "dataset": prepared.dataset,
            **dict(provenance),
            "raw_sha256": raw_manifest["sha256"],
            "split_manifest_sha256": split_hash,
            "transform_manifest_sha256": transform_hash,
            "transform_state_sha256": prepared.transforms.state_sha256,
        }
        _exclusive_json(data_attempt / "provenance_manifest.json", provenance_manifest)
        data_hashes = _sha256_tree_files(data_attempt)
        checksum_manifest = {
            "schema_version": "external-materialization-checksums-v1",
            "dataset": prepared.dataset,
            "data_files": data_hashes,
        }
        _exclusive_json(artifact_attempt / "checksum_manifest.json", checksum_manifest)
        artifact_index = {
            "schema_version": "external-materialization-artifact-index-v1",
            "dataset": prepared.dataset,
            "data_attempt": str(data_attempt),
            "data_tree_files": data_hashes,
            "checksum_manifest_sha256": _sha256_file(
                artifact_attempt / "checksum_manifest.json"
            ),
        }
        _exclusive_json(artifact_attempt / "artifact_index.json", artifact_index)
        complete = {
            "schema_version": "external-materialization-complete-v1",
            "status": "COMPLETE",
            "dataset": prepared.dataset,
            "completed_at": _utc_now(),
            "artifact_index_sha256": _sha256_file(
                artifact_attempt / "artifact_index.json"
            ),
            "checksum_manifest_sha256": _sha256_file(
                artifact_attempt / "checksum_manifest.json"
            ),
        }
        _exclusive_json(artifact_attempt / "COMPLETE.json", complete)
        return complete
    except BaseException as error:
        if not (artifact_attempt / "FAILED.json").exists():
            _exclusive_json(
                artifact_attempt / "FAILED.json",
                {
                    "schema_version": "external-materialization-failed-v1",
                    "status": "FAILED",
                    "dataset": prepared.dataset,
                    "failed_at": _utc_now(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                },
            )
        raise


def _validate_config(repo_root: Path, config: Mapping[str, Any]) -> Path:
    if config["status"] != "source_only_not_authorized":
        raise ExternalMaterializationError("materialization execution is not authorized")
    protocol_path = repo_root / config["protocol"]["path"]
    observed = _sha256_file(protocol_path)
    if observed != config["protocol"]["sha256"]:
        raise ExternalMaterializationError("external protocol config hash mismatch")
    runtime = config["runtime"]
    if (
        runtime["attempt"] != "attempt_001"
        or runtime["append_only"] is not True
        or runtime["overwrite"] != "forbidden"
        or runtime["authorization_required_for_execute"] is not True
    ):
        raise ExternalMaterializationError("append-only runtime contract mismatch")
    if config["datasets"] != ["amlsim", "sparkov"]:
        raise ExternalMaterializationError("materialization dataset scope changed")
    return protocol_path


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _verify_development_header(
    source: Mapping[str, Any],
    required_columns: Sequence[str],
) -> Mapping[str, Any]:
    path = Path(source["path"])
    if not path.is_file():
        raise ExternalMaterializationError(f"raw source missing: {path}")
    observed = _sha256_file(path)
    if observed != source["sha256"]:
        raise ExternalMaterializationError(f"raw source hash mismatch: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            columns = next(csv.reader(handle))
        except StopIteration as error:
            raise ExternalMaterializationError(f"raw source has no header: {path}") from error
    missing = sorted(set(required_columns) - set(columns))
    if missing:
        raise ExternalMaterializationError(
            f"raw schema is missing protocol columns: {missing}"
        )
    return {
        "path": str(path),
        "sha256": observed,
        "declared_rows": int(source["rows"]),
        "columns": columns,
        "required_columns": list(required_columns),
        "schema_contract": "PASS",
        "access": "read_only_hash_and_header",
    }


def _expected_counts_from_frozen_audit(repo_root: Path) -> Mapping[str, Any]:
    """Return body-free planning ranges from the already frozen feasibility audit."""

    audit_path = repo_root / "docs/benchmark_v2/external_validation_feasibility_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records = {
        "amlsim": audit["datasets"]["amlsim"]["audit"],
        "sparkov": audit["datasets"]["sparkov"]["train_audit"],
    }
    result: dict[str, Any] = {
        "source": "frozen_external_validation_feasibility_audit",
        "source_path": str(audit_path),
        "source_sha256": _sha256_file(audit_path),
        "exact_split_and_window_counts": "deferred_to_authorized_body_read",
        "reason": (
            "the frozen audit retained all nonempty tails, while protocol v1 drops "
            "tails shorter than 16 and uses a different label-stratified entity split"
        ),
    }
    for dataset, record in records.items():
        candidate_windows = int(record["fixed_event_windows"]["window_count"])
        entities = int(record["entity_transaction_counts"]["count"])
        result[dataset] = {
            "raw_rows": int(record["rows"]),
            "raw_entities": entities,
            "pre_minimum_tail_candidate_windows": candidate_windows,
            "protocol_retained_window_conservative_range": {
                "minimum": max(0, candidate_windows - entities),
                "maximum": candidate_windows,
            },
            "planned_entity_split_fractions": {
                "train": 0.70,
                "validation": 0.15,
                "internal_test": 0.15,
            },
            "exact_counts_artifact": "entity_split_manifest.json and window_manifest.json",
        }
    return result


def validate_materialization_authorization(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact future execute scope without accessing raw bodies."""

    config = _load_config(config_path)
    protocol_path = _validate_config(repo_root, config)
    protocol = _load_config(protocol_path)
    materializer_path = Path(__file__).resolve()
    expected_forbidden = {
        "gpu_inventory_query",
        "cuda",
        "model_fit",
        "model_sample",
        "fidelity",
        "coherence",
        "tstr",
        "privacy",
        "external_test_evaluation",
        "sparkov_public_fraudTest_parse",
    }
    fixed = (
        authorization.get("schema_version")
        == "external-frozen-materialization-authorization-v1"
        and authorization.get("scope")
        == "external_frozen_data_materialization_v1_only"
        and authorization.get("source_commit") == _git_head(repo_root)
        and authorization.get("materializer_source_sha256")
        == _sha256_file(materializer_path)
        and authorization.get("materialization_config_sha256")
        == _sha256_file(config_path)
        and authorization.get("protocol_config_sha256")
        == _sha256_file(protocol_path)
        and authorization.get("attempt") == config["runtime"]["attempt"]
        and authorization.get("allowed_operations") == ["frozen_data_materialization"]
        and set(authorization.get("forbidden_operations", [])) == expected_forbidden
    )
    if not fixed:
        raise ExternalMaterializationError("authorization provenance or scope mismatch")
    expected_datasets = {
        "amlsim": {
            "path": protocol["datasets"]["amlsim"]["development_source"]["path"],
            "sha256": protocol["datasets"]["amlsim"]["development_source"]["sha256"],
            "source_role": "transactions_development",
        },
        "sparkov": {
            "path": protocol["datasets"]["sparkov"]["development_source"]["path"],
            "sha256": protocol["datasets"]["sparkov"]["development_source"]["sha256"],
            "source_role": "fraudTrain_development",
        },
    }
    if authorization.get("datasets") != expected_datasets:
        raise ExternalMaterializationError("authorization dataset mismatch")
    return {
        "status": "PASS",
        "source_commit": authorization["source_commit"],
        "attempt": authorization["attempt"],
        "datasets": sorted(expected_datasets),
        "authorization_sha256": hashlib.sha256(
            json.dumps(
                authorization,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "raw_body_reads": 0,
    }


def execute_authorized_materialization(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Materialize both development datasets under an exact authorization.

    This entry point is intentionally not invoked by plan/dry-run.  It has no
    model or accelerator dependency and refuses any existing attempt before it
    reads either raw CSV body.
    """

    authorization_check = validate_materialization_authorization(
        repo_root=repo_root,
        config_path=config_path,
        authorization=authorization,
    )
    config = _load_config(config_path)
    protocol_path = _validate_config(repo_root, config)
    protocol = _load_config(protocol_path)
    compatibility = audit_frozen_cof_compatibility(repo_root)
    if compatibility["status"] != "PASS":
        raise ExternalMaterializationError("frozen non-v3 CoF compatibility blocker")
    attempt = config["runtime"]["attempt"]
    data_root = repo_root / config["runtime"]["data_root"]
    artifact_root = repo_root / config["runtime"]["artifact_root"]
    targets = {
        dataset: (
            data_root / dataset / attempt,
            artifact_root / dataset / attempt,
        )
        for dataset in config["datasets"]
    }
    if any(path.exists() for pair in targets.values() for path in pair):
        raise ExternalMaterializationError("materialization attempt already exists")

    from data.external_sequence_adapter import prepare_external_dataset

    results: dict[str, Any] = {}
    for dataset in config["datasets"]:
        authorized_source = authorization["datasets"][dataset]
        raw_path = Path(authorized_source["path"])
        if _sha256_file(raw_path) != authorized_source["sha256"]:
            raise ExternalMaterializationError(f"raw source hash mismatch: {dataset}")
        raw = load_materialization_raw(
            raw_path,
            dataset=dataset,
            source_role=authorized_source["source_role"],
        )
        if _sha256_file(raw_path) != authorized_source["sha256"]:
            raise ExternalMaterializationError(
                f"raw source changed while being read: {dataset}"
            )
        expected_rows = int(protocol["datasets"][dataset]["development_source"]["rows"])
        if len(raw) != expected_rows:
            raise ExternalMaterializationError(f"raw row count mismatch: {dataset}")
        prepared = prepare_external_dataset(
            raw,
            dataset=dataset,
            source_role=authorized_source["source_role"],
            split_seed=int(protocol["split"]["seed"]),
            gap_bins=int(protocol["transforms"]["gap"]["bins"]),
        )
        raw_manifest = {
            "schema_version": "external-raw-schema-manifest-v1",
            "dataset": dataset,
            "source_role": authorized_source["source_role"],
            "path": str(raw_path),
            "sha256": authorized_source["sha256"],
            "bytes": raw_path.stat().st_size,
            "rows": len(raw),
            "columns": raw.columns.tolist(),
            "dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
            "read_only": True,
        }
        results[dataset] = write_frozen_dataset_attempt(
            prepared=prepared,
            data_attempt=targets[dataset][0],
            artifact_attempt=targets[dataset][1],
            raw_manifest=raw_manifest,
            provenance={
                "source_commit": authorization_check["source_commit"],
                "materializer_source_sha256": authorization[
                    "materializer_source_sha256"
                ],
                "materialization_config_sha256": _sha256_file(config_path),
                "protocol_config_sha256": _sha256_file(protocol_path),
                "authorization_sha256": authorization_check["authorization_sha256"],
                "frozen_cof_compatibility": compatibility,
            },
        )
    return {
        "status": "COMPLETE",
        "attempt": attempt,
        "datasets": results,
        "authorization_sha256": authorization_check["authorization_sha256"],
    }


def build_materialization_plan(
    *,
    repo_root: Path,
    config_path: Path,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run"}:
        raise ExternalMaterializationError("source-only mode must be plan or dry-run")
    config = _load_config(config_path)
    protocol_path = _validate_config(repo_root, config)
    protocol_plan = build_external_protocol_plan(
        repo_root=repo_root,
        config_path=protocol_path,
        mode="plan",
    )
    protocol = _load_config(protocol_path)
    verified_raw_sources: dict[str, Any] = {}
    raw_reads = {"raw_hash_reads": 0, "raw_header_reads": 0, "raw_body_reads": 0}
    if mode == "dry-run":
        for dataset, result_name in (
            ("amlsim", "amlsim_development"),
            ("sparkov", "sparkov_development"),
        ):
            dataset_config = protocol["datasets"][dataset]
            required = [
                dataset_config[key]
                for key in (
                    "entity",
                    "timestamp",
                    "transaction_id",
                    "receiver",
                    "amount",
                    "fraud_label",
                )
            ]
            verified_raw_sources[result_name] = _verify_development_header(
                dataset_config["development_source"],
                required,
            )
            raw_reads["raw_hash_reads"] += 1
            raw_reads["raw_header_reads"] += 1
    compatibility = audit_frozen_cof_compatibility(repo_root)
    attempt = config["runtime"]["attempt"]
    data_root = Path(config["runtime"]["data_root"])
    artifact_root = Path(config["runtime"]["artifact_root"])
    datasets = {
        dataset: {
            "data_attempt_path": str(data_root / dataset / attempt),
            "artifact_attempt_path": str(artifact_root / dataset / attempt),
            "planned_outputs": dict(config["output_contract"]),
        }
        for dataset in config["datasets"]
    }
    return {
        "schema_version": "external-frozen-materialization-plan-v1",
        "mode": mode,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "protocol_config_path": str(protocol_path),
        "protocol_config_sha256": _sha256_file(protocol_path),
        "attempt": attempt,
        "datasets": datasets,
        "raw_read_counts": raw_reads,
        "verified_raw_sources": verified_raw_sources,
        "sparkov_public_test_access": {
            "hash_reads": 0,
            "header_reads": 0,
            "body_reads": 0,
            "csv_parse_calls": 0,
        },
        "expected_counts": _expected_counts_from_frozen_audit(repo_root),
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "runtime_artifacts_created": False,
        "execution_authorized": False,
        "frozen_cof_static_compatibility": compatibility,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "dry-run", "execute"), required=True)
    parser.add_argument("--authorization", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "execute":
        if args.authorization is None:
            raise ExternalMaterializationError(
                "execute requires an append-only authorization manifest"
            )
        authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
        result = execute_authorized_materialization(
            repo_root=args.repo_root.resolve(),
            config_path=args.config.resolve(),
            authorization=authorization,
        )
    else:
        if args.authorization is not None:
            raise ExternalMaterializationError(
                "plan/dry-run do not accept execution authorization"
            )
        result = build_materialization_plan(
            repo_root=args.repo_root.resolve(),
            config_path=args.config.resolve(),
            mode=args.mode,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
