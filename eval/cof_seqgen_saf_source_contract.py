"""Fail-closed source-definition contract for the new CoFSeqGen-SAF family."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


class SAFSourceContractError(RuntimeError):
    pass


TRUE_SAFETY_FLAGS = (
    "data_download_authorized",
    "raw_data_body_read_authorized",
    "preprocessing_authorized",
    "materialization_authorized",
    "canonical_test_structural_materialization_authorized",
    "metric_implementation_authorized",
    "train_only_metric_audit_authorized",
    "train_only_privacy_metric_audit_authorized",
    "baseline_compatibility_design_authorized",
    "model_implementation_authorized",
    "cpu_structural_unit_sampling_authorized",
)

FALSE_SAFETY_FLAGS = (
    "baseline_install_authorized",
    "runner_implementation_authorized",
    "gpu_query_authorized",
    "cuda_authorized",
    "fit_authorized",
    "sample_authorized",
    "evaluation_authorized",
    "validation_authorized",
    "test_authorized",
    "privacy_authorized",
    "claim_authorized",
)

EXPECTED_HYPOTHESES = {
    "SAF-H1": "support_alignment_improves_gap_fidelity",
    "SAF-H2": "ordered_hazard_outperforms_unordered_categorical_gap_decoding",
    "SAF-H3": "gap_to_mark_routing_improves_joint_dependency_recovery",
    "SAF-H4": "full_cofseqgen_saf_outperforms_current_sequential_generators_on_temporal_relational_fidelity",
    "SAF-H5": "marginal_fidelity_utility_and_privacy_are_noninferior",
}

EXPECTED_DATASETS = {
    "controlled_coupling_dgp": "REGISTERED_GENERATED_AND_CANONICAL_MATERIALIZED",
    "amlsim": "ACQUIRED_AND_CANONICAL_MATERIALIZED",
    "sparkov": "ACQUIRED_FRAUDTRAIN_AND_CANONICAL_MATERIALIZED",
    "berka": "ACQUIRED_PINNED_REVISION_AND_CANONICAL_MATERIALIZED",
    "hm": "ACQUIRED_HASH_PINNED_CSV_ONLY_AND_CANONICAL_MATERIALIZED",
    "citi_bike": "ACQUIRED_OFFICIAL_ARCHIVE_AND_CANONICAL_MATERIALIZED",
}

EXPECTED_BASELINES = (
    "tabularargn",
    "tabdit",
    "cpar",
    "realtabformer",
    "ctgan",
    "tvae",
    "gaussian_copula",
    "empirical_sequence_sampler",
    "non_v3_cof",
    "ccmtpp_c1",
    "hcmttpp_h1",
)


@dataclass(frozen=True)
class SAFSourceDefinition:
    repository_root: Path
    config_path: Path
    config_sha256: str
    raw: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SAFSourceContractError(f"cannot hash source contract: {path}") from error
    return digest.hexdigest()


def validate_saf_source_definition(raw: Mapping[str, Any]) -> None:
    if (
        raw.get("schema_version")
        != "cof-seqgen-saf-preexecution-source-protocol-v2"
        or raw.get("mode")
        != "METRIC_AUDIT_BASELINE_COMPATIBILITY_AND_MODEL_SOURCE_IMPLEMENTATION"
        or raw.get("family") != "cof_seqgen_saf"
        or raw.get("display_name") != "CoFSeqGen-SAF"
        or raw.get("current_state")
        != "PREEXECUTION_SOURCE_IMPLEMENTED_METRIC_AUDIT_COMPLETE"
    ):
        raise SAFSourceContractError("SAF family identity or authorized boundary changed")

    separation = raw.get("family_separation")
    if not isinstance(separation, Mapping) or any(
        (
            separation.get("h1_permanent_state")
            != "STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL",
            separation.get("h1_artifacts_mutable") is not False,
            separation.get("d1_family") != "cof_zdh_v3",
            separation.get("d1_status") != "SPECIFIED_NOT_IMPLEMENTED",
            separation.get("d1_role") != "PLANNED_DECODER_ABLATION_REFERENCE_ONLY",
            separation.get("d1_source_config")
            != "configs/benchmark_v2/cof_zdh_v3_d1_source_only.yaml",
        )
    ):
        raise SAFSourceContractError("H1/D1 separation boundary changed")

    hypotheses = raw.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 5:
        raise SAFSourceContractError("exactly five untested hypotheses are required")
    observed_hypotheses = {
        item.get("id"): item.get("statement") for item in hypotheses
    }
    if observed_hypotheses != EXPECTED_HYPOTHESES or any(
        item.get("status") != "UNTESTED" for item in hypotheses
    ):
        raise SAFSourceContractError("SAF hypothesis family changed or claims results")

    schema = raw.get("canonical_data_contract")
    if not isinstance(schema, Mapping) or any(
        (
            schema.get("schema_version")
            != "cofseqgen-saf-canonical-entity-sequence-v1",
            schema.get("split_before_preprocessing") is not True,
            schema.get("split_unit") != "entity_id",
            schema.get("split_roles") != ["train", "validation", "test"],
            schema.get("first_event_gap") != "MISSING_NOT_ZERO",
            schema.get("transform_fit_split") != "train",
            schema.get("entity_id_model_input") != "FORBIDDEN",
            schema.get("static_context_required") is not False,
            schema.get("fraud_label_required") is not False,
            schema.get("reserved_category_codes")
            != {"PAD": 0, "UNK": 1, "MISSING": 2, "FIRST_LEARNED": 3},
        )
    ):
        raise SAFSourceContractError("canonical data contract changed")
    if schema.get("event_columns") != [
        "entity_id",
        "event_id",
        "event_index",
        "timestamp",
        "gap",
        "receiver_or_mark",
        "amount_or_numeric_value",
    ]:
        raise SAFSourceContractError("canonical event columns changed")

    factorization = raw.get("planned_model_factorization")
    if not isinstance(factorization, Mapping) or factorization != {
        "semantics": "NON_ANTICIPATIVE_NOT_CAUSAL_EFFECT_IDENTIFICATION",
        "history": "PAST_VALID_EVENTS_ONLY",
        "within_event_order": [
            "gap_given_history_and_optional_static_context",
            "mark_given_history_optional_static_context_and_current_gap",
            "numeric_value_given_history_optional_static_context_current_gap_and_mark",
        ],
        "sequence_termination": "FUTURE_MODEL_COMPONENT_UNIMPLEMENTED",
        "implementation_status": "CORE_AND_FIVE_ABLATIONS_IMPLEMENTED_CPU_TESTED_NOT_TRAINED",
        "training_route": "OBSERVED_CURRENT_GAP_AND_MARK_TEACHER_FORCED",
        "generation_route": "GENERATED_CURRENT_GAP_AND_MARK",
        "first_gap_likelihood": "MASKED",
    }:
        raise SAFSourceContractError("planned SAF factorization changed")

    datasets = raw.get("datasets")
    if not isinstance(datasets, list) or {
        item.get("id"): item.get("acquisition_state") for item in datasets
    } != EXPECTED_DATASETS:
        raise SAFSourceContractError("planned dataset registry changed")
    if any(
        not isinstance(item.get("raw_body_read_count"), int)
        or item.get("raw_body_read_count") < 0
        for item in datasets
    ):
        raise SAFSourceContractError("raw-body read counters must be nonnegative integers")

    baselines = raw.get("planned_baselines")
    if baselines != list(EXPECTED_BASELINES):
        raise SAFSourceContractError("planned baseline registry changed")

    endpoint = raw.get("endpoint_policy")
    if not isinstance(endpoint, Mapping) or any(
        (
            endpoint.get("primary_family") != "temporal_relational_fidelity",
            endpoint.get("required_noninferiority_families")
            != ["marginal_fidelity", "utility", "privacy"],
            endpoint.get("exact_metrics_status")
            != "FIXED_BY_TRAIN_ONLY_VALIDITY_AUDIT",
            endpoint.get("metric_config")
            != "configs/benchmark_v2/cof_seqgen_saf_metric_audit.yaml",
            endpoint.get("numeric_margins_status")
            != "MARGINAL_AND_STRUCTURAL_Q95_FROM_31_TRAIN_ONLY_PSEUDO_SPLITS",
            endpoint.get("utility_decision_status")
            != "FROZEN_TSTR_TASKS_PAIRED_BASELINE_COMPARISON",
            endpoint.get("privacy_decision_status")
            != "FROZEN_COPY_EXPOSURE_GUARDS_AND_PAIRED_BASELINE_COMPARISON",
            endpoint.get("metric_audit_artifact")
            != "artifacts/cof_seqgen_saf/metric_validity_audit_train_only.json",
            endpoint.get("metric_audit_artifact_sha256")
            != "4c0cec50af9dceccad8a709cbf7393aa1523e8f1ff55e8f43db4d65638395ce4",
            endpoint.get("candidate_output_accessed_during_audit") is not False,
            endpoint.get("result_based_endpoint_switching") != "FORBIDDEN",
            endpoint.get("held_out_test_during_development") != "FORBIDDEN",
        )
    ):
        raise SAFSourceContractError("metric endpoint or audit provenance changed")

    safety = raw.get("safety")
    if (
        not isinstance(safety, Mapping)
        or any(safety.get(flag) is not True for flag in TRUE_SAFETY_FLAGS)
        or any(safety.get(flag) is not False for flag in FALSE_SAFETY_FLAGS)
    ):
        raise SAFSourceContractError("SAF acquisition safety boundary changed")

    implementation = raw.get("implementation")
    if not isinstance(implementation, Mapping) or any(
        (
            implementation.get("status")
            != "PREEXECUTION_SOURCE_IMPLEMENTED_METRIC_AUDIT_COMPLETE",
            implementation.get("model_path") != "models/cof_seqgen_saf.py",
            implementation.get("model_config_path")
            != "configs/benchmark_v2/cof_seqgen_saf_model.yaml",
            implementation.get("model_test_path")
            != "tests/test_cof_seqgen_saf_model.py",
            implementation.get("metric_path")
            != "benchmarks/cof_seqgen_saf_metrics.py",
            implementation.get("metric_config_path")
            != "configs/benchmark_v2/cof_seqgen_saf_metric_audit.yaml",
            implementation.get("metric_test_path")
            != "tests/test_cof_seqgen_saf_metric_audit.py",
            implementation.get("metric_audit_runner_path")
            != "scripts/audit_cof_seqgen_saf_metrics.py",
            implementation.get("metric_audit_artifact_path")
            != "artifacts/cof_seqgen_saf/metric_validity_audit_train_only.json",
            implementation.get("baseline_contract_path")
            != "generators/cof_seqgen_saf_baselines.py",
            implementation.get("baseline_config_path")
            != "configs/benchmark_v2/cof_seqgen_saf_baseline_compatibility.yaml",
            implementation.get("baseline_test_path")
            != "tests/test_cof_seqgen_saf_baseline_compatibility.py",
            implementation.get("runner_path") is not None,
            implementation.get("download_script_path")
            != "scripts/acquire_cof_seqgen_saf.py",
            implementation.get("acquisition_config_path")
            != "configs/benchmark_v2/cof_seqgen_saf_acquisition.yaml",
            implementation.get("runtime_root") != "data/cof_seqgen_saf",
            implementation.get("canonical_contract_path")
            != "data/cof_seqgen_saf_contract.py",
            implementation.get("contract_test_path")
            != "tests/test_cof_seqgen_saf_contract.py",
            implementation.get("adapter_path")
            != "data/cof_seqgen_saf_adapters.py",
            implementation.get("adapter_test_path")
            != "tests/test_cof_seqgen_saf_adapters.py",
            implementation.get("dataset_config_path")
            != "configs/benchmark_v2/cof_seqgen_saf_datasets.yaml",
            implementation.get("materializer_path")
            != "scripts/materialize_cof_seqgen_saf.py",
            implementation.get("verifier_path")
            != "scripts/verify_cof_seqgen_saf_data.py",
        )
    ):
        raise SAFSourceContractError("implementation scope changed beyond authorization")


def load_saf_source_definition(
    repository_root: str | Path,
    config_path: str | Path = "configs/benchmark_v2/cof_seqgen_saf_source_protocol.yaml",
) -> SAFSourceDefinition:
    root = Path(repository_root).resolve()
    path = (root / config_path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SAFSourceContractError(f"cannot read SAF source config: {path}") from error
    if not isinstance(raw, Mapping):
        raise SAFSourceContractError("SAF source config must be a mapping")
    validate_saf_source_definition(raw)
    documents = raw.get("documents")
    if documents != {
        "protocol": "docs/benchmark_v2/preregistered_cof_seqgen_saf_source_protocol.md",
        "data_collection_report": "docs/benchmark_v2/cof_seqgen_saf_data_collection_report_2026_08_26.md",
        "implementation_report": "docs/benchmark_v2/cof_seqgen_saf_preexecution_implementation_report_2026_08_26.md",
    }:
        raise SAFSourceContractError("SAF protocol document registry changed")
    required_paths = (
        documents["protocol"],
        documents["data_collection_report"],
        documents["implementation_report"],
        raw["implementation"]["canonical_contract_path"],
        raw["implementation"]["contract_test_path"],
        raw["implementation"]["adapter_path"],
        raw["implementation"]["adapter_test_path"],
        raw["implementation"]["dataset_config_path"],
        raw["implementation"]["materializer_path"],
        raw["implementation"]["verifier_path"],
        raw["implementation"]["acquisition_config_path"],
        raw["implementation"]["download_script_path"],
        raw["implementation"]["model_path"],
        raw["implementation"]["model_config_path"],
        raw["implementation"]["model_test_path"],
        raw["implementation"]["metric_path"],
        raw["implementation"]["metric_config_path"],
        raw["implementation"]["metric_test_path"],
        raw["implementation"]["metric_audit_runner_path"],
        raw["implementation"]["metric_audit_artifact_path"],
        raw["implementation"]["baseline_contract_path"],
        raw["implementation"]["baseline_config_path"],
        raw["implementation"]["baseline_test_path"],
        raw["family_separation"]["d1_source_config"],
    )
    for relative_path in required_paths:
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise SAFSourceContractError(
                f"required SAF source file escapes the repository: {relative_path}"
            ) from error
        if not candidate.is_file():
            raise SAFSourceContractError(f"required SAF source file is missing: {relative_path}")
    artifact_path = (root / raw["endpoint_policy"]["metric_audit_artifact"]).resolve()
    if _sha256(artifact_path) != raw["endpoint_policy"]["metric_audit_artifact_sha256"]:
        raise SAFSourceContractError("metric audit artifact hash does not match protocol")
    return SAFSourceDefinition(
        repository_root=root,
        config_path=path,
        config_sha256=_sha256(path),
        raw=raw,
    )
