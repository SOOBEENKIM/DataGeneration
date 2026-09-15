from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
import ctgan

from eval.full_statistics_v2_5 import (
    FLOOR_EPSILON,
    PRIMARY_COMPARATORS,
    PRIMARY_ENDPOINT,
    SECONDARY_COMPARATORS,
    SUPPORT_DIAGNOSTIC_BINS,
)
from generators.full_registry_v2_5 import baseline_registry


CONFIG_PATH = Path("configs/benchmark_v2/full_v2_5.yaml")
REQUIRED_DOCUMENTS = (
    Path("docs/benchmark_v2/preregistered_full_experiment_v2_5.md"),
    Path("docs/benchmark_v2/baseline_definitions_v2_5.md"),
    Path("docs/benchmark_v2/full_experiment_artifact_contract_v2_5.md"),
    Path(
        "docs/benchmark_v2/"
        "preregistered_full_experiment_v2_5_amendment_1.md"
    ),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    errors: list[str] = []
    if raw.get("schema_version") != "benchmark-v2.5-full-preregistered":
        errors.append("schema_version")
    if raw.get("configuration_role") != "full_experiment":
        errors.append("full configuration role")
    execution = raw.get("execution", {})
    if execution.get("full_experiment_authorized") is not False:
        errors.append("full experiment must remain unauthorized")
    if execution.get("five_seed_run_started") is not False:
        errors.append("five-seed run must remain unstarted")
    scope = raw.get("scope", {})
    if scope.get("scenarios") != ["joint_semimarkov_v2b"]:
        errors.append("scenario scope")
    if scope.get("kappas") != [1.0]:
        errors.append("kappa scope")
    if scope.get("model_seeds") != [1, 2, 3, 4, 5]:
        errors.append("seed scope")
    paths = raw.get("paths", {})
    if paths.get("artifact_root") != "artifacts/benchmark_v2_5":
        errors.append("artifact root")
    if paths.get("data_root") != "data/benchmark_v2_5":
        errors.append("data root")
    data = raw.get("data", {})
    if data.get("n_train") != 31_951:
        errors.append("learned train N")
    if data.get("test_usage") != "evaluation_only":
        errors.append("test usage")
    plan = raw.get("sampling_plan", {})
    if plan.get("test_labels_or_lengths_used") is not False:
        errors.append("sampling plan leaks test labels or lengths")
    if (
        plan.get("seed") != 10_001
        or plan.get("generated_once_from_train") is not True
        or plan.get("shared_by_all_generators_and_model_seeds") is not True
        or plan.get("model_seed_may_not_change_plan") is not True
        or plan.get("manifest_requires_full_sha256") is not True
    ):
        errors.append("shared SamplingPlan")
    endpoint = raw.get("evaluation", {}).get("primary_endpoint", {})
    if (
        endpoint.get("name")
        != PRIMARY_ENDPOINT
        or endpoint.get("direction") != "lower_is_better"
        or endpoint.get("sole_primary_endpoint") is not True
    ):
        errors.append("primary endpoint")
    evaluation = raw.get("evaluation", {})
    if tuple(evaluation.get("support_diagnostic_bins", ())) != (
        SUPPORT_DIAGNOSTIC_BINS
    ):
        errors.append("support diagnostic bins")
    if float(evaluation.get("floor_epsilon", -1)) != FLOOR_EPSILON:
        errors.append("floor epsilon")
    registry_ids = list(baseline_registry())
    configured_ids = list(raw.get("baselines", {}))
    if registry_ids != configured_ids:
        errors.append("baseline registry/config mismatch")
    for generator in (
        "ctgan_separate_class",
        "tvae_separate_class",
    ):
        model = raw["baselines"][generator]
        total = float(model["max_wall_seconds_total"])
        by_class = {
            int(label): float(value)
            for label, value in model["max_wall_seconds_per_class"].items()
        }
        if total != 7200 or by_class != {0: 3600.0, 1: 3600.0}:
            errors.append(f"{generator} class budget")
        steps = {
            int(label): int(value)
            for label, value in model["requested_steps_per_class"].items()
        }
        if sum(steps.values()) != int(model["requested_steps_total"]):
            errors.append(f"{generator} class steps")
        if model.get("package_version") != "ctgan==0.12.1":
            errors.append(f"{generator} package pin")
    if getattr(ctgan, "__version__", None) != "0.12.1":
        errors.append("installed ctgan version")
    for generator in ("neural_sequence", "cof_seqgen"):
        model = raw["baselines"][generator]
        if float(model["max_wall_seconds"]) != 7200:
            errors.append(f"{generator} budget")
    cof = raw["baselines"]["cof_seqgen"]
    if (
        int(cof.get("d_model", 0)) != 128
        or int(cof.get("n_layers", 0)) != 2
        or int(cof.get("diffusion_steps", 0)) != 50
        or int(cof.get("sampling_chunk_size", 0)) != 256
        or int(cof.get("batch_size", 0)) != 256
        or float(cof.get("optimizer", {}).get("lr", 0)) != 0.001
    ):
        errors.append("CoF full engineering configuration")
    for generator in (
        "independent_markov",
        "joint_markov",
        "plug_in_hmm",
        "plug_in_hsmm",
    ):
        if float(raw["baselines"][generator].get("max_wall_seconds", 0)) != 7200:
            errors.append(f"{generator} hard wall cap")
    for generator in ("plug_in_hmm", "plug_in_hsmm"):
        model = raw["baselines"][generator]
        if (
            model.get("uses_em") is not False
            or model.get("uses_latent_state_restarts") is not False
            or model.get("uses_test_data_for_fit") is not False
            or model.get("state_fit_split") != "train"
        ):
            errors.append(f"{generator} plug-in definition")
    for generator, model in raw.get("baselines", {}).items():
        requested = model.get(
            "requested_steps_total",
            model.get("requested_steps"),
        )
        interval = model.get("checkpoint_interval_steps")
        if requested is not None and interval is not None:
            maximum_interval = min(100, max(1, int(requested) // 10))
            if int(interval) > maximum_interval:
                errors.append(f"{generator} checkpoint interval")
    statistics = raw.get("statistics", {})
    if tuple(statistics.get("c2_primary_family", ())) != PRIMARY_COMPARATORS:
        errors.append("C2 primary Holm family")
    if (
        tuple(statistics.get("secondary_holm_family", ()))
        != SECONDARY_COMPARATORS
    ):
        errors.append("secondary Holm family")
    if statistics.get("sole_primary_endpoint") != PRIMARY_ENDPOINT:
        errors.append("statistics primary endpoint")
    preflight = raw.get("capacity_preflight", {})
    if (
        preflight.get("mode") != "timing_only"
        or preflight.get("full_run_authorized") is not False
        or int(preflight.get("max_updates", 0)) != 500
        or int(preflight.get("warmup_updates_excluded", 0)) != 20
        or int(preflight.get("measured_updates", 0)) != 480
        or int(preflight.get("sampling_chunk_size", 0)) != 256
        or float(preflight.get("fixed_overhead_seconds", 0)) != 600
        or preflight.get("durable_checkpoint_forbidden") is not True
        or preflight.get("durable_sample_forbidden") is not True
        or preflight.get("quality_metrics_forbidden") is not True
    ):
        errors.append("timing-only capacity preflight")
    if errors:
        raise ValueError(
            "invalid v2.5 preparation: " + ", ".join(errors)
        )
    return {
        "status": "PASS",
        "generator_count": len(registry_ids),
        "generator_ids": registry_ids,
        "full_experiment_authorized": False,
    }


def verify_preparation(
    config_path: Path = CONFIG_PATH,
    *,
    repository_root: Path = Path("."),
) -> Mapping[str, Any]:
    config_path = repository_root / config_path
    raw = yaml.safe_load(config_path.read_text())
    result = dict(validate_config(raw))
    smoke_suite = yaml.safe_load(
        (repository_root / "configs/benchmark_v2/smoke.yaml").read_text()
    )
    smoke_cof = yaml.safe_load(
        (repository_root / "configs/benchmark_v2/cof.yaml").read_text()
    )
    if (
        smoke_suite.get("schema_version")
        == raw.get("schema_version")
        or smoke_cof.get("smoke", {}).get("diffusion_steps") == raw[
            "baselines"
        ]["cof_seqgen"]["diffusion_steps"]
        or smoke_cof.get("smoke", {}).get("training_steps")
        == raw["baselines"]["cof_seqgen"]["requested_steps"]
    ):
        raise ValueError("smoke and full CoF configurations are not separated")
    missing = [
        str(path)
        for path in REQUIRED_DOCUMENTS
        if not (repository_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing v2.5 documents: {missing}")
    freeze = raw["freeze"]
    index_path = repository_root / freeze["v2_4_artifact_index"]["path"]
    gate_path = repository_root / freeze["v2_4_gate_report"]["path"]
    actual_index_hash = sha256_file(index_path)
    actual_gate_hash = sha256_file(gate_path)
    if actual_index_hash != freeze["v2_4_artifact_index"]["sha256"]:
        raise ValueError("preserved v2.4 artifact index hash changed")
    if actual_gate_hash != freeze["v2_4_gate_report"]["actual_sha256"]:
        raise ValueError("preserved v2.4 gate report hash changed")
    index = json.loads(index_path.read_text())
    if index["artifact_count"] != 1679 or len(index["artifacts"]) != 1679:
        raise ValueError("preserved v2.4 artifact index count changed")
    gate_record = next(
        record
        for record in index["artifacts"]
        if record["path"] == "gates/gate_report.json"
    )
    if gate_record["sha256"] != actual_gate_hash:
        raise ValueError("v2.4 indexed gate hash differs from actual")
    gate = json.loads(gate_path.read_text())
    if gate.get("full_experiment_authorized") is not False:
        raise ValueError("preserved v2.4 full authorization changed")
    result.update(
        {
            "config_sha256": sha256_file(config_path),
            "v2_4_artifact_index_sha256": actual_index_hash,
            "v2_4_artifact_count": 1679,
            "v2_4_gate_report_sha256": actual_gate_hash,
            "v2_4_gate_report_hash_matches_index": True,
            "v2_4_full_experiment_authorized": False,
        }
    )
    return result


def main() -> None:
    print(json.dumps(verify_preparation(), indent=2))


if __name__ == "__main__":
    main()
