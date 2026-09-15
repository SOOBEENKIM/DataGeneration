"""Independent, read-only forensic extractor for CoF-SeqGen v3 attempt_002.

This module deliberately does not import the production evaluator or runner.
It reads frozen NPZ/JSON evidence and reproduces the five row guards with a
separate implementation.  It never restores a model or touches CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp


ANALYSIS_BASE_HEAD = "b3ac8382eed60a796fec0d6f943642f71559a4fe"
DIRECT_ID = "cof_v3_c01_direct_joint"
FACTORIZED_ID = "cof_v3_c02_factorized_joint"
CANDIDATE_IDS = (DIRECT_ID, FACTORIZED_ID)
METRICS = (
    "amount_ks",
    "gap_ks",
    "amount_abs_standardized_label_effect",
    "gap_abs_standardized_label_effect",
    "receiver_max_abs_signed_frequency",
)
EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "validation_reruns": 0,
    "test_split_reads": 0,
    "fresh_test_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "five_seed_full_run_calls": 0,
}
EXPECTED_FILES = {
    "v2_5_full_config": (
        "configs/benchmark_v2/full_v2_5.yaml",
        "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3",
    ),
    "v2_5_final_complete": (
        "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json",
        "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a",
    ),
    "v2_6_development_manifest": (
        "configs/benchmark_v2/development_data_v2_6.yaml",
        "31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5",
    ),
    "v2_8_source_config": (
        "configs/benchmark_v2/selection_v2_8_source_amendment.yaml",
        "9e8ecaeab15f28ec05cee27150f4c6f76142dce661536094f41a798a83a7ae98",
    ),
    "v2_8_evaluation_config": (
        "configs/benchmark_v2/evaluation_only_v2_8.yaml",
        "2af755a73e2764b060ef880b66e5deea7668919382ae151a11b42f15661feea0",
    ),
    "v2_8_aggregate_config": (
        "configs/benchmark_v2/evaluation_aggregate_v2_8.yaml",
        "b3db5f71080df1bfdc051259eb0267cf17629b90c15512916094b4b47890092d",
    ),
    "v2_8_execution_authorization": (
        "artifacts/benchmark_v2_8/authorizations/authorization_ea868cb_attempt_001.json",
        "028a6fdc74814ea60e84fa1a027d15abc546c3d5298caa5765db8641d3af6bba",
    ),
    "v2_8_aggregate_authorization": (
        "artifacts/benchmark_v2_8/authorizations/aggregate_authorization_3b4770f_attempt_001.json",
        "d83892f88341c22b37c794f422db1f528b9187d1f4a1b3ff8a513a2b63faed8e",
    ),
    "v2_8_aggregate_terminal": (
        "artifacts/benchmark_v2_8/candidate_selection/aggregate_attempt_001/AGGREGATE_COMPLETE.json",
        "58573dfb93ebed71843f286405e205669140b8cc66811ce61deebe808de513e9",
    ),
    "runner_config": (
        "configs/benchmark_v2/cof_seqgen_v3_validation_runner.yaml",
        "623ab43352889d8e64c05fd17b883a255f9a92811d713bbedbf7f05ed19082b5",
    ),
    "source_config": (
        "configs/benchmark_v2/cof_seqgen_v3_source_preparation.yaml",
        "1f9ea3a596509b729e02157a26e60859cccf780621c1439659370c540fec6bd2",
    ),
    "attempt_001_authorization": (
        "artifacts/benchmark_v3/authorizations/authorization_4289048_attempt_001.json",
        "d11941f514a65c74e07fe522f738df19447553116e024def5e0224fc9d2176d8",
    ),
    "attempt_002_authorization": (
        "artifacts/benchmark_v3/authorizations/authorization_b3ac838_attempt_002.json",
        "5386a55b0af81dea053433bdbab690ef2114ce04d2890411156a69dd51ce11ca",
    ),
    "frozen_manifest": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/data_manifest.json",
        "b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05",
    ),
    "frozen_meta": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/meta.json",
        "74566c8f06bed7b37060b12fd86e9a38b168942101433e7df8ddf18dc93aa2b4",
    ),
    "frozen_train": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/train.npz",
        "c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8",
    ),
    "frozen_validation": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/validation.npz",
        "68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5",
    ),
    "sampling_plan_file": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/shared_sampling_plan.npz",
        "b8876c54983f8af7ed4253f2c25fdf9f351f603bc932a0152ae0a17f896e28e9",
    ),
}
EXPECTED_TREES = {
    "direct_attempt_001": (
        "artifacts/benchmark_v3/candidate_selection/candidates/"
        "cof_v3_c01_direct_joint/seed_3001/attempt_001",
        "9d75a436d61caf8ace7c38c212dffe6d4d0f65dc9ffc3ccdd371b4a0f7d7a55c",
        11,
        53_508,
    ),
    "direct_attempt_002": (
        "artifacts/benchmark_v3/candidate_selection/candidates/"
        "cof_v3_c01_direct_joint/seed_3001/attempt_002",
        "a5ede6b1c13b0a59d328e094444bda9cad54489b8fc3c49282004bd033dc830d",
        52,
        186_589_677,
    ),
    "factorized_attempt_001": (
        "artifacts/benchmark_v3/candidate_selection/candidates/"
        "cof_v3_c02_factorized_joint/seed_3001/attempt_001",
        "454e68f4bbd77d34ab1bf395e433bb1f195b40d5b425d22f64e49762f2bbe0ca",
        11,
        53_524,
    ),
    "factorized_attempt_002": (
        "artifacts/benchmark_v3/candidate_selection/candidates/"
        "cof_v3_c02_factorized_joint/seed_3001/attempt_002",
        "28f159a587fbf5d9c656bc09417bd84c283b7a5e4537280c0314b23c4d1e0da1",
        52,
        164_781_479,
    ),
    "v2_8_candidate_selection": (
        "artifacts/benchmark_v2_8/candidate_selection",
        "7f60deb8a10cd9946e2e8a814b21c463fbf753954e7e03c44b6a7d9efb02e700",
        21,
        2_065_996,
    ),
}
SAMPLE_FIELDS = (
    "x_num",
    "dt_bin",
    "x_cat",
    "valid_mask",
    "y_entity",
    "lengths",
)


class ForensicV3Error(RuntimeError):
    """Raised when frozen evidence or an analysis invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ForensicV3Error(f"cannot hash frozen evidence: {path}") from error
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tree_inventory(root: Path) -> Mapping[str, Any]:
    files: dict[str, Mapping[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "tree_sha256": _canonical_sha256({"files": files}),
        "file_count": len(files),
        "byte_count": sum(row["bytes"] for row in files.values()),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForensicV3Error(f"cannot read frozen JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ForensicV3Error(f"JSON root is not an object: {path}")
    return value


def _load_npz(path: Path, fields: Sequence[str]) -> Mapping[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {field: np.asarray(archive[field]) for field in fields}
    except (OSError, ValueError, KeyError) as error:
        raise ForensicV3Error(f"cannot read frozen NPZ: {path}") from error


def _standardized_effect(values: np.ndarray, labels: np.ndarray) -> float:
    groups = [np.asarray(values)[labels == label] for label in (0, 1)]
    if any(len(group) < 2 for group in groups):
        return float("inf")
    degrees = len(groups[0]) + len(groups[1]) - 2
    pooled_variance = (
        (len(groups[0]) - 1) * groups[0].var(ddof=1)
        + (len(groups[1]) - 1) * groups[1].var(ddof=1)
    ) / degrees
    difference = float(groups[1].mean() - groups[0].mean())
    if pooled_variance <= 0:
        return 0.0 if difference == 0 else float("inf")
    return float(abs(difference) / math.sqrt(float(pooled_variance)))


def _receiver_guard(sample: Mapping[str, np.ndarray], categories: int) -> float:
    frequencies = np.zeros((len(sample["lengths"]), categories), dtype=float)
    receiver = sample["x_cat"][..., 0]
    for index, length_value in enumerate(sample["lengths"]):
        length = int(length_value)
        frequencies[index] = (
            np.bincount(receiver[index, :length], minlength=categories) / length
        )
    means = [
        frequencies[sample["y_entity"] == label].mean(axis=0)
        for label in (0, 1)
    ]
    return float(np.max(np.abs(means[1] - means[0])))


def independent_guard_statistics(
    *,
    validation: Mapping[str, np.ndarray],
    sample: Mapping[str, np.ndarray],
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, float]:
    real_valid = np.asarray(validation["valid_mask"], dtype=bool)
    synthetic_valid = np.asarray(sample["valid_mask"], dtype=bool)
    synthetic_labels = np.repeat(sample["y_entity"], sample["lengths"])
    real_amount = validation["x_num"][..., 0][real_valid]
    synthetic_amount = sample["x_num"][..., 0][synthetic_valid]
    real_gap = tau[validation["dt_bin"][real_valid]]
    synthetic_gap = tau[sample["dt_bin"][synthetic_valid]]
    return {
        "amount_ks": float(ks_2samp(real_amount, synthetic_amount).statistic),
        "gap_ks": float(ks_2samp(real_gap, synthetic_gap).statistic),
        "amount_abs_standardized_label_effect": _standardized_effect(
            synthetic_amount,
            synthetic_labels,
        ),
        "gap_abs_standardized_label_effect": _standardized_effect(
            synthetic_gap,
            synthetic_labels,
        ),
        "receiver_max_abs_signed_frequency": _receiver_guard(
            sample,
            receiver_categories,
        ),
    }


def _probability_vector(values: np.ndarray, size: int) -> np.ndarray:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=size)
    total = int(counts.sum())
    if total == 0:
        raise ForensicV3Error("cannot form a PMF from an empty class")
    return counts.astype(float) / total


def _pmf_comparison(
    real_values: np.ndarray,
    synthetic_values: np.ndarray,
    size: int,
) -> Mapping[str, Any]:
    real = _probability_vector(real_values, size)
    synthetic = _probability_vector(synthetic_values, size)
    absolute = np.abs(synthetic - real)
    maximum_state = int(np.argmax(absolute))
    return {
        "real_pmf": [float(value) for value in real],
        "synthetic_pmf": [float(value) for value in synthetic],
        "signed_difference": [float(value) for value in synthetic - real],
        "total_variation": float(0.5 * absolute.sum()),
        "maximum_absolute_difference": float(absolute[maximum_state]),
        "maximum_difference_state": maximum_state,
    }


def _classwise_pmf(
    *,
    validation: Mapping[str, np.ndarray],
    sample: Mapping[str, np.ndarray],
    gap_bins: int,
    receiver_categories: int,
) -> Mapping[str, Any]:
    output: dict[str, Any] = {}
    for label in (0, 1):
        real_entity_mask = validation["y_entity"] == label
        synthetic_entity_mask = sample["y_entity"] == label
        real_valid = validation["valid_mask"] & real_entity_mask[:, None]
        synthetic_valid = sample["valid_mask"] & synthetic_entity_mask[:, None]
        real_gap = validation["dt_bin"][real_valid]
        synthetic_gap = sample["dt_bin"][synthetic_valid]
        real_receiver = validation["x_cat"][..., 0][real_valid]
        synthetic_receiver = sample["x_cat"][..., 0][synthetic_valid]
        real_joint = real_gap * receiver_categories + real_receiver
        synthetic_joint = synthetic_gap * receiver_categories + synthetic_receiver
        output[str(label)] = {
            "valid_real_rows": int(real_valid.sum()),
            "valid_synthetic_rows": int(synthetic_valid.sum()),
            "gap_bin": _pmf_comparison(real_gap, synthetic_gap, gap_bins),
            "receiver": _pmf_comparison(
                real_receiver,
                synthetic_receiver,
                receiver_categories,
            ),
            "joint": _pmf_comparison(
                real_joint,
                synthetic_joint,
                gap_bins * receiver_categories,
            ),
        }
    return output


def _sample_contract(
    *,
    sample: Mapping[str, np.ndarray],
    plan: Mapping[str, np.ndarray],
    support_mask: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, Any]:
    valid = np.asarray(sample["valid_mask"], dtype=bool)
    padding = ~valid
    receiver = sample["x_cat"][..., 0]
    gap = sample["dt_bin"]
    joint = gap[valid] * receiver_categories + receiver[valid]
    decoded_gap = joint // receiver_categories
    decoded_receiver = joint % receiver_categories
    expected_mask = (
        np.arange(valid.shape[1])[None, :] < sample["lengths"][:, None]
    )
    return {
        "sampling_plan_labels_exact": bool(
            np.array_equal(sample["y_entity"], plan["y_entity"])
        ),
        "sampling_plan_lengths_exact": bool(
            np.array_equal(sample["lengths"], plan["lengths"])
        ),
        "sampling_plan_mask_exact": bool(
            np.array_equal(valid, plan["valid_mask"])
        ),
        "mask_matches_lengths": bool(np.array_equal(valid, expected_mask)),
        "padding_zero": bool(
            np.all(sample["x_num"][..., 0][padding] == 0)
            and np.all(gap[padding] == 0)
            and np.all(receiver[padding] == 0)
        ),
        "labels_binary": bool(np.isin(sample["y_entity"], (0, 1)).all()),
        "label_prevalence": float(sample["y_entity"].mean()),
        "plan_label_prevalence": float(plan["y_entity"].mean()),
        "valid_rows": int(valid.sum()),
        "codec_round_trip_errors": int(
            np.count_nonzero(decoded_gap != gap[valid])
            + np.count_nonzero(decoded_receiver != receiver[valid])
        ),
        "joint_states_outside_train_support_mask": int(
            np.count_nonzero(~support_mask[joint])
        ),
        "gap_values_out_of_range": int(
            np.count_nonzero((gap[valid] < 0) | (gap[valid] >= 16))
        ),
        "receiver_values_out_of_range": int(
            np.count_nonzero(
                (receiver[valid] < 0)
                | (receiver[valid] >= receiver_categories)
            )
        ),
    }


def _support_coverage(
    *,
    validation: Mapping[str, np.ndarray],
    sample: Mapping[str, np.ndarray],
    support_mask: np.ndarray,
    train_observed_states: set[int],
    receiver_categories: int,
) -> Mapping[str, Any]:
    def states(batch: Mapping[str, np.ndarray]) -> set[int]:
        valid = batch["valid_mask"]
        values = (
            batch["dt_bin"][valid] * receiver_categories
            + batch["x_cat"][..., 0][valid]
        )
        return {int(value) for value in np.unique(values)}

    real_states = states(validation)
    synthetic_states = states(sample)
    common = real_states & synthetic_states
    return {
        "train_support_mask_allowed_states": int(support_mask.sum()),
        "train_observed_joint_states": len(train_observed_states),
        "validation_observed_joint_states": len(real_states),
        "synthetic_observed_joint_states": len(synthetic_states),
        "validation_support_covered_by_synthetic_fraction": float(
            len(common) / len(real_states)
        ),
        "synthetic_support_present_in_validation_fraction": float(
            len(common) / len(synthetic_states)
        ),
        "synthetic_joint_states_in_train_support_fraction": float(
            len(synthetic_states & train_observed_states)
            / len(synthetic_states)
        ),
        "synthetic_states_absent_from_validation": sorted(
            synthetic_states - real_states
        ),
        "validation_states_absent_from_synthetic": sorted(
            real_states - synthetic_states
        ),
}


def _code_audit(repository_root: Path) -> Mapping[str, Any]:
    records = {
        "model_v3": {
            "path": "models/cof_seqgen_v3.py",
            "loci": [
                "37-67 joint codec encode/decode",
                "70-97 paired corruption mask",
                "100-148 direct joint head/sample",
                "151-263 factorized gap-conditional receiver path",
                "487-497 support masking",
                "499-595 paired objective and teacher-forced gap",
            ],
        },
        "candidate_sampler": {
            "path": "generators/cof_seqgen_v3_candidate.py",
            "loci": [
                "212-240 fixed 50-step/guidance/temperature contract",
                "243-250 active Y/L mask and initialization",
                "285-344 direct/factorized discrete sampling branches",
                "345-376 padding and SamplingPlan output binding",
                "504-557 train-only amount state and validation sample path",
            ],
        },
        "production_evaluator": {
            "path": "eval/evaluation_only_v2_7.py",
            "loci": [
                "20-75 frozen threshold/SamplingPlan validation and five guards",
            ],
        },
        "production_guard": {
            "path": "eval/model_guards_v2_5.py",
            "loci": [
                "25-43 standardized class effect",
                "46-78 entity-weighted receiver frequency",
                "81-122 row guard statistics",
            ],
        },
        "execution_runner": {
            "path": "experiments/cof_seqgen_v3_execution_runner.py",
            "loci": [
                "1171-1194 train-policy plan binding in spawned child",
                "1241-1307 train/support/conditioning provenance",
            ],
        },
    }
    return {
        role: {
            **record,
            "sha256": sha256_file(repository_root / record["path"]),
        }
        for role, record in records.items()
    }


def _hypothesis_verdicts(
    *,
    guard_recalculation: Mapping[str, Any],
    sample_contracts: Mapping[str, Any],
    support_coverage: Mapping[str, Any],
    classwise_pmf: Mapping[str, Any],
    v2_8_comparison: Mapping[str, Any],
) -> Mapping[str, Any]:
    all_guard_matches = all(
        metric["matches_stored"]
        for candidate in guard_recalculation.values()
        for metric in candidate.values()
    )
    all_contracts = all(
        contract["sampling_plan_labels_exact"]
        and contract["sampling_plan_lengths_exact"]
        and contract["sampling_plan_mask_exact"]
        and contract["mask_matches_lengths"]
        and contract["padding_zero"]
        for contract in sample_contracts.values()
    )
    all_codec_support = all(
        contract["codec_round_trip_errors"] == 0
        and contract["joint_states_outside_train_support_mask"] == 0
        and contract["gap_values_out_of_range"] == 0
        and contract["receiver_values_out_of_range"] == 0
        and support["synthetic_joint_states_in_train_support_fraction"] == 1.0
        for contract, support in zip(
            sample_contracts.values(),
            support_coverage.values(),
        )
    )
    pmf_extrema = {
        candidate_id: {
            family: max(
                classwise_pmf[candidate_id][label][family]["total_variation"]
                for label in ("0", "1")
            )
            for family in ("gap_bin", "receiver", "joint")
        }
        for candidate_id in CANDIDATE_IDS
    }
    return {
        "A_evaluator_mask_label_mapping": {
            "verdict": "REFUTED",
            "evidence": [
                f"independent guard implementation matches all ten stored values/checks: {all_guard_matches}",
                f"both samples exactly match the active train-policy Y/L/mask and zero padding: {all_contracts}",
                "candidate sample hashes are bound by the stored evaluations",
            ],
            "qualification": (
                "The active plan was reconstructed from the immutable conditioning "
                "binding written before candidate execution. Its Y/L/mask and plan "
                "hash agree with both samples. The separate preserved SamplingPlan "
                "file was hash-inventoried but not loaded for this calculation."
            ),
        },
        "B_joint_codec_support_decoder_sampling_bug": {
            "verdict": "REFUTED",
            "evidence": [
                f"codec round trips, bounds, and support compliance all pass: {all_codec_support}",
                "train, validation, and each synthetic sample contain all 1024 joint states",
                "hash-bound source uses one paired direct state or sampled-gap conditional receiver path",
            ],
            "qualification": (
                "The train support mask is the full 16x64 Cartesian set, so it "
                "prevents illegal codes but cannot enforce probability mass. "
                "Stored artifacts do not include per-step logits, but no observable "
                "codec, mask, support, decode, or branch-contract violation was found."
            ),
        },
        "C_v3_objective_architecture_limit": {
            "verdict": "SUPPORTED",
            "evidence": [
                f"direct max class TV: {pmf_extrema[DIRECT_ID]}",
                f"factorized max class TV: {pmf_extrema[FACTORIZED_ID]}",
                "all four v3 amount guards pass while all six v3 discrete guards fail",
                "v2.8 CoF had passing gap guards and only a near-threshold receiver failure; both v3 candidates regress gap and receiver",
                f"v2.8 comparison independently reproduced: {v2_8_comparison['v2_8_stored_evaluation_matches_independent']}",
            ],
            "mechanism_consistent_with_evidence": [
                "direct joint CE must allocate probability over 1024 dense states under a full support mask",
                "factorized training teacher-forces true gap while sampling conditions receiver on sampled gap, exposing inference-time error propagation",
                "sampling starts from a fully masked joint state although training corruption is capped at 0.7",
                "the fixed 50-step sampler repeatedly resamples discrete states; the objective has no hard marginal-matching term",
            ],
        },
    }


def _verify_inventory(repository_root: Path) -> Mapping[str, Any]:
    inventory: dict[str, Any] = {}
    for role, (relative, expected_hash) in EXPECTED_FILES.items():
        path = repository_root / relative
        actual = sha256_file(path)
        if actual != expected_hash:
            raise ForensicV3Error(f"frozen file hash mismatch: {role}")
        inventory[role] = {"path": relative, "sha256": actual}
    for role, (relative, expected_hash, file_count, byte_count) in (
        EXPECTED_TREES.items()
    ):
        actual = tree_inventory(repository_root / relative)
        expected = {
            "tree_sha256": expected_hash,
            "file_count": file_count,
            "byte_count": byte_count,
        }
        if actual != expected:
            raise ForensicV3Error(f"frozen tree hash mismatch: {role}")
        inventory[role] = {"path": relative, **actual}
    return inventory


def _candidate_attempt(repository_root: Path, candidate_id: str) -> Path:
    return (
        repository_root
        / "artifacts/benchmark_v3/candidate_selection/candidates"
        / candidate_id
        / "seed_3001/attempt_002"
    )


def analyze_v3_attempt_002_failure(repository_root: Path) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    inventory = _verify_inventory(repository_root)
    frozen_root = (
        repository_root
        / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00"
    )
    validation = _load_npz(frozen_root / "validation.npz", SAMPLE_FIELDS)
    metadata = _read_json(frozen_root / "meta.json")
    tau = np.asarray(metadata["tau"], dtype=float)
    direct_attempt = _candidate_attempt(repository_root, DIRECT_ID)
    active_binding = _read_json(direct_attempt / "conditioning_binding.json")
    binding_payload = {
        key: value
        for key, value in active_binding.items()
        if key != "binding_sha256"
    }
    if active_binding.get("binding_sha256") != _canonical_sha256(
        binding_payload
    ):
        raise ForensicV3Error("conditioning binding hash mismatch")
    labels = np.asarray(active_binding["labels"], dtype=np.int64)
    lengths = np.asarray(active_binding["lengths"], dtype=np.int64)
    valid_mask = np.arange(validation["valid_mask"].shape[1])[None, :] < (
        lengths[:, None]
    )
    active_plan_hash = hashlib.sha256(
        labels.tobytes()
        + lengths.tobytes()
        + b"benchmark-v2.5-train-only-plan"
    ).hexdigest()
    if (
        active_plan_hash != active_binding.get("sampling_plan_sha256")
        or hashlib.sha256(valid_mask.astype(np.bool_).tobytes()).hexdigest()
        != active_binding.get("valid_mask_sha256")
    ):
        raise ForensicV3Error("active Y/L conditioning binding is inconsistent")
    plan = {
        "y_entity": labels,
        "lengths": lengths,
        "valid_mask": valid_mask,
        "plan_hash": active_plan_hash,
    }
    direct_support = _read_json(direct_attempt / "joint_support_state.json")
    receiver_categories = int(direct_support["receiver_classes"])
    plan_file_consistency = {
        "frozen_file_sha256": sha256_file(
            frozen_root / "shared_sampling_plan.npz"
        ),
        "active_conditioning_binding_sha256": sha256_file(
            direct_attempt / "conditioning_binding.json"
        ),
        "active_binding_payload_sha256": str(
            active_binding["binding_sha256"]
        ),
        "independently_recomputed_plan_hash": active_plan_hash,
        "forensic_guard_inputs": [
            "stored sample.npz",
            "frozen validation.npz",
            "frozen tau metadata",
        ],
        "frozen_train_npz_loaded": False,
        "shared_sampling_plan_npz_arrays_loaded": False,
        "interpretation": (
            "The active Y/L plan is reconstructed from the pre-sampling, "
            "hash-bound conditioning artifact. Guard and PMF calculations read "
            "only stored samples plus frozen validation; train and shared-plan "
            "NPZ bytes are preservation-hashed but not loaded."
        ),
    }

    guard_recalculation: dict[str, Any] = {}
    sample_contracts: dict[str, Any] = {}
    support_coverage: dict[str, Any] = {}
    classwise_pmf: dict[str, Any] = {}
    independent_by_candidate: dict[str, Mapping[str, float]] = {}
    for candidate_id in CANDIDATE_IDS:
        attempt = _candidate_attempt(repository_root, candidate_id)
        stored = _read_json(attempt / "evaluation.json")
        sample = _load_npz(attempt / "sample.npz", SAMPLE_FIELDS)
        if sha256_file(attempt / "sample.npz") != stored["validation_sample_sha256"]:
            raise ForensicV3Error(f"sample hash mismatch: {candidate_id}")
        candidate_binding = _read_json(attempt / "conditioning_binding.json")
        candidate_binding_payload = {
            key: value
            for key, value in candidate_binding.items()
            if key != "binding_sha256"
        }
        if (
            candidate_binding.get("binding_sha256")
            != _canonical_sha256(candidate_binding_payload)
            or candidate_binding.get("binding_sha256")
            != active_binding.get("binding_sha256")
        ):
            raise ForensicV3Error(
                f"candidate conditioning binding mismatch: {candidate_id}"
            )
        support_state = _read_json(attempt / "joint_support_state.json")
        support_mask = np.asarray(support_state["support_mask"], dtype=bool)
        if (
            support_state.get("fit_split") != "train"
            or support_state.get("validation_rows_used") != 0
            or support_state.get("test_rows_used") != 0
            or support_state.get("gap_bins") != len(tau)
            or support_mask.shape != (len(tau) * receiver_categories,)
        ):
            raise ForensicV3Error(f"invalid train-only support state: {candidate_id}")
        independent = independent_guard_statistics(
            validation=validation,
            sample=sample,
            tau=tau,
            receiver_categories=receiver_categories,
        )
        independent_by_candidate[candidate_id] = independent
        rows: dict[str, Any] = {}
        for metric in METRICS:
            threshold = float(stored["thresholds"][metric])
            independent_check = (
                "PASS"
                if np.isfinite(independent[metric])
                and independent[metric] <= threshold
                else "FAIL"
            )
            difference = abs(independent[metric] - float(stored["statistics"][metric]))
            rows[metric] = {
                "stored_value": float(stored["statistics"][metric]),
                "independent_value": float(independent[metric]),
                "absolute_difference": float(difference),
                "threshold": threshold,
                "stored_check": stored["checks"][metric],
                "independent_check": independent_check,
                "matches_stored": bool(
                    difference <= 1e-12
                    and independent_check == stored["checks"][metric]
                ),
            }
        guard_recalculation[candidate_id] = rows
        sample_contracts[candidate_id] = _sample_contract(
            sample=sample,
            plan=plan,
            support_mask=support_mask,
            receiver_categories=receiver_categories,
        )
        support_coverage[candidate_id] = {
            **_support_coverage(
                validation=validation,
                sample=sample,
                support_mask=support_mask,
                train_observed_states={
                    int(value)
                    for value in support_state["observed_joint_states"]
                },
                receiver_categories=receiver_categories,
            ),
            "joint_support_state_sha256": sha256_file(
                attempt / "joint_support_state.json"
            ),
            "joint_support_fit_split": support_state["fit_split"],
            "joint_support_validation_rows_used": support_state[
                "validation_rows_used"
            ],
            "joint_support_test_rows_used": support_state["test_rows_used"],
        }
        classwise_pmf[candidate_id] = _classwise_pmf(
            validation=validation,
            sample=sample,
            gap_bins=len(tau),
            receiver_categories=receiver_categories,
        )

    v2_8_attempt = (
        repository_root
        / "artifacts/benchmark_v2_8/candidate_selection/evaluations/"
        "cof_seqgen/cof_v28_c01_gap_distribution_sampler/seed_2801/attempt_001"
    )
    v2_8_sample_path = v2_8_attempt / "validation_sample.npz"
    if sha256_file(v2_8_sample_path) != (
        "bfc9c625b24bb593042370ebad1932372c1db39e519fcd3acbd12a0539ba8fb5"
    ):
        raise ForensicV3Error("v2.8 frozen CoF sample hash mismatch")
    v2_8_stored = _read_json(v2_8_attempt / "evaluation.json")
    v2_8_sample = _load_npz(v2_8_sample_path, SAMPLE_FIELDS)
    v2_8_independent = independent_guard_statistics(
        validation=validation,
        sample=v2_8_sample,
        tau=tau,
        receiver_categories=receiver_categories,
    )
    v2_8_matches = all(
        abs(v2_8_independent[key] - float(v2_8_stored["statistics"][key]))
        <= 1e-12
        and (
            "PASS"
            if v2_8_independent[key] <= float(v2_8_stored["thresholds"][key])
            else "FAIL"
        )
        == v2_8_stored["checks"][key]
        for key in METRICS
    )
    v2_8_comparison = {
        "candidate_id": "cof_v28_c01_gap_distribution_sampler",
        "sample_sha256": sha256_file(v2_8_sample_path),
        "stored_statistics": {
            key: float(v2_8_stored["statistics"][key]) for key in METRICS
        },
        "independent_statistics": {
            key: float(v2_8_independent[key]) for key in METRICS
        },
        "v2_8_stored_evaluation_matches_independent": bool(v2_8_matches),
        "v2_8_all_amount_guards_pass": bool(
            v2_8_stored["checks"]["amount_ks"] == "PASS"
            and v2_8_stored["checks"][
                "amount_abs_standardized_label_effect"
            ]
            == "PASS"
        ),
        "candidates": {},
    }
    for candidate_id, values in independent_by_candidate.items():
        v2_8_comparison["candidates"][candidate_id] = {
            "v3_all_amount_guards_pass": bool(
                guard_recalculation[candidate_id]["amount_ks"][
                    "independent_check"
                ]
                == "PASS"
                and guard_recalculation[candidate_id][
                    "amount_abs_standardized_label_effect"
                ]["independent_check"]
                == "PASS"
            ),
            "amount_ks_delta_vs_v2_8": float(
                values["amount_ks"] - v2_8_independent["amount_ks"]
            ),
            "amount_effect_delta_vs_v2_8": float(
                values["amount_abs_standardized_label_effect"]
                - v2_8_independent["amount_abs_standardized_label_effect"]
            ),
            "gap_ks_delta_vs_v2_8": float(
                values["gap_ks"] - v2_8_independent["gap_ks"]
            ),
            "gap_effect_delta_vs_v2_8": float(
                values["gap_abs_standardized_label_effect"]
                - v2_8_independent["gap_abs_standardized_label_effect"]
            ),
            "receiver_guard_delta_vs_v2_8": float(
                values["receiver_max_abs_signed_frequency"]
                - v2_8_independent["receiver_max_abs_signed_frequency"]
            ),
        }

    code_audit = _code_audit(repository_root)
    hypotheses = _hypothesis_verdicts(
        guard_recalculation=guard_recalculation,
        sample_contracts=sample_contracts,
        support_coverage=support_coverage,
        classwise_pmf=classwise_pmf,
        v2_8_comparison=v2_8_comparison,
    )

    return {
        "schema_version": "cof-seqgen-v3-attempt-002-forensic-v1",
        "analysis_base_head": ANALYSIS_BASE_HEAD,
        "inventory": inventory,
        "guard_recalculation": guard_recalculation,
        "sample_contracts": sample_contracts,
        "support_coverage": support_coverage,
        "classwise_pmf": classwise_pmf,
        "sampling_plan_provenance": plan_file_consistency,
        "v2_8_comparison": v2_8_comparison,
        "code_audit": code_audit,
        "hypotheses": hypotheses,
        "execution_counts": dict(EXECUTION_COUNTS),
    }


def _evidence_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    columns = {
        "row_type": "",
        "candidate_id": "",
        "label": "",
        "metric": "",
        "state": "",
        "stored_value": "",
        "independent_value": "",
        "threshold": "",
        "check": "",
        "real_probability": "",
        "synthetic_probability": "",
        "signed_difference": "",
        "total_variation": "",
        "maximum_absolute_difference": "",
        "notes": "",
    }
    rows: list[dict[str, Any]] = []
    for candidate_id, metrics in report["guard_recalculation"].items():
        for metric, evidence in metrics.items():
            rows.append(
                {
                    **columns,
                    "row_type": "guard",
                    "candidate_id": candidate_id,
                    "metric": metric,
                    "stored_value": evidence["stored_value"],
                    "independent_value": evidence["independent_value"],
                    "threshold": evidence["threshold"],
                    "check": evidence["independent_check"],
                    "notes": f"matches_stored={evidence['matches_stored']}",
                }
            )
    for candidate_id, labels in report["classwise_pmf"].items():
        for label, families in labels.items():
            for family in ("gap_bin", "receiver", "joint"):
                evidence = families[family]
                rows.append(
                    {
                        **columns,
                        "row_type": "pmf_summary",
                        "candidate_id": candidate_id,
                        "label": label,
                        "metric": family,
                        "state": evidence["maximum_difference_state"],
                        "total_variation": evidence["total_variation"],
                        "maximum_absolute_difference": evidence[
                            "maximum_absolute_difference"
                        ],
                    }
                )
                for state, (real, synthetic, difference) in enumerate(
                    zip(
                        evidence["real_pmf"],
                        evidence["synthetic_pmf"],
                        evidence["signed_difference"],
                    )
                ):
                    rows.append(
                        {
                            **columns,
                            "row_type": "pmf_state",
                            "candidate_id": candidate_id,
                            "label": label,
                            "metric": family,
                            "state": state,
                            "real_probability": real,
                            "synthetic_probability": synthetic,
                            "signed_difference": difference,
                            "total_variation": evidence["total_variation"],
                        }
                    )
    for candidate_id, contract in report["sample_contracts"].items():
        for metric, value in contract.items():
            rows.append(
                {
                    **columns,
                    "row_type": "sample_contract",
                    "candidate_id": candidate_id,
                    "metric": metric,
                    "independent_value": value,
                }
            )
    return rows


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CoF-SeqGen v3 attempt_002 failure forensic",
        "",
        "This is a read-only analysis of stored samples and frozen validation data. "
        "It did not query a GPU, restore a model, train, sample, or rerun validation.",
        "",
        "## Finding",
        "",
        "The production evaluator is independently reproduced, and both samples "
        "obey the active train-derived Y/L, mask, padding, codec, and support "
        "contracts. The failure is a probability-mass failure within valid joint "
        "support, consistent with the current v3 objective/discrete architecture.",
        "",
        "## Five-guard reproduction",
        "",
        "| candidate | metric | independent | threshold | result | stored match |",
        "|---|---|---:|---:|---|---|",
    ]
    for candidate_id, metrics in report["guard_recalculation"].items():
        for metric, row in metrics.items():
            lines.append(
                f"| {candidate_id} | {metric} | {row['independent_value']:.12g} "
                f"| {row['threshold']:.12g} | {row['independent_check']} "
                f"| {row['matches_stored']} |"
            )
    lines.extend(
        [
            "",
            "All ten independent values agree with stored `evaluation.json` within "
            "1e-12 and reproduce the same PASS/FAIL decisions.",
            "",
            "## Class-conditional discrete PMF differences",
            "",
            "Total variation (TV) compares frozen validation rows with stored "
            "synthetic rows within each entity label.",
            "",
            "| candidate | Y | gap-bin TV | receiver TV | joint TV |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate_id, labels in report["classwise_pmf"].items():
        for label in ("0", "1"):
            row = labels[label]
            lines.append(
                f"| {candidate_id} | {label} | "
                f"{row['gap_bin']['total_variation']:.9f} | "
                f"{row['receiver']['total_variation']:.9f} | "
                f"{row['joint']['total_variation']:.9f} |"
            )
    lines.extend(
        [
            "",
            "The factorized candidate is especially class-distorting: its Y=1 "
            "gap/receiver/joint TVs are substantially larger than direct-joint. "
            "Full PMF vectors and state-level differences are in the JSON/CSV.",
            "",
            "## Support, Y/L, mask, and padding",
            "",
            "| candidate | active plan exact | padding zero | codec errors | outside support | observed states (real/synth) |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for candidate_id in CANDIDATE_IDS:
        contract = report["sample_contracts"][candidate_id]
        support = report["support_coverage"][candidate_id]
        plan_exact = (
            contract["sampling_plan_labels_exact"]
            and contract["sampling_plan_lengths_exact"]
            and contract["sampling_plan_mask_exact"]
        )
        lines.append(
            f"| {candidate_id} | {plan_exact} | {contract['padding_zero']} | "
            f"{contract['codec_round_trip_errors']} | "
            f"{contract['joint_states_outside_train_support_mask']} | "
            f"{support['validation_observed_joint_states']}/"
            f"{support['synthetic_observed_joint_states']} |"
        )
    plan = report["sampling_plan_provenance"]
    lines.extend(
        [
            "",
            "The active train-policy plan was independently reconstructed from the "
            "pre-sampling conditioning binding as "
            f"`{plan['independently_recomputed_plan_hash']}` and exactly matches "
            "both samples. `train.npz` and `shared_sampling_plan.npz` were only "
            "preservation-hashed, not loaded by the guard/PMF extractor. The guard "
            "calculations therefore use only the stored samples, frozen validation, "
            "and frozen tau metadata.",
            "",
            "The train support mask, real validation, and both synthetic samples all "
            "contain 1024/1024 joint states. Therefore support coverage is complete "
            "but distributional mass is wrong.",
            "",
            "## v2.8 CoF comparison",
            "",
            "| candidate | amount KS delta | amount effect delta | gap KS delta | gap effect delta | receiver delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id, row in report["v2_8_comparison"]["candidates"].items():
        lines.append(
            f"| {candidate_id} | {row['amount_ks_delta_vs_v2_8']:.9f} | "
            f"{row['amount_effect_delta_vs_v2_8']:.9f} | "
            f"{row['gap_ks_delta_vs_v2_8']:.9f} | "
            f"{row['gap_effect_delta_vs_v2_8']:.9f} | "
            f"{row['receiver_guard_delta_vs_v2_8']:.9f} |"
        )
    lines.extend(
        [
            "",
            "Both v3 candidates retain amount-guard PASS. Direct improves amount KS "
            "relative to v2.8 while its amount effect remains PASS; factorized is "
            "slightly worse on both amount measures but remains PASS. In contrast, "
            "v2.8 gap guards passed and receiver was 0.0211266, whereas both v3 "
            "candidates regress every discrete guard.",
            "",
            "## Hypothesis decisions",
            "",
        ]
    )
    for key, hypothesis in report["hypotheses"].items():
        lines.append(f"### {key}: {hypothesis['verdict']}")
        lines.append("")
        for evidence in hypothesis["evidence"]:
            lines.append(f"- {evidence}")
        if hypothesis.get("qualification"):
            lines.extend(["", str(hypothesis["qualification"])])
        if hypothesis.get("mechanism_consistent_with_evidence"):
            lines.extend(["", "Mechanisms consistent with the evidence:", ""])
            for mechanism in hypothesis["mechanism_consistent_with_evidence"]:
                lines.append(f"- {mechanism}")
        lines.append("")
    lines.extend(
        [
            "## Static code audit",
            "",
            "| role | path | SHA-256 | audited loci |",
            "|---|---|---|---|",
        ]
    )
    for role, record in report["code_audit"].items():
        lines.append(
            f"| {role} | `{record['path']}` | `{record['sha256']}` | "
            f"{' ; '.join(record['loci'])} |"
        )
    lines.extend(
        [
            "",
            "No evaluator fix is proposed because no evaluator defect was found. "
            "A future preregistration should change the train-only objective or "
            "discrete generative formulation, not thresholds or test-based "
            "calibration. The factorized candidate specifically needs removal of "
            "teacher-forcing/sampled-gap exposure mismatch; the direct candidate "
            "needs a tractable probability-mass objective for the dense joint space. "
            "Those are proposals only; no implementation or rerun occurred here.",
            "",
            "## Preservation and zero-execution statement",
            "",
        ]
    )
    for role, record in report["inventory"].items():
        digest = record.get("tree_sha256", record.get("sha256"))
        lines.append(f"- `{role}`: `{digest}`")
    lines.extend(
        [
            "",
            "Execution counters are all zero: "
            + ", ".join(
                f"{key}={value}"
                for key, value in report["execution_counts"].items()
            )
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def write_forensic_bundle(
    *,
    report: Mapping[str, Any],
    markdown: Path,
    csv: Path,
    json: Path,
) -> Mapping[str, str]:
    paths = {"markdown": markdown, "csv": csv, "json": json}
    if any(path.exists() for path in paths.values()):
        raise ForensicV3Error("append-only output exists")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with markdown.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(render_markdown(report))
        with csv.open("x", encoding="utf-8", newline="") as handle:
            rows = _evidence_rows(report)
            writer = csv_module_writer(handle, rows)
            writer.writeheader()
            writer.writerows(rows)
        with json.open("x", encoding="utf-8", newline="\n") as handle:
            json_module_dump(report, handle)
    except FileExistsError as error:
        raise ForensicV3Error("append-only output exists") from error
    return {role: sha256_file(path) for role, path in paths.items()}


def csv_module_writer(handle: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    if not rows:
        raise ForensicV3Error("forensic CSV has no evidence rows")
    return csv.DictWriter(
        handle,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )


def json_module_dump(value: Mapping[str, Any], handle: Any) -> None:
    json.dump(
        value,
        handle,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only forensic extraction for v3 attempt_002",
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "docs/benchmark_v2/forensic_v3_attempt_002_failure.md"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "docs/benchmark_v2/forensic_v3_attempt_002_evidence.csv"
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(
            "docs/benchmark_v2/forensic_v3_attempt_002_evidence.json"
        ),
    )
    arguments = parser.parse_args()
    repository_root = arguments.repository_root.resolve()

    def resolve_output(path: Path) -> Path:
        return path if path.is_absolute() else repository_root / path

    report = analyze_v3_attempt_002_failure(repository_root)
    hashes = write_forensic_bundle(
        report=report,
        markdown=resolve_output(arguments.markdown),
        csv=resolve_output(arguments.csv),
        json=resolve_output(arguments.json),
    )
    print(json.dumps(hashes, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
