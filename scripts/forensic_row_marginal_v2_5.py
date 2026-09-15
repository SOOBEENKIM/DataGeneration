from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp


GENERATOR_ORDER = (
    "empirical_iid",
    "block_2",
    "block_4",
    "block_8",
    "full_sequence_reference",
    "independent_markov",
    "joint_markov",
    "plug_in_hmm",
    "plug_in_hsmm",
    "ctgan_separate_class",
    "tvae_separate_class",
    "neural_sequence",
    "cof_seqgen",
)
LEARNED_GENERATORS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "neural_sequence",
    "cof_seqgen",
)
SEEDS = (1, 2, 3, 4, 5)
COMPONENTS = (
    "amount_ks",
    "gap_ks",
    "amount_abs_standardized_label_effect",
    "gap_abs_standardized_label_effect",
    "receiver_max_abs_signed_frequency",
)
HARD_GUARDS = (
    "canonical_mask_and_zero_padding",
    "train_discrete_support",
    "c0_c1_reference_contract",
    "row_marginal_guards",
)
TERMINAL_MARKERS = (
    "COMPLETE.json",
    "FAILED.json",
    "INVALID.json",
    "UNAVAILABLE.json",
    "CANCELLED.json",
)
FORENSIC_SCHEMA = "benchmark-v2.5-row-marginal-forensic-v1"


class ForensicContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForensicContractError(f"cannot read JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ForensicContractError(f"JSON root is not an object: {path}")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as error:
        raise ForensicContractError(f"cannot read NPZ: {path}") from error


def _mtime(path: Path) -> str | None:
    if not path.is_file():
        return None
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()


def _file_evidence(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "bytes": None,
            "sha256": None,
            "mtime_utc": None,
        }
    return {
        "path": str(path),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mtime_utc": _mtime(path),
    }


def attempt_tree_digest(
    full_root: Path,
    *,
    attempt: int,
    terminal_markers_only: bool = False,
) -> Mapping[str, Any]:
    attempt_part = f"attempt_{attempt:03d}"
    paths = sorted(
        path
        for path in full_root.rglob("*")
        if path.is_file()
        and attempt_part in path.parts
        and (
            not terminal_markers_only
            or path.name in TERMINAL_MARKERS
        )
    )
    digest = hashlib.sha256()
    for path in paths:
        line = (
            f"{sha256_file(path)}  "
            f"{path.relative_to(full_root).as_posix()}\n"
        )
        digest.update(line.encode())
    return {
        "sha256": digest.hexdigest(),
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "paths": [
            path.relative_to(full_root).as_posix()
            for path in paths
        ],
    }


def _signed_standardized_label_effect(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(values)
    labels = np.asarray(labels, dtype=np.int64)
    groups = [values[labels == label] for label in (0, 1)]
    if any(len(group) < 2 for group in groups):
        return float("inf")
    degrees = len(groups[0]) + len(groups[1]) - 2
    pooled_variance = (
        (len(groups[0]) - 1) * groups[0].var(ddof=1)
        + (len(groups[1]) - 1) * groups[1].var(ddof=1)
    ) / degrees
    difference = float(groups[1].mean() - groups[0].mean())
    if pooled_variance <= 0:
        return 0.0 if difference == 0 else float(
            np.copysign(np.inf, difference)
        )
    return float(difference / np.sqrt(pooled_variance))


def _row_labels(batch: Mapping[str, np.ndarray]) -> np.ndarray:
    repeated = np.repeat(batch["y_entity"], batch["lengths"])
    broadcast = np.broadcast_to(
        batch["y_entity"][:, None],
        batch["valid_mask"].shape,
    )[batch["valid_mask"]]
    if not np.array_equal(repeated, broadcast):
        raise ForensicContractError(
            "np.repeat row labels differ from canonical mask traversal"
        )
    return repeated


def _receiver_signed_frequencies(
    batch: Mapping[str, np.ndarray],
    *,
    categories: int,
) -> np.ndarray:
    output = np.zeros((len(batch["lengths"]), categories), dtype=float)
    for index, length_value in enumerate(batch["lengths"]):
        length = int(length_value)
        counts = np.bincount(
            batch["x_cat"][index, :length, 0],
            minlength=categories,
        )
        output[index] = counts / length
    y = batch["y_entity"]
    return output[y == 1].mean(0) - output[y == 0].mean(0)


def summarize_batch_rows(
    batch: Mapping[str, np.ndarray],
    *,
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, Any]:
    valid = batch["valid_mask"]
    labels = _row_labels(batch)
    amount = np.asarray(batch["x_num"][..., 0][valid], dtype=float)
    gap = np.asarray(tau)[batch["dt_bin"][valid]]
    signed_receiver = _receiver_signed_frequencies(
        batch,
        categories=receiver_categories,
    )
    receiver_category = int(np.argmax(np.abs(signed_receiver)))

    def continuous(values: np.ndarray) -> Mapping[str, Any]:
        effect = _signed_standardized_label_effect(values, labels)
        return {
            "rows": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q01": float(np.quantile(values, 0.01)),
            "q50": float(np.quantile(values, 0.50)),
            "q99": float(np.quantile(values, 0.99)),
            "mean_y0": float(values[labels == 0].mean()),
            "mean_y1": float(values[labels == 1].mean()),
            "signed_standardized_label_effect": effect,
            "abs_standardized_label_effect": abs(effect),
        }

    return {
        "amount": continuous(amount),
        "gap": continuous(gap),
        "receiver": {
            "categories": receiver_categories,
            "max_abs_signed_frequency": float(
                np.max(np.abs(signed_receiver))
            ),
            "max_category": receiver_category,
            "signed_frequency_at_max_category": float(
                signed_receiver[receiver_category]
            ),
            "signed_frequency_by_category": signed_receiver.tolist(),
        },
        "label_prevalence": float(batch["y_entity"].mean()),
        "entity_count": int(len(batch["lengths"])),
        "valid_rows": int(valid.sum()),
    }


def independent_row_guard_statistics(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, float]:
    reference_valid = reference["valid_mask"]
    candidate_valid = candidate["valid_mask"]
    amount_reference = reference["x_num"][..., 0][reference_valid]
    amount_candidate = candidate["x_num"][..., 0][candidate_valid]
    gap_reference = np.asarray(tau)[
        reference["dt_bin"][reference_valid]
    ]
    gap_candidate = np.asarray(tau)[
        candidate["dt_bin"][candidate_valid]
    ]
    labels = _row_labels(candidate)
    receiver = _receiver_signed_frequencies(
        candidate,
        categories=receiver_categories,
    )
    return {
        "amount_ks": float(
            ks_2samp(amount_reference, amount_candidate).statistic
        ),
        "gap_ks": float(
            ks_2samp(gap_reference, gap_candidate).statistic
        ),
        "amount_abs_standardized_label_effect": abs(
            _signed_standardized_label_effect(
                amount_candidate,
                labels,
            )
        ),
        "gap_abs_standardized_label_effect": abs(
            _signed_standardized_label_effect(
                gap_candidate,
                labels,
            )
        ),
        "receiver_max_abs_signed_frequency": float(
            np.max(np.abs(receiver))
        ),
    }


def audit_sample_contract(
    sample: Mapping[str, np.ndarray],
    *,
    plan: Mapping[str, np.ndarray],
    train: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    valid = sample["valid_mask"]
    padding = ~valid
    labels_equal = np.array_equal(
        sample["y_entity"],
        plan["y_entity"],
    )
    lengths_equal = np.array_equal(
        sample["lengths"],
        plan["lengths"],
    )
    masks_equal = np.array_equal(
        sample["valid_mask"],
        plan["valid_mask"],
    )
    canonical = np.array_equal(
        valid,
        np.arange(valid.shape[1])[None, :] < sample["lengths"][:, None],
    )
    padding_zero = bool(
        np.all(sample["x_num"][padding] == 0)
        and np.all(sample["dt_bin"][padding] == 0)
        and np.all(sample["x_cat"][padding] == 0)
    )
    finite = bool(np.isfinite(sample["x_num"][valid]).all())
    train_gap_max = int(train["dt_bin"][train["valid_mask"]].max())
    gap_support = bool(
        np.all(sample["dt_bin"][valid] >= 0)
        and np.all(sample["dt_bin"][valid] <= train_gap_max)
    )
    category_support = True
    for channel in range(train["x_cat"].shape[-1]):
        train_max = int(
            train["x_cat"][..., channel][train["valid_mask"]].max()
        )
        values = sample["x_cat"][..., channel][valid]
        category_support = bool(
            category_support
            and np.all(values >= 0)
            and np.all(values <= train_max)
        )
    row_labels_align = np.array_equal(
        np.repeat(sample["y_entity"], sample["lengths"]),
        np.broadcast_to(
            sample["y_entity"][:, None],
            valid.shape,
        )[valid],
    )
    checks = {
        "labels_equal_plan": labels_equal,
        "lengths_equal_plan": lengths_equal,
        "mask_equal_plan": masks_equal,
        "canonical_prefix_mask": canonical,
        "padding_zero": padding_zero,
        "finite_amount": finite,
        "gap_train_support": gap_support,
        "receiver_train_support": category_support,
        "repeat_labels_equal_mask_order": row_labels_align,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "label_prevalence": float(sample["y_entity"].mean()),
        "entity_count": int(len(sample["lengths"])),
        "valid_rows": int(valid.sum()),
    }


def latest_terminal_attempt(seed_root: Path) -> tuple[Path, str, Path]:
    attempts = sorted(
        path
        for path in seed_root.glob("attempt_[0-9][0-9][0-9]")
        if path.is_dir()
    )
    if not attempts:
        raise ForensicContractError(f"seed has no attempts: {seed_root}")
    latest = attempts[-1]
    terminals = [
        latest / name
        for name in TERMINAL_MARKERS
        if (latest / name).is_file()
    ]
    if len(terminals) != 1:
        raise ForensicContractError(
            f"latest attempt is not singly terminal: {latest}"
        )
    marker = _json(terminals[0])
    status = str(marker.get("status"))
    if terminals[0].name != f"{status}.json":
        raise ForensicContractError(
            f"terminal filename/status mismatch: {terminals[0]}"
        )
    return latest, status, terminals[0]


def _component_values(
    component: str,
    *,
    real: Mapping[str, Any],
    synthetic: Mapping[str, Any],
) -> tuple[float, float, str, str, int | None]:
    if component == "amount_ks":
        return (
            float(real["amount"]["mean"]),
            float(synthetic["amount"]["mean"]),
            "real_test row-weighted amount mean",
            "synthetic row-weighted amount mean",
            None,
        )
    if component == "gap_ks":
        return (
            float(real["gap"]["mean"]),
            float(synthetic["gap"]["mean"]),
            "real_test row-weighted frozen-tau gap mean",
            "synthetic row-weighted frozen-tau gap mean",
            None,
        )
    if component == "amount_abs_standardized_label_effect":
        return (
            float(real["amount"]["abs_standardized_label_effect"]),
            float(synthetic["amount"]["abs_standardized_label_effect"]),
            "real_test absolute standardized y1-y0 amount effect",
            "synthetic absolute standardized y1-y0 amount effect",
            None,
        )
    if component == "gap_abs_standardized_label_effect":
        return (
            float(real["gap"]["abs_standardized_label_effect"]),
            float(synthetic["gap"]["abs_standardized_label_effect"]),
            "real_test absolute standardized y1-y0 gap effect",
            "synthetic absolute standardized y1-y0 gap effect",
            None,
        )
    if component == "receiver_max_abs_signed_frequency":
        category = int(synthetic["receiver"]["max_category"])
        return (
            float(real["receiver"]["max_abs_signed_frequency"]),
            float(synthetic["receiver"]["max_abs_signed_frequency"]),
            "real_test maximum entity-balanced absolute y1-y0 frequency",
            "synthetic maximum entity-balanced absolute y1-y0 frequency",
            category,
        )
    raise KeyError(component)


def _training_budget_summary(
    runtime: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if runtime is None:
        return {
            "requested_steps": None,
            "actual_steps": None,
            "wall_cap_reached": None,
            "all_requested_steps_completed": None,
        }
    budget = runtime.get("actual_training_budget", {})
    if not isinstance(budget, Mapping):
        budget = {}
    if "actual_steps" in budget:
        requested = budget.get("requested_steps")
        actual = budget.get("actual_steps")
        capped = bool(budget.get("wall_cap_reached", False))
    else:
        members = [
            value
            for value in budget.values()
            if isinstance(value, Mapping) and "actual_steps" in value
        ]
        requested = (
            sum(int(value["requested_steps"]) for value in members)
            if members
            else None
        )
        actual = (
            sum(int(value["actual_steps"]) for value in members)
            if members
            else None
        )
        capped = (
            any(bool(value.get("wall_cap_reached", False)) for value in members)
            if members
            else None
        )
    return {
        "requested_steps": requested,
        "actual_steps": actual,
        "wall_cap_reached": capped,
        "all_requested_steps_completed": (
            bool(actual == requested and not capped)
            if requested is not None and actual is not None
            else None
        ),
    }


def _aggregate_component_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    output: dict[str, Any] = {}
    evaluated = [
        row
        for row in rows
        if row["component"] in COMPONENTS
        and row["effect_size"] is not None
    ]
    for group_name, group_rows in (
        ("terminal_complete", [
            row for row in evaluated if row["terminal_status"] == "COMPLETE"
        ]),
        ("terminal_invalid", [
            row for row in evaluated if row["terminal_status"] == "INVALID"
        ]),
    ):
        output[group_name] = {}
        for component in COMPONENTS:
            values = [
                float(row["effect_size"])
                for row in group_rows
                if row["component"] == component
            ]
            output[group_name][component] = {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
    by_generator = {}
    for generator in GENERATOR_ORDER:
        by_generator[generator] = {}
        for component in COMPONENTS:
            selected = [
                row
                for row in evaluated
                if row["generator"] == generator
                and row["component"] == component
            ]
            values = [float(row["effect_size"]) for row in selected]
            by_generator[generator][component] = {
                "n": len(values),
                "failures": sum(
                    row["component_status"] == "FAIL"
                    for row in selected
                ),
                "failure_rate": (
                    sum(
                        row["component_status"] == "FAIL"
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else None
                ),
                "mean": float(np.mean(values)) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
    output["by_generator"] = by_generator
    return output


def verify_finalization(full_root: Path) -> Mapping[str, Any]:
    final_path = full_root / "FINAL_COMPLETE.json"
    final = _json(final_path)
    if final.get("status") != "FINAL_COMPLETE":
        raise ForensicContractError("full experiment is not FINAL_COMPLETE")
    stage = full_root / str(final["finalization_attempt"])
    index_path = stage / "artifact_index.json"
    checksum_path = stage / "checksum_manifest_report.json"
    if sha256_file(index_path) != final.get("artifact_index_sha256"):
        raise ForensicContractError("final artifact index hash mismatch")
    if sha256_file(checksum_path) != final.get("checksum_sha256"):
        raise ForensicContractError("final checksum hash mismatch")
    index = _json(index_path)
    entries = index.get("artifacts")
    if not isinstance(entries, list):
        raise ForensicContractError("final artifact index has no entries")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ForensicContractError("malformed final artifact entry")
        target = stage / str(entry["path"])
        if (
            not target.is_file()
            or target.stat().st_size != int(entry["bytes"])
            or sha256_file(target) != entry["sha256"]
        ):
            raise ForensicContractError(
                f"final artifact mismatch: {target}"
            )
    return {
        "FINAL_COMPLETE": _file_evidence(final_path),
        "finalization_attempt": stage.name,
        "artifact_index": _file_evidence(index_path),
        "checksum_manifest": _file_evidence(checksum_path),
        "indexed_artifacts": len(entries),
        "all_indexed_artifacts_verified": True,
    }


def _static_sampling_path_audit(
    repository_root: Path,
) -> Mapping[str, Any]:
    runner_path = repository_root / "scripts/run_full_experiment_v2_5.py"
    runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    child = next(
        (
            node
            for node in runner_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_execute_full_job_child"
        ),
        None,
    )
    if child is None:
        raise ForensicContractError(
            "runner has no _execute_full_job_child function"
        )

    sample_assignment: ast.Assign | None = None
    evaluation_call: ast.Call | None = None
    assignments_between: list[int] = []
    for node in ast.walk(child):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "sample"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "adapter"
            and node.value.func.attr == "sample"
        ):
            sample_assignment = node
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "evaluate_full_seed"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "sample"
        ):
            evaluation_call = node
    if sample_assignment is None or evaluation_call is None:
        raise ForensicContractError(
            "runner does not pass adapter.sample output to evaluate_full_seed"
        )
    for node in ast.walk(child):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        if not (
            sample_assignment.lineno < node.lineno < evaluation_call.lineno
        ):
            continue
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if any(
            isinstance(target, ast.Name) and target.id == "sample"
            for target in targets
        ):
            assignments_between.append(node.lineno)

    implementation_groups = {
        "ctgan_separate_class": (
            "generators.conditional_ctgan.ConditionalCTGAN.sample"
        ),
        "tvae_separate_class": (
            "generators.conditional_ctgan.ConditionalCTGAN.sample"
        ),
        "neural_sequence": (
            "generators.joint_sequence_baseline."
            "NeuralSequenceBaseline.sample"
        ),
        "cof_seqgen": (
            "generators.cof_seqgen_adapter.CoFSeqGenAdapter.sample"
        ),
    }
    required_definitions = (
        (
            repository_root / "generators/conditional_ctgan.py",
            "ConditionalCTGAN",
            "sample",
        ),
        (
            repository_root / "generators/joint_sequence_baseline.py",
            "NeuralSequenceBaseline",
            "sample",
        ),
        (
            repository_root / "generators/cof_seqgen_adapter.py",
            "CoFSeqGenAdapter",
            "sample",
        ),
    )
    definitions_present: dict[str, bool] = {}
    for path, class_name, method_name in required_definitions:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == class_name
            ),
            None,
        )
        present = bool(
            class_node is not None
            and any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method_name
                for node in class_node.body
            )
        )
        definitions_present[
            f"{path.relative_to(repository_root)}:{class_name}.{method_name}"
        ] = present
    tvae_tree = ast.parse(
        (
            repository_root / "generators/conditional_tvae.py"
        ).read_text(encoding="utf-8")
    )
    tvae_class = next(
        (
            node
            for node in tvae_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ConditionalTVAE"
        ),
        None,
    )
    tvae_inherits_ctgan = bool(
        tvae_class is not None
        and any(
            isinstance(base, ast.Name) and base.id == "ConditionalCTGAN"
            for base in tvae_class.bases
        )
        and not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "sample"
            for node in tvae_class.body
        )
    )
    direct = bool(
        sample_assignment.lineno < evaluation_call.lineno
        and not assignments_between
    )
    return {
        "runner_sample_assignment_line": sample_assignment.lineno,
        "runner_evaluation_call_line": evaluation_call.lineno,
        "sample_reassignments_between": assignments_between,
        "runner_passes_adapter_sample_directly_to_evaluator": direct,
        "learned_sample_implementations": implementation_groups,
        "distinct_learned_sample_implementation_groups": len(
            set(implementation_groups.values())
        ),
        "required_sample_definitions_present": definitions_present,
        "tvae_inherits_ctgan_sample": tvae_inherits_ctgan,
        "audit_pass": bool(
            direct
            and all(definitions_present.values())
            and tvae_inherits_ctgan
            and len(set(implementation_groups.values())) == 3
        ),
    }


def build_forensic_report(
    *,
    repository_root: Path,
    artifact_root: Path,
    frozen_root: Path,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    artifact_root = artifact_root.resolve()
    frozen_root = frozen_root.resolve()
    full_root = artifact_root / "full"
    finalization = verify_finalization(full_root)
    preservation_path = (
        repository_root
        / "docs/benchmark_v2/"
        "attempt_002_preservation_manifest_v2_5.json"
    )
    preservation = _json(preservation_path)
    expected_attempt_002 = preservation["attempt_002_inventory"]
    attempt_002_tree = attempt_tree_digest(full_root, attempt=2)
    attempt_002_terminals = attempt_tree_digest(
        full_root,
        attempt=2,
        terminal_markers_only=True,
    )
    if (
        attempt_002_tree["sha256"]
        != expected_attempt_002["tree_sha256"]
        or attempt_002_tree["files"] != expected_attempt_002["files"]
        or attempt_002_tree["bytes"] != expected_attempt_002["bytes"]
    ):
        raise ForensicContractError(
            "attempt_002 preservation tree does not match its manifest"
        )
    if (
        attempt_002_terminals["sha256"]
        != expected_attempt_002["terminal_marker_tree_sha256"]
        or attempt_002_terminals["files"]
        != expected_attempt_002["terminal_marker_files"]
        or attempt_002_terminals["bytes"]
        != expected_attempt_002["terminal_marker_bytes"]
    ):
        raise ForensicContractError(
            "attempt_002 terminal tree does not match its manifest"
        )
    manifest_path = frozen_root / "data_manifest.json"
    meta_path = frozen_root / "meta.json"
    plan_path = frozen_root / "shared_sampling_plan.npz"
    data_manifest = _json(manifest_path)
    meta = _json(meta_path)
    tau = np.asarray(meta["tau"], dtype=float)
    train = _load_npz(frozen_root / "train.npz")
    real_test = _load_npz(frozen_root / "test.npz")
    plan = _load_npz(plan_path)
    receiver_categories = int(
        train["x_cat"][..., 0][train["valid_mask"]].max()
    ) + 1
    real_summary = summarize_batch_rows(
        real_test,
        tau=tau,
        receiver_categories=receiver_categories,
    )
    run_root = (
        full_root / "joint_semimarkov_v2b" / "kappa_1.00"
    )
    records: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for generator in GENERATOR_ORDER:
        for seed in SEEDS:
            attempt, status, terminal_path = latest_terminal_attempt(
                run_root / generator / f"seed_{seed}"
            )
            manifest_file = attempt / "manifest.json"
            evaluation_path = attempt / "evaluation.json"
            metrics_path = attempt / "metrics.json"
            runtime_path = attempt / "runtime.json"
            sample_path = attempt / "sample.npz"
            manifest = _json(manifest_file)
            evaluation = (
                _json(evaluation_path)
                if evaluation_path.is_file()
                else None
            )
            metrics = (
                _json(metrics_path)
                if metrics_path.is_file()
                else None
            )
            runtime = (
                _json(runtime_path)
                if runtime_path.is_file()
                else None
            )
            hard_guards = (
                dict(evaluation.get("hard_guards", {}))
                if evaluation is not None
                else {}
            )
            row_report = (
                evaluation.get("row_marginal_guard_report")
                if evaluation is not None
                else None
            )
            evidence = {
                name: _file_evidence(path)
                for name, path in {
                    "manifest": manifest_file,
                    "sample": sample_path,
                    "evaluation": evaluation_path,
                    "metrics": metrics_path,
                    "runtime": runtime_path,
                    "terminal": terminal_path,
                }.items()
            }
            budget = _training_budget_summary(runtime)
            record: dict[str, Any] = {
                "generator": generator,
                "seed": seed,
                "attempt": int(attempt.name.rsplit("_", 1)[1]),
                "terminal_status": status,
                "hard_guards": {
                    guard: hard_guards.get(guard)
                    for guard in HARD_GUARDS
                },
                "association_recovery_error": (
                    evaluation.get("association_recovery_error")
                    if evaluation is not None
                    else None
                ),
                "manifest_provenance": {
                    key: manifest.get(key)
                    for key in (
                        "git_commit",
                        "code_hash",
                        "config_hash",
                        "data_hashes",
                        "sampling_plan_hash",
                        "evaluation_version",
                        "baseline_definition_version",
                    )
                },
                "files": evidence,
                "training_budget": budget,
                "artifact_index_sha256": (
                    sha256_file(attempt / "artifact_index.jsonl")
                    if (attempt / "artifact_index.jsonl").is_file()
                    else None
                ),
            }
            if (
                evaluation is None
                or row_report is None
                or not sample_path.is_file()
            ):
                record["sample_contract"] = None
                record["independent_statistics"] = None
                record["synthetic_row_summary"] = None
                record["artifact_replay_max_abs_delta"] = None
                records.append(record)
                component_rows.append(
                    {
                        "generator": generator,
                        "seed": seed,
                        "attempt": record["attempt"],
                        "terminal_status": status,
                        "component": "NOT_COMPUTED",
                        "component_status": "NOT_COMPUTED",
                        "threshold": None,
                        "effect_size": None,
                        "artifact_statistic": None,
                        "artifact_replay_abs_delta": None,
                        "real_value": None,
                        "synthetic_value": None,
                        "real_value_definition": None,
                        "synthetic_value_definition": None,
                        "real_signed_label_effect": None,
                        "synthetic_signed_label_effect": None,
                        "receiver_max_category": None,
                        "failure_reason": (
                            f"{status} has no completed evaluation artifact"
                        ),
                        **{
                            f"hard_guard_{guard}": hard_guards.get(guard)
                            for guard in HARD_GUARDS
                        },
                        "sample_sha256": evidence["sample"]["sha256"],
                        "evaluation_sha256": evidence["evaluation"]["sha256"],
                        "metrics_sha256": evidence["metrics"]["sha256"],
                        "runtime_sha256": evidence["runtime"]["sha256"],
                        "terminal_sha256": evidence["terminal"]["sha256"],
                    }
                )
                continue
            sample = _load_npz(sample_path)
            contract = audit_sample_contract(
                sample,
                plan=plan,
                train=train,
            )
            independent = independent_row_guard_statistics(
                real_test,
                sample,
                tau=tau,
                receiver_categories=receiver_categories,
            )
            artifact_statistics = row_report["statistics"]
            deltas = {
                component: abs(
                    float(independent[component])
                    - float(artifact_statistics[component])
                )
                for component in COMPONENTS
            }
            synthetic_summary = summarize_batch_rows(
                sample,
                tau=tau,
                receiver_categories=receiver_categories,
            )
            record["sample_contract"] = contract
            record["independent_statistics"] = independent
            record["synthetic_row_summary"] = synthetic_summary
            record["artifact_replay_max_abs_delta"] = max(deltas.values())
            record["row_marginal_guard_report"] = row_report
            records.append(record)
            for component in COMPONENTS:
                threshold = float(row_report["thresholds"][component])
                statistic = float(independent[component])
                component_status = str(row_report["checks"][component])
                real_value, synthetic_value, real_definition, (
                    synthetic_definition
                ), receiver_category = _component_values(
                    component,
                    real=real_summary,
                    synthetic=synthetic_summary,
                )
                component_rows.append(
                    {
                        "generator": generator,
                        "seed": seed,
                        "attempt": record["attempt"],
                        "terminal_status": status,
                        "component": component,
                        "component_status": component_status,
                        "threshold": threshold,
                        "effect_size": statistic,
                        "artifact_statistic": float(
                            artifact_statistics[component]
                        ),
                        "artifact_replay_abs_delta": deltas[component],
                        "real_value": real_value,
                        "synthetic_value": synthetic_value,
                        "real_value_definition": real_definition,
                        "synthetic_value_definition": synthetic_definition,
                        "real_signed_label_effect": (
                            real_summary["amount"][
                                "signed_standardized_label_effect"
                            ]
                            if component.startswith("amount")
                            else (
                                real_summary["gap"][
                                    "signed_standardized_label_effect"
                                ]
                                if component.startswith("gap")
                                else None
                            )
                        ),
                        "synthetic_signed_label_effect": (
                            synthetic_summary["amount"][
                                "signed_standardized_label_effect"
                            ]
                            if component.startswith("amount")
                            else (
                                synthetic_summary["gap"][
                                    "signed_standardized_label_effect"
                                ]
                                if component.startswith("gap")
                                else None
                            )
                        ),
                        "receiver_max_category": receiver_category,
                        "failure_reason": (
                            f"{statistic:.17g} > {threshold:.17g}"
                            if component_status == "FAIL"
                            else f"{statistic:.17g} <= {threshold:.17g}"
                        ),
                        **{
                            f"hard_guard_{guard}": hard_guards.get(guard)
                            for guard in HARD_GUARDS
                        },
                        "sample_sha256": evidence["sample"]["sha256"],
                        "evaluation_sha256": evidence["evaluation"]["sha256"],
                        "metrics_sha256": evidence["metrics"]["sha256"],
                        "runtime_sha256": evidence["runtime"]["sha256"],
                        "terminal_sha256": evidence["terminal"]["sha256"],
                    }
                )
    evaluated = [
        record
        for record in records
        if record["independent_statistics"] is not None
    ]
    learned_evaluated = [
        record
        for record in evaluated
        if record["generator"] in LEARNED_GENERATORS
    ]
    common_learned_failures = [
        component
        for component in COMPONENTS
        if learned_evaluated
        and all(
            record["row_marginal_guard_report"]["checks"][component]
            == "FAIL"
            for record in learned_evaluated
        )
    ]
    all_replays_match = all(
        float(record["artifact_replay_max_abs_delta"]) <= 1e-12
        for record in evaluated
    )
    all_contracts_pass = all(
        record["sample_contract"]["status"] == "PASS"
        for record in evaluated
    )
    all_reference_and_support_pass = all(
        record["hard_guards"]["c0_c1_reference_contract"] == "PASS"
        and record["hard_guards"]["train_discrete_support"] == "PASS"
        for record in evaluated
    )
    learned_budget_records = [
        record
        for record in learned_evaluated
        if record["training_budget"]["all_requested_steps_completed"]
        is not None
    ]
    learned_steps_complete = all(
        record["training_budget"]["all_requested_steps_completed"] is True
        for record in learned_budget_records
    )
    learned_all_invalid = all(
        record["terminal_status"] == "INVALID"
        for record in learned_evaluated
    )
    sampling_path_audit = _static_sampling_path_audit(repository_root)
    attempt_003_terminal_records = [
        record for record in records if record["attempt"] == 3
    ]
    attempt_003_terminal_paths = [
        record["files"]["terminal"]["path"]
        for record in attempt_003_terminal_records
    ]
    expected_attempt_003_jobs = set(preservation["pending_jobs"])
    actual_attempt_003_jobs = {
        f"{record['generator']}/seed_{record['seed']}"
        for record in attempt_003_terminal_records
    }
    if (
        len(attempt_003_terminal_records) != 17
        or actual_attempt_003_jobs != expected_attempt_003_jobs
        or not all(
            record["files"]["terminal"]["present"]
            for record in attempt_003_terminal_records
        )
    ):
        raise ForensicContractError(
            "attempt_003 does not contain all 17 continuation terminals"
        )
    hypotheses = {
        "A_evaluator_or_guard_defect": {
            "verdict": (
                "REFUTED"
                if all_replays_match
                and all_contracts_pass
                and all_reference_and_support_pass
                else "INCONCLUSIVE"
            ),
            "facts": {
                "evaluated_seeds": len(evaluated),
                "all_independent_replays_match_artifacts": all_replays_match,
                "maximum_abs_replay_delta": max(
                    float(record["artifact_replay_max_abs_delta"])
                    for record in evaluated
                ),
                "all_sample_contracts_pass": all_contracts_pass,
                "all_c0_c1_and_train_support_pass": (
                    all_reference_and_support_pass
                ),
            },
        },
        "B_common_learned_sampling_or_adapter_defect": {
            "verdict": (
                "REFUTED"
                if all_contracts_pass
                and all_replays_match
                and sampling_path_audit["audit_pass"]
                else "INCONCLUSIVE"
            ),
            "facts": {
                "shared_plan_contract_passes": all_contracts_pass,
                "common_failed_components": common_learned_failures,
                "all_five_components_common": (
                    len(common_learned_failures) == len(COMPONENTS)
                ),
                "learned_generator_count": len(LEARNED_GENERATORS),
                "evaluated_learned_seeds": len(learned_evaluated),
                "static_sampling_path_audit": sampling_path_audit,
            },
        },
        "C_model_distribution_fidelity_at_frozen_configuration": {
            "verdict": (
                "SUPPORTED"
                if learned_all_invalid
                and learned_steps_complete
                and bool(common_learned_failures)
                and all_replays_match
                else "INCONCLUSIVE"
            ),
            "facts": {
                "all_evaluated_learned_seeds_invalid": learned_all_invalid,
                "all_recorded_learned_steps_completed_without_wall_cap": (
                    learned_steps_complete
                ),
                "common_failed_components": common_learned_failures,
                "model_specific_failure_profiles_preserved": True,
            },
        },
    }
    source_paths = (
        "eval/full_evaluation_v2_5.py",
        "eval/model_guards_v2_5.py",
        "generators/contracts_v2_5.py",
        "generators/sampling_plan.py",
        "generators/conditional_ctgan.py",
        "generators/conditional_tvae.py",
        "generators/joint_sequence_baseline.py",
        "generators/cof_seqgen_adapter.py",
        "scripts/run_full_experiment_v2_5.py",
    )
    source_hashes = {
        path: sha256_file(repository_root / path)
        for path in source_paths
    }
    report = {
        "schema_version": FORENSIC_SCHEMA,
        "scope": {
            "scenario": "joint_semimarkov_v2b",
            "kappa": 1.0,
            "generators": list(GENERATOR_ORDER),
            "seeds": list(SEEDS),
            "runtime_mutation": False,
            "model_or_evaluator_execution": False,
        },
        "provenance": {
            "source_head": _git_head(repository_root),
            "config": _file_evidence(
                repository_root
                / "configs/benchmark_v2/full_v2_5.yaml"
            ),
            "frozen_manifest": _file_evidence(manifest_path),
            "frozen_data_hashes": data_manifest.get("data_hashes"),
            "sampling_plan": {
                **_file_evidence(plan_path),
                "content_sha256": str(
                    np.asarray(plan["plan_hash"]).item()
                ),
            },
            "row_guard_calibration": _file_evidence(
                repository_root
                / "artifacts/benchmark_v2_5/prerun_calibration/"
                "row_guard_thresholds.json"
            ),
            "finalization": finalization,
            "attempt_002_preservation": {
                "manifest": _file_evidence(preservation_path),
                "tree": attempt_002_tree,
                "terminal_tree": attempt_002_terminals,
                "matches_preservation_manifest": True,
            },
            "attempt_003_continuation": {
                "terminal_artifacts": len(
                    attempt_003_terminal_records
                ),
                "all_expected_jobs_terminal": True,
                "terminal_paths": attempt_003_terminal_paths,
            },
            "source_file_sha256": source_hashes,
        },
        "logical_artifact_sequence": [
            "manifest/RUNNING allocation",
            "training/checkpoint/progress",
            "adapter.sample returns in memory",
            "evaluate_full_seed computes contract, association, diagnostics, and row guards in memory",
            "sample/checkpoint/metrics/runtime/evaluation artifacts are written and indexed",
            "COMPLETE or INVALID terminal marker is written last",
        ],
        "static_sampling_path_audit": sampling_path_audit,
        "real_test_row_summary": real_summary,
        "summary": {
            "planned_cells": len(GENERATOR_ORDER) * len(SEEDS),
            "terminal_cells": len(records),
            "evaluated_cells": len(evaluated),
            "terminal_counts": {
                status: sum(
                    record["terminal_status"] == status
                    for record in records
                )
                for status in (
                    "COMPLETE",
                    "INVALID",
                    "CANCELLED",
                    "FAILED",
                    "UNAVAILABLE",
                )
            },
            "hard_guard_pass_counts": {
                guard: sum(
                    record["hard_guards"].get(guard) == "PASS"
                    for record in evaluated
                )
                for guard in HARD_GUARDS
            },
            "common_learned_failure_components": (
                common_learned_failures
            ),
            "all_independent_replays_match": all_replays_match,
            "all_sample_contracts_pass": all_contracts_pass,
            "all_c0_c1_and_train_support_pass": (
                all_reference_and_support_pass
            ),
        },
        "valid_invalid_comparison": _aggregate_component_rows(
            component_rows
        ),
        "hypotheses": hypotheses,
        "records": records,
        "component_rows": component_rows,
    }
    return report


def _git_head(repository_root: Path) -> str:
    head_path = repository_root / ".git/HEAD"
    if not head_path.is_file():
        raise ForensicContractError("repository has no .git/HEAD")
    value = head_path.read_text().strip()
    if value.startswith("ref: "):
        ref_path = repository_root / ".git" / value[5:]
        if ref_path.is_file():
            return ref_path.read_text().strip()
        packed = repository_root / ".git/packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(f" {value[5:]}"):
                    return line.split()[0]
        raise ForensicContractError("cannot resolve Git HEAD ref")
    return value


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ForensicContractError("forensic CSV has no rows")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ForensicContractError("forensic CSV row schemas differ")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_forensic_artifacts(
    report: Mapping[str, Any],
    *,
    csv_path: Path,
    json_path: Path,
    repository_root: Path,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    forbidden = {
        (repository_root / "artifacts").resolve(),
        (repository_root / "data").resolve(),
    }
    for path in (csv_path, json_path):
        resolved = path.resolve()
        if any(
            resolved == root or root in resolved.parents
            for root in forbidden
        ):
            raise ForensicContractError(
                "forensic outputs cannot modify runtime artifact/data roots"
            )
    csv_payload = csv_bytes(report["component_rows"])
    json_payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_payload)
    json_path.write_bytes(json_payload)
    return {
        "csv": {
            "path": str(csv_path),
            "sha256": hashlib.sha256(csv_payload).hexdigest(),
            "bytes": len(csv_payload),
        },
        "json": {
            "path": str(json_path),
            "sha256": hashlib.sha256(json_payload).hexdigest(),
            "bytes": len(json_payload),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only v2.5 row-marginal forensic extractor"
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/benchmark_v2_5"),
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=Path(
            "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
            "kappa_1.00"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "docs/benchmark_v2/"
            "forensic_row_marginal_failure_v2_5.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "docs/benchmark_v2/"
            "forensic_row_marginal_failure_v2_5.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    report = build_forensic_report(
        repository_root=root,
        artifact_root=resolve(args.artifact_root),
        frozen_root=resolve(args.frozen_root),
    )
    written = write_forensic_artifacts(
        report,
        csv_path=resolve(args.output_csv),
        json_path=resolve(args.output_json),
        repository_root=root,
    )
    print(json.dumps(written, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
