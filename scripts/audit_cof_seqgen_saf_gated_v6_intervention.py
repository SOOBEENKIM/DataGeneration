#!/usr/bin/env python3
"""Audit whether a trained SAF-v6 gap route is selective to history/static state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.cof_seqgen_saf_tensorizer import (  # noqa: E402
    SAFTensorizer,
    SAFTensorizerState,
    load_canonical_dataset,
)
from experiments.cof_seqgen_saf_training import (  # noqa: E402
    SequenceWindowDataset,
    collate_windows,
)
from models.cof_seqgen_saf import (  # noqa: E402
    MODEL_IMPLEMENTATION_VERSION,
    CoFSeqGenSAF,
    SAFModelConfig,
)


class InterventionAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_lookup(state: SAFTensorizerState) -> tuple[int, Mapping[int, int]]:
    fields = list(state.static_categorical_codecs)
    names = [name for name, _ in fields]
    if "entity_label" not in names:
        raise InterventionAuditError("entity_label static field is required")
    field = names.index("entity_label")
    codec = fields[field][1]
    mapping: Dict[int, int] = {}
    for code in range(codec.vocab_size):
        decoded = codec.decode([code])[0]
        if decoded in {"0", "1"}:
            mapping[code] = int(decoded)
    if set(mapping.values()) != {0, 1}:
        raise InterventionAuditError("entity_label codec must contain labels 0 and 1")
    return field, mapping


def _build_model(
    checkpoint: Mapping[str, Any],
    state: SAFTensorizerState,
    *,
    candidate_id: str,
    device: torch.device,
) -> CoFSeqGenSAF:
    raw_config = dict(checkpoint["model_config"])
    raw_config["candidate_id"] = candidate_id
    model = CoFSeqGenSAF(SAFModelConfig(**raw_config), state.gap_support)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _history_bank(
    *,
    model: CoFSeqGenSAF,
    tensorizer: SAFTensorizer,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    max_histories_per_label: int,
) -> Dict[int, torch.Tensor]:
    sequences = tensorizer.transform_split(dataset, "validation")
    loader = DataLoader(
        SequenceWindowDataset(sequences, context_window=model.config.context_window),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )
    label_field, label_by_code = _label_lookup(tensorizer.state)
    banks: Dict[int, list[torch.Tensor]] = {0: [], 1: []}
    counts = {0: 0, 1: 0}
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: (
                    tuple(item.to(device) for item in value)
                    if isinstance(value, tuple)
                    else value.to(device)
                )
                for key, value in raw.items()
            }
            hidden = model.encoder(
                batch["gap"],
                batch["receiver"],
                batch["numeric_value"],
                batch["valid_mask"],
                batch["static"],
                batch["static_categorical"],
                batch["auxiliary_categorical"],
                batch["auxiliary_numeric"],
            )
            has_previous = torch.zeros_like(batch["valid_mask"])
            has_previous[:, 1:] = (
                batch["valid_mask"][:, 1:] & batch["valid_mask"][:, :-1]
            )
            target = batch["target_mask"] & has_previous
            codes = batch["static_categorical"][label_field]
            for label in (0, 1):
                matching_codes = [
                    code for code, decoded in label_by_code.items() if decoded == label
                ]
                entity_mask = torch.zeros_like(codes, dtype=torch.bool)
                for code in matching_codes:
                    entity_mask |= codes == code
                mask = target & entity_mask.unsqueeze(1)
                remaining = max_histories_per_label - counts[label]
                selected = hidden[mask][:remaining].detach().cpu()
                if selected.numel():
                    banks[label].append(selected)
                    counts[label] += len(selected)
            if all(counts[label] >= max_histories_per_label for label in (0, 1)):
                break
    output = {
        label: torch.cat(parts) if parts else torch.empty(0, model.config.hidden_dim)
        for label, parts in banks.items()
    }
    if any(len(output[label]) == 0 for label in (0, 1)):
        raise InterventionAuditError("validation histories are missing a label")
    return output


def _sensitivity(
    *,
    model: CoFSeqGenSAF,
    histories: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> Dict[str, float | int]:
    gap_values = torch.as_tensor(
        model.support.representatives,
        dtype=torch.float32,
        device=device,
    )
    ranges = []
    gates = []
    with torch.no_grad():
        for start in range(0, len(histories), chunk_size):
            hidden = histories[start : start + chunk_size].to(device)
            n_histories = len(hidden)
            n_gaps = len(gap_values)
            expanded_hidden = hidden[:, None, :].expand(
                n_histories,
                n_gaps,
                hidden.shape[-1],
            )
            expanded_gap = gap_values[None, :].expand(n_histories, n_gaps)
            probability = torch.sigmoid(
                model._copy_logits(
                    expanded_hidden.reshape(-1, hidden.shape[-1]),
                    expanded_gap.reshape(-1),
                )
            ).reshape(n_histories, n_gaps)
            ranges.append((probability.max(1).values - probability.min(1).values).cpu())
            gates.append(
                torch.sigmoid(model.copy_gap_gate_head(hidden).squeeze(-1)).cpu()
            )
    sensitivity = torch.cat(ranges).numpy()
    gate = torch.cat(gates).numpy()
    return {
        "history_count": int(len(sensitivity)),
        "mean_copy_probability_range": float(np.mean(sensitivity)),
        "median_copy_probability_range": float(np.median(sensitivity)),
        "max_copy_probability_range": float(np.max(sensitivity)),
        "mean_gate_activation": float(np.mean(gate)),
    }


def _audit_checkpoint(
    *,
    checkpoint_path: Path,
    dataset_dir: Path,
    routed_candidate: str,
    control_candidate: str,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
    max_histories_per_label: int,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        raise InterventionAuditError("checkpoint implementation version mismatch")
    if checkpoint.get("test_accessed", True):
        raise InterventionAuditError("checkpoint does not certify sealed test content")
    if checkpoint["model_config"]["candidate_id"] != routed_candidate:
        raise InterventionAuditError("checkpoint candidate does not match config")
    state = SAFTensorizerState.from_dict(checkpoint["tensorizer_state"])
    tensorizer = SAFTensorizer(state)
    dataset = load_canonical_dataset(
        dataset_dir,
        allowed_splits=("train", "validation"),
    )
    tensorizer._validate_dataset(dataset)
    routed = _build_model(
        checkpoint,
        state,
        candidate_id=routed_candidate,
        device=device,
    )
    control = _build_model(
        checkpoint,
        state,
        candidate_id=control_candidate,
        device=device,
    )
    histories = _history_bank(
        model=routed,
        tensorizer=tensorizer,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        max_histories_per_label=max_histories_per_label,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "routed_candidate": routed_candidate,
        "matched_control_candidate": control_candidate,
        "routed": {
            str(label): _sensitivity(
                model=routed,
                histories=bank,
                device=device,
                chunk_size=chunk_size,
            )
            for label, bank in histories.items()
        },
        "matched_zero_gap_control": {
            str(label): _sensitivity(
                model=control,
                histories=bank,
                device=device,
                chunk_size=chunk_size,
            )
            for label, bank in histories.items()
        },
        "test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model_implementation_version"] != MODEL_IMPLEMENTATION_VERSION:
        raise InterventionAuditError("audit config implementation version mismatch")
    device = torch.device(config.get("device", "cpu"))
    results: Dict[str, Any] = {}
    for kappa_name, cell in config["cells"].items():
        results[kappa_name] = {}
        for branch, spec in cell["branches"].items():
            results[kappa_name][branch] = _audit_checkpoint(
                checkpoint_path=Path(spec["checkpoint"]),
                dataset_dir=Path(cell["dataset_dir"]),
                routed_candidate=spec["routed_candidate"],
                control_candidate=spec["control_candidate"],
                device=device,
                batch_size=int(config["batch_size"]),
                chunk_size=int(config["chunk_size"]),
                max_histories_per_label=int(config["max_histories_per_label"]),
            )

    criteria = config["criteria"]
    checks: Dict[str, Any] = {}
    for branch in sorted(results["kappa_1"]):
        k1 = results["kappa_1"][branch]
        k0 = results["kappa_0"][branch]
        k1_label1 = k1["routed"]["1"]["mean_copy_probability_range"]
        nuisance = max(
            k1["routed"]["0"]["mean_copy_probability_range"],
            k0["routed"]["0"]["mean_copy_probability_range"],
            k0["routed"]["1"]["mean_copy_probability_range"],
        )
        control_max = max(
            result["matched_zero_gap_control"][str(label)][
                "max_copy_probability_range"
            ]
            for result in (k0, k1)
            for label in (0, 1)
        )
        branch_checks = {
            "kappa_1_label_1_material": (
                k1_label1 >= float(criteria["min_kappa_1_label_1_mean_range"])
            ),
            "noncausal_cells_small": (
                nuisance <= float(criteria["max_noncausal_mean_range"])
            ),
            "kappa_label_selectivity": (
                k1_label1
                >= float(criteria["min_selectivity_ratio"]) * max(nuisance, 1e-12)
            ),
            "matched_control_invariant": (
                control_max <= float(criteria["max_control_range"])
            ),
        }
        checks[branch] = {
            **branch_checks,
            "pass": all(branch_checks.values()),
            "kappa_1_label_1_mean_range": k1_label1,
            "largest_noncausal_mean_range": nuisance,
            "observed_selectivity_ratio": k1_label1 / max(nuisance, 1e-12),
            "matched_control_max_range": control_max,
        }
    decision = "PASS" if all(item["pass"] for item in checks.values()) else "FAIL"
    artifact = {
        "schema_version": "cof-seqgen-saf-gated-v6-intervention-audit-v1",
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "decision": decision,
        "criteria": criteria,
        "checks": checks,
        "results": results,
        "config_sha256": _sha256(args.config),
        "loaded_content_splits": ["train", "validation"],
        "test_accessed": False,
    }
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "checks": checks,
                "output_path": str(output_path),
                "sha256": _sha256(output_path),
                "test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
