"""Authorization-only execution backend for ``cof_ccmtpp_v1``.

This module is lazy-imported only after an execution authorization has passed.
Plan and dry-run never import it. It does not query GPU inventory: the runner
exposes exactly one externally selected device as ``cuda:0``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.cof_ccmtpp_v1_contract import (
    CCMTPPContractError,
    evaluate_candidate_stop,
    sha256_file,
)
from eval.external_validation_metrics_v1 import (
    ExternalMetricError,
    compute_external_validation_metrics,
    validate_external_hard_contract,
)
from experiments.cof_ccmtpp_v1_execution_runner import (
    AMOUNT_CONTRACT_NAME,
    CCMTPPAttemptStore,
    ExecutionDependencies,
    ExecutionJob,
    ExecutionPlan,
    build_checkpoint_provenance,
    canonical_sha256,
    fit_train_only_states,
    validate_runtime_candidate,
    resolve_dataset_access,
)
from generators.sampling_plan import SamplingPlan


@dataclass(frozen=True)
class AuthorizedTrainingData:
    train: SequenceBatch
    transform: Mapping[str, Any]
    states: Any
    validation_path: Path


def _load_sequence_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as archive:
        values = {
            field: archive[field]
            for field in SequenceBatch.__dataclass_fields__
        }
    return SequenceBatch(**values)


def load_authorized_train_body(
    *, plan: ExecutionPlan, job: ExecutionJob, authorization: Mapping[str, Any]
) -> AuthorizedTrainingData:
    if authorization.get("candidate_id") != job.candidate_id:
        raise CCMTPPContractError("authorized train candidate mismatch")
    train_path = resolve_dataset_access(
        plan=plan,
        dataset=job.dataset,
        split="train",
        purpose="fit",
        fit_complete=False,
    )
    record = plan.raw["datasets"][job.dataset]
    transform_path = plan.repository_root / str(record["transform_path"])
    if sha256_file(transform_path) != record["transform_sha256"]:
        raise CCMTPPContractError("authorized train transform changed")
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    train = _load_sequence_batch(train_path)
    states = fit_train_only_states(
        plan=plan,
        job=job,
        batch=train,
        transform_state=transform,
        transform_file_sha256=record["transform_sha256"],
        split="train",
    )
    # The validation body is neither resolved nor opened here. The relative
    # binding is carried to the post-fit phase only.
    return AuthorizedTrainingData(
        train=train,
        transform=transform,
        states=states,
        validation_path=plan.repository_root / str(record["validation_path"]),
    )


def build_authorized_model(
    *, plan: ExecutionPlan, job: ExecutionJob, data: AuthorizedTrainingData
):
    from models.cof_ccmtpp_v1 import CoFCCMTPPV1

    model_config = plan.source_definition.raw["model"]
    hierarchy = data.states.hierarchy if job.candidate_id in {"C3", "C4"} else None
    model = CoFCCMTPPV1(
        candidate=job.candidate_id,
        receiver_classes=int(data.states.receiver_vocabulary["receiver_classes"]),
        hierarchy=hierarchy,
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        n_layers=int(model_config["n_layers"]),
        max_length=int(model_config["max_length"]),
        gap_components=int(model_config["gap_mixture_components"]),
        gap_max=float(data.states.gap_support["max_gap"]),
        dropout=float(model_config["dropout"]),
    )
    training = {
        key: model_config[key]
        for key in (
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "requested_updates",
            "max_wall_seconds",
            "checkpoint_interval_updates",
            "early_stopping",
            "update_sweep",
        )
    }
    validate_runtime_candidate(
        plan=plan,
        job=job,
        model=model,
        train_state_binding=data.states.binding,
        training_hyperparameters=training,
        enabled_hooks=(),
    )
    return model


def select_explicit_device(
    *, plan: ExecutionPlan, job: ExecutionJob, authorization: Mapping[str, Any]
) -> str:
    del job
    if (
        plan.raw["execution"]["explicit_device"] != "cuda:0"
        or plan.raw["execution"]["gpu_inventory_query"] != "FORBIDDEN"
        or authorization.get("scope", {}).get("gpu_inventory_query") is not False
    ):
        raise CCMTPPContractError("explicit device/no-inventory contract changed")
    return "cuda:0"


def _torch_bytes(value: Any) -> bytes:
    import torch

    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _npz_bytes(values: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def _training_config(plan: ExecutionPlan) -> Mapping[str, Any]:
    model = plan.source_definition.raw["model"]
    if model["optimizer"] != "adamw":
        raise CCMTPPContractError("only the preregistered AdamW optimizer is allowed")
    return model


def _checkpoint_bundle(model: Any, *, data: AuthorizedTrainingData, plan: ExecutionPlan):
    from models.cof_ccmtpp_v1 import build_checkpoint_bundle

    binding = data.states.binding
    return build_checkpoint_bundle(
        model=model,
        provenance={
            "source_sha256": plan.source_definition.config_sha256,
            "config_sha256": plan.config_sha256,
            "train_manifest_sha256": binding["train_sha256"],
            "transform_state_sha256": binding["amount_transform_sha256"],
            "sampling_plan_sha256": binding["sampling_plan_sha256"],
            "amount_contract_sha256": canonical_sha256(
                {"amount_contract": AMOUNT_CONTRACT_NAME}
            ),
            "receiver_hierarchy_sha256": binding[
                "receiver_hierarchy_sha256"
            ],
            "validation_rows_used": 0,
            "internal_test_rows_used": 0,
            "fraud_test_rows_used": 0,
        },
    )


def _fixed_validation_plan(
    *, plan: ExecutionPlan, job: ExecutionJob, validation: SequenceBatch
) -> SamplingPlan:
    record = plan.raw["datasets"][job.dataset]
    plan_path = (
        plan.repository_root
        / str(record["c0_attempt"])
        / "validation_sampling_plan.npz"
    )
    if sha256_file(plan_path) != record["sampling_plan_sha256"]:
        raise CCMTPPContractError("frozen validation SamplingPlan changed")
    with np.load(plan_path, allow_pickle=False) as archive:
        fixed = SamplingPlan(
            y_entity=archive["y_entity"],
            lengths=archive["lengths"],
            valid_mask=archive["valid_mask"],
            plan_hash=str(archive["plan_hash"].item()),
        )
    observed = SamplingPlan.from_batch(validation)
    if (
        fixed.plan_hash != observed.plan_hash
        or not np.array_equal(fixed.y_entity, validation.y_entity)
        or not np.array_equal(fixed.lengths, validation.lengths)
        or not np.array_equal(fixed.valid_mask, validation.valid_mask)
    ):
        raise CCMTPPContractError("validation conditioning plan mismatch")
    return fixed


def _numeric_gap_edges(transform: Mapping[str, Any]) -> np.ndarray:
    raw = transform.get("gap_edges")
    if not isinstance(raw, list) or len(raw) != len(transform.get("gap_tau", [])) + 1:
        raise CCMTPPContractError("train-only gap edges are invalid")
    decoded = []
    for value in raw:
        if value == "+inf":
            decoded.append(float("inf"))
        elif value == "-inf":
            decoded.append(float("-inf"))
        else:
            decoded.append(float(value))
    result = np.asarray(decoded, dtype=np.float64)
    if np.isnan(result).any() or np.any(result[1:] < result[:-1]):
        raise CCMTPPContractError("train-only gap edges are not monotone")
    return result


def _sample_validation(
    *, model: Any, plan: SamplingPlan, transform: Mapping[str, Any], device: str,
    seed: int, batch_size: int, event_queue: Any,
) -> tuple[SyntheticBatch, np.ndarray]:
    import torch

    n, width = plan.valid_mask.shape
    amount_out = np.zeros((n, width, 1), dtype=np.float32)
    gap_out = np.zeros((n, width), dtype=np.float64)
    receiver_out = np.zeros((n, width, 1), dtype=np.int64)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            valid = torch.from_numpy(plan.valid_mask[start:end]).to(device)
            lengths = torch.from_numpy(plan.lengths[start:end]).to(device)
            y = torch.from_numpy(plan.y_entity[start:end]).to(device)
            amount = torch.zeros((end - start, width, 1), device=device)
            gap = torch.zeros((end - start, width), device=device)
            receiver = torch.where(
                valid,
                torch.ones_like(valid, dtype=torch.long),
                torch.zeros_like(valid, dtype=torch.long),
            )
            for position in range(width):
                hidden = model.event_decoder(
                    amount=amount,
                    gap=gap,
                    receiver=receiver,
                    y=y,
                    lengths=lengths,
                    valid_mask=valid,
                )
                mark = model.sample_mark_from_hidden(
                    hidden=hidden,
                    history_receiver=receiver,
                    valid_mask=valid,
                    generator=generator,
                )
                active = valid[:, position]
                amount[:, position, 0] = torch.where(
                    active, mark.normalized_amount[:, position], amount[:, position, 0]
                )
                gap[:, position] = torch.where(
                    active, mark.gap[:, position], gap[:, position]
                )
                receiver[:, position] = torch.where(
                    active, mark.receiver[:, position], receiver[:, position]
                )
            amount_out[start:end] = amount.cpu().numpy().astype(np.float32)
            gap_out[start:end] = gap.cpu().numpy()
            receiver_out[start:end, :, 0] = receiver.cpu().numpy()
            event_queue.put({"kind": "progress", "phase": "sampling", "entities": end})
    edges = _numeric_gap_edges(transform)
    bins = np.searchsorted(edges[1:-1], gap_out, side="right").astype(np.int64)
    bins[~plan.valid_mask] = 0
    amount_out[~plan.valid_mask] = 0
    receiver_out[~plan.valid_mask] = 0
    return (
        SyntheticBatch(
            x_num=amount_out,
            dt_bin=bins,
            x_cat=receiver_out,
            valid_mask=plan.valid_mask.copy(),
            y_entity=plan.y_entity.copy(),
            lengths=plan.lengths.copy(),
        ),
        gap_out,
    )


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        raise CCMTPPContractError("classwise gap KS has an empty class")
    support = np.sort(np.concatenate((left, right)))
    return float(
        np.max(
            np.abs(
                np.searchsorted(np.sort(left), support, side="right") / len(left)
                - np.searchsorted(np.sort(right), support, side="right") / len(right)
            )
        )
    )


def _classwise_gap_ks(
    *, real: SequenceBatch, continuous_synthetic: np.ndarray, tau: np.ndarray
) -> Mapping[str, float]:
    real_gap = tau[real.dt_bin]
    result = {}
    for label in (0, 1):
        selected = real.valid_mask & (real.y_entity[:, None] == label)
        result[f"y{label}"] = _ks(real_gap[selected], continuous_synthetic[selected])
    return result


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    support = np.union1d(left, right)
    left_count = np.zeros(len(support), dtype=float)
    right_count = np.zeros(len(support), dtype=float)
    left_values, counts = np.unique(left, return_counts=True)
    left_count[np.searchsorted(support, left_values)] = counts / len(left)
    right_values, counts = np.unique(right, return_counts=True)
    right_count[np.searchsorted(support, right_values)] = counts / len(right)
    return float(np.abs(left_count - right_count).sum() / 2)


def _receiver_fidelity_decomposition(
    *, real: SequenceBatch, synthetic: SyntheticBatch, hierarchy: Any
) -> Mapping[str, float]:
    real_values = real.x_cat[..., 0][real.valid_mask]
    synthetic_values = synthetic.x_cat[..., 0][synthetic.valid_mask]
    head = np.asarray(hierarchy.head_codes, dtype=np.int64)
    tail = np.asarray(
        [code for cluster in hierarchy.tail_clusters for code in cluster],
        dtype=np.int64,
    )
    result = {"full_receiver_tv": _tv(real_values, synthetic_values)}
    for name, codes in (("head", head), ("tail", tail)):
        left = real_values[np.isin(real_values, codes)]
        right = synthetic_values[np.isin(synthetic_values, codes)]
        result[f"{name}_receiver_tv"] = (
            _tv(left, right) if len(left) and len(right) else 1.0
        )
    result["unk_rate_error"] = abs(
        float(np.mean(real_values == 1)) - float(np.mean(synthetic_values == 1))
    )
    return result


def _teacher_forced_diagnostics(
    *, model: Any, data: AuthorizedTrainingData, device: str, batch_size: int,
    seed: int,
) -> Mapping[str, Any]:
    """Compute exact train-only likelihood strata without validation fitting."""

    import torch
    from models.cof_ccmtpp_v1 import (
        FlatCopyReceiverDecoder,
        FlatReceiverDecoder,
        HierarchicalCopyReceiverDecoder,
    )

    group_names = ("overall", "head", "tail", "unk", "repeat", "new")
    receiver_sums = {name: 0.0 for name in group_names}
    receiver_counts = {name: 0 for name in group_names}
    gap_sums = {0: 0.0, 1: 0.0}
    gap_counts = {0: 0, 1: 0}
    amount_sum = 0.0
    finite_density_count = 0
    hierarchy = data.states.hierarchy
    head_codes = torch.tensor(hierarchy.head_codes, dtype=torch.long, device=device)
    tail_codes = torch.tensor(
        [code for cluster in hierarchy.tail_clusters for code in cluster],
        dtype=torch.long,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.train.lengths), batch_size):
            end = min(start + batch_size, len(data.train.lengths))
            amount = torch.from_numpy(data.train.x_num[start:end]).to(device)
            gap = torch.from_numpy(data.states.continuous_gap[start:end]).float().to(device)
            receiver = torch.from_numpy(data.train.x_cat[start:end, :, 0]).to(device)
            y = torch.from_numpy(data.train.y_entity[start:end]).to(device)
            lengths = torch.from_numpy(data.train.lengths[start:end]).to(device)
            valid = torch.from_numpy(data.train.valid_mask[start:end]).to(device)
            hidden = model.event_decoder(
                amount=amount,
                gap=gap,
                receiver=receiver,
                y=y,
                lengths=lengths,
                valid_mask=valid,
            )
            gap_parameters = model.gap_head(hidden)
            gap_log_prob = model.gap_head.log_prob_gap(gap, gap_parameters)
            sampled_gap = model.gap_head.sample(gap_parameters, generator=generator)
            decoder = model.receiver_decoder
            if isinstance(decoder, FlatReceiverDecoder):
                target_probability = torch.gather(
                    decoder(hidden), -1, receiver.unsqueeze(-1)
                ).squeeze(-1)
            elif isinstance(decoder, FlatCopyReceiverDecoder):
                target_probability = torch.gather(
                    decoder(
                        hidden=hidden,
                        history_receiver=receiver,
                        valid_mask=valid,
                        sampled_gap=sampled_gap,
                    ).probabilities,
                    -1,
                    receiver.unsqueeze(-1),
                ).squeeze(-1)
            elif isinstance(decoder, HierarchicalCopyReceiverDecoder):
                target_probability = decoder(
                    hidden=hidden,
                    history_receiver=receiver,
                    valid_mask=valid,
                    sampled_gap=sampled_gap,
                ).target_probability(receiver)
            else:  # pragma: no cover - guarded by validate_runtime_candidate
                raise CCMTPPContractError("unregistered receiver decoder")
            receiver_nll = -torch.log(target_probability.clamp_min(1e-12))
            amount_prediction = model.amount_head(
                hidden=hidden,
                sampled_gap=sampled_gap,
                receiver=receiver,
            )
            amount_error = (amount_prediction - amount.squeeze(-1)).square()
            width = receiver.shape[1]
            positions = torch.arange(width, device=device)
            prior = positions[None, None, :] < positions[None, :, None]
            repeated = (
                (receiver[:, None, :] == receiver[..., None])
                & prior
                & valid[:, None, :]
            ).any(dim=-1)
            groups = {
                "overall": valid,
                "head": torch.isin(receiver, head_codes) & valid,
                "tail": torch.isin(receiver, tail_codes) & valid,
                "unk": (receiver == 1) & valid,
                "repeat": repeated & valid,
                "new": (~repeated) & valid,
            }
            for name, selected in groups.items():
                receiver_counts[name] += int(selected.sum().item())
                receiver_sums[name] += float(receiver_nll[selected].sum().cpu())
            for label in (0, 1):
                selected = valid & (y == label)[:, None]
                gap_counts[label] += int(selected.sum().item())
                gap_sums[label] += float((-gap_log_prob[selected]).sum().cpu())
            amount_sum += float(amount_error[valid].sum().cpu())
            finite_density_count += int(torch.isfinite(gap_log_prob[valid]).sum().item())
    if receiver_counts["overall"] < 1 or min(gap_counts.values()) < 1:
        raise CCMTPPContractError("train likelihood diagnostics have empty support")
    receiver_result: dict[str, float | int] = {}
    for name in group_names:
        count = receiver_counts[name]
        receiver_result[f"{name}_nll"] = receiver_sums[name] / count if count else 0.0
        receiver_result[f"{name}_count"] = count
    return {
        "receiver": receiver_result,
        "gap": {
            "nll": sum(gap_sums.values()) / sum(gap_counts.values()),
            "class_0_nll": gap_sums[0] / gap_counts[0],
            "class_1_nll": gap_sums[1] / gap_counts[1],
            "finite_density_count": finite_density_count,
        },
        "amount_normalized_mse": amount_sum / receiver_counts["overall"],
    }


def _c0_gate_parent(
    *, plan: ExecutionPlan, job: ExecutionJob, validation: SequenceBatch,
    tau: np.ndarray,
) -> Mapping[str, Any]:
    record = plan.raw["datasets"][job.dataset]
    attempt = plan.repository_root / str(record["c0_attempt"])
    sample_path = attempt / "sample.npz"
    evaluation_path = attempt / "evaluation.json"
    if (
        sha256_file(sample_path) != record["c0_sample_sha256"]
        or sha256_file(evaluation_path) != record["c0_evaluation_sha256"]
    ):
        raise CCMTPPContractError("frozen C0 gate evidence changed")
    with np.load(sample_path, allow_pickle=False) as archive:
        sample = SyntheticBatch(
            x_num=archive["x_num"],
            dt_bin=archive["dt_bin"],
            x_cat=archive["x_cat"],
            valid_mask=archive["valid_mask"],
            y_entity=archive["y_entity"],
            lengths=archive["lengths"],
        )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    coherence = evaluation["metrics"]["coherence"]
    surrogate = tau[sample.dt_bin]
    return {
        "gap_ks": _classwise_gap_ks(
            real=validation,
            continuous_synthetic=surrogate,
            tau=tau,
        ),
        "short_gap_repeat_error": max(float(value) for value in coherence.values()),
        "classwise_short_gap_repeat_error": {
            f"y{label}": float(
                coherence[f"short_gap_receiver_repeat_error_y{label}"]
            )
            for label in (0, 1)
        },
    }


def _sequential_gate_decision(
    *, candidate_id: str, parent: Mapping[str, Any], current: Mapping[str, Any]
) -> Mapping[str, Any]:
    if candidate_id in {"C1", "C2"}:
        parent_coherence = parent.get("classwise_short_gap_repeat_error")
        current_coherence = current.get("classwise_short_gap_repeat_error")
        if (
            not isinstance(parent_coherence, Mapping)
            or not isinstance(current_coherence, Mapping)
            or set(parent_coherence) != {"y0", "y1"}
            or set(current_coherence) != {"y0", "y1"}
        ):
            raise CCMTPPContractError("classwise coherence gate evidence is incomplete")
        if candidate_id == "C1":
            parent_gap = parent.get("gap_ks")
            current_gap = current.get("gap_ks")
            if (
                not isinstance(parent_gap, Mapping)
                or not isinstance(current_gap, Mapping)
                or set(parent_gap) != {"y0", "y1"}
                or set(current_gap) != {"y0", "y1"}
            ):
                raise CCMTPPContractError("classwise gap gate evidence is incomplete")
            checks = {
                "classwise_gap_noninferior": all(
                    float(current_gap[key]) <= float(parent_gap[key])
                    for key in ("y0", "y1")
                ),
                "classwise_coherence_noninferior": all(
                    float(current_coherence[key]) <= float(parent_coherence[key])
                    for key in ("y0", "y1")
                ),
            }
        else:
            checks = {
                "classwise_short_gap_repeat_improved": all(
                    float(current_coherence[key]) < float(parent_coherence[key])
                    for key in ("y0", "y1")
                ),
                "receiver_tv_noninferior": float(current["receiver_tv"])
                <= float(parent["receiver_tv"]),
            }
        return {
            "candidate_id": candidate_id,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return evaluate_candidate_stop(candidate_id, parent=parent, current=current)


def run_authorized_job(
    *, plan: ExecutionPlan, job: ExecutionJob, authorization: Mapping[str, Any],
    data: AuthorizedTrainingData, model: Any, device: str, attempt_path: Path,
    event_queue: Any,
) -> Mapping[str, Any]:
    import torch

    runtime_root = Path(str(plan.raw["runtime_root"]))
    if not runtime_root.is_absolute():
        runtime_root = plan.repository_root / runtime_root
    ownership = runtime_root / "workers" / job.dataset / job.candidate_id / "ownership.lock"
    store = CCMTPPAttemptStore.attach_existing(
        runtime_root=runtime_root,
        job=job,
        ownership_path=ownership,
        attempt_path=attempt_path,
    )
    config = _training_config(plan)
    store.write_json("receiver_vocabulary_state.json", data.states.receiver_vocabulary)
    store.write_json("receiver_hierarchy_state.json", asdict(data.states.hierarchy))
    store.write_json("gap_support_state.json", data.states.gap_support)
    store.write_json("amount_transform_state.json", data.states.amount_transform)
    started = time.monotonic()
    torch.manual_seed(job.seed)
    rng = np.random.default_rng(job.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    n = len(data.train.lengths)
    batch_size = int(config["batch_size"])
    if n < batch_size:
        raise CCMTPPContractError("train split is smaller than frozen batch size")
    progress_interval = int(plan.raw["execution"]["progress_interval_updates"])
    checkpoint_interval = int(config["checkpoint_interval_updates"])
    actual_updates = 0
    last_diagnostics: Mapping[str, Any] = {}
    generator = torch.Generator(device=device).manual_seed(job.seed + 1)
    model.train()
    for step in range(1, int(config["requested_updates"]) + 1):
        indices = rng.choice(n, size=batch_size, replace=False)
        optimizer.zero_grad(set_to_none=True)
        valid = torch.from_numpy(data.train.valid_mask[indices]).to(device)
        loss, last_diagnostics = model.compute_loss(
            amount=torch.from_numpy(data.train.x_num[indices]).to(device),
            gap=torch.from_numpy(data.states.continuous_gap[indices]).float().to(device),
            receiver=torch.from_numpy(data.train.x_cat[indices, :, 0]).to(device),
            y=torch.from_numpy(data.train.y_entity[indices]).to(device),
            lengths=torch.from_numpy(data.train.lengths[indices]).to(device),
            valid_mask=valid,
            generator=generator,
        )
        if not bool(torch.isfinite(loss)):
            raise CCMTPPContractError("non-finite CCMTPP training loss")
        loss.backward()
        optimizer.step()
        actual_updates = step
        if step == 1 or step % progress_interval == 0:
            progress = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "elapsed_seconds": time.monotonic() - started,
            }
            store.append_progress(progress)
            event_queue.put({"kind": "progress", **progress})
        if step % checkpoint_interval == 0:
            checkpoint = _checkpoint_bundle(model, data=data, plan=plan)
            store.write_bytes(
                f"checkpoints/step_{step:06d}.pt", _torch_bytes(checkpoint)
            )
            store.write_json(
                f"checkpoints/latest_step_{step:06d}.json",
                {"step": step, "path": f"step_{step:06d}.pt"},
            )
    final_bundle = _checkpoint_bundle(model, data=data, plan=plan)
    final_path = store.write_bytes("checkpoints/final.pt", _torch_bytes(final_bundle))
    checkpoint_provenance = build_checkpoint_provenance(
        plan=plan,
        job=job,
        authorization_sha256=str(authorization["authorization_sha256"]),
        train_state_binding=data.states.binding,
        actual_updates=actual_updates,
        checkpoint_sha256=sha256_file(final_path),
    )
    store.write_json("checkpoint_provenance.json", checkpoint_provenance)
    event_queue.put({"kind": "progress", "phase": "fit_complete", "step": actual_updates})

    validation_path = resolve_dataset_access(
        plan=plan,
        dataset=job.dataset,
        split="validation",
        purpose="evaluation",
        fit_complete=True,
    )
    if validation_path != data.validation_path.resolve():
        raise CCMTPPContractError("post-fit validation path binding changed")
    validation = _load_sequence_batch(validation_path)
    sampling_plan = _fixed_validation_plan(plan=plan, job=job, validation=validation)
    store.write_json(
        "conditioning_plan.json",
        {
            "fit_split": "train",
            "validation_use": "post_fit_generation_and_evaluation_only",
            "sampling_plan_sha256": plan.raw["datasets"][job.dataset][
                "sampling_plan_sha256"
            ],
            "plan_hash": sampling_plan.plan_hash,
            "validation_rows_used_for_fit": 0,
            "internal_test_rows_used": 0,
            "fraud_test_rows_used": 0,
        },
    )
    synthetic, continuous_gap = _sample_validation(
        model=model,
        plan=sampling_plan,
        transform=data.transform,
        device=device,
        seed=job.seed + 100_000,
        batch_size=int(plan.raw["execution"]["validation_sample_batch_size"]),
        event_queue=event_queue,
    )
    sample_path = store.write_bytes(
        "validation_sample.npz",
        _npz_bytes(
            {
                "x_num": synthetic.x_num,
                "dt_bin": synthetic.dt_bin,
                "x_cat": synthetic.x_cat,
                "valid_mask": synthetic.valid_mask,
                "y_entity": synthetic.y_entity,
                "lengths": synthetic.lengths,
                "continuous_gap": continuous_gap,
            }
        ),
    )
    tau = np.asarray(data.transform["gap_tau"], dtype=np.float64)
    threshold_path = (
        plan.repository_root
        / str(plan.raw["datasets"][job.dataset]["c0_attempt"])
        / "train_bootstrap_thresholds.json"
    )
    if sha256_file(threshold_path) != plan.raw["datasets"][job.dataset][
        "threshold_sha256"
    ]:
        raise CCMTPPContractError("frozen train-only threshold changed")
    threshold_state = json.loads(threshold_path.read_text(encoding="utf-8"))
    try:
        hard = validate_external_hard_contract(
            real_validation=validation,
            synthetic=synthetic,
            gap_cardinality=len(tau),
            receiver_cardinality=int(data.states.receiver_vocabulary["receiver_classes"]),
        )
        external_metrics = compute_external_validation_metrics(
            real=validation,
            synthetic=synthetic,
            gap_tau=tau,
            short_gap_threshold=float(threshold_state["short_gap_threshold"]),
        )
        classwise_gap = _classwise_gap_ks(
            real=validation,
            continuous_synthetic=continuous_gap,
            tau=tau,
        )
        fidelity = _receiver_fidelity_decomposition(
            real=validation,
            synthetic=synthetic,
            hierarchy=data.states.hierarchy,
        )
        current = {
            "gap_ks": classwise_gap,
            "short_gap_repeat_error": max(
                float(value) for value in external_metrics["coherence"].values()
            ),
            "receiver_tv": fidelity["full_receiver_tv"],
            "head_tv": fidelity["head_receiver_tv"],
            "tail_tv": fidelity["tail_receiver_tv"],
            "unk_error": fidelity["unk_rate_error"],
        }
        class_receiver_tv = {}
        for label in (0, 1):
            real_selected = validation.valid_mask & (
                validation.y_entity[:, None] == label
            )
            synth_selected = synthetic.valid_mask & (
                synthetic.y_entity[:, None] == label
            )
            class_receiver_tv[f"y{label}"] = _tv(
                validation.x_cat[..., 0][real_selected],
                synthetic.x_cat[..., 0][synth_selected],
            )
        coherence_by_class = {
            f"y{label}": float(
                external_metrics["coherence"][
                    f"short_gap_receiver_repeat_error_y{label}"
                ]
            )
            for label in (0, 1)
        }
        current.update(
            {
                "classwise_receiver_tv": class_receiver_tv,
                "classwise_short_gap_repeat_error": coherence_by_class,
                "y0_composite": max(
                    classwise_gap["y0"],
                    class_receiver_tv["y0"],
                    coherence_by_class["y0"],
                ),
                "y1_composite": max(
                    classwise_gap["y1"],
                    class_receiver_tv["y1"],
                    coherence_by_class["y1"],
                ),
            }
        )
        if job.candidate_id == "C1":
            parent = _c0_gate_parent(
                plan=plan, job=job, validation=validation, tau=tau
            )
        else:
            parent_id = {"C2": "C1", "C3": "C2", "C4": "C3"}[job.candidate_id]
            parent_path = runtime_root / job.dataset / parent_id / f"seed_{job.seed}" / "attempt_001"
            parent = json.loads((parent_path / "metrics.json").read_text(encoding="utf-8"))[
                "gate_values"
            ]
        decision = _sequential_gate_decision(
            candidate_id=job.candidate_id,
            parent=parent,
            current=current,
        )
        gate = {
            "candidate_id": job.candidate_id,
            "stop_criterion_id": job.candidate_id,
            "status": "PASS" if decision["passed"] else "FAIL",
            "checks": decision["checks"],
            "parent_values": parent,
            "current_values": current,
        }
        evaluation_status = "VALID"
    except ExternalMetricError as error:
        hard = {"status": "INVALID", "message": str(error)}
        external_metrics = {"fidelity": {}, "coherence": {}}
        classwise_gap = {}
        fidelity = {
            "full_receiver_tv": 1.0,
            "head_receiver_tv": 1.0,
            "tail_receiver_tv": 1.0,
            "unk_rate_error": 1.0,
        }
        current = {}
        gate = {
            "candidate_id": job.candidate_id,
            "stop_criterion_id": job.candidate_id,
            "status": "NOT_EVALUABLE",
            "reason": str(error),
        }
        evaluation_status = "INVALID"
    likelihood_diagnostics = _teacher_forced_diagnostics(
        model=model,
        data=data,
        device=device,
        batch_size=int(config["batch_size"]),
        seed=job.seed + 200_000,
    )
    diagnostics = {
        "gap": {
            **likelihood_diagnostics["gap"],
            "sampled_min": float(continuous_gap[synthetic.valid_mask].min()),
            "sampled_max": float(continuous_gap[synthetic.valid_mask].max()),
        },
        "receiver": likelihood_diagnostics["receiver"],
        "amount": {
            "normalized_mse": likelihood_diagnostics["amount_normalized_mse"],
            "decode_contract_sha256": canonical_sha256(
                {"amount_contract": AMOUNT_CONTRACT_NAME}
            ),
        },
        "fidelity": {"classwise_gap_ks": classwise_gap, **fidelity},
    }
    store.write_json("diagnostics.json", diagnostics)
    store.write_json(
        "metrics.json",
        {
            "external_metrics": external_metrics,
            "gate_values": current,
            "validation_sample_sha256": sha256_file(sample_path),
        },
    )
    store.write_json(
        "evaluation.json",
        {
            "status": evaluation_status,
            "hard_contract": hard,
            "test_accessed": False,
            "sparkov_fraud_test_accessed": False,
        },
    )
    store.write_json("gate_decision.json", gate)
    store.write_json(
        "runtime.json",
        {
            "requested_updates": int(config["requested_updates"]),
            "actual_updates": actual_updates,
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "device": device,
        },
    )
    return {"terminal_status": "COMPLETE" if evaluation_status == "VALID" else "INVALID"}


def build_default_execution_dependencies() -> ExecutionDependencies:
    return ExecutionDependencies(
        load_data_body=load_authorized_train_body,
        build_model=build_authorized_model,
        query_device=select_explicit_device,
        run_job=run_authorized_job,
    )
