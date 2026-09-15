"""Lazy, authorization-only backend for CoF-HCMTTPP-v2 H1.

The module is imported only after the runner validates an exact dataset-scoped
authorization.  It never inventories GPUs and exposes only ``cuda:0`` selected
by the user's shell through ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import yaml

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.external_validation_metrics_v1 import (
    ExternalMetricError,
    compute_external_validation_metrics,
    validate_external_hard_contract,
)
from experiments.cof_hcmttpp_v2_execution_runner import (
    ExecutionDependencies,
    ExecutionJob,
    ExecutionPlan,
    H1AttemptStore,
    H1RunnerContractError,
    build_checkpoint_provenance,
    canonical_sha256,
    evaluate_h1_gate,
    resolve_dataset_access,
    sha256_file,
)
from generators.sampling_plan import SamplingPlan


@dataclass(frozen=True)
class AuthorizedTrainingData:
    train: SequenceBatch
    transform: Mapping[str, Any]
    continuous_gap: np.ndarray
    receiver_classes: int
    receiver_reference: Mapping[str, Any]
    amount_reference: Mapping[str, Any]
    tail_state: Any
    binding: Mapping[str, Any]
    validation_relative_path: str


def _load_sequence_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as archive:
        values = {field: archive[field] for field in SequenceBatch.__dataclass_fields__}
    return SequenceBatch(**values)


def _validate_train_batch(batch: SequenceBatch) -> None:
    n, width = batch.valid_mask.shape
    if (
        batch.x_num.shape != (n, width, 1)
        or batch.dt_bin.shape != (n, width)
        or batch.x_cat.shape != (n, width, 1)
        or batch.y_entity.shape != (n,)
        or batch.lengths.shape != (n,)
        or batch.valid_mask.dtype != np.bool_
        or not np.array_equal(
            batch.valid_mask, np.arange(width)[None, :] < batch.lengths[:, None]
        )
    ):
        raise H1RunnerContractError("train sequence/mask contract is invalid")


def _numeric_gap_edges(transform: Mapping[str, Any]) -> np.ndarray:
    raw = transform.get("gap_edges")
    tau = transform.get("gap_tau")
    if not isinstance(raw, list) or not isinstance(tau, list) or len(raw) != len(tau) + 1:
        raise H1RunnerContractError("train-only gap edges are invalid")
    decoded = []
    for item in raw:
        if item == "+inf":
            decoded.append(float("inf"))
        elif item == "-inf":
            decoded.append(float("-inf"))
        else:
            decoded.append(float(item))
    result = np.asarray(decoded, dtype=np.float64)
    if np.isnan(result).any() or np.any(result[1:] < result[:-1]):
        raise H1RunnerContractError("train-only gap edges are not monotone")
    return result


def load_authorized_train_body(
    *, plan: ExecutionPlan, job: ExecutionJob, authorization: Mapping[str, Any]
) -> AuthorizedTrainingData:
    """Open train first and fit every H1 state from train rows only."""

    if authorization.get("dataset_scope") != job.dataset:
        raise H1RunnerContractError("authorized train dataset mismatch")
    train_path = resolve_dataset_access(
        plan=plan,
        split="train",
        purpose="tail_state_fit_and_training",
        final_checkpoint_complete=False,
    )
    record = plan.raw["datasets"][job.dataset]
    transform_path = plan.repository_root / str(record["transform_path"])
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    if (
        transform.get("fit_role") != "train"
        or transform.get("pad_code") != 0
        or transform.get("unk_code") != 1
    ):
        raise H1RunnerContractError("external transform is not frozen train-only")
    train = _load_sequence_batch(train_path)
    _validate_train_batch(train)
    vocabulary = transform.get("receiver_vocabulary")
    if not isinstance(vocabulary, list):
        raise H1RunnerContractError("train receiver vocabulary is missing")
    codes = sorted(int(item["code"]) for item in vocabulary)
    if codes != list(range(2, len(codes) + 2)):
        raise H1RunnerContractError("train receiver vocabulary is non-contiguous")
    receiver_classes = len(codes) + 2
    receiver = train.x_cat[..., 0]
    if np.any(receiver[train.valid_mask] < 2) or np.any(
        receiver[train.valid_mask] >= receiver_classes
    ):
        raise H1RunnerContractError("train receiver violates PAD/UNK vocabulary")
    tau = np.asarray(transform.get("gap_tau"), dtype=np.float64)
    if (
        tau.ndim != 1
        or not len(tau)
        or not np.isfinite(tau).all()
        or np.any(tau < 0)
        or np.any(train.dt_bin[train.valid_mask] < 0)
        or np.any(train.dt_bin[train.valid_mask] >= len(tau))
    ):
        raise H1RunnerContractError("train gap support is invalid")
    gap = np.zeros(train.dt_bin.shape, dtype=np.float64)
    gap[train.valid_mask] = tau[train.dt_bin[train.valid_mask]]
    from models.cof_hcmttpp_v2 import fit_train_only_h1_tail_state

    import torch

    tail_state = fit_train_only_h1_tail_state(
        gap=torch.from_numpy(gap),
        y=torch.from_numpy(train.y_entity).long(),
        valid_mask=torch.from_numpy(train.valid_mask),
        fit_split="train",
        provenance={
            "train_manifest_sha256": str(record["train_sha256"]),
            "transform_state_sha256": str(record["transform_sha256"]),
        },
    )
    receiver_payload = {
        "schema_version": "cof-hcmttpp-v2-h1-receiver-reference-v1",
        "fit_split": "train",
        "path": "flat_no_copy",
        "pad_code": 0,
        "unk_code": 1,
        "receiver_classes": receiver_classes,
        "codes": codes,
        "train_sha256": record["train_sha256"],
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    receiver_counts = np.bincount(
        receiver[train.valid_mask].astype(np.int64), minlength=receiver_classes
    )
    ordered = sorted(range(2, receiver_classes), key=lambda code: (-receiver_counts[code], code))
    head_codes = [code for code in ordered if receiver_counts[code] >= 32][:512]
    receiver_payload["diagnostic_head_rule"] = {
        "algorithm": "descending_frequency_then_code",
        "head_min_count": 32,
        "max_head_categories": 512,
        "fit_split": "train",
    }
    receiver_payload["diagnostic_head_codes"] = head_codes
    receiver_payload["diagnostic_tail_codes"] = [
        code for code in range(2, receiver_classes) if code not in set(head_codes)
    ]
    receiver_reference = {
        **receiver_payload,
        "state_sha256": canonical_sha256(receiver_payload),
    }
    amount_payload = {
        "schema_version": "cof-hcmttpp-v2-h1-amount-reference-v1",
        "fit_split": "train",
        "contract": "frozen_non_v3_train_only_encode_inverse_decode_v1",
        "amount_log_mean": float(transform["amount_log_mean"]),
        "amount_log_std": float(transform["amount_log_std"]),
        "transform_sha256": record["transform_sha256"],
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    if (
        not np.isfinite(amount_payload["amount_log_mean"])
        or not np.isfinite(amount_payload["amount_log_std"])
        or amount_payload["amount_log_std"] <= 0
    ):
        raise H1RunnerContractError("amount transform is invalid")
    amount_reference = {
        **amount_payload,
        "state_sha256": canonical_sha256(amount_payload),
    }
    binding = {
        "fit_split": "train",
        "train_sha256": record["train_sha256"],
        "transform_sha256": record["transform_sha256"],
        "receiver_vocabulary_sha256": receiver_reference["state_sha256"],
        "tail_state_sha256": tail_state.state_sha256,
        "amount_contract_sha256": amount_reference["state_sha256"],
        "sampling_plan_sha256": record["sampling_plan_sha256"],
        "threshold_sha256": record["threshold_sha256"],
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    return AuthorizedTrainingData(
        train=train,
        transform=transform,
        continuous_gap=gap,
        receiver_classes=receiver_classes,
        receiver_reference=receiver_reference,
        amount_reference=amount_reference,
        tail_state=tail_state,
        binding=binding,
        # Deliberately retain the configured string without resolving/opening it.
        validation_relative_path=str(record["validation_path"]),
    )


def _source_budget(plan: ExecutionPlan) -> Mapping[str, Any]:
    path = plan.repository_root / str(plan.raw["source_config_path"])
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    budget = raw.get("fixed_model_and_budget") if isinstance(raw, Mapping) else None
    if not isinstance(budget, Mapping):
        raise H1RunnerContractError("H1 frozen model budget is missing")
    for key in (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "requested_updates",
        "max_wall_seconds",
        "checkpoint_interval_updates",
        "early_stopping",
        "retry",
        "sweep",
    ):
        if budget.get(key) != plan.raw["training"].get(key):
            raise H1RunnerContractError(f"H1 training factor changed: {key}")
    return budget


def validate_runtime_h1_factor_isolation(
    *, model: Any, plan: ExecutionPlan, enabled_hooks: tuple[str, ...] = ()
) -> Mapping[str, Any]:
    config = model.model_config()
    forbidden_hooks = {
        "pointer",
        "hierarchy",
        "y_balanced_likelihood",
        "coherence_loss",
        "structure_loss",
        "posthoc_calibration",
        "c1_v1_retry",
        "C2",
        "C3",
        "C4",
        "H2",
    }
    if (
        model.__class__.__name__ != "CoFHCMTTPPV2H1"
        or config.get("candidate") != "H1"
        or config.get("changed_factors") != ["gap_decoder"]
        or config.get("central_bins") != 16
        or config.get("receiver_path") != "flat_no_copy"
        or config.get("y_balanced_likelihood") is not False
        or config.get("structure_loss") is not None
        or config.get("permanent_v1_chain_state")
        != "STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL"
        or forbidden_hooks.intersection(enabled_hooks)
        or model.receiver_decoder.__class__.__name__ != "FlatReceiverDecoder"
        or model.gap_decoder.__class__.__name__ != "H1HurdleRQSGapDecoder"
    ):
        raise H1RunnerContractError("H1 factor-isolation contract changed")
    _source_budget(plan)
    return {"status": "PASS", "changed_factors": ["gap_decoder"]}


def build_authorized_model(
    *, plan: ExecutionPlan, job: ExecutionJob, data: AuthorizedTrainingData
):
    del job
    from models.cof_hcmttpp_v2 import CoFHCMTTPPV2H1

    budget = _source_budget(plan)
    model = CoFHCMTTPPV2H1(
        candidate="H1",
        receiver_classes=data.receiver_classes,
        tail_state=data.tail_state,
        d_model=int(budget["d_model"]),
        n_heads=int(budget["n_heads"]),
        n_layers=int(budget["n_layers"]),
        max_length=int(budget["max_length"]),
        dropout=float(budget["dropout"]),
    )
    validate_runtime_h1_factor_isolation(model=model, plan=plan)
    return model


def attach_cpu_readiness_attempt_003(payload: Mapping[str, Any]) -> H1AttemptStore:
    """Exercise the production parent/child ownership boundary in scratch space.

    This entry point is intentionally incapable of loading data, selecting a
    device, or running an authorized job.  It exists only for the exact
    attempt_003 CPU-synthetic execution-readiness integration gate.
    """

    job = ExecutionJob(**dict(payload["job"]))
    return H1AttemptStore.attach_existing(
        runtime_root=Path(str(payload["runtime_root"])),
        job=job,
        ownership_path=Path(str(payload["ownership_path"])),
        attempt_path=Path(str(payload["attempt_path"])),
        readiness_only=True,
        expected_ownership_id=str(payload["ownership_id"]),
        expected_authorization_sha256=str(payload["authorization_sha256"]),
    )


def run_cpu_readiness_attempt_003(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one real H1 NLL/backward/update boundary on CPU synthetic tensors."""

    store = attach_cpu_readiness_attempt_003(payload)
    if store.job.attempt != "attempt_003":
        raise H1RunnerContractError("readiness backend accepts only attempt_003")

    import torch
    from models.cof_hcmttpp_v2 import (
        CoFHCMTTPPV2H1,
        InvalidH1GapStateError,
        fit_train_only_h1_tail_state,
    )

    frozen = dict(payload["frozen_model_and_optimizer"])
    expected_frozen = {
        "seed": 4001,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "max_length": 64,
        "dropout": 0.0,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    if frozen != expected_frozen:
        raise H1RunnerContractError("readiness frozen model/optimizer mismatch")

    torch.manual_seed(4001)
    train_width = 64
    train_y = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
    train_valid = torch.ones((10, train_width), dtype=torch.bool)
    base = torch.linspace(0.01, 7.0, train_width, dtype=torch.float64)
    train_gap = torch.stack(
        [torch.expm1(base + 0.003 * row) for row in range(10)], dim=0
    )
    tail_state = fit_train_only_h1_tail_state(
        gap=train_gap,
        y=train_y,
        valid_mask=train_valid,
        fit_split="train",
        provenance={
            "authorization_sha256": str(payload["authorization_sha256"]),
            "dataset": store.job.dataset,
            "scope": "cpu_synthetic_execution_readiness",
        },
    )
    if (
        tail_state.fit_split != "train"
        or tail_state.validation_rows_used != 0
        or tail_state.internal_test_rows_used != 0
        or tail_state.fraud_test_rows_used != 0
    ):
        raise H1RunnerContractError("readiness tail state is not train-only")

    model = CoFHCMTTPPV2H1(
        candidate="H1",
        receiver_classes=12,
        tail_state=tail_state,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_length=64,
        dropout=0.0,
    ).cpu()
    model.train()

    lengths = torch.tensor([8, 7, 6, 5], dtype=torch.long)
    position = torch.arange(8)
    valid = position[None, :] < lengths[:, None]
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    amount = torch.zeros((4, 8, 1), dtype=torch.float32)
    amount[valid] = torch.linspace(
        -1.0, 1.0, int(valid.sum()), dtype=torch.float32
    ).unsqueeze(-1)
    receiver = torch.zeros((4, 8), dtype=torch.long)
    receiver[valid] = 2 + torch.arange(int(valid.sum()), dtype=torch.long) % 10
    gap = torch.zeros((4, 8), dtype=torch.float32)
    for row in range(4):
        u_tail = tail_state.u_tail[int(y[row])]
        beta = tail_state.beta_base[int(y[row])]
        route_u = (0.0, 0.5 * u_tail, u_tail, u_tail + beta)
        for column in range(int(lengths[row])):
            gap[row, column] = 0.0 if column % 4 == 0 else math.expm1(
                route_u[column % 4]
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.zero_grad(set_to_none=True)
    hidden = model.event_decoder(
        amount=amount,
        gap=gap,
        receiver=receiver,
        y=y,
        lengths=lengths,
        valid_mask=valid,
    )
    parameters = model.gap_decoder(hidden, y=y)
    event_nll = model.gap_decoder.event_nll(gap, parameters, valid_mask=valid)
    loss = event_nll[valid].mean()
    if not bool(torch.isfinite(loss)):
        raise H1RunnerContractError("readiness H1 loss is non-finite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    if not finite_gradients:
        raise H1RunnerContractError("readiness H1 gradients are non-finite")
    optimizer.step()
    finite_parameters = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    parameters_changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    if not finite_parameters or not parameters_changed:
        raise H1RunnerContractError("readiness optimizer step is invalid")

    with torch.no_grad():
        hidden_after = model.event_decoder(
            amount=amount,
            gap=gap,
            receiver=receiver,
            y=y,
            lengths=lengths,
            valid_mask=valid,
        )
        parameters_after = model.gap_decoder(hidden_after, y=y)
        q = torch.full_like(parameters_after.zero_logits, 0.5)
        forward = model.gap_decoder.central_quantile(q, parameters_after)
        inverse = model.gap_decoder.central_inverse(
            forward.value, parameters_after
        )
        q_low = torch.full_like(
            parameters_after.zero_logits,
            torch.finfo(parameters_after.zero_logits.dtype).eps,
        )
        q_high = torch.full_like(
            parameters_after.zero_logits,
            1.0 - torch.finfo(parameters_after.zero_logits.dtype).eps,
        )
        open_endpoint_forward_low = model.gap_decoder.central_quantile(
            q_low, parameters_after
        )
        open_endpoint_forward_high = model.gap_decoder.central_quantile(
            q_high, parameters_after
        )
        open_endpoint_inverse_low = model.gap_decoder.central_inverse(
            open_endpoint_forward_low.value, parameters_after
        )
        open_endpoint_inverse_high = model.gap_decoder.central_inverse(
            open_endpoint_forward_high.value, parameters_after
        )
        endpoint = model.gap_decoder.central_inverse(
            parameters_after.u_tail, parameters_after
        )
        left = torch.nextafter(
            parameters_after.u_tail,
            torch.zeros_like(parameters_after.u_tail),
        )
        right = torch.nextafter(
            parameters_after.u_tail,
            torch.full_like(parameters_after.u_tail, float("inf")),
        )
        boundary_cdf = torch.stack(
            (
                model.gap_decoder.positive_cdf_u(left, parameters_after),
                model.gap_decoder.positive_cdf_u(
                    parameters_after.u_tail, parameters_after
                ),
                model.gap_decoder.positive_cdf_u(right, parameters_after),
            )
        )
        boundary_log_prob = torch.stack(
            (
                model.gap_decoder.positive_log_prob_u(left, parameters_after),
                model.gap_decoder.positive_log_prob_u(
                    parameters_after.u_tail, parameters_after
                ),
                model.gap_decoder.positive_log_prob_u(right, parameters_after),
            )
        )
        rqs_finite = all(
            bool(torch.isfinite(value).all())
            for value in (
                forward.value,
                forward.log_abs_det,
                inverse.value,
                inverse.log_abs_det,
                open_endpoint_forward_low.value,
                open_endpoint_forward_low.log_abs_det,
                open_endpoint_forward_high.value,
                open_endpoint_forward_high.log_abs_det,
                open_endpoint_inverse_low.value,
                open_endpoint_inverse_low.log_abs_det,
                open_endpoint_inverse_high.value,
                open_endpoint_inverse_high.log_abs_det,
                endpoint.value,
                endpoint.log_abs_det,
            )
        ) and bool(
            torch.allclose(
                inverse.value,
                q.to(inverse.value),
                atol=2.0e-5,
                rtol=2.0e-5,
            )
        ) and bool(
            torch.allclose(
                open_endpoint_inverse_low.value,
                q_low.to(open_endpoint_inverse_low.value),
                atol=2.0e-5,
                rtol=2.0e-5,
            )
        ) and bool(
            torch.allclose(
                open_endpoint_inverse_high.value,
                q_high.to(open_endpoint_inverse_high.value),
                atol=2.0e-5,
                rtol=2.0e-5,
            )
        )
        boundary_finite = bool(torch.isfinite(boundary_cdf).all()) and bool(
            torch.isfinite(boundary_log_prob).all()
        )

    invalid_domain_rejected = False
    try:
        model.gap_decoder.central_inverse(
            torch.zeros_like(parameters_after.u_tail), parameters_after
        )
    except (ValueError, InvalidH1GapStateError):
        invalid_domain_rejected = True
    if not rqs_finite or not boundary_finite or not invalid_domain_rejected:
        raise H1RunnerContractError("readiness RQS boundary contract failed")

    return {
        "spawn_payload_round_trip": "PASS",
        "ownership_attach": "PASS",
        "tail_state_fit_split": tail_state.fit_split,
        "tail_state_nontrain_rows": 0,
        "optimizer_steps": 1,
        "finite_loss": True,
        "finite_gradients": finite_gradients,
        "finite_parameters": finite_parameters,
        "parameters_changed": parameters_changed,
        "routes": {
            "zero": True,
            "positive_body": True,
            "central_endpoint": True,
            "positive_tail": True,
            "y0": bool((y == 0).any()),
            "y1": bool((y == 1).any()),
            "padding": bool((~valid).any()),
        },
        "rqs_forward_inverse_finite": rqs_finite,
        "central_tail_boundary_finite": boundary_finite,
        "invalid_domain_rejected": invalid_domain_rejected,
        "loss": float(loss.detach()),
    }


def select_explicit_device(
    *, plan: ExecutionPlan, job: ExecutionJob, authorization: Mapping[str, Any]
) -> str:
    del job
    if (
        plan.raw["execution"].get("explicit_device") != "cuda:0"
        or plan.raw["execution"].get("gpu_inventory_query") != "FORBIDDEN"
        or authorization.get("scope", {}).get("gpu_inventory_query") is not False
    ):
        raise H1RunnerContractError("explicit device/no-inventory contract changed")
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


def _h1_loss(model: Any, *, amount: Any, gap: Any, receiver: Any, y: Any,
             lengths: Any, valid: Any, generator: Any):
    import torch
    from models.cof_ccmtpp_v1 import SampledGap

    hidden = model.event_decoder(
        amount=amount,
        gap=gap,
        receiver=receiver,
        y=y,
        lengths=lengths,
        valid_mask=valid,
    )
    parameters = model.gap_decoder(hidden, y=y)
    gap_event = model.gap_decoder.event_nll(gap, parameters, valid_mask=valid)
    sampled = model.gap_decoder.sample(parameters, generator=generator)
    receiver_probability = model.receiver_decoder(hidden)
    receiver_event = -torch.log(
        torch.gather(receiver_probability, -1, receiver.unsqueeze(-1))
        .squeeze(-1)
        .clamp_min(1e-12)
    )
    preserved_gap = SampledGap(
        gap=sampled.gap,
        u=sampled.u,
        source="cof_ccmtpp_v1_continuous_gap_head",
    )
    amount_prediction = model.amount_head(
        hidden=hidden, sampled_gap=preserved_gap, receiver=receiver
    )
    amount_event = (amount_prediction - amount.squeeze(-1)).square()
    total = (gap_event + receiver_event + amount_event)[valid].mean()
    return total, {
        "total_loss": float(total.detach()),
        "gap_nll": float(gap_event[valid].mean().detach()),
        "receiver_nll": float(receiver_event[valid].mean().detach()),
        "amount_loss": float(amount_event[valid].mean().detach()),
        "structure_loss": 0.0,
        "y_balanced_likelihood": 0,
    }


def _fixed_validation_plan(
    *, plan: ExecutionPlan, job: ExecutionJob, validation: SequenceBatch
) -> SamplingPlan:
    record = plan.raw["datasets"][job.dataset]
    path = plan.repository_root / str(record["sampling_plan_path"])
    with np.load(path, allow_pickle=False) as archive:
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
        raise H1RunnerContractError("validation conditioning plan mismatch")
    return fixed


def _sample_validation(
    *, model: Any, plan: SamplingPlan, transform: Mapping[str, Any], device: str,
    seed: int, batch_size: int, event_queue: Any,
) -> tuple[SyntheticBatch, np.ndarray]:
    import torch
    from models.cof_ccmtpp_v1 import SampledGap

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
                gap_parameters = model.gap_decoder(hidden, y=y)
                sampled_h1 = model.gap_decoder.sample(gap_parameters, generator=generator)
                receiver_probability = model.receiver_decoder(hidden)
                sampled_receiver = torch.multinomial(
                    receiver_probability.reshape(-1, model.receiver_classes),
                    1,
                    generator=generator,
                ).reshape(hidden.shape[:-1])
                sampled_gap = SampledGap(
                    gap=sampled_h1.gap,
                    u=sampled_h1.u,
                    source="cof_ccmtpp_v1_continuous_gap_head",
                )
                sampled_amount = model.amount_head(
                    hidden=hidden,
                    sampled_gap=sampled_gap,
                    receiver=sampled_receiver,
                )
                active = valid[:, position]
                gap[:, position] = torch.where(active, sampled_h1.gap[:, position], gap[:, position])
                receiver[:, position] = torch.where(active, sampled_receiver[:, position], receiver[:, position])
                amount[:, position, 0] = torch.where(active, sampled_amount[:, position], amount[:, position, 0])
            amount_out[start:end] = amount.cpu().numpy().astype(np.float32)
            gap_out[start:end] = gap.cpu().numpy()
            receiver_out[start:end, :, 0] = receiver.cpu().numpy()
            event_queue.put({"kind": "progress", "phase": "sampling", "entities": end})
    edges = _numeric_gap_edges(transform)
    bins = np.searchsorted(edges[1:-1], gap_out, side="right").astype(np.int64)
    bins[~plan.valid_mask] = 0
    amount_out[~plan.valid_mask] = 0
    receiver_out[~plan.valid_mask] = 0
    return SyntheticBatch(
        x_num=amount_out,
        dt_bin=bins,
        x_cat=receiver_out,
        valid_mask=plan.valid_mask.copy(),
        y_entity=plan.y_entity.copy(),
        lengths=plan.lengths.copy(),
    ), gap_out


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        raise H1RunnerContractError("gap KS class is empty")
    support = np.sort(np.concatenate((left, right)))
    return float(np.max(np.abs(
        np.searchsorted(np.sort(left), support, side="right") / len(left)
        - np.searchsorted(np.sort(right), support, side="right") / len(right)
    )))


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return 1.0
    support = np.union1d(left, right)
    l = np.zeros(len(support), dtype=float)
    r = np.zeros(len(support), dtype=float)
    values, counts = np.unique(left, return_counts=True)
    l[np.searchsorted(support, values)] = counts / len(left)
    values, counts = np.unique(right, return_counts=True)
    r[np.searchsorted(support, values)] = counts / len(right)
    return float(np.abs(l - r).sum() / 2)


def _gate_observations(
    *, validation: SequenceBatch, synthetic: SyntheticBatch,
    continuous_gap: np.ndarray, tau: np.ndarray, external: Mapping[str, Any],
) -> Mapping[str, Any]:
    real_gap = tau[validation.dt_bin]
    gap_ks: dict[str, float] = {}
    positive_ks: dict[str, float] = {}
    receiver_tv: dict[str, float] = {}
    for label in (0, 1):
        selected = validation.valid_mask & (validation.y_entity[:, None] == label)
        gap_ks[f"y{label}"] = _ks(real_gap[selected], continuous_gap[selected])
        real_positive = real_gap[selected & (real_gap > 0)]
        synthetic_positive = continuous_gap[selected & (continuous_gap > 0)]
        positive_ks[f"y{label}"] = _ks(real_positive, synthetic_positive)
        receiver_tv[f"y{label}"] = _tv(
            validation.x_cat[..., 0][selected], synthetic.x_cat[..., 0][selected]
        )
    coherence = {
        f"y{label}": float(
            external["coherence"][f"short_gap_receiver_repeat_error_y{label}"]
        )
        for label in (0, 1)
    }
    return {
        "gap_ks": gap_ks,
        "positive_gap_ks": positive_ks,
        "coherence": coherence,
        "receiver_tv": receiver_tv,
        "full_receiver_tv": _tv(
            validation.x_cat[..., 0][validation.valid_mask],
            synthetic.x_cat[..., 0][synthetic.valid_mask],
        ),
        "hard_validity": True,
    }


def _conditional_h1_diagnostics(
    *, model: Any, validation: SequenceBatch, real_gap: np.ndarray,
    device: str,
) -> Mapping[str, Any]:
    """Teacher-forced validation diagnostics only; no state is fitted here."""

    import torch

    amount = torch.from_numpy(validation.x_num).to(device)
    gap = torch.from_numpy(real_gap).float().to(device)
    receiver = torch.from_numpy(validation.x_cat[..., 0]).to(device)
    y = torch.from_numpy(validation.y_entity).to(device)
    lengths = torch.from_numpy(validation.lengths).to(device)
    valid = torch.from_numpy(validation.valid_mask).to(device)
    with torch.no_grad():
        hidden = model.event_decoder(
            amount=amount,
            gap=gap,
            receiver=receiver,
            y=y,
            lengths=lengths,
            valid_mask=valid,
        )
        parameters = model.gap_decoder(hidden, y=y)
        event_nll = model.gap_decoder.event_nll(gap, parameters, valid_mask=valid)
        epsilon = torch.finfo(parameters.u_tail.dtype).eps * 32
        left_u = torch.clamp(parameters.u_tail - epsilon, min=epsilon)
        right_u = parameters.u_tail + epsilon
        cdf_left = model.gap_decoder.positive_cdf_u(left_u, parameters)
        cdf_at = 1.0 - parameters.pi_tail
        cdf_right = model.gap_decoder.positive_cdf_u(right_u, parameters)
        far_u = parameters.u_tail + parameters.beta * 50.0
        cdf_far = model.gap_decoder.positive_cdf_u(far_u, parameters)
    position = torch.arange(valid.shape[1], device=device)[None, :]
    strata = {"first": position == 0, "later": position > 0}
    gate_summary: dict[str, Any] = {}
    scale_summary: dict[str, Any] = {}
    signed_summary: dict[str, Any] = {}
    for label in (0, 1):
        for stratum, position_mask in strata.items():
            selected = valid & (y == label)[:, None] & position_mask
            key = f"y{label}_{stratum}"
            if bool(selected.any()):
                gate_summary[key] = float(parameters.pi_tail[selected].mean().cpu())
                scale_summary[key] = float(parameters.beta[selected].mean().cpu())
                signed_summary[key] = {
                    "mean_r_beta": float(parameters.r_beta[selected].mean().cpu()),
                    "mean_beta_over_baseline": float(
                        (
                            parameters.beta[selected]
                            / model.gap_decoder.beta_base_by_y[label].to(parameters.beta)
                        ).mean().cpu()
                    ),
                    "below_baseline_count": int(
                        (
                            parameters.beta[selected]
                            < model.gap_decoder.beta_base_by_y[label].to(parameters.beta)
                        ).sum().cpu()
                    ),
                    "above_baseline_count": int(
                        (
                            parameters.beta[selected]
                            > model.gap_decoder.beta_base_by_y[label].to(parameters.beta)
                        ).sum().cpu()
                    ),
                }
    positive = gap > 0
    tail = torch.log1p(gap) > parameters.u_tail
    route_counts = {
        "zero": int((valid & ~positive).sum().cpu()),
        "central": int((valid & positive & ~tail).sum().cpu()),
        "tail": int((valid & positive & tail).sum().cpu()),
        "finite": int(torch.isfinite(event_nll[valid]).sum().cpu()),
    }
    return {
        "conditional_tail_gate_by_class_and_history_stratum": gate_summary,
        "conditional_tail_scale_by_class_and_history_stratum": scale_summary,
        "signed_scale_diagnostics": signed_summary,
        "positive_cdf_total_mass_error": float(
            torch.abs(1.0 - cdf_far[valid]).max().cpu()
        ),
        "threshold_cdf_continuity_error": float(
            torch.maximum(torch.abs(cdf_left - cdf_at), torch.abs(cdf_right - cdf_at))[valid]
            .max()
            .cpu()
        ),
        "finite_nll_count_by_route": route_counts,
        "finite_density_count": route_counts["finite"],
    }


def _full_diagnostics(
    *, model: Any, data: AuthorizedTrainingData, validation: SequenceBatch,
    synthetic: SyntheticBatch, continuous: np.ndarray, observed: Mapping[str, Any],
    hard: Mapping[str, Any], device: str, last_training: Mapping[str, Any],
) -> Mapping[str, Any]:
    tau = np.asarray(data.transform["gap_tau"], dtype=np.float64)
    real_gap = tau[validation.dt_bin]
    zero_rate: dict[str, Any] = {}
    zero_error: dict[str, Any] = {}
    quantile_diff: dict[str, Any] = {}
    central_tail: dict[str, Any] = {}
    tail_error: dict[str, Any] = {}
    frozen_mass: dict[str, Any] = {}
    mapping_difference: dict[str, Any] = {}
    quantiles = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    for label in (0, 1):
        selected = validation.valid_mask & (validation.y_entity[:, None] == label)
        real = real_gap[selected]
        generated = continuous[selected]
        real_positive = real[real > 0]
        generated_positive = generated[generated > 0]
        zero_rate[f"y{label}"] = {
            "real": float(np.mean(real == 0)),
            "synthetic": float(np.mean(generated == 0)),
        }
        zero_error[f"y{label}"] = abs(
            zero_rate[f"y{label}"]["real"] - zero_rate[f"y{label}"]["synthetic"]
        )
        quantile_diff[f"y{label}"] = {
            f"q{int(q * 100):02d}": float(
                np.quantile(generated_positive, q) - np.quantile(real_positive, q)
            )
            for q in quantiles
        }
        threshold = float(np.expm1(data.tail_state.u_tail[label]))
        real_tail = int(np.sum(real_positive > threshold))
        generated_tail = int(np.sum(generated_positive > threshold))
        central_tail[f"y{label}"] = {
            "real_central": int(len(real_positive) - real_tail),
            "real_tail": real_tail,
            "synthetic_central": int(len(generated_positive) - generated_tail),
            "synthetic_tail": generated_tail,
        }
        tail_error[f"y{label}"] = abs(
            real_tail / len(real_positive) - generated_tail / len(generated_positive)
        )
        real_bins = np.bincount(
            validation.dt_bin[selected], minlength=len(tau)
        ).astype(float)
        generated_bins = np.bincount(
            synthetic.dt_bin[selected], minlength=len(tau)
        ).astype(float)
        frozen_mass[f"y{label}"] = {
            "real": (real_bins / real_bins.sum()).tolist(),
            "synthetic": (generated_bins / generated_bins.sum()).tolist(),
        }
        mapped_ks = _ks(real, tau[synthetic.dt_bin[selected]])
        mapping_difference[f"y{label}"] = float(
            mapped_ks - observed["gap_ks"][f"y{label}"]
        )
    conditional = _conditional_h1_diagnostics(
        model=model, validation=validation, real_gap=real_gap, device=device
    )
    real_receiver = validation.x_cat[..., 0][validation.valid_mask]
    generated_receiver = synthetic.x_cat[..., 0][synthetic.valid_mask]
    head = np.asarray(data.receiver_reference["diagnostic_head_codes"], dtype=np.int64)
    tail = np.asarray(data.receiver_reference["diagnostic_tail_codes"], dtype=np.int64)
    receiver = {
        "classwise_receiver_tv": observed.get("receiver_tv", {}),
        "full_receiver_tv": observed.get("full_receiver_tv"),
        "head_receiver_tv": _tv(
            real_receiver[np.isin(real_receiver, head)],
            generated_receiver[np.isin(generated_receiver, head)],
        ),
        "tail_receiver_tv": _tv(
            real_receiver[np.isin(real_receiver, tail)],
            generated_receiver[np.isin(generated_receiver, tail)],
        ),
        "unk_rate_error": abs(
            float(np.mean(real_receiver == 1)) - float(np.mean(generated_receiver == 1))
        ),
        "head": {"real_count": int(np.isin(real_receiver, head).sum()), "synthetic_count": int(np.isin(generated_receiver, head).sum())},
        "tail": {"real_count": int(np.isin(real_receiver, tail).sum()), "synthetic_count": int(np.isin(generated_receiver, tail).sum())},
        "unk": {"real_count": int((real_receiver == 1).sum()), "synthetic_count": int((generated_receiver == 1).sum())},
        "repeat": {"preserved_metric": "short_gap_receiver_repeat"},
        "new": {"path": "flat_no_copy"},
    }
    valid_generated = continuous[synthetic.valid_mask]
    return {
        "gap": {
            "zero_rate_by_class": zero_rate,
            "zero_rate_absolute_error_by_class": zero_error,
            "overall_gap_ks_by_class": observed.get("gap_ks", {}),
            "positive_only_gap_ks_by_class": observed.get("positive_gap_ks", {}),
            "positive_quantile_differences_by_class": quantile_diff,
            "central_tail_counts_by_class": central_tail,
            "tail_mass_error_by_class": tail_error,
            **conditional,
            "sampled_min_max": {"min": float(valid_generated.min()), "max": float(valid_generated.max())},
            "lower_boundary_rate": float(np.mean(valid_generated == 0)),
            "nonfinite_sample_count": int((~np.isfinite(valid_generated)).sum()),
            "exact_upper_clip_count": 0,
            "frozen_bin_mass": frozen_mass,
            "raw_vs_mapped_ks_difference": mapping_difference,
            "training": dict(last_training),
        },
        "receiver": receiver,
        "amount": {"contract": data.amount_reference["contract"]},
        "hard_validity": dict(hard),
        "forbidden_access": {"internal_test": 0, "sparkov_fraud_test": 0},
    }


def _checkpoint_bundle(model: Any, *, plan: ExecutionPlan, data: AuthorizedTrainingData):
    from models.cof_hcmttpp_v2 import build_h1_checkpoint_bundle

    return build_h1_checkpoint_bundle(
        model=model,
        provenance={
            "source_sha256": plan.raw["model_sha256"],
            "config_sha256": plan.config_sha256,
            "train_manifest_sha256": data.binding["train_sha256"],
            "transform_state_sha256": data.binding["transform_sha256"],
            "sampling_plan_sha256": data.binding["sampling_plan_sha256"],
            "amount_contract_sha256": data.binding["amount_contract_sha256"],
            "receiver_vocabulary_sha256": data.binding["receiver_vocabulary_sha256"],
            "tail_state_sha256": data.binding["tail_state_sha256"],
            "validation_rows_used": 0,
            "internal_test_rows_used": 0,
            "fraud_test_rows_used": 0,
        },
    )


def run_authorized_job(
    *, plan: ExecutionPlan, job: ExecutionJob, authorization: Mapping[str, Any],
    data: AuthorizedTrainingData, model: Any, device: str,
    ownership_path: Path, attempt_path: Path, event_queue: Any,
) -> Mapping[str, Any]:
    import torch

    runtime_root = Path(str(plan.raw["runtime_root"]))
    if not runtime_root.is_absolute():
        runtime_root = plan.repository_root / runtime_root
    store = H1AttemptStore.attach_existing(
        runtime_root=runtime_root,
        job=job,
        ownership_path=ownership_path,
        attempt_path=attempt_path,
    )
    record = plan.raw["datasets"][job.dataset]
    store.write_json("frozen_parent_references.json", {
        "schema_version": "cof-hcmttpp-v2-h1-frozen-parents-v1",
        "c0": record["c0"],
        "c1": record["c1"],
        "gate_references": record["gate_references"],
    })
    store.write_json("gap_hurdle_state.json", {
        "schema_version": "cof-hcmttpp-v2-h1-hurdle-state-v1",
        "fit_split": "train",
        "target_rule": "exact_gap_equals_zero",
        "epsilon_relabeling": "FORBIDDEN",
        "zero_head_parameters": "checkpoint_only",
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    })
    store.write_json("positive_gap_spline_state.json", {
        "schema_version": "cof-hcmttpp-v2-h1-rqs-state-v1",
        "fit_split": "train",
        "central_bins": 16,
        "probability_knot_grid": "uniform_conditional_central_0_to_1",
        "minimum_height_fraction_per_bin": 0.00000625,
        "minimum_derivative": 0.001,
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    })
    store.write_json("positive_gap_tail_state.json", {
        "schema_version": "cof-hcmttpp-v2-h1-tail-state-artifact-v1",
        **asdict(data.tail_state),
        "support": "positive_unbounded",
        "hard_upper_clip": "FORBIDDEN",
        "redraw_nonfinite": "FORBIDDEN",
        "conditional_tail_gate": "sigmoid_logit_p_base_plus_r_tail",
        "conditional_scale": "beta_base_times_exp_r_beta",
    })
    store.write_json("amount_transform_reference.json", data.amount_reference)
    store.write_json("receiver_vocabulary_reference.json", data.receiver_reference)
    budget = _source_budget(plan)
    validate_runtime_h1_factor_isolation(model=model, plan=plan)
    started = time.monotonic()
    torch.manual_seed(job.seed)
    rng = np.random.default_rng(job.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(budget["learning_rate"]),
        weight_decay=float(budget["weight_decay"]),
    )
    n = len(data.train.lengths)
    batch_size = int(budget["batch_size"])
    if n < batch_size:
        raise H1RunnerContractError("train split is smaller than batch size")
    progress_interval = int(plan.raw["training"]["progress_interval_updates"])
    checkpoint_interval = int(budget["checkpoint_interval_updates"])
    generator = torch.Generator(device=device).manual_seed(job.seed + 1)
    actual_updates = 0
    last_diagnostics: Mapping[str, Any] = {}
    model.train()
    for step in range(1, int(budget["requested_updates"]) + 1):
        indices = rng.choice(n, size=batch_size, replace=False)
        optimizer.zero_grad(set_to_none=True)
        valid = torch.from_numpy(data.train.valid_mask[indices]).to(device)
        loss, last_diagnostics = _h1_loss(
            model,
            amount=torch.from_numpy(data.train.x_num[indices]).to(device),
            gap=torch.from_numpy(data.continuous_gap[indices]).float().to(device),
            receiver=torch.from_numpy(data.train.x_cat[indices, :, 0]).to(device),
            y=torch.from_numpy(data.train.y_entity[indices]).to(device),
            lengths=torch.from_numpy(data.train.lengths[indices]).to(device),
            valid=valid,
            generator=generator,
        )
        if not bool(torch.isfinite(loss)):
            raise H1RunnerContractError("non-finite H1 training loss")
        loss.backward()
        optimizer.step()
        actual_updates = step
        if step == 1 or step % progress_interval == 0:
            progress = {"step": step, "loss": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - started}
            store.append_progress(progress)
            event_queue.put({"kind": "progress", **progress})
        if step % checkpoint_interval == 0:
            store.write_bytes(
                f"checkpoints/step_{step:06d}.pt",
                _torch_bytes(_checkpoint_bundle(model, plan=plan, data=data)),
            )
    final = store.write_bytes(
        "checkpoints/final.pt", _torch_bytes(_checkpoint_bundle(model, plan=plan, data=data))
    )
    store.write_bytes("checkpoints/latest", b"final.pt\n")
    store.write_json(
        "checkpoint_provenance.json",
        build_checkpoint_provenance(
            plan=plan,
            job=job,
            authorization_sha256=str(authorization["authorization_sha256"]),
            train_state_binding=data.binding,
            actual_updates=actual_updates,
            checkpoint_sha256=sha256_file(final),
        ),
    )
    event_queue.put({"kind": "progress", "phase": "final_checkpoint", "step": actual_updates})

    validation_path = resolve_dataset_access(
        plan=plan,
        split="validation",
        purpose="fixed_plan_sampling_and_evaluation",
        final_checkpoint_complete=True,
    )
    if validation_path.relative_to(plan.repository_root).as_posix() != data.validation_relative_path:
        raise H1RunnerContractError("validation binding changed after training")
    validation = _load_sequence_batch(validation_path)
    sampling_plan = _fixed_validation_plan(plan=plan, job=job, validation=validation)
    store.write_json("conditioning_plan.json", {
        "fit_split": "train",
        "validation_use": "post_final_checkpoint_only",
        "sampling_plan_sha256": data.binding["sampling_plan_sha256"],
        "plan_hash": sampling_plan.plan_hash,
        "validation_rows_used_for_fit": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    })
    synthetic, continuous = _sample_validation(
        model=model,
        plan=sampling_plan,
        transform=data.transform,
        device=device,
        seed=job.seed + 100_000,
        batch_size=int(plan.raw["execution"].get("validation_sample_batch_size", 128)),
        event_queue=event_queue,
    )
    sample = store.write_bytes("validation_sample.npz", _npz_bytes({
        "x_num": synthetic.x_num,
        "dt_bin": synthetic.dt_bin,
        "x_cat": synthetic.x_cat,
        "valid_mask": synthetic.valid_mask,
        "y_entity": synthetic.y_entity,
        "lengths": synthetic.lengths,
        "continuous_gap": continuous,
    }))
    tau = np.asarray(data.transform["gap_tau"], dtype=np.float64)
    threshold_path = plan.repository_root / str(plan.raw["datasets"][job.dataset]["threshold_path"])
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    try:
        hard = validate_external_hard_contract(
            real_validation=validation,
            synthetic=synthetic,
            gap_cardinality=len(tau),
            receiver_cardinality=data.receiver_classes,
        )
        external = compute_external_validation_metrics(
            real=validation,
            synthetic=synthetic,
            gap_tau=tau,
            short_gap_threshold=float(threshold["short_gap_threshold"]),
        )
        observed = _gate_observations(
            validation=validation,
            synthetic=synthetic,
            continuous_gap=continuous,
            tau=tau,
            external=external,
        )
        gate = evaluate_h1_gate(
            observed, plan.raw["datasets"][job.dataset]["gate_references"]
        )
        evaluation_status = "VALID"
    except (ExternalMetricError, H1RunnerContractError, ValueError) as error:
        hard = {"status": "INVALID", "message": str(error)}
        external = {"fidelity": {}, "coherence": {}}
        observed = {"hard_validity": False}
        gate = evaluate_h1_gate(
            observed, plan.raw["datasets"][job.dataset]["gate_references"]
        )
        evaluation_status = "INVALID"
    if evaluation_status == "VALID":
        diagnostics = _full_diagnostics(
            model=model,
            data=data,
            validation=validation,
            synthetic=synthetic,
            continuous=continuous,
            observed=observed,
            hard=hard,
            device=device,
            last_training=last_diagnostics,
        )
    else:
        unavailable = {"status": "INVALID", "reason": hard.get("message", "hard validity")}
        diagnostics = {
            "gap": {
                key: dict(unavailable)
                for key in H1AttemptStore.REQUIRED_GAP_DIAGNOSTICS
            },
            "receiver": {
                key: dict(unavailable)
                for key in H1AttemptStore.REQUIRED_RECEIVER_DIAGNOSTICS
            },
            "amount": dict(unavailable),
            "hard_validity": dict(hard),
            "forbidden_access": {"internal_test": 0, "sparkov_fraud_test": 0},
        }
    store.write_json("diagnostics.json", diagnostics)
    store.write_json("metrics.json", {
        "external_metrics": external,
        "gate_values": observed,
        "validation_sample_sha256": sha256_file(sample),
    })
    store.write_json("evaluation.json", {
        "status": evaluation_status,
        "hard_contract": hard,
        "internal_test_accessed": False,
        "sparkov_fraud_test_accessed": False,
    })
    store.write_json("gate_decision.json", gate)
    store.write_json("runtime.json", {
        "requested_updates": int(budget["requested_updates"]),
        "actual_updates": actual_updates,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "device": device,
    })
    return {"terminal_status": "COMPLETE" if evaluation_status == "VALID" else "INVALID"}


def build_default_execution_dependencies() -> ExecutionDependencies:
    return ExecutionDependencies(
        load_train_body=load_authorized_train_body,
        build_model=build_authorized_model,
        select_device=select_explicit_device,
        run_job=run_authorized_job,
    )
