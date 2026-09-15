"""Authorized-only CoF-SeqGen v3 candidate backend.

Importing and constructing this module is side-effect free.  Device access,
optimizer updates, checkpoint writes, and sampling occur only through the
explicit methods invoked by the separately authorized execution runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.cof_seqgen_v3_contract import (
    V3ContractError,
    build_train_joint_support_state,
    canonical_sha256,
)
from generators.sampling_plan import SamplingPlan
from models.cof_seqgen_v3 import (
    CoFSeqDenoiserV3,
    CoFSeqGenV3,
    DirectJointDiscretePath,
    FactorizedJointDiscretePath,
    JointStateCodec,
)


@dataclass
class V3FittedCandidate:
    model: CoFSeqGenV3
    optimizer: torch.optim.Optimizer
    actual_updates: int
    elapsed_seconds: float
    loss_history: list[float]
    joint_support_state: Mapping[str, Any]
    amount_fit_state: Mapping[str, Any]
    numpy_rng_state: Mapping[str, Any]
    torch_rng_state: Tensor


def _row_labels(batch: SequenceBatch | SyntheticBatch) -> np.ndarray:
    return np.broadcast_to(
        batch.y_entity[:, None],
        batch.valid_mask.shape,
    )[batch.valid_mask]


def fit_train_only_amount_state(
    *,
    train: SequenceBatch,
    calibration_sample: SyntheticBatch,
    train_file_sha256: str,
    train_content_sha256: str,
    sampling_plan_sha256: str,
    quantile_grid_size: int = 257,
) -> Mapping[str, Any]:
    """Fit the frozen centered empirical-residual path using train only."""

    if quantile_grid_size != 257:
        raise V3ContractError("v3 amount quantile grid changed")
    target = train.x_num[..., 0][train.valid_mask].astype(np.float64)
    source = calibration_sample.x_num[..., 0][
        calibration_sample.valid_mask
    ].astype(np.float64)
    target_labels = _row_labels(train)
    source_labels = _row_labels(calibration_sample)
    levels = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    maps: dict[str, Mapping[str, Any]] = {}
    for label in (0, 1):
        source_values = source[source_labels == label]
        target_values = target[target_labels == label]
        if (
            not len(source_values)
            or not len(target_values)
            or not np.isfinite(source_values).all()
            or not np.isfinite(target_values).all()
        ):
            raise V3ContractError(
                "train-only amount fit requires finite rows per class"
            )
        source_center = float(source_values.mean())
        target_center = float(target_values.mean())
        maps[str(label)] = {
            "levels": levels.tolist(),
            "source_quantiles": np.quantile(
                source_values - source_center,
                levels,
            ).tolist(),
            "target_quantiles": np.quantile(
                target_values - target_center,
                levels,
            ).tolist(),
            "source_center": source_center,
            "target_center": target_center,
        }
    parameters = {
        "class_quantile_maps": maps,
        "parameters_sha256": canonical_sha256(maps),
    }
    payload = {
        "schema_version": "cof-seqgen-v3-amount-fit-state-v1",
        "algorithm": "train_fitted_centered_empirical_residual",
        "quantile_grid_size": 257,
        "fit_split": "train",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "train_file_sha256": train_file_sha256,
        "train_content_sha256": train_content_sha256,
        "sampling_plan_sha256": sampling_plan_sha256,
        **parameters,
    }
    return {**payload, "state_sha256": canonical_sha256(payload)}


def apply_amount_state(
    sample: SyntheticBatch,
    state: Mapping[str, Any],
) -> SyntheticBatch:
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    if (
        state.get("fit_split") != "train"
        or state.get("validation_rows_used") != 0
        or state.get("test_rows_used") != 0
        or state.get("state_sha256") != canonical_sha256(payload)
    ):
        raise V3ContractError("v3 amount fit-state identity mismatch")
    maps = state.get("class_quantile_maps")
    if not isinstance(maps, Mapping):
        raise V3ContractError("v3 amount fit-state has no quantile maps")
    amount = sample.x_num.copy()
    for label in (0, 1):
        mask = sample.valid_mask & (sample.y_entity == label)[:, None]
        mapping = maps[str(label)]
        source = np.asarray(mapping["source_quantiles"], dtype=np.float64)
        target = np.asarray(mapping["target_quantiles"], dtype=np.float64)
        values = amount[..., 0][mask].astype(np.float64)
        values = values - float(mapping["source_center"])
        amount[..., 0][mask] = (
            np.interp(values, source, target)
            + float(mapping["target_center"])
        ).astype(np.float32)
    amount[~sample.valid_mask] = 0
    return SyntheticBatch(
        x_num=amount,
        dt_bin=sample.dt_bin.copy(),
        x_cat=sample.x_cat.copy(),
        valid_mask=sample.valid_mask.copy(),
        y_entity=sample.y_entity.copy(),
        lengths=sample.lengths.copy(),
    )


def build_v3_candidate_model(
    *,
    architecture: str,
    joint_support_mask: Tensor,
    joint_support_sha256: str,
    max_length: int,
    d_num: int = 1,
    gap_bins: int = 16,
    receiver_classes: int = 64,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    cfg_dropout: float = 0.15,
) -> CoFSeqGenV3:
    codec = JointStateCodec(
        gap_bins=gap_bins,
        receiver_classes=receiver_classes,
    )
    denoiser = CoFSeqDenoiserV3(
        d_num=d_num,
        codec=codec,
        candidate=architecture,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_length=max_length,
    )
    return CoFSeqGenV3(
        denoiser=denoiser,
        codec=codec,
        joint_support_mask=joint_support_mask,
        joint_support_sha256=joint_support_sha256,
        coherence_lambda=0.0,
        cfg_dropout=cfg_dropout,
    )


def _choice(
    logits: Tensor,
    *,
    temperature: float,
    generator: torch.Generator,
) -> Tensor:
    if temperature != 1.0:
        raise V3ContractError("v3 discrete temperature changed")
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        1,
        generator=generator,
    ).reshape(logits.shape[:-1])


@torch.no_grad()
def sample_v3_candidate(
    *,
    model: CoFSeqGenV3,
    plan: SamplingPlan,
    device: str,
    seed: int,
    diffusion_steps: int,
    guidance_scale: float,
    discrete_temperature: float,
    chunk_size: int = 256,
) -> SyntheticBatch:
    """Generate one sample with paired direct or factorized joint decoding."""

    if (
        diffusion_steps != 50
        or guidance_scale != 2.0
        or discrete_temperature != 1.0
        or chunk_size < 1
    ):
        raise V3ContractError("v3 sampling schedule changed")
    model.eval()
    count, length = plan.valid_mask.shape
    amount = np.zeros((count, length, 1), dtype=np.float32)
    gap = np.zeros((count, length), dtype=np.int64)
    receiver = np.zeros((count, length, 1), dtype=np.int64)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    schedule = torch.linspace(1.0, 0.0, diffusion_steps + 1)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        valid = torch.from_numpy(plan.valid_mask[start:stop]).to(device)
        labels = torch.from_numpy(plan.y_entity[start:stop]).to(device)
        joint = model.initial_joint_state(valid)
        numerical = torch.randn(
            (stop - start, length, 1),
            device=device,
            generator=generator,
        ) * valid[..., None]
        gap_sample = torch.zeros_like(joint)
        receiver_sample = torch.zeros_like(joint)
        for index in range(diffusion_steps):
            current = float(schedule[index])
            following = float(schedule[index + 1])
            time_tensor = torch.full(
                (stop - start,),
                current,
                dtype=numerical.dtype,
                device=device,
            )
            hidden_conditional = model.denoiser.encode_hidden(
                numerical,
                joint,
                time_tensor,
                valid_mask=valid,
                y_cond=labels,
            )
            hidden_unconditional = model.denoiser.encode_hidden(
                numerical,
                joint,
                time_tensor,
                valid_mask=valid,
                y_cond=None,
            )
            amount_conditional = model.denoiser.amount_head(
                hidden_conditional
            )
            amount_unconditional = model.denoiser.amount_head(
                hidden_unconditional
            )
            amount_hat = amount_unconditional + guidance_scale * (
                amount_conditional - amount_unconditional
            )
            path = model.denoiser.discrete_path
            if isinstance(path, DirectJointDiscretePath):
                conditional_logits = path.logits(hidden_conditional)
                unconditional_logits = path.logits(hidden_unconditional)
                logits = model._mask_direct_logits(
                    unconditional_logits
                    + guidance_scale
                    * (conditional_logits - unconditional_logits)
                )
                joint_sample = _choice(
                    logits,
                    temperature=discrete_temperature,
                    generator=generator,
                )
                gap_sample, receiver_sample = model.codec.decode(
                    joint_sample
                )
            elif isinstance(path, FactorizedJointDiscretePath):
                support = model._support_matrix(hidden_conditional.device)
                gap_logits = (
                    path.gap_logits(hidden_unconditional)
                    + guidance_scale
                    * (
                        path.gap_logits(hidden_conditional)
                        - path.gap_logits(hidden_unconditional)
                    )
                ).masked_fill(~support.any(dim=-1), float("-inf"))
                gap_sample = _choice(
                    gap_logits,
                    temperature=discrete_temperature,
                    generator=generator,
                )
                receiver_logits = (
                    path.receiver_logits(
                        hidden_unconditional,
                        gap_sample,
                    )
                    + guidance_scale
                    * (
                        path.receiver_logits(
                            hidden_conditional,
                            gap_sample,
                        )
                        - path.receiver_logits(
                            hidden_unconditional,
                            gap_sample,
                        )
                    )
                ).masked_fill(~support[gap_sample], float("-inf"))
                receiver_sample = _choice(
                    receiver_logits,
                    temperature=discrete_temperature,
                    generator=generator,
                )
                joint_sample = model.codec.encode(
                    gap_sample,
                    receiver_sample,
                )
            else:
                raise V3ContractError("legacy discrete path is forbidden")
            joint = torch.where(valid, joint_sample, torch.zeros_like(joint))
            gap_sample = torch.where(
                valid,
                gap_sample,
                torch.zeros_like(gap_sample),
            )
            receiver_sample = torch.where(
                valid,
                receiver_sample,
                torch.zeros_like(receiver_sample),
            )
            alpha_current, sigma_current = model._alpha_sigma(current)
            alpha_following, sigma_following = model._alpha_sigma(following)
            epsilon_hat = (
                numerical - alpha_current * amount_hat
            ) / (sigma_current + 1e-8)
            numerical = (
                alpha_following * amount_hat
                + sigma_following * epsilon_hat
            ) * valid[..., None]
        amount[start:stop] = numerical.cpu().numpy().astype(np.float32)
        gap[start:stop] = gap_sample.cpu().numpy().astype(np.int64)
        receiver[start:stop, :, 0] = (
            receiver_sample.cpu().numpy().astype(np.int64)
        )
    return SyntheticBatch(
        x_num=amount,
        dt_bin=gap,
        x_cat=receiver,
        valid_mask=plan.valid_mask.copy(),
        y_entity=plan.y_entity.copy(),
        lengths=plan.lengths.copy(),
    )


class CoFSeqGenV3CandidateAdapter:
    """Fixed candidate backend used only by an authorized child process."""

    def __init__(
        self,
        *,
        architecture: str,
        config: Mapping[str, Any],
        sampling_plan: SamplingPlan,
        train_file_sha256: str,
        train_content_sha256: str,
        device: str,
        checkpoint_callback: Callable[
            [int, Mapping[str, Any]], None
        ]
        | None = None,
    ) -> None:
        if architecture not in {"direct_joint", "factorized_joint"}:
            raise V3ContractError("unknown v3 candidate architecture")
        self.architecture = architecture
        self.config = dict(config)
        self.sampling_plan = sampling_plan
        self.train_file_sha256 = train_file_sha256
        self.train_content_sha256 = train_content_sha256
        self.device = device
        self.checkpoint_callback = checkpoint_callback

    def fit_train_only(
        self,
        batch: SequenceBatch,
        *,
        split: str,
        operation: Any,
    ) -> V3FittedCandidate:
        if split != "train" or not isinstance(batch, SequenceBatch):
            raise V3ContractError("v3 adapter fit is train-only")
        if operation.architecture != self.architecture:
            raise V3ContractError("v3 adapter operation mismatch")
        if (
            operation.requested_updates != 20_000
            or float(operation.max_wall_seconds) != 7_200
        ):
            raise V3ContractError("v3 frozen training budget changed")
        torch.manual_seed(operation.seed)
        rng = np.random.default_rng(operation.seed)
        codec = JointStateCodec(16, 64)
        support = build_train_joint_support_state(
            codec=codec,
            gap=torch.from_numpy(batch.dt_bin),
            receiver=torch.from_numpy(batch.x_cat[..., 0]),
            valid_mask=torch.from_numpy(batch.valid_mask),
            fit_split="train",
            provenance={
                "train_file_sha256": self.train_file_sha256,
                "train_content_sha256": self.train_content_sha256,
                "sampling_plan_sha256": self.sampling_plan.plan_hash,
            },
        )
        model = build_v3_candidate_model(
            architecture=self.architecture,
            joint_support_mask=torch.tensor(
                support["support_mask"],
                dtype=torch.bool,
            ),
            joint_support_sha256=str(support["state_sha256"]),
            max_length=batch.valid_mask.shape[1],
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        batch_size = min(int(self.config["batch_size"]), len(batch.lengths))
        interval = int(self.config["checkpoint_interval_updates"])
        if batch_size != 256 or interval != 1_000:
            raise V3ContractError("v3 batch/checkpoint schedule changed")
        history: list[float] = []
        started = time.monotonic()
        model.train()
        for step in range(1, operation.requested_updates + 1):
            indices = rng.choice(
                len(batch.lengths),
                size=batch_size,
                replace=False,
            )
            optimizer.zero_grad()
            loss, _ = model.compute_loss(
                x_num=torch.from_numpy(batch.x_num[indices]).to(self.device),
                gap=torch.from_numpy(batch.dt_bin[indices]).to(self.device),
                receiver=torch.from_numpy(
                    batch.x_cat[indices, :, 0]
                ).to(self.device),
                y=torch.from_numpy(
                    batch.y_position()[indices]
                ).to(self.device),
                valid_mask=torch.from_numpy(
                    batch.valid_mask[indices]
                ).to(self.device),
                time_fraction=step / (operation.requested_updates + 1),
            )
            if not bool(torch.isfinite(loss)):
                raise V3ContractError("non-finite v3 training loss")
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu()))
            if self.checkpoint_callback and (
                step % interval == 0
                or step == operation.requested_updates
            ):
                self.checkpoint_callback(
                    step,
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "actual_updates": step,
                        "requested_updates": operation.requested_updates,
                        "loss": history[-1],
                        "elapsed_seconds": time.monotonic() - started,
                        "joint_support_state_sha256": support[
                            "state_sha256"
                        ],
                        "amount_contract": model.AMOUNT_CONTRACT,
                    },
                )
        base_sample = sample_v3_candidate(
            model=model,
            plan=self.sampling_plan,
            device=self.device,
            seed=operation.seed + 100_000,
            diffusion_steps=int(self.config["diffusion_steps"]),
            guidance_scale=float(self.config["guidance_scale"]),
            discrete_temperature=float(
                self.config["discrete_temperature"]
            ),
        )
        amount_state = fit_train_only_amount_state(
            train=batch,
            calibration_sample=base_sample,
            train_file_sha256=self.train_file_sha256,
            train_content_sha256=self.train_content_sha256,
            sampling_plan_sha256=self.sampling_plan.plan_hash,
        )
        return V3FittedCandidate(
            model=model,
            optimizer=optimizer,
            actual_updates=operation.requested_updates,
            elapsed_seconds=time.monotonic() - started,
            loss_history=history,
            joint_support_state=support,
            amount_fit_state=amount_state,
            numpy_rng_state=rng.bit_generator.state,
            torch_rng_state=torch.get_rng_state(),
        )

    def sample_validation(
        self,
        plan: SamplingPlan,
        *,
        checkpoint: V3FittedCandidate,
        operation: Any,
    ) -> SyntheticBatch:
        if (
            plan.plan_hash != self.sampling_plan.plan_hash
            or operation.architecture != self.architecture
        ):
            raise V3ContractError("v3 validation SamplingPlan mismatch")
        base = sample_v3_candidate(
            model=checkpoint.model,
            plan=plan,
            device=self.device,
            seed=operation.seed + 200_000,
            diffusion_steps=int(self.config["diffusion_steps"]),
            guidance_scale=float(self.config["guidance_scale"]),
            discrete_temperature=float(
                self.config["discrete_temperature"]
            ),
        )
        return apply_amount_state(base, checkpoint.amount_fit_state)
