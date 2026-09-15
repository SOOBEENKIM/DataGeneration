from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.behavior_summaries_v2 import (
    compute_behavior_summaries_v2,
    fit_short_gap_threshold,
)
from eval.coherence_v2 import (
    CoherenceReference,
    coherence_bin_diagnostics,
    fit_coherence_reference,
)
from eval.joint_association_v2 import association_statistics
from generators.contracts_v2_5 import validate_synthetic_contract
from generators.sampling_plan import SamplingPlan
from .model_guards_v2_5 import (
    RowGuardThresholds,
    evaluate_row_guards,
)


@dataclass(frozen=True)
class FullEvaluationReference:
    short_gap_threshold: float
    coherence_references: Mapping[int, CoherenceReference]
    real_test_delta_joint: float
    real_test_summaries: Mapping[str, np.ndarray]
    row_guard_thresholds: RowGuardThresholds
    v2_4_reference_contract_verified: bool

    @property
    def coherence_reference(self) -> CoherenceReference:
        """Backward-compatible accessor for the confirmatory 8-bin view."""
        return self.coherence_references[8]


def fit_full_evaluation_reference(
    train: SequenceBatch,
    real_test: SequenceBatch,
    *,
    tau: np.ndarray,
    window_width: float,
    minimum_bin_count: int,
    row_guard_thresholds: RowGuardThresholds,
    v2_4_reference_contract_verified: bool,
) -> FullEvaluationReference:
    if not v2_4_reference_contract_verified:
        raise ValueError("preserved v2.4 C0/C1/support gate is not verified")
    cutoff = fit_short_gap_threshold(train, tau=tau)
    train_summaries = compute_behavior_summaries_v2(
        train,
        tau=tau,
        short_gap_threshold=cutoff,
        window_width=window_width,
    )
    real_summaries = compute_behavior_summaries_v2(
        real_test,
        tau=tau,
        short_gap_threshold=cutoff,
        window_width=window_width,
    )
    coherence_references = {
        n_bins: fit_coherence_reference(
            train_summaries,
            train.y_entity,
            n_bins=n_bins,
            min_bin_count=minimum_bin_count,
        )
        for n_bins in (4, 8)
    }
    real_delta = float(
        association_statistics(
            real_summaries["joint_alignment"],
            real_test.y_entity,
        )["delta_joint"]
    )
    return FullEvaluationReference(
        cutoff,
        coherence_references,
        real_delta,
        real_summaries,
        row_guard_thresholds,
        True,
    )


def evaluate_full_seed(
    synthetic: SyntheticBatch,
    *,
    train: SequenceBatch,
    real_test: SequenceBatch,
    plan: SamplingPlan,
    tau: np.ndarray,
    window_width: float,
    reference: FullEvaluationReference,
) -> Mapping[str, Any]:
    try:
        contract = validate_synthetic_contract(
            synthetic,
            plan=plan,
            train=train,
        )
    except (TypeError, ValueError) as error:
        return {
            "status": "INVALID",
            "association_recovery_error": None,
            "hard_guards": {
                "canonical_mask_and_zero_padding": "FAIL",
                "train_discrete_support": "FAIL",
                "c0_c1_reference_contract": (
                    "PASS"
                    if reference.v2_4_reference_contract_verified
                    else "FAIL"
                ),
                "row_marginal_guards": "FAIL",
            },
            "diagnostics": {
                "support_diagnostics": {},
                "confirmatory_support": {
                    "status": "NOT_COMPUTED",
                    "all_bins_valid": False,
                },
            },
            "invalid_reason": str(error),
        }
    summaries = compute_behavior_summaries_v2(
        synthetic,
        tau=tau,
        short_gap_threshold=reference.short_gap_threshold,
        window_width=window_width,
    )
    association = association_statistics(
        summaries["joint_alignment"],
        synthetic.y_entity,
        real_delta=reference.real_test_delta_joint,
    )
    support_diagnostics = {}
    for n_bins, coherence_reference in sorted(
        reference.coherence_references.items()
    ):
        edges = coherence_reference.thresholds_by_channel[
            "joint_alignment"
        ]
        support_diagnostics[f"{n_bins}_bin"] = coherence_bin_diagnostics(
            reference.real_test_summaries["joint_alignment"],
            real_test.y_entity,
            summaries["joint_alignment"],
            synthetic.y_entity,
            edges,
            coherence_reference.min_bin_count,
        )
    confirmatory_support = support_diagnostics["8_bin"]
    row_guards = evaluate_row_guards(
        real_test,
        synthetic,
        tau=tau,
        receiver_categories=int(
            train.x_cat[..., 0][train.valid_mask].max()
        )
        + 1,
        thresholds=reference.row_guard_thresholds,
    )
    hard_guards = {
        "canonical_mask_and_zero_padding": contract["mask_contract"],
        "train_discrete_support": contract["support_contract"],
        "c0_c1_reference_contract": (
            "PASS"
            if reference.v2_4_reference_contract_verified
            else "FAIL"
        ),
        "row_marginal_guards": row_guards["status"],
    }
    valid = all(value == "PASS" for value in hard_guards.values())
    return {
        "status": "VALID" if valid else "INVALID",
        "association_recovery_error": (
            association["association_recovery_error"] if valid else None
        ),
        "delta_joint_real_test": reference.real_test_delta_joint,
        "delta_joint_synthetic": association["delta_joint"],
        "association": association,
        "hard_guards": hard_guards,
        "diagnostics": {
            "support_diagnostics": support_diagnostics,
            "confirmatory_support": confirmatory_support,
        },
        "row_marginal_guard_report": row_guards,
        "invalid_reason": (
            None
            if valid
            else "one or more preregistered hard guards failed"
        ),
    }
