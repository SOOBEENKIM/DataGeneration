import copy
from pathlib import Path

import pytest
import yaml

from eval.cof_ccmtpp_v1_contract import (
    CCMTPPContractError,
    DIAGNOSTIC_FIELDS,
    evaluate_candidate_stop,
    load_ccmtpp_definition,
    validate_ccmtpp_definition,
    validate_ccmtpp_io_path,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/benchmark_v2/cof_ccmtpp_v1_source_only.yaml"


def test_finite_candidate_family_and_deferred_c5_are_preregistered():
    definition = load_ccmtpp_definition(CONFIG)
    assert [candidate.candidate_id for candidate in definition.candidates] == [
        "C0", "C1", "C2", "C3", "C4"
    ]
    assert definition.candidates[0].implemented is False
    assert all(candidate.implemented for candidate in definition.candidates[1:])
    assert all(candidate.execute is False for candidate in definition.candidates)
    assert definition.raw["deferred_C5"] == {
        "status": "LOCKED_UNIMPLEMENTED",
        "implementation_allowed_only_after": "C4_PASS",
        "selected_loss": None,
        "mutually_exclusive_options": [
            "class_conditional_gap_quantile_x_receiver_frequency_bin_pair_loss",
            "class_conditional_short_gap_x_repeat_coherence_loss",
        ],
        "simultaneous_losses": "FORBIDDEN",
    }
    assert definition.raw["common_contract"]["structure_loss"] is None
    assert definition.raw["safety"]["entity_id_input"] == "FORBIDDEN"


def test_contract_rejects_gap_bins_entity_ids_test_fit_and_candidate_expansion():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutations = (
        lambda value: value["common_contract"].update(gap_representation="bins"),
        lambda value: value["safety"].update(entity_id_input="allowed"),
        lambda value: value["safety"].update(internal_test_fit_rows=1),
        lambda value: value["candidates"].append({"candidate_id": "C6"}),
        lambda value: value["deferred_C5"].update(
            selected_loss="class_conditional_short_gap_x_repeat_coherence_loss"
        ),
        lambda value: value["common_contract"].update(structure_loss="coherence"),
        lambda value: value["stop_criteria"]["C4"].update(
            y0_composite_relative_tolerance=0.10
        ),
    )
    for mutate in mutations:
        changed = copy.deepcopy(raw)
        mutate(changed)
        with pytest.raises(CCMTPPContractError):
            validate_ccmtpp_definition(changed)


def test_stop_criteria_are_sequential_and_fail_closed():
    assert evaluate_candidate_stop(
        "C1",
        parent={"gap_ks": {"0": 0.2, "1": 0.3}, "short_gap_repeat_error": 0.1},
        current={"gap_ks": {"0": 0.19, "1": 0.3}, "short_gap_repeat_error": 0.1},
    )["passed"] is True
    assert evaluate_candidate_stop(
        "C2",
        parent={"short_gap_repeat_error": 0.1, "receiver_tv": 0.2},
        current={"short_gap_repeat_error": 0.09, "receiver_tv": 0.2},
    )["passed"] is True
    assert evaluate_candidate_stop(
        "C3",
        parent={"receiver_tv": 0.2, "head_tv": 0.1, "tail_tv": 0.3, "unk_error": 0.01},
        current={"receiver_tv": 0.19, "head_tv": 0.1, "tail_tv": 0.29, "unk_error": 0.01},
    )["passed"] is True
    assert evaluate_candidate_stop(
        "C4",
        parent={"y1_composite": 0.2, "y0_composite": 0.1},
        current={"y1_composite": 0.19, "y0_composite": 0.105},
    )["passed"] is True
    assert evaluate_candidate_stop(
        "C2",
        parent={"short_gap_repeat_error": 0.1, "receiver_tv": 0.2},
        current={"short_gap_repeat_error": 0.1, "receiver_tv": 0.19},
    )["passed"] is False
    with pytest.raises(CCMTPPContractError, match="sequential"):
        evaluate_candidate_stop("C3", parent=None, current={})


def test_diagnostic_schema_and_io_boundary_are_fixed(tmp_path):
    assert DIAGNOSTIC_FIELDS["receiver"] == (
        "overall_nll", "head_nll", "tail_nll", "unk_nll",
        "repeat_nll", "new_nll", "head_count", "tail_count",
        "unk_count", "repeat_count", "new_count",
    )
    repository = tmp_path / "repo"
    train = repository / "data/external/frozen/train.npz"
    validation = train.with_name("validation.npz")
    internal = train.with_name("internal_test.npz")
    fraud = repository / "fraudTest.csv"
    for path in (train, validation, internal, fraud):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    assert validate_ccmtpp_io_path(
        repository_root=repository,
        path=train,
        purpose="fit",
        access="read",
        source_only=True,
    ) == train.resolve()
    for forbidden in (validation, internal, fraud):
        with pytest.raises(CCMTPPContractError):
            validate_ccmtpp_io_path(
                repository_root=repository,
                path=forbidden,
                purpose="fit",
                access="read",
                source_only=True,
            )
    with pytest.raises(CCMTPPContractError, match="source-only"):
        validate_ccmtpp_io_path(
            repository_root=repository,
            path=repository / "artifacts/cof_ccmtpp_v1/attempt_001",
            purpose="artifact",
            access="write",
            source_only=True,
        )
