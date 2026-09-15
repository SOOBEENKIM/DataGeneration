import pytest

from scripts.run_benchmark_v2 import validate_smoke_scope


def valid(**overrides):
    values = {
        "mode": "smoke",
        "generators": ["cof"],
        "seeds": [1],
        "kappas": [1.0],
        "n_train": 512,
        "steps": 100,
    }
    values.update(overrides)
    return values


def test_common_runner_accepts_only_bounded_single_seed_smoke():
    validate_smoke_scope(**valid())


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "full"},
        {"generators": ["cof", "conditional_ctgan"]},
        {"seeds": [1, 2, 3, 4, 5]},
        {"kappas": [0.0, 1.0]},
        {"n_train": 513},
        {"steps": 99},
        {"steps": 501},
    ],
)
def test_common_runner_rejects_full_experiment_or_sweep(overrides):
    with pytest.raises(ValueError):
        validate_smoke_scope(**valid(**overrides))
