from pathlib import Path

import yaml

from scripts.run_cof_capacity_preflight_v2_5 import (
    DEFAULT_REPORT,
    FORBIDDEN_DURABLE_OUTPUTS,
    _report_text,
    projected_seconds,
)


def test_capacity_projection_uses_frozen_formula():
    value = projected_seconds(
        p95_update_seconds=0.1,
        p95_sampling_seconds_per_entity=0.01,
        requested_updates=20_000,
        entity_count=7_989,
        fixed_overhead_seconds=600,
        safety_multiplier=1.15,
    )
    assert value == (0.1 * 20_000 + 0.01 * 7_989 + 600) * 1.15


def test_timing_preflight_contract_forbids_quality_and_sample_artifacts():
    raw = yaml.safe_load(
        Path("configs/benchmark_v2/full_v2_5.yaml").read_text()
    )
    preflight = raw["capacity_preflight"]
    assert preflight["max_updates"] == 500
    assert preflight["warmup_updates_excluded"] == 20
    assert preflight["measured_updates"] == 480
    assert preflight["sampling_chunk_size"] == 256
    assert preflight["fixed_overhead_seconds"] == 600
    assert preflight["durable_checkpoint_forbidden"] is True
    assert preflight["durable_sample_forbidden"] is True
    assert preflight["quality_metrics_forbidden"] is True
    assert DEFAULT_REPORT == Path(
        "artifacts/benchmark_v2_5/capacity_preflight/"
        "capacity_preflight_report.md"
    )
    assert {"checkpoint", "sample", "quality_metric"} <= (
        FORBIDDEN_DURABLE_OUTPUTS
    )


def test_runtime_report_states_no_performance_artifacts():
    result = {
        "status": "UNAVAILABLE",
        "source_commit": "a" * 40,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
        "gpu": {},
        "cuda_version": None,
        "pytorch_version": "test",
        "selection_reason": "test",
        "reason": "no idle GPU",
    }
    text = _report_text(result)
    assert "No timing probe or full experiment was run." in text
    assert "performance or sample-quality result" in text
