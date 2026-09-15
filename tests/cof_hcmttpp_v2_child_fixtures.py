"""Spawn-safe, CPU-only child fixtures for the H1 runner watchdog."""

from pathlib import Path
import time

from experiments.cof_hcmttpp_v2_execution_runner import (
    ExecutionJob,
    H1AttemptStore,
)


def startup_hang(payload, event_queue):
    del payload, event_queue
    time.sleep(5)
    return {"terminal_status": "COMPLETE"}


def progress_hang(payload, event_queue):
    del payload
    event_queue.put({"kind": "setup_complete"})
    time.sleep(5)
    return {"terminal_status": "COMPLETE"}


def child_exception(payload, event_queue):
    del payload
    event_queue.put({"kind": "setup_complete"})
    raise RuntimeError("synthetic child failure")


def _write_complete_fixture(store, event_queue):
    store.append_progress({"step": 1, "status": "TRAINING"})
    event_queue.put({"kind": "progress", "step": 1})
    store.write_json("frozen_parent_references.json", {"status": "PASS"})
    store.write_json("gap_hurdle_state.json", {"fit_split": "train"})
    store.write_json("positive_gap_spline_state.json", {"bins": 16})
    store.write_json("positive_gap_tail_state.json", {"fit_split": "train"})
    store.write_json("amount_transform_reference.json", {"rows": "train"})
    store.write_json("receiver_vocabulary_reference.json", {"path": "flat"})
    store.write_json("conditioning_plan.json", {"source": "fixed"})
    store.write_bytes("checkpoints/final.pt", b"checkpoint")
    store.write_bytes("checkpoints/latest", b"final.pt\n")
    store.write_json("checkpoint_provenance.json", {"status": "PASS"})
    store.write_bytes("validation_sample.npz", b"sample")
    store.write_json(
        "diagnostics.json",
        {
            "gap": {
                key: {}
                for key in (
                    "zero_rate_by_class", "zero_rate_absolute_error_by_class",
                    "overall_gap_ks_by_class", "positive_only_gap_ks_by_class",
                    "positive_quantile_differences_by_class", "central_tail_counts_by_class",
                    "tail_mass_error_by_class", "conditional_tail_gate_by_class_and_history_stratum",
                    "conditional_tail_scale_by_class_and_history_stratum", "signed_scale_diagnostics",
                    "positive_cdf_total_mass_error", "threshold_cdf_continuity_error",
                    "finite_nll_count_by_route", "finite_density_count", "sampled_min_max",
                    "lower_boundary_rate", "nonfinite_sample_count", "exact_upper_clip_count",
                    "frozen_bin_mass", "raw_vs_mapped_ks_difference",
                )
            },
            "receiver": {
                key: {}
                for key in (
                    "classwise_receiver_tv", "full_receiver_tv", "head_receiver_tv",
                    "tail_receiver_tv", "unk_rate_error", "head", "tail", "unk",
                    "repeat", "new",
                )
            },
            "amount": {"contract": "frozen"},
            "hard_validity": {"status": "PASS"},
            "forbidden_access": {"internal_test": 0, "sparkov_fraud_test": 0},
        },
    )
    store.write_json("metrics.json", {"finite": True})
    store.write_json("evaluation.json", {"status": "VALID"})
    store.write_json("runtime.json", {"status": "COMPLETE"})
    store.write_json("gate_decision.json", {"status": "PASS"})
    return {"terminal_status": "COMPLETE"}


def successful_child(payload, event_queue):
    job = ExecutionJob(**payload["job"])
    store = H1AttemptStore.attach_existing(
        runtime_root=Path(payload["runtime_root"]),
        job=job,
        ownership_path=Path(payload["ownership_path"]),
        attempt_path=Path(payload["attempt_path"]),
    )
    event_queue.put({"kind": "setup_complete"})
    return _write_complete_fixture(store, event_queue)


def boundary_load_train_body(**_kwargs):
    return {"fixture": "no_data_body_read"}


def boundary_build_model(**_kwargs):
    return {"fixture": "no_model_build"}


def boundary_select_device(**_kwargs):
    return "cpu-fixture-no-cuda"


def boundary_run_job(
    *, plan, job, authorization, data, model, device, attempt_path,
    ownership_path, event_queue,
):
    del plan, authorization, data, model, device
    ownership_path = Path(ownership_path)
    runtime_root = ownership_path.parents[3]
    store = H1AttemptStore.attach_existing(
        runtime_root=runtime_root,
        job=job,
        ownership_path=ownership_path,
        attempt_path=Path(attempt_path),
    )
    return _write_complete_fixture(store, event_queue)
