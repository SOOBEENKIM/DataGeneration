"""Lightweight spawn targets for CCMTPP watchdog regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import time


def child_never_reports(payload, event_queue):
    del event_queue
    time.sleep(float(payload["sleep_seconds"]))


def child_setup_then_hangs(payload, event_queue):
    event_queue.put({"kind": "setup_complete"})
    time.sleep(float(payload["sleep_seconds"]))


def child_raises(payload, event_queue):
    event_queue.put({"kind": "setup_complete"})
    raise RuntimeError(str(payload["message"]))


def child_interrupts(payload, event_queue):
    event_queue.put({"kind": "setup_complete"})
    raise KeyboardInterrupt(str(payload["message"]))


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")


def _bytes(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def child_completes(payload, event_queue):
    attempt = Path(payload["attempt_path"])
    event_queue.put({"kind": "setup_complete"})
    with (attempt / "progress.jsonl").open("ab") as handle:
        handle.write(b'{"step":1,"loss":0.5}\n')
    event_queue.put({"kind": "progress", "step": 1})
    _bytes(attempt / "checkpoints/final.pt", b"checkpoint")
    values = {
        "receiver_vocabulary_state.json": {"state_sha256": "1" * 64},
        "receiver_hierarchy_state.json": {"state_sha256": "2" * 64},
        "gap_support_state.json": {"state_sha256": "3" * 64},
        "amount_transform_state.json": {"state_sha256": "4" * 64},
        "conditioning_plan.json": {"plan_sha256": "5" * 64},
        "checkpoint_provenance.json": {"checkpoint_sha256": "6" * 64},
        "runtime.json": {"actual_updates": 1, "elapsed_seconds": 0.1},
        "metrics.json": {"loss": 0.5},
        "evaluation.json": {"status": "VALID"},
        "diagnostics.json": {
            "receiver": {
                "overall_nll": 1.0,
                "head_nll": 1.0,
                "tail_nll": 1.0,
                "unk_nll": 1.0,
                "repeat_nll": 1.0,
                "new_nll": 1.0,
                "head_count": 1,
                "tail_count": 1,
                "unk_count": 0,
                "repeat_count": 1,
                "new_count": 1,
            },
            "fidelity": {
                "full_receiver_tv": 0.2,
                "head_receiver_tv": 0.1,
                "tail_receiver_tv": 0.3,
                "unk_rate_error": 0.0,
            },
        },
        "gate_decision.json": {
            "candidate_id": payload["candidate_id"],
            "status": "PASS",
            "stop_criterion_id": payload["candidate_id"],
        },
    }
    for name, value in values.items():
        _json(attempt / name, value)
    _bytes(attempt / "validation_sample.npz", b"sample")
    return {"terminal_status": "COMPLETE"}


def load_fixture_data(*, plan, job, authorization):
    del plan, authorization
    return {"dataset": job.dataset, "fit_split": "train"}


def build_fixture_model(*, plan, job, data):
    del plan
    return {"candidate_id": job.candidate_id, "dataset": data["dataset"]}


def query_fixture_device(*, plan, job, authorization):
    del plan, job, authorization
    return "fixture:0"


def run_fixture_job(
    *, plan, job, authorization, data, model, device, attempt_path, event_queue
):
    del plan, authorization
    if (
        data["fit_split"] != "train"
        or model["candidate_id"] != job.candidate_id
        or device != "fixture:0"
    ):
        raise AssertionError("fixture execution wiring changed")
    return child_completes(
        {"attempt_path": str(attempt_path), "candidate_id": job.candidate_id},
        event_queue,
    )
