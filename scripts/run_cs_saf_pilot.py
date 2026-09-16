"""Execute prepared CS-SAF CPU gates and the frozen single-seed pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cs_saf_pilot import (
    prepare_data, train_job, audit_model, decide_pilot, frozen_source,
)
from scripts.materialize_cs_saf_prevalence import write_json
from scripts.audit_cs_saf_oracle import sha256


def cpu_gate(cache_root, output):
    frozen_source()
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml").read_text())
    reports = []
    audits = []
    smoke_cfg = dict(cfg, generation_entities=32, generation_batch_size=32)
    for repeat in (1, 2):
        model, payload, report = train_job(cache_root/"pi_0.05_kappa_1.pt", output/f"run_{repeat}",
                                         "CS-B1", torch.device("cpu"), cpu_gate=True)
        audits.append(audit_model(model, payload, smoke_cfg, torch.device("cpu"),
                                   sample_output=output/f"run_{repeat}"/"generated_sample.pt"))
        reports.append(report)
    first, second = reports
    drop = (first["history"][0]["train_objective"]-min(x["train_objective"] for x in first["history"]))/abs(first["history"][0]["train_objective"])
    checks = {"training_history_identical": first["history"] == second["history"],
              "best_model_bitwise_identical": first["best_state_sha256"] == second["best_state_sha256"],
              "loss_decrease": drop >= cfg["cpu_gate"]["minimum_relative_train_loss_decrease"],
              "generation_support_preserved": all(a["gap_support_violations"] == 0 for a in audits),
              "generation_valid": all(a["invalid_reserved_marks"] == 0 and a["finite_generated_values"] for a in audits),
              "zero_gap_invariant": all(a["zero_gap_control_max_range"] <= 1e-8 for a in audits)}
    result = {"decision": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "relative_loss_decrease": drop, "best_state_sha256": first["best_state_sha256"],
              "source_commit": first["source_commit"], "test_accessed": False,
              "audits": audits, "config_sha256": sha256(ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml")}
    write_json(output/"COMPLETE.json", result)
    print(json.dumps(result), flush=True)
    return result


def job(cache, output, candidate, device):
    if output.exists():
        raise FileExistsError("choose a new job directory; prior artifacts are immutable")
    try:
        model, payload, report = train_job(cache, output, candidate, torch.device(device))
        cfg = yaml.safe_load((ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml").read_text())
        audit = audit_model(model, payload, cfg, torch.device(device), sample_output=output/"generated_sample.pt")
        write_json(output/"intervention_audit.json", audit)
        write_json(output/"COMPLETE.json", {"status": "COMPLETE", "source_commit": report["source_commit"],
            "training_report_sha256": sha256(output/"training_report.json"),
            "intervention_audit_sha256": sha256(output/"intervention_audit.json"), "test_accessed": False})
    except Exception as exc:
        if output.is_dir() and not (output/"COMPLETE.json").exists():
            write_json(output/"FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


def pilot(cache_root, cpu_gate_path, output, gpus):
    commit = frozen_source()
    cfg = yaml.safe_load((ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml").read_text())
    cpu = json.loads(cpu_gate_path.read_text())
    if cpu["decision"] != "PASS" or cpu["source_commit"] != commit:
        raise RuntimeError("current source CPU gate is required")
    if cpu["config_sha256"] != sha256(ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml"):
        raise RuntimeError("CPU gate config mismatch")
    prepared = json.loads((cache_root/"COMPLETE.json").read_text())
    if not all(g["decision"] == "PASS" for g in prepared["oracle_gates"].values()):
        raise RuntimeError("all prevalence oracle gates must pass before training")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("four distinct GPU IDs required for the frozen four-job stage")
    output.mkdir(parents=True, exist_ok=False)
    result = {"source_commit": commit, "model_seed": cfg["model_seed"], "test_accessed": False,
              "five_seed_started": False, "stages": {}, "decision": "IN_PROGRESS"}
    for pi in cfg["prevalences"]:
        processes = []
        logs = []
        jobs = []
        try:
            for index, (kappa, candidate) in enumerate((k, c) for k in (0, 1) for c in cfg["pilot_candidates"]):
                name = f"pi_{pi:.2f}_kappa_{kappa}"
                destination = output/name/candidate
                destination.parent.mkdir(parents=True, exist_ok=True)
                log = (destination.parent/f"{candidate}.log").open("w")
                logs.append(log)
                command = [sys.executable, "-m", "scripts.run_cs_saf_pilot", "job", "--cache", str(cache_root/f"{name}.pt"),
                           "--output", str(destination), "--candidate", candidate, "--device", f"cuda:{gpus[index]}"]
                processes.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
                jobs.append((kappa, candidate, destination))
            codes = [process.wait() for process in processes]
        finally:
            for log in logs:
                log.close()
        if any(codes):
            write_json(output/"FAILED.json", {"source_commit": commit, "pi": pi, "exit_codes": codes})
            raise RuntimeError("pilot worker failed; inspect preserved logs")
        audits, training = {}, {}
        for kappa, candidate, destination in jobs:
            terminal = json.loads((destination/"COMPLETE.json").read_text())
            if terminal["intervention_audit_sha256"] != sha256(destination/"intervention_audit.json"):
                raise RuntimeError("audit checksum mismatch")
            if terminal["training_report_sha256"] != sha256(destination/"training_report.json"):
                raise RuntimeError("training checksum mismatch")
            audits[(kappa, candidate)] = json.loads((destination/"intervention_audit.json").read_text())
            training[(kappa, candidate)] = json.loads((destination/"training_report.json").read_text())
        full = {k: audits[(k, "CS-B1")] for k in (0, 1)}
        decision = decide_pilot(full, cfg["pilot_gate"])
        matched = all(training[(k, "CS-U1")]["initial_state_sha256"] == training[(k, "CS-B1")]["initial_state_sha256"]
                      and audits[(k, "CS-U1")]["sampling_plan_sha256"] == audits[(k, "CS-B1")]["sampling_plan_sha256"] for k in (0, 1))
        decision["checks"]["matched_initialization_and_sampling"] = matched
        decision["decision"] = "PASS" if all(decision["checks"].values()) else "FAIL"
        stage = {**decision, "audits": {f"kappa_{k}_{c}": a for (k, c), a in audits.items()},
                 "training": {f"kappa_{k}_{c}": {key: r[key] for key in
                     ("best_epoch", "best_validation_base_nll", "parameters", "seconds", "checkpoint_sha256", "initial_state_sha256")}
                     for (k, c), r in training.items()}}
        result["stages"][f"{pi:.2f}"] = stage
        write_json(output/"progress.json", result)
        print(f"pi={pi:.2f} pilot {decision['decision']}: {decision['checks']}", flush=True)
        if decision["decision"] == "FAIL":
            result["decision"] = "FAIL"
            result["stop_reason"] = f"frozen_full_candidate_gate_failed_at_pi_{pi:.2f}"
            break
    else:
        result["decision"] = "PASS"
    write_json(output/"COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--cache-root", type=Path, required=True)
    p = sub.add_parser("cpu"); p.add_argument("--cache-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("job"); p.add_argument("--cache", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", choices=["CS-U1", "CS-B1"], required=True); p.add_argument("--device", required=True)
    p = sub.add_parser("pilot"); p.add_argument("--cache-root", type=Path, required=True); p.add_argument("--cpu-gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_data(args.data_root, args.cache_root)
    elif args.phase == "cpu":
        if cpu_gate(args.cache_root, args.output)["decision"] != "PASS":
            raise SystemExit(2)
    elif args.phase == "job":
        job(args.cache, args.output, args.candidate, args.device)
    else:
        if pilot(args.cache_root, args.cpu_gate, args.output, [int(x) for x in args.gpus.split(",")])["decision"] != "PASS":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
