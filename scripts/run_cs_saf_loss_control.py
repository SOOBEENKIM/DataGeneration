"""Execute the separately preregistered CS-SAF three-objective diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from experiments.cs_saf_loss_control import (
    ROOT, CONTRACT_SHA, CANDIDATES, load_contract, train, make_model,
    save_diagnostics, paired_contrasts, frozen_source,
)
from experiments.cs_saf_pilot import audit_model, decide_pilot, batch, state_digest
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json


def cpu_gate(cache_root, output):
    commit = frozen_source()
    _, cfg = load_contract()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for candidate in CANDIDATES:
        runs, audits, reload_checks = [], [], []
        for repeat in (1, 2):
            folder = output/candidate/f"run_{repeat}"
            model, payload, report = train(cache_root/"pi_0.05_kappa_1.pt", folder,
                                           candidate, torch.device("cpu"), cpu=True)
            copy = make_model(payload, candidate, torch.device("cpu")).eval()
            copy.load_state_dict(torch.load(folder/"checkpoint_best.pt", map_location="cpu")["model_state"])
            x = batch(payload["validation"], torch.arange(len(payload["validation"]["lengths"])), torch.device("cpu"))
            with torch.no_grad():
                a, b = model.loss_terms(**x), copy.loss_terms(**x)
            reload_checks.append(state_digest(copy) == report["best_state_sha256"]
                                 and all(torch.equal(a[k], b[k]) for k in a))
            audits.append(audit_model(copy, payload, dict(cfg, generation_entities=32, generation_batch_size=32),
                                      torch.device("cpu"), sample_output=folder/"generated_sample.pt"))
            runs.append(report)
        first, second = runs
        drop = (first["history"][0]["train_objective"]-min(x["train_objective"] for x in first["history"]))/abs(first["history"][0]["train_objective"])
        checks = {"training_history_identical": first["history"] == second["history"],
                  "best_model_identical": first["best_state_sha256"] == second["best_state_sha256"],
                  "loss_decrease": drop >= cfg["cpu_gate"]["minimum_relative_train_loss_decrease"],
                  "checkpoint_reload_identical": all(reload_checks),
                  "generation_support": all(a["gap_support_violations"] == 0 for a in audits),
                  "generation_valid": all(a["invalid_reserved_marks"] == 0 and a["finite_generated_values"] for a in audits),
                  "zero_gap_invariant": all(a["zero_gap_control_max_range"] <= 1e-8 for a in audits)}
        results[candidate] = {"checks": checks, "decision": "PASS" if all(checks.values()) else "FAIL",
                              "relative_loss_decrease": drop, "best_state_sha256": first["best_state_sha256"],
                              "initial_state_sha256": first["initial_state_sha256"], "audits": audits}
    paired = len({r["initial_state_sha256"] for r in results.values()}) == 1
    result = {"decision": "PASS" if paired and all(r["decision"] == "PASS" for r in results.values()) else "FAIL",
              "source_commit": commit, "config_sha256": CONTRACT_SHA, "candidates": results,
              "matched_initialization": paired, "test_accessed": False}
    write_json(output/"COMPLETE.json", result)
    print(json.dumps(result), flush=True)
    return result


def validate_cpu(path):
    cpu = json.loads(path.read_text())
    if cpu["source_commit"] != frozen_source() or cpu["config_sha256"] != CONTRACT_SHA or cpu["decision"] != "PASS":
        raise RuntimeError("same-source passing CPU gate required")


def job(cache, output, candidate, device, cpu_path):
    validate_cpu(cpu_path)
    if output.exists():
        raise FileExistsError("prior output is immutable")
    try:
        model, payload, report = train(cache, output, candidate, device)
        _, cfg = load_contract()
        audit = audit_model(model, payload, cfg, device, sample_output=output/"generated_sample.pt")
        write_json(output/"intervention_audit.json", audit)
        diagnostic = save_diagnostics(model, payload, output, device)
        for label in ("0", "1"):
            for metric in ("copy", "repeat"):
                detailed = diagnostic["checkpoints"]["best"]["splits"]["validation"]["groups"][label]["metrics"][metric+"_range"]["mean"]
                if abs(detailed-audit["responses"][label][f"mean_{metric}_range"]) > 1e-6:
                    raise RuntimeError("independent diagnostic and pilot aggregation disagree")
        if state_digest(model) != report["best_state_sha256"]:
            raise RuntimeError("analysis changed model state")
        write_json(output/"COMPLETE.json", {"status": "COMPLETE", "source_commit": report["source_commit"],
            "config_sha256": CONTRACT_SHA, "test_accessed": False,
            "artifact_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    except Exception as exc:
        if output.is_dir() and not (output/"COMPLETE.json").exists():
            write_json(output/"FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


def historical_reproduction(training, audits, contract):
    old = json.loads((ROOT/"docs/cs_saf/v2_pilot_v1_result.json").read_text())
    checks = {}
    for kappa in (0, 1):
        for candidate in ("CS2-U1", "CS2-B1"):
            key = f"pi_0.05_kappa_{kappa}/{candidate}"
            a, b = training[key], old["training_runs"][key]
            old_audit = old["pilot"]["stages"]["0.05"]["audits"][f"kappa_{kappa}_{candidate}"]
            delta = max(abs(audits[key]["responses"][str(y)][f"mean_{m}_range"]-old_audit["responses"][str(y)][f"mean_{m}_range"])
                        for y in (0, 1) for m in ("copy", "repeat"))
            nll_delta = abs(a["best_validation_base_nll"]-b["best_validation_base_nll"])
            checks[key] = {"initial_state_identical": a["initial_state_sha256"] == b["initial_state_sha256"],
                "best_state_identical": a["best_state_sha256"] == b["best_state_sha256"],
                "best_epoch_identical": a["best_epoch"] == b["best_epoch"],
                "maximum_response_absolute_difference": delta, "base_nll_absolute_difference": nll_delta}
            checks[key]["comparison_valid"] = (checks[key]["initial_state_identical"]
                and checks[key]["best_epoch_identical"]
                and delta <= contract["historical_U_B_reproduction"]["response_absolute_tolerance"]
                and nll_delta <= contract["historical_U_B_reproduction"]["nll_absolute_tolerance"])
    return checks


def run(cache_root, cpu_path, output, gpus):
    validate_cpu(cpu_path)
    commit = frozen_source()
    contract, cfg = load_contract()
    if len(gpus) != 3 or len(set(gpus)) != 3:
        raise ValueError("three distinct GPUs required for the fixed three-arm comparison")
    output.mkdir(parents=True, exist_ok=False)
    for kappa in (0, 1):
        processes, handles = [], []
        try:
            for gpu, candidate in zip(gpus, CANDIDATES):
                destination = output/f"pi_0.05_kappa_{kappa}"/candidate
                destination.parent.mkdir(parents=True, exist_ok=True)
                log = (destination.parent/f"{candidate}.log").open("w")
                handles.append(log)
                command = [sys.executable, "-m", "scripts.run_cs_saf_loss_control", "job",
                    "--cache", str(cache_root/f"pi_0.05_kappa_{kappa}.pt"), "--output", str(destination),
                    "--candidate", candidate, "--device", f"cuda:{gpu}", "--cpu-gate", str(cpu_path)]
                processes.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            codes = [p.wait() for p in processes]
        finally:
            for h in handles:
                h.close()
        if any(codes):
            write_json(output/"FAILED.json", {"kappa": kappa, "exit_codes": codes, "source_commit": commit})
            raise RuntimeError("technical worker failure; retain all artifacts")
        print(f"Completed kappa={kappa}: all three objective fits and diagnostics", flush=True)
    training, audits, diagnostics, terminals, contrasts = {}, {}, {}, {}, {}
    for kappa in (0, 1):
        folders = {}
        for candidate in CANDIDATES:
            key = f"pi_0.05_kappa_{kappa}/{candidate}"
            folder = output/key; folders[candidate] = folder
            terminal = json.loads((folder/"COMPLETE.json").read_text())
            if terminal["source_commit"] != commit or terminal["config_sha256"] != CONTRACT_SHA:
                raise RuntimeError("worker provenance mismatch")
            for name, expected in terminal["artifact_sha256"].items():
                if sha256(folder/name) != expected:
                    raise RuntimeError(f"artifact mismatch: {key}/{name}")
            terminals[key] = terminal
            training[key] = json.loads((folder/"training_report.json").read_text())
            audits[key] = json.loads((folder/"intervention_audit.json").read_text())
            diagnostics[key] = json.loads((folder/"null_diagnostics.json").read_text())
        contrasts[str(kappa)] = paired_contrasts(folders)
        reports = [training[f"pi_0.05_kappa_{kappa}/{c}"] for c in CANDIDATES]
        if len({r["initial_state_sha256"] for r in reports}) != 1:
            raise RuntimeError("unmatched initial states")
        common_epochs = min(len(r["history"]) for r in reports)
        if any(len({r["history"][e]["entity_order_prefix_sha256"] for r in reports}) != 1 for e in range(common_epochs)):
            raise RuntimeError("entity orders differ")
        if len({audits[f"pi_0.05_kappa_{kappa}/{c}"]["sampling_plan_sha256"] for c in CANDIDATES}) != 1:
            raise RuntimeError("generation plans differ")
    reproduction = historical_reproduction(training, audits, contract)
    valid = all(r["comparison_valid"] for r in reproduction.values())
    gates = {candidate: decide_pilot({k: audits[f"pi_0.05_kappa_{k}/{candidate}"] for k in (0, 1)}, cfg["pilot_gate"])
             for candidate in CANDIDATES}
    result = {"schema_version": "cs-saf-loss-control-v1-result", "source_commit": commit,
              "config_sha256": CONTRACT_SHA, "status": "COMPLETE" if valid else "REPRODUCTION_MISMATCH",
              "scientific_fits": len(training), "prevalence": .05,
              "candidate_local_response_gates": gates, "historical_reproduction": reproduction,
              "training": training, "audits": audits, "diagnostics": diagnostics,
              "paired_contrasts": contrasts, "worker_terminals": terminals,
              "cpu_gate_sha256": sha256(cpu_path), "prepared_index_sha256": sha256(cache_root/"COMPLETE.json"),
              "matched_initialization_order_and_sampling": True,
              "test_accessed": False, "five_seed_started": False, "later_prevalences_started": False,
              "full_pilot_success_claim": False}
    write_json(output/"COMPLETE.json", result)
    print(json.dumps({"status": result["status"], "candidate_gates": gates, "historical_reproduction": reproduction}), flush=True)
    if not valid:
        raise RuntimeError("historical U/B mismatch requires investigation before interpretation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("cpu")
    p.add_argument("--cache-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("job")
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", choices=CANDIDATES, required=True); p.add_argument("--device", required=True)
    p.add_argument("--cpu-gate", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--cache-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-gate", type=Path, required=True); p.add_argument("--gpus", default="0,1,2")
    args = parser.parse_args()
    if args.phase == "cpu":
        if cpu_gate(args.cache_root, args.output)["decision"] != "PASS":
            raise SystemExit(2)
    elif args.phase == "job":
        job(args.cache, args.output, args.candidate, torch.device(args.device), args.cpu_gate)
    else:
        run(args.cache_root, args.cpu_gate, args.output, [int(x) for x in args.gpus.split(",")])


if __name__ == "__main__":
    main()
