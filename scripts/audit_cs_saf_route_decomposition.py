"""Preregistered read-only analysis of 12 fixed CS-SAF checkpoint snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

from experiments.cs_saf_pilot import ROOT, frozen_source, state_digest, subset
from experiments.cs_saf_loss_control import make_model, load_cache
from experiments.cs_saf_route_decomposition import reference_measure, audit_payload, analyze
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONTRACT_SHA = "f188d987db85e4dcac3c17fce11bf8aee3f16b57a1f6495d48c06edb9bad26e2"


def contract_and_parent():
    path = ROOT/"configs/benchmark_v2/cs_saf_route_decomposition_v1.yaml"
    if sha256(path) != CONTRACT_SHA:raise ValueError("preregistration changed")
    cfg = yaml.safe_load(path.read_text())
    if sha256(ROOT/cfg["parent_result"]) != cfg["parent_result_sha256"]:raise ValueError("parent evidence changed")
    return cfg, json.loads((ROOT/cfg["parent_result"]).read_text())["result"]


def load_snapshot(cfg, parent, kappa, candidate, name, device):
    key = f"pi_0.05_kappa_{kappa}/{candidate}"
    path = ROOT/cfg["checkpoint_root"]/key/("checkpoint_best.pt" if name=="best" else "checkpoint_epoch_9.pt")
    expected = parent["worker_terminals"][key]["artifact_sha256"][path.name]
    if sha256(path) != expected:raise ValueError("checkpoint checksum mismatch")
    cache = ROOT/cfg["prepared_root"]/f"pi_0.05_kappa_{kappa}.pt"
    payload = load_cache(cache)
    if sha256(cache) != parent["training"][key]["cache_sha256"]:raise ValueError("cache identity mismatch")
    cp = torch.load(path, map_location="cpu")
    if cp["tensorizer_state"] != payload["tensorizer_state"] or cp["source_commit"] != parent["source_commit"]:
        raise ValueError("checkpoint provenance mismatch")
    if cp["candidate"] != candidate or cp["data_manifest_sha256"] != payload["data_manifest_sha256"]:
        raise ValueError("candidate/data mismatch")
    model = make_model(payload, candidate, device).eval()
    model.load_state_dict(cp["model_state"])
    expected_state = parent["diagnostics"][key]["checkpoints"][name]["state_sha256"]
    if state_digest(model) != expected_state:raise ValueError("checkpoint tensor identity mismatch")
    return model,payload,path,expected


def check_errors(summary, cfg):
    for key,value in summary["mechanical_errors"].items():
        if value > cfg["mechanical_checks"][key]:raise RuntimeError(f"mechanical identity failed: {key}={value}")


def cpu(output):
    source = frozen_source();cfg,parent = contract_and_parent()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model,payload,path,expected = load_snapshot(cfg,parent,0,"CS2-U1","best",torch.device("cpu"))
    weights,ref = reference_measure(model,payload["train"])
    small = subset(payload["train"],16)
    a,raw_a = analyze(model,small,weights,torch.device("cpu"))
    b,raw_b = analyze(model,small,weights,torch.device("cpu"))
    check_errors(a,cfg);check_errors(b,cfg)
    if a!=b or not all(np.array_equal(raw_a[k],raw_b[k]) for k in raw_a):raise RuntimeError("CPU analysis not deterministic")
    if sha256(path)!=expected:raise RuntimeError("CPU analysis changed checkpoint")
    result = {"status":"PASS","source_commit":source,"config_sha256":CONTRACT_SHA,
              "deterministic_arrays":True,"checkpoint_unchanged":True,"new_fits":0,
              "data_audit":audit_payload(model,payload),"reference_measure":ref,"smoke_summary":a}
    write_json(output/"COMPLETE.json",result);print("CPU diagnostic PASS",flush=True)


def validate_cpu(path):
    r=json.loads(path.read_text())
    if r["status"]!="PASS" or r["source_commit"]!=frozen_source() or r["config_sha256"]!=CONTRACT_SHA:
        raise RuntimeError("same-source passing CPU diagnostic required")


def job(kappa,candidate,output,device,cpu_path):
    validate_cpu(cpu_path);source=frozen_source();cfg,parent=contract_and_parent()
    if kappa not in cfg["kappas"] or candidate not in cfg["candidates"]:raise ValueError("unregistered snapshot")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    results,arrays={},{ }
    try:
        for name in cfg["checkpoint_names"]:
            model,payload,path,expected=load_snapshot(cfg,parent,kappa,candidate,name,device)
            before=state_digest(model)
            weights,reference=reference_measure(model,payload["train"])
            item={"checkpoint_sha256":expected,"checkpoint_path":str(path),"state_sha256":before,
                  "data_audit":audit_payload(model,payload),"reference_measure":reference,"splits":{}}
            for split in cfg["content_splits"]:
                summary,raw=analyze(model,payload[split],weights,device)
                check_errors(summary,cfg)
                historical=parent["diagnostics"][f"pi_0.05_kappa_{kappa}/{candidate}"]["checkpoints"][name]["splits"][split]
                error=0.
                for label in ("0","1"):
                    new=summary["groups"][label]["metrics"];old=historical["groups"][label]["metrics"]
                    for a,b in [("F_repeat_BCE","observed_repeat_BCE"),("Z_repeat_BCE","zero_gap_repeat_BCE")]:
                        error=max(error,abs(new[a]["mean"]-old[b]["mean"]))
                if error>cfg["mechanical_checks"]["historical_full_zero_BCE_match_max_absolute_error"]:
                    raise RuntimeError("historical full/zero loss mismatch")
                summary["historical_full_zero_BCE_max_absolute_difference"]=error
                item["splits"][split]=summary
                arrays.update({f"{name}_{split}_{key}":v for key,v in raw.items()})
            if state_digest(model)!=before or sha256(path)!=expected:raise RuntimeError("checkpoint changed during analysis")
            results[name]=item
        np.savez_compressed(output/"entity_metrics.npz",**arrays)
        result={"source_commit":source,"config_sha256":CONTRACT_SHA,"kappa":kappa,"candidate":candidate,
                "checkpoints":results,"entity_arrays_sha256":sha256(output/"entity_metrics.npz"),
                "status":"COMPLETE","new_fits":0,"optimizer_steps":0,"test_accessed":False}
        write_json(output/"COMPLETE.json",result)
        print(f"COMPLETE kappa={kappa} {candidate}",flush=True)
    except Exception as exc:
        write_json(output/"FAILED.json",{"error":type(exc).__name__,"message":str(exc)})
        raise


def run(output,cpu_path,gpus):
    validate_cpu(cpu_path);source=frozen_source();cfg,parent=contract_and_parent()
    if len(gpus)!=3 or len(set(gpus))!=3:raise ValueError("three distinct GPUs required")
    output.mkdir(parents=True,exist_ok=False)
    results={}
    for kappa in cfg["kappas"]:
        processes,handles=[],[]
        try:
            for candidate,gpu in zip(cfg["candidates"],gpus):
                folder=output/f"kappa_{kappa}"/candidate;folder.parent.mkdir(parents=True,exist_ok=True)
                log=(folder.parent/f"{candidate}.log").open("w");handles.append(log)
                processes.append(subprocess.Popen([sys.executable,"-m","scripts.audit_cs_saf_route_decomposition","job",
                    "--kappa",str(kappa),"--candidate",candidate,"--output",str(folder),"--device",f"cuda:{gpu}",
                    "--cpu-gate",str(cpu_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
            exits=[p.wait() for p in processes]
        finally:
            for h in handles:h.close()
        if any(exits):
            write_json(output/"FAILED.json",{"kappa":kappa,"exit_codes":exits});raise RuntimeError("worker failure")
        for candidate in cfg["candidates"]:
            folder=output/f"kappa_{kappa}"/candidate
            r=json.loads((folder/"COMPLETE.json").read_text())
            if r["source_commit"]!=source or sha256(folder/"entity_metrics.npz")!=r["entity_arrays_sha256"]:
                raise RuntimeError("worker provenance mismatch")
            results[f"kappa_{kappa}/{candidate}"]=r
        print(f"All kappa={kappa} checkpoint decompositions complete",flush=True)
    directional={}
    for name,kappa,metric,sign in [("mean_useful_in_primary_null",0,"M_minus_Z_repeat_BCE",-1),
        ("residual_harmful_in_primary_null",0,"M_minus_F_repeat_BCE",-1),
        ("residual_useful_in_primary_active",1,"M_minus_F_repeat_BCE",1)]:
        observations={f"{cp}_{split}":results[f"kappa_{kappa}/CS2-U1"]["checkpoints"][cp]["splits"][split]["groups"]["1"]["metrics"][metric]
                      for cp in cfg["checkpoint_names"] for split in cfg["content_splits"]}
        directional[name]={"all_registered_signs_observed":all(sign*x["mean"]>0 for x in observations.values()),"observations":observations}
    assert sum(len(r["checkpoints"]) for r in results.values())==cfg["checkpoint_count"]
    result={"status":"COMPLETE","source_commit":source,"config_sha256":CONTRACT_SHA,
            "parent_result_sha256":cfg["parent_result_sha256"],"cpu_gate_sha256":sha256(cpu_path),
            "checkpoint_count":12,"results":results,"directional_hypotheses":directional,
            "new_fits":0,"optimizer_steps":0,"test_accessed":False,"new_model_success":False}
    write_json(output/"COMPLETE.json",result)
    print(json.dumps({"status":"COMPLETE","hypotheses":directional}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="phase",required=True)
    a=sub.add_parser("cpu");a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("job");a.add_argument("--output",type=Path,required=True)
    a.add_argument("--cpu-gate",type=Path,required=True);a.add_argument("--kappa",type=int,choices=[0,1],required=True)
    a.add_argument("--candidate",choices=["CS2-U1","CS2-A1","CS2-B1"],required=True);a.add_argument("--device",required=True)
    a=sub.add_parser("run");a.add_argument("--output",type=Path,required=True);a.add_argument("--cpu-gate",type=Path,required=True)
    a.add_argument("--gpus",default="0,1,2");args=p.parse_args()
    if args.phase=="cpu":cpu(args.output)
    elif args.phase=="job":job(args.kappa,args.candidate,args.output,torch.device(args.device),args.cpu_gate)
    else:run(args.output,args.cpu_gate,[int(x) for x in args.gpus.split(',')])


if __name__=="__main__":main()
