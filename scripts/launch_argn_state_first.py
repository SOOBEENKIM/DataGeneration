"""Launch two isolated, durable GPU queues after recorded preflight checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from run_argn_state_first import ROOT, OUT, CFG, check_manifest, digest, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", nargs=2, type=int, required=True)
    args = p.parse_args()
    assert len(set(args.gpus)) == 2
    check_manifest()
    smoke = json.loads((OUT / "smoke/RESULT.json").read_text())
    assert set(smoke["arms"]) == set(CFG["arms"])
    assert all(r["all_weights_finite"] and r["generated_events"] == 16 for r in smoke["arms"].values())
    assert not (OUT / "LAUNCH.json").exists(), "do not duplicate launch"
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    gpus = {}
    for line in query.strip().splitlines():
        index, uuid, memory, utilization = [v.strip() for v in line.split(",")]
        gpus[int(index)] = dict(uuid=uuid, used_mib=int(memory), utilization=int(utilization))
    for index in args.gpus:
        assert gpus[index]["used_mib"] < 500 and gpus[index]["utilization"] <= 5, f"GPU {index} is busy"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": CFG,
        "launcher_sha256": digest(__file__), "smoke_sha256": digest(OUT / "smoke/RESULT.json"),
        "queues": {}, "prelaunch_gpu_state": gpus,
    }
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for arm, index in zip(CFG["arms"], args.gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[index]["uuid"], PYTHONUNBUFFERED="1")
        command = [sys.executable, str(ROOT / "scripts/run_argn_state_first.py"), "queue", "--arm", arm, "--device", "cuda:0"]
        log_path = logs / f"{arm}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest["queues"][arm] = dict(pid=process.pid, gpu_index=index, gpu_uuid=gpus[index]["uuid"],
                                       log=str(log_path), command=command)
        write(OUT / "LAUNCH.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
