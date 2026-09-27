"""Bounded, task-specific GPU queue. Never stop or share an occupied GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/argn_event_weight_control_v1"


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(wait_minutes):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    lock = OUT / "DISPATCH_STARTED.json"
    with lock.open("x") as f:
        json.dump(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                       max_wait_minutes=wait_minutes, dispatcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), f)
    pending = [20260930, 20261001]
    running, finished, expired, idle_counts = {}, {}, [], {}
    launch = dict(queues={}, purpose="adaptive objective control after evaluation")
    deadline = time.monotonic() + wait_minutes * 60
    while pending or running:
        for fs, entry in list(running.items()):
            code = entry["process"].poll()
            if code is not None:
                finished[fs] = dict(exit_code=code, pid=entry["process"].pid)
                del running[fs]
        latest = []
        if pending and time.monotonic() >= deadline:
            expired = pending[:]
            for fs in pending:
                write(OUT / f"queue_{fs}.json", dict(stage="gpu_wait_timeout", seed=fs, training_started=False))
            pending.clear()
        if pending:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    index, uuid, memory, utilization = [s.strip() for s in line.split(",")]
                    free = int(memory) < 500 and int(utilization) <= 5
                    idle_counts[uuid] = idle_counts.get(uuid, 0) + 1 if free else 0
                    latest.append(dict(index=int(index), uuid=uuid, used_mib=int(memory), utilization=int(utilization)))
                occupied_by_us = {entry["uuid"] for entry in running.values()}
                for gpu in latest:
                    if not pending: break
                    if idle_counts.get(gpu["uuid"], 0) < 2 or gpu["uuid"] in occupied_by_us:
                        continue
                    fs = pending.pop(0)
                    command = [sys.executable, str(ROOT / "scripts/run_argn_event_weight_control.py"), "--seed", str(fs), "--device", "cuda:0"]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
                    with (OUT / "logs" / f"{fs}.log").open("x") as log:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    running[fs] = dict(process=process, uuid=gpu["uuid"])
                    occupied_by_us.add(gpu["uuid"])
                    launch["queues"][fs] = dict(pid=process.pid, gpu=gpu, command=command, started_utc=datetime.now(timezone.utc).isoformat())
                    write(OUT / "LAUNCH.json", launch)
                    print(f"LAUNCHED seed={fs} pid={process.pid} gpu={gpu['index']}", flush=True)
            else:
                idle_counts.clear()  # never infer idleness from a failed query
            for fs in pending:
                write(OUT / f"queue_{fs}.json", dict(stage="waiting_for_idle_gpu", seed=fs, training_started=False))
        write(OUT / "DISPATCH_STATUS.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
              pid=os.getpid(), pending=pending, running={fs:entry["process"].pid for fs,entry in running.items()},
              finished=finished, expired=expired, last_gpu_state=latest))
        if pending or running:
            time.sleep(30)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--wait-minutes", type=float, default=120)
    main(p.parse_args().wait_minutes)
