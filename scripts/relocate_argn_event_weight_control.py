"""One-time, user-requested relocation from GPUs 0/1 to idle GPUs 2/3.

Preserves interrupted attempts and restarts their registered seeds from scratch.
Fully completed fits instead keep their weights and continue generation only.
Never signals processes outside the exact recorded original task commands.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/argn_event_weight_control_v1"
TARGETS = {20260930: 2, 20261001: 3}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_state():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in output.splitlines():
        index, uuid, memory, utilization = [s.strip() for s in line.split(",")]
        result[int(index)] = dict(index=int(index), uuid=uuid,
            used_mib=int(memory), utilization=int(utilization))
    return result


def command(pid):
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
        return payload.decode().rstrip("\0").split("\0") if payload else []
    except FileNotFoundError:
        return []


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = OUT / "relocation_history" / timestamp
    archive.mkdir(parents=True, exist_ok=False)
    with (OUT / "GPU_RELOCATION_LOCK.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), archive=str(archive)), stream)
    old_launch = json.loads((OUT / "LAUNCH.json").read_text())
    dispatcher = json.loads((OUT / "DISPATCH_STARTED.json").read_text())["pid"]
    dispatcher_command = command(dispatcher)
    if dispatcher_command:
        assert dispatcher_command[1] == str(ROOT / "scripts/dispatch_argn_event_weight_control.py")
    gpu = gpu_state()
    for index in TARGETS.values():
        assert gpu[index]["used_mib"] < 500 and gpu[index]["utilization"] <= 5, gpu[index]
    paused = []
    terminated = set()
    try:
        if dispatcher_command:
            os.kill(dispatcher, signal.SIGSTOP)
            paused.append(dispatcher)
        for fs in TARGETS:
            pid = old_launch["queues"][str(fs)]["pid"]
            argv = command(pid)
            if argv:
                assert argv == old_launch["queues"][str(fs)]["command"], (pid, argv)
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
        for pid in paused:
            for _ in range(100):
                status = Path(f"/proc/{pid}/status")
                if not status.exists() or "\nState:\tT" in status.read_text():
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError(f"Could not confirm stopped process {pid}")
        for name in ["LAUNCH.json", "DISPATCH_STARTED.json", "DISPATCH_STATUS.json"]:
            shutil.copy2(OUT / name, archive / name)
        plan = {}
        for fs, index in TARGETS.items():
            status_path = OUT / f"queue_{fs}.json"
            status = json.loads(status_path.read_text())
            shutil.copy2(status_path, archive / status_path.name)
            folder = OUT / "runs" / f"B_event_weighted_{fs}"
            fit_path = folder / "FIT.json"
            mode = "complete" if status["stage"] == "complete" else (
                "generation_only" if fit_path.exists() else "fresh_restart")
            if fit_path.exists():
                fit = json.loads(fit_path.read_text())
                weights = folder / "workspace/ModelStore/model-data/model-weights.pt"
                assert hashlib.sha256(weights.read_bytes()).hexdigest() == fit["weights_sha256"]
            plan[fs] = dict(mode=mode, old_pid=old_launch["queues"][str(fs)]["pid"],
                previous_stage=status["stage"], gpu=gpu[index])
        write(archive / "PLAN.json", plan)
        # Recheck before changing process placement; a foreign allocation may have appeared.
        gpu = gpu_state()
        for index in TARGETS.values():
            assert gpu[index]["used_mib"] < 500 and gpu[index]["utilization"] <= 5
        for pid in paused:
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            terminated.add(pid)
        for pid in paused:
            for _ in range(100):
                if not command(pid):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(f"Original task process did not exit: {pid}")
        processes = {}
        launch = dict(queues={}, purpose="user-requested relocation to idle GPUs 2 and 3",
            archive=str(archive), controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for fs, entry in plan.items():
            if entry["mode"] == "complete":
                continue
            folder = OUT / "runs" / f"B_event_weighted_{fs}"
            old_log = OUT / "logs" / f"{fs}.log"
            old_log.rename(archive / old_log.name)
            if entry["mode"] == "fresh_restart":
                folder.rename(archive / folder.name)
                script = ROOT / "scripts/run_argn_event_weight_control.py"
            else:
                script = ROOT / "scripts/continue_argn_event_weight_generation.py"
                for gs in [20261011, 20261012]:
                    if not (folder / f"generation_validation_{gs}.json").exists():
                        for prefix in ["native", "generated"]:
                            partial = folder / f"{prefix}_validation_{gs}.parquet"
                            if partial.exists():
                                partial.rename(archive / f"{fs}_{partial.name}")
                if (folder / "workspace/SyntheticData").exists():
                    shutil.copytree(folder / "workspace/SyntheticData", archive / f"{fs}_SyntheticData")
            argv = [sys.executable, str(script), "--seed", str(fs), "--device", "cuda:0"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=entry["gpu"]["uuid"],
                PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
            write(OUT / f"queue_{fs}.json", dict(stage="relocating", seed=fs, gpu=entry["gpu"], mode=entry["mode"]))
            with old_log.open("x") as log:
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[fs] = process
            launch["queues"][fs] = dict(pid=process.pid, gpu=entry["gpu"], command=argv,
                started_utc=datetime.now(timezone.utc).isoformat(), mode=entry["mode"])
            print(f"RELOCATED seed={fs} gpu={entry['gpu']['index']} pid={process.pid} mode={entry['mode']}", flush=True)
        write(OUT / "LAUNCH.json", launch)
        write(OUT / "DISPATCH_STARTED.json", dict(pid=os.getpid(),
            started_utc=datetime.now(timezone.utc).isoformat(), relocation_archive=str(archive)))
        write(OUT / "GPU_RELOCATION.json", dict(plan=plan, launch=launch,
            reason="User requested idle GPUs after foreign jobs occupied the original GPUs",
            interrupted_training_is_not_exactly_resumable=True, original_attempts_preserved=True))
        finished = {}
        while processes:
            for fs, process in list(processes.items()):
                code = process.poll()
                if code is not None:
                    finished[fs] = dict(exit_code=code, pid=process.pid)
                    del processes[fs]
            write(OUT / "DISPATCH_STATUS.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                pid=os.getpid(), pending=[], running={fs:p.pid for fs,p in processes.items()},
                finished=finished, expired=[], relocation_archive=str(archive), gpu_assignment=TARGETS))
            if processes:
                time.sleep(15)
    except BaseException as error:
        for pid in paused:
            if pid not in terminated:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
        write(archive / "ERROR.json", dict(error=repr(error), terminated=list(terminated)))
        raise


if __name__ == "__main__":
    main()
