from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any

import numpy as np
import torch
import yaml

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from benchmarks.types import SyntheticBatch
from experiments.artifact_store import ArtifactStore
from generators.cof_seqgen_adapter import CoFSeqGenAdapter
from generators.conditional_ctgan import ConditionalCTGAN
from generators.conditional_tvae import ConditionalTVAE
from generators.joint_sequence_baseline import NeuralSequenceBaseline
from generators.sampling_plan import SamplingPlan


ALLOWED_SMOKE_GENERATORS = {
    "conditional_ctgan",
    "conditional_tvae",
    "neural_sequence_baseline",
    "cof",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_batch(batch) -> str:
    digest = hashlib.sha256()
    for value in (
        batch.x_num,
        batch.dt_bin,
        batch.x_cat,
        batch.valid_mask,
        batch.y_entity,
        batch.lengths,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def validate_smoke_scope(
    *,
    mode: str,
    generators: list[str],
    seeds: list[int],
    kappas: list[float],
    n_train: int,
    steps: int,
) -> None:
    if mode != "smoke":
        raise ValueError("only IMPLEMENT_AND_SMOKE mode is authorized")
    if len(generators) != 1 or generators[0] not in ALLOWED_SMOKE_GENERATORS:
        raise ValueError("exactly one authorized learned generator is required")
    if len(seeds) != 1:
        raise ValueError("smoke requires exactly one seed")
    if len(kappas) != 1:
        raise ValueError("smoke requires exactly one kappa")
    if n_train > 512:
        raise ValueError("smoke train entities cannot exceed 512")
    if not 100 <= steps <= 500:
        raise ValueError("smoke training steps must be between 100 and 500")


def generator_for(name: str):
    if name == "conditional_ctgan":
        return ConditionalCTGAN()
    if name == "conditional_tvae":
        return ConditionalTVAE()
    if name == "neural_sequence_baseline":
        return NeuralSequenceBaseline()
    if name == "cof":
        return CoFSeqGenAdapter()
    raise ValueError(name)


def save_sample(path: Path, sample: SyntheticBatch) -> None:
    np.savez_compressed(
        path,
        x_num=sample.x_num,
        dt_bin=sample.dt_bin,
        x_cat=sample.x_cat,
        valid_mask=sample.valid_mask,
        y_entity=sample.y_entity,
        lengths=sample.lengths,
    )


def basic_metrics(sample: SyntheticBatch, reference) -> dict[str, Any]:
    valid = sample.valid_mask
    reference_valid = reference.valid_mask
    gap_classes = max(
        int(sample.dt_bin.max()),
        int(reference.dt_bin.max()),
    ) + 1
    receiver_classes = max(
        int(sample.x_cat[..., 0].max()),
        int(reference.x_cat[..., 0].max()),
    ) + 1
    gap_real = np.bincount(
        reference.dt_bin[reference_valid],
        minlength=gap_classes,
    )
    gap_fake = np.bincount(sample.dt_bin[valid], minlength=gap_classes)
    receiver_real = np.bincount(
        reference.x_cat[..., 0][reference_valid],
        minlength=receiver_classes,
    )
    receiver_fake = np.bincount(
        sample.x_cat[..., 0][valid],
        minlength=receiver_classes,
    )
    gap_real = gap_real / gap_real.sum()
    gap_fake = gap_fake / gap_fake.sum()
    receiver_real = receiver_real / receiver_real.sum()
    receiver_fake = receiver_fake / receiver_fake.sum()
    return {
        "contract_valid": True,
        "entities": len(sample.lengths),
        "valid_rows": int(valid.sum()),
        "amount_mean": float(sample.x_num[..., 0][valid].mean()),
        "amount_std": float(sample.x_num[..., 0][valid].std()),
        "gap_tvd": float(0.5 * np.abs(gap_real - gap_fake).sum()),
        "receiver_tvd": float(
            0.5 * np.abs(receiver_real - receiver_fake).sum()
        ),
        "label_prevalence": float(sample.y_entity.mean()),
    }


def save_model_checkpoints(generator, name: str, root: Path) -> list[str]:
    checkpoints = []
    if name in ("conditional_ctgan", "conditional_tvae"):
        for label, model in generator.models.items():
            path = root / f"label_{label}.pkl"
            model.save(path)
            checkpoints.append(path.name)
    else:
        expected = root / "model.pt"
        if not expected.exists():
            raise FileNotFoundError(f"missing checkpoint {expected}")
        checkpoints.append(expected.name)
    return checkpoints


def run_smoke(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    raw = yaml.safe_load(config_bytes)
    validate_smoke_scope(
        mode=args.mode,
        generators=args.generators,
        seeds=args.seeds,
        kappas=args.kappas,
        n_train=int(raw["data"]["n_train"]),
        steps=args.steps,
    )
    generator_name = args.generators[0]
    seed = args.seeds[0]
    kappa = args.kappas[0]
    scenario = args.scenario
    if scenario not in raw["scenarios"]:
        raise ValueError(f"scenario {scenario} absent from smoke config")
    device = args.device
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("learned smoke requires an available CUDA device")
    torch.cuda.set_device(torch.device(device))
    torch.cuda.reset_peak_memory_stats()

    config = BenchmarkConfig.from_mapping(raw, scenario, kappa)
    dgp_started = time.perf_counter()
    bundle = generate_benchmark(config, int(raw["seeds"]["base"]) + seed)
    dgp_seconds = time.perf_counter() - dgp_started
    plan = SamplingPlan.from_batch(bundle.test)
    output = (
        Path(args.artifact_root)
        / generator_name
        / scenario
        / f"kappa_{kappa:.2f}"
        / f"seed_{seed}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"smoke output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir()

    model_config: dict[str, Any] = {
        "device": device,
        "cuda": True,
        "steps": args.steps,
        "checkpoint_path": str(checkpoint_root / "model.pt"),
    }
    if generator_name in ("conditional_ctgan", "conditional_tvae"):
        valid_rows = int(bundle.train.valid_mask.sum())
        epochs = max(1, int(np.ceil(args.steps * 500 / valid_rows)))
        model_config.update({"epochs": epochs, "batch_size": 500})
        model_config.pop("checkpoint_path")
    elif generator_name == "neural_sequence_baseline":
        model_config.update({"hidden_size": 64, "lr": 1e-3})
    else:
        model_config.update(
            {
                "d_model": 64,
                "n_layers": 1,
                "lr": 1e-3,
                "tau": bundle.metadata["tau"],
                "window_width": float(raw["data"]["window_width"]),
                "temperature": float(raw["data"]["soft_g_temperature"]),
                "coherence_lambda": 0.0,
            }
        )

    generator = generator_for(generator_name)
    fit_started = time.perf_counter()
    generator.fit(bundle.train, config=model_config, seed=seed)
    fit_seconds = time.perf_counter() - fit_started
    checkpoint_files = save_model_checkpoints(
        generator,
        generator_name,
        checkpoint_root,
    )
    sample_started = time.perf_counter()
    sample = generator.sample(plan, seed=seed + 10_000)
    sample_seconds = time.perf_counter() - sample_started
    sample_path = output / "sample.npz"
    save_sample(sample_path, sample)
    metrics = basic_metrics(sample, bundle.test)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    runtime = {
        "dgp_cpu_seconds": dgp_seconds,
        "fit_gpu_seconds": fit_seconds,
        "sample_gpu_seconds": sample_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "process_peak_rss_kb": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "requested_training_steps": args.steps,
        "adapter_epochs": model_config.get("epochs"),
    }
    (output / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True)
        .strip()
    )
    manifest = {
        "schema_version": "benchmark-v2.4-learned-smoke",
        "status": "complete",
        "mode": "smoke",
        "generator": generator_name,
        "scenario": scenario,
        "kappa": kappa,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "dataset_hash": hash_batch(bundle.train),
        "config_hash": sha256_bytes(config_bytes),
        "sampling_plan_id": plan.plan_id,
        "model_seed": seed,
        "sampling_seed": seed + 10_000,
        "bootstrap_seed": None,
        "sample_path": sample_path.name,
        "checkpoint_paths": [
            str(Path("checkpoints") / name) for name in checkpoint_files
        ],
        "metrics_path": "metrics.json",
        "runtime_path": "runtime.json",
        "full_experiment_authorized": False,
    }
    ArtifactStore(output).complete(manifest)
    return {"output": str(output), "manifest": manifest, "runtime": runtime}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke",), required=True)
    parser.add_argument(
        "--config",
        default="configs/benchmark_v2/smoke.yaml",
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--scenario", default="joint_semimarkov_v2b")
    parser.add_argument("--kappas", type=float, nargs="+", required=True)
    parser.add_argument("--generators", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
