from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark


def _save_batch(path: Path, batch) -> None:
    np.savez_compressed(
        path,
        x_num=batch.x_num,
        dt_bin=batch.dt_bin,
        x_cat=batch.x_cat,
        valid_mask=batch.valid_mask,
        y_entity=batch.y_entity,
        lengths=batch.lengths,
        entity_ids=batch.entity_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="data/benchmark_v2")
    args = parser.parse_args()
    raw_bytes = Path(args.config).read_bytes()
    raw = yaml.safe_load(raw_bytes)
    root = Path(args.output_root)
    config_schema = str(raw.get("schema_version", "benchmark_v2.0"))
    manifest = {
        "schema_version": f"{config_schema}-manifest",
        "benchmark_schema_version": config_schema,
        "config_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "datasets": [],
    }
    for scenario in raw["scenarios"]:
        reference_cfg = BenchmarkConfig.from_mapping(raw, scenario, 0.0)
        reference = generate_benchmark(reference_cfg, int(raw["seeds"]["base"]))
        edges = np.asarray(reference.metadata["bin_edges"])
        scenario_root = root / scenario
        scenario_root.mkdir(parents=True, exist_ok=True)
        (scenario_root / "binning.json").write_text(
            json.dumps(
                {
                    "fit_split": "train",
                    "reference_kappa": 0.0,
                    "edges": edges.tolist(),
                    "tau": reference.metadata["tau"],
                    "effective_bins": len(edges) - 1,
                },
                indent=2,
            )
            + "\n"
        )
        for kappa in raw["coupling"]["kappas"]:
            cfg = replace(
                BenchmarkConfig.from_mapping(raw, scenario, float(kappa)),
                bin_edges=edges,
            )
            bundle = generate_benchmark(cfg, int(raw["seeds"]["base"]))
            out = scenario_root / f"kappa_{float(kappa):.2f}"
            out.mkdir(parents=True, exist_ok=True)
            _save_batch(out / "train.npz", bundle.train)
            _save_batch(out / "test.npz", bundle.test)
            for split in ("train", "test"):
                latent = bundle.metadata[f"{split}_latent"]
                np.savez_compressed(
                    out / f"{split}_audit_latent.npz",
                    raw_gap=latent["raw_gap"],
                    gap_state=latent["gap_state"],
                    receiver_state=latent["receiver_state"],
                    receiver_repeat=latent["receiver_repeat"],
                )
            meta = {
                "schema_version": f"{config_schema}-dataset",
                "benchmark_schema_version": config_schema,
                "scenario": scenario,
                "kappa": float(kappa),
                "kappa_semantics": (
                    "normalized_persistence_contrast"
                    if scenario.endswith("v2a")
                    else "normalized_cross_channel_synchronization"
                ),
                "padding_side": "right",
                "bin_edges": bundle.metadata["bin_edges"],
                "tau": bundle.metadata["tau"],
                "seed": int(raw["seeds"]["base"]),
                "n_train": len(bundle.train.lengths),
                "n_test": len(bundle.test.lengths),
                "train_seed_metadata": bundle.metadata["train_latent"]["metadata"],
                "test_seed_metadata": bundle.metadata["test_latent"]["metadata"],
            }
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            manifest["datasets"].append(str(out.relative_to(root)))
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
