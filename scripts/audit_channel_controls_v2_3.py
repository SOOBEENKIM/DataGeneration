from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from benchmarks.types import SequenceBatch
from eval.channel_controls_v2 import channel_only_summaries
from eval.joint_association_v2 import association_statistics


def load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as values:
        return SequenceBatch(
            **{key: values[key] for key in SequenceBatch.__dataclass_fields__}
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main_v2_3_candidate.yaml")
    parser.add_argument("--data-root", default="data/benchmark_v2_2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2_3/channel_controls")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for kappa in raw["coupling"]["kappas"]:
        folder = (
            Path(args.data_root)
            / "joint_semimarkov_v2b"
            / f"kappa_{float(kappa):.2f}"
        )
        batch = load_batch(folder / "test.npz")
        meta = json.loads((folder / "meta.json").read_text())
        summaries = channel_only_summaries(
            batch,
            tau=np.asarray(meta["tau"]),
            window_width=float(raw["data"]["window_width"]),
        )
        for control, values in summaries.items():
            rows.append(
                {
                    "kappa": float(kappa),
                    "control": control,
                    **association_statistics(values, batch.y_entity),
                }
            )
    with (output / "channel_only_negative_controls.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    threshold = float(
        raw["evaluation"][
            "channel_negative_control_abs_standardized_delta_max"
        ]
    )
    maximum = max(abs(float(row["standardized_delta"])) for row in rows)
    (output / "channel_control_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "channel_only_negative_controls_v2.3-candidate",
                "status": "PASS" if maximum <= threshold else "FAIL",
                "maximum_abs_standardized_delta": maximum,
                "threshold": threshold,
                "fanout_excluded": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
