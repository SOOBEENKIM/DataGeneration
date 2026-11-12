"""Standalone research figure; dots retain fit/draw identities."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/argn_state_first_v1/evaluation"


def main():
    data = pd.read_csv(DOCS / "metrics.csv")
    reference = pd.read_csv(DOCS / "real_only_variability.csv")
    real = data[data.run.eq("real_validation")].iloc[0]
    models = data[data.arm.isin(["B", "B_S"])]
    seeds = sorted(models.fit_seed.unique())
    draws = sorted(models.generation_seed.unique())
    colors = ["#2563a6", "#d66a27"]
    panels = [
        ("fraud_rate", "Fraud prevalence (%)", 100, True),
        ("mean_run_length", "Observed fraud-run mean length", 1, True),
        ("termination_rate", "Observed-next termination (%)", 100, True),
        ("fraud_ratio_log_w1", "Fraud amount / past median: log-W1", 1, False),
        ("normal_amount_log_w1", "Normal amount: log-W1", 1, False),
        ("customer_length_log_w1", "Customer length: log-W1", 1, False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (metric, title, scale, raw_metric) in zip(axes.flat, panels):
        for x, arm in enumerate(["B", "B_S"]):
            arm_data = models[models.arm.eq(arm)]
            for i, seed in enumerate(seeds):
                for j, draw in enumerate(draws):
                    row = arm_data[(arm_data.fit_seed == seed) & (arm_data.generation_seed == draw)].iloc[0]
                    offset = (i - .5) * .16 + (j - .5) * .045
                    ax.scatter(x + offset, row[metric] * scale, c=colors[i], marker=["o", "s"][j], s=50, zorder=3)
            ax.scatter(x, arm_data[metric].mean() * scale, marker="D", c="black", s=34, zorder=4)
        if raw_metric:
            ax.axhline(real[metric] * scale, color="#555555", linestyle="--", linewidth=1.2)
            key = "continuation_rate" if metric == "termination_rate" else metric
            match = reference[(reference.metric == key) & reference.kind.eq("customer-bootstrap scalar")]
            if len(match):
                lo, hi = match.iloc[0][["ci_low", "ci_high"]]
                if metric == "termination_rate":
                    lo, hi = 1-hi, 1-lo
                ax.axhspan(lo * scale, hi * scale, color="#d8d8d8", alpha=.45)
        else:
            ax.axhline(0, color="#aaaaaa", linewidth=.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 1], ["B: ARGN", "B+S: past state"])
        ax.set_xlim(-.45, 1.45)
        ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color=colors[i], marker="o", linestyle="", label=f"Fit seed {int(seed)}") for i, seed in enumerate(seeds)]
    handles += [Line2D([], [], color="black", marker="D", linestyle="", label="Descriptive mean"),
                Line2D([], [], color="#555555", linestyle="--", label="Real development value")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .96), ncol=4, frameon=False)
    fig.suptitle("Sparkov: fresh ARGN baseline vs explicit past-state features", fontsize=15, y=.995)
    fig.text(.04, .015, "Each arm: 2 independent fits x 2 free-generation draws. Shading: real customer-bootstrap 95% intervals.\n"
             "Distances: lower is closer. Four dots are not four independent fitted models. No real future lengths are supplied.", fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .915])
    fig.savefig(DOCS / "comparison.png", dpi=180)
    fig.savefig(DOCS / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
