"""
Generate ablation table at κ=1.0 (Table 2 in paper).

Columns: vel | gap | fanout | amt | mean  (+ n_seeds)

Rows (ablation stages):
  C0_floor        real vs real noise floor
  row_shuffle     marginal-only baseline (=C1)
  cof_classcond   λ=2.0, guidance=1.0, no discrete diffusion  (class-conditional)
  cof_cfg_nomask  λ=0.0, guidance=3.0, no absorbing masking   (CFG only)
  cof_diff_p10    λ=0.0, guidance=3.0, p_max=1.0 (unstable)
  cof_diff_p07    λ=0.0, guidance=3.0, p_max=0.7 (final)      ← proposed
  C1_upper        label-shuffled upper bound
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
R1   = ROOT / "results" / "kappa_1.00"

CHANNELS = ["vel", "gap", "fanout", "amt"]

def mean_ci(df, col):
    v = df[col].values
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)) * 1.96 if len(v) > 1 else (v[0], 0.0)

def row_stats(df, label, note=""):
    row = {"label": label, "note": note, "n": len(df)}
    ch_means = []
    for ch in CHANNELS:
        m, _ = mean_ci(df, f"coh_gap_{ch}")
        row[ch] = m
        ch_means.append(m)
    row["mean"] = np.mean(ch_means)
    return row

# ── Load data ──────────────────────────────────────────────────────────────────
k1   = pd.read_csv(R1 / "kappa_curve.csv")
cfg3 = pd.read_csv(R1 / "cfg_guidance3.csv")
disc = pd.read_csv(R1 / "discrete_diffusion.csv")
mk07 = pd.read_csv(R1 / "discrete_diffusion_maxmask07.csv")
cond = pd.read_csv(R1 / "diag_classconditional.csv")

c0   = k1[k1.generator == "C0_real_split"]
c1   = k1[k1.generator == "C1_label_shuf"]
rs   = k1[k1.generator == "row_shuffle"]
cof5_g3 = k1[k1.generator == "cof_cfg"]

# Load g=2.0 results (final recommended)
g2_path = R1 / "guidance2_sweep.csv"
cof5_g2 = pd.read_csv(g2_path)[lambda df: df.generator == "cof_cfg"] if g2_path.exists() else cof5_g3

# ── Build table ────────────────────────────────────────────────────────────────
rows = [
    row_stats(c0,      "C0 (real vs real)",                   note="noise floor"),
    row_stats(rs,      "Row-shuffle",                         note="no sequence model"),
    row_stats(cond,    "CoF  λ=2.0, g=1.0",                  note="class-cond only"),
    row_stats(cfg3,    "CoF  CFG g=3.0, no mask",             note="CFG only, no diffusion"),
    row_stats(disc,    "CoF  CFG + AbsDiff p_max=1.0",        note="max-mask unstable"),
    row_stats(mk07,    "CoF  CFG + AbsDiff p_max=0.7 (3s)",   note="3-seed pilot, g=3.0"),
    row_stats(cof5_g3, "CoF  CFG + AbsDiff p_max=0.7 (5s)",   note="5 seeds, g=3.0"),
    row_stats(cof5_g2, "CoF  CFG + AbsDiff p_max=0.7 (5s) ★", note="final, g=2.0"),
    row_stats(c1,      "C1 (label shuffle)",                  note="upper bound"),
]

df_ab = pd.DataFrame(rows)

# ── Pretty-print ───────────────────────────────────────────────────────────────
HDR = f"{'Variant':<38} {'vel':>6} {'gap':>6} {'fanout':>7} {'amt':>7} {'mean':>7}  n  note"
SEP = "-" * len(HDR)
print(f"\n{'Ablation Table — κ=1.0':^{len(HDR)}}")
print(SEP)
print(HDR)
print(SEP)
for _, r in df_ab.iterrows():
    print(f"{r['label']:<38} {r['vel']:>6.4f} {r['gap']:>6.4f} {r['fanout']:>7.4f} "
          f"{r['amt']:>7.4f} {r['mean']:>7.4f}  {r['n']}  {r['note']}")
print(SEP)

# ── Per-channel seed detail for final g=2.0 ───────────────────────────────────
print("\n[Per-seed detail — final 5-seed CoF g=2.0 at κ=1.0]")
print(f"{'seed':>5} {'vel':>7} {'gap':>7} {'fanout':>8} {'amt':>8} {'mean':>7}")
for _, r in cof5_g2.iterrows():
    print(f"{int(r.seed):>5} {r.coh_gap_vel:>7.4f} {r.coh_gap_gap:>7.4f} "
          f"{r.coh_gap_fanout:>8.4f} {r.coh_gap_amt:>8.4f} {r.coh_gap_mean:>7.4f}")
seed2 = cof5_g2[cof5_g2.seed == 2]
if len(seed2):
    rest = cof5_g2[cof5_g2.seed != 2]
    print(f"→ seed2 amt: {seed2.coh_gap_amt.values[0]:.4f} (others ≤ {rest.coh_gap_amt.max():.4f})")

# ── LaTeX output ───────────────────────────────────────────────────────────────
OUT = ROOT / "figs" / "ablation_table.tex"
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Ablation: coherence-gap at $\kappa=1.0$ (lower is better). "
    r"C0 = real vs.\ real noise floor; C1 = label-shuffled upper bound. "
    r"$\star$ = proposed method.}",
    r"\label{tab:ablation}",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"\textbf{Variant} & vel & gap & fanout & amt & \textbf{mean} & $n$ \\",
    r"\midrule",
]
for _, r in df_ab.iterrows():
    star = r"$\star$" if "★" in r["label"] else ""
    lbl  = r["label"].replace("★", "").strip()
    # Replace ASCII lambda with LaTeX
    lbl  = lbl.replace("λ", r"$\lambda$")
    lbl  = lbl.replace("_", r"\_")
    if "C0" in r["label"] or "C1" in r["label"]:
        lbl = rf"\textit{{{lbl}}}"
    lines.append(
        rf"  {lbl}{star} & {r['vel']:.4f} & {r['gap']:.4f} & {r['fanout']:.4f} "
        rf"& {r['amt']:.4f} & \textbf{{{r['mean']:.4f}}} & {int(r['n'])} \\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"\nLaTeX saved → {OUT}")

# ── CSV for record ─────────────────────────────────────────────────────────────
CSV_OUT = ROOT / "results" / "ablation_table.csv"
df_ab.to_csv(CSV_OUT, index=False)
print(f"CSV saved  → {CSV_OUT}")
