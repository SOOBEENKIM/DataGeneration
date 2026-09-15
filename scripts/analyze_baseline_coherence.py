"""
Analyzer for baseline_coherence.csv — C3 verification.

Reporting:
  1. Full table: all generators with coherence_gap metrics and 95% CI
  2. Expected ordering check: C0 < CoF < row-baselines (C3 GO criterion)
  3. Actual ordering vs hypothesis
  4. Coupling strength diagnostic (C1 - C0 = signal in real data)
  5. GO/NO-GO judgment with interpretation

C3 GO criterion:
  - coh_gap_mean(CoF) < coh_gap_mean(all row-baselines, P2) AND
  - Δgap CI lower bound > 0 (CoF clearly better)

C3 negative finding interpretation:
  - If CoF ≥ row-baselines: coherence loss doesn't improve (or hurts) the coupling
  - Check coupling_signal = gap(C1) - gap(C0): if < 0.005, coupling is too weak to measure
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "results" / "baseline_coherence.csv"))
    p.add_argument("--n_bootstrap", type=int, default=2000)
    return p.parse_args()


def load_rows(path):
    if not Path(path).exists():
        print(f"CSV not found: {path}")
        return []
    return list(csv.DictReader(open(path)))


def float_or_nan(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    if not rows:
        return

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("── Coherence Gap Summary (lower = better) ──")
    print(f"{'Generator':>16} {'Assembly':>6} {'Seeds':>5}  "
          f"{'coh_vel':>8}  {'coh_gap':>8}  {'coh_fanout':>10}  {'coh_amt':>8}  "
          f"{'mean':>8}  {'CI_lo':>8}  {'CI_hi':>8}  {'fraud%':>6}")
    print("─" * 110)

    by_gen = defaultdict(list)
    for r in rows:
        key = (r["generator"], r["assembly"])
        by_gen[key].append(r)

    gen_order = ["C0_real_split", "C1_label_shuf", "cof", "cof_cfg",
                 "row_shuffle", "ctgan", "tvae"]
    printed_keys = set()
    gen_stats = {}   # key → {"mean": float, "ci_lo": float, "ci_hi": float}

    for gen in gen_order:
        for asm in ["native", "P2", "P1"]:
            key = (gen, asm)
            if key not in by_gen or key in printed_keys:
                continue
            printed_keys.add(key)
            rs = by_gen[key]

            # Average across seeds (for multi-seed entries)
            def avg(col):
                vals = [float_or_nan(r.get(col, "")) for r in rs
                        if not np.isnan(float_or_nan(r.get(col, "")))]
                return float(np.mean(vals)) if vals else float("nan")

            m_vel    = avg("coh_gap_vel")
            m_gap    = avg("coh_gap_gap")
            m_fanout = avg("coh_gap_fanout")
            m_amt    = avg("coh_gap_amt")
            m_mean   = avg("coh_gap_mean")
            m_ci_lo  = avg("coh_gap_mean_ci_lo")
            m_ci_hi  = avg("coh_gap_mean_ci_hi")
            m_fraud  = avg("fraud_rate_synth")

            # 3-channel mean (vel + fanout + amt): comparable with CoF CSV which lacks gap
            vals_3ch = [v for v in [m_vel, m_fanout, m_amt] if not np.isnan(v)]
            m_mean_3ch = float(np.mean(vals_3ch)) if vals_3ch else float("nan")

            def fmt(v):
                return f"{v:.5f}" if not np.isnan(v) else "   NaN "

            print(f"{gen:>16} {asm:>6} {len(rs):>5}  "
                  f"{fmt(m_vel):>8}  {fmt(m_gap):>8}  {fmt(m_fanout):>10}  {fmt(m_amt):>8}  "
                  f"{fmt(m_mean):>8}  {fmt(m_ci_lo):>8}  {fmt(m_ci_hi):>8}  {m_fraud*100:>5.2f}%")

            gen_stats[key] = {
                "mean": m_mean, "ci_lo": m_ci_lo, "ci_hi": m_ci_hi,
                "vel": m_vel, "gap_ch": m_gap, "fanout": m_fanout, "amt": m_amt,
                "mean_3ch": m_mean_3ch,   # comparable with CoF (no gap channel)
                "fraud_rate": m_fraud, "n_seeds": len(rs),
            }

    # Print any remaining generators not in gen_order
    for key, rs in by_gen.items():
        if key in printed_keys:
            continue
        gen, asm = key
        m_vel   = np.nanmean([float_or_nan(r.get("coh_gap_vel","")) for r in rs])
        m_fanout= np.nanmean([float_or_nan(r.get("coh_gap_fanout","")) for r in rs])
        m_amt   = np.nanmean([float_or_nan(r.get("coh_gap_amt","")) for r in rs])
        m_mean  = np.nanmean([float_or_nan(r["coh_gap_mean"]) for r in rs])
        m_ci_lo = np.nanmean([float_or_nan(r["coh_gap_mean_ci_lo"]) for r in rs])
        m_ci_hi = np.nanmean([float_or_nan(r["coh_gap_mean_ci_hi"]) for r in rs])
        m_fraud = np.nanmean([float_or_nan(r["fraud_rate_synth"]) for r in rs])
        m_mean_3ch = float(np.nanmean([v for v in [m_vel, m_fanout, m_amt] if not np.isnan(v)]))
        print(f"{gen:>16} {asm:>6} {len(rs):>5}  {'':>8}  {'':>8}  {'':>10}  {'':>8}  "
              f"{m_mean:.5f}  {m_ci_lo:.5f}  {m_ci_hi:.5f}  {m_fraud*100:>5.2f}%")
        gen_stats[key] = {
            "mean": m_mean, "ci_lo": m_ci_lo, "ci_hi": m_ci_hi,
            "mean_3ch": m_mean_3ch,
            "fraud_rate": m_fraud, "n_seeds": len(rs),
        }

    print()

    # ── Coupling strength diagnostic ──────────────────────────────────────────
    c0 = gen_stats.get(("C0_real_split", "native"))
    c1 = gen_stats.get(("C1_label_shuf", "native"))
    cof = gen_stats.get(("cof", "native"))

    if c0 and c1:
        coupling_signal = c1["mean"] - c0["mean"]
        print(f"── Coupling Strength Diagnostic ──")
        print(f"  C0 (real-vs-real floor) :  {c0['mean']:.5f}")
        print(f"  C1 (label-shuffle UB)   :  {c1['mean']:.5f}")
        print(f"  Signal = C1 - C0        : +{coupling_signal:.5f}")
        if coupling_signal < 0.002:
            print(f"  ⚠  Signal < 0.002: g→fraud coupling is negligibly weak in real data.")
            print(f"     Coherence_gap cannot discriminate generators in this regime.")
        elif coupling_signal < 0.005:
            print(f"  ⚠  Signal < 0.005: g→fraud coupling is weak.")
            print(f"     Any generator matching prevalence will score near C0.")
        else:
            print(f"  ✓ Signal ≥ 0.005: coupling detectable; metric is discriminative.")
        print()

    # ── C3 ordering check ─────────────────────────────────────────────────────
    row_baselines = {}
    for key in [("row_shuffle", "P2"), ("ctgan", "P2"), ("tvae", "P2"),
                ("row_shuffle", "P1"), ("ctgan", "P1"), ("tvae", "P1")]:
        if key in gen_stats:
            row_baselines[key] = gen_stats[key]

    cof_cfg = gen_stats.get(("cof_cfg", "native"))

    print(f"── C3 Hypothesis Check ──")
    print(f"  Hypothesis: coh_gap_mean(CoF) < coh_gap_mean(row-baselines)")
    print(f"  Note: Old-CoF CSV lacks gap channel → comparing on 3ch (vel+fanout+amt) for old-CoF.")
    print(f"        CFG-CoF has all 4 channels — using 4ch mean for cof_cfg.")
    print()

    if not cof and not cof_cfg:
        print("  [CoF / CFG-CoF] Neither available in CSV.")
        return

    if cof:
        cof_mean3 = cof.get("mean_3ch", cof["mean"])
        print(f"  Old-CoF 3ch mean : {cof_mean3:.5f}  CI=[{cof['ci_lo']:.5f},{cof['ci_hi']:.5f}]")
    if cof_cfg:
        print(f"  CFG-CoF 4ch mean : {cof_cfg['mean']:.5f}  "
              f"CI=[{cof_cfg['ci_lo']:.5f},{cof_cfg['ci_hi']:.5f}]  "
              f"(vel={cof_cfg.get('vel',float('nan')):.5f} "
              f"gap={cof_cfg.get('gap_ch',float('nan')):.5f} "
              f"fanout={cof_cfg.get('fanout',float('nan')):.5f} "
              f"amt={cof_cfg.get('amt',float('nan')):.5f})")
    print()

    # Old-CoF vs row-baselines (skip if old CoF absent)
    any_go = False
    n_nogo = 0
    if cof:
        for key, stats in row_baselines.items():
            gen, asm = key
            base_mean3 = stats.get("mean_3ch", stats["mean"])
            delta = base_mean3 - cof_mean3
            # Approximate CI on Δgap using independent CI bounds
            delta_ci_lo = stats["ci_lo"] - cof["ci_hi"]
            delta_ci_hi = stats["ci_hi"] - cof["ci_lo"]

            if delta > 0 and delta_ci_lo > 0:
                verdict = "✓ GO (CI>0)"
                any_go = True
            elif delta > 0:
                verdict = "~ Directional (CI∋0)"
            else:
                verdict = "✗ NO-GO"
                n_nogo += 1

            print(f"  {gen:>12}/{asm:<3}: Δgap(3ch)={delta:+.5f}  "
                  f"approx CI=[{delta_ci_lo:+.5f},{delta_ci_hi:+.5f}]  {verdict}")
            print(f"    baseline: 3ch={base_mean3:.5f}  4ch={stats['mean']:.5f}  "
                  f"CI=[{stats['ci_lo']:.5f},{stats['ci_hi']:.5f}]")
        print()

    # ── Final judgment ────────────────────────────────────────────────────────
    print("── C3 Final Judgment ──")
    coupling_signal = (c1["mean"] - c0["mean"]) if (c0 and c1) else float("nan")

    print(f"  Coupling signal (C1−C0): {coupling_signal:+.5f}")
    print(f"  C0 floor:  {c0['mean']:.5f}" if c0 else "  C0: N/A")
    print(f"  C1 UB:     {c1['mean']:.5f}" if c1 else "  C1: N/A")
    if cof:
        print(f"  CoF 3ch:   {cof_mean3:.5f}")

    if cof:
        if any_go:
            print(f"\n  ✓ C3 CONFIRMED (old-CoF): CoF 3ch < row-baselines with CI>0.")
            print(f"    CoF preserves behavior→fraud coupling better than row-level generators.")
        elif n_nogo > 0:
            print(f"\n  ✗ C3 NOT CONFIRMED (old-CoF):")
            print(f"    CoF 3ch={cof_mean3:.5f} is NOT lower than row-level baselines.")
            if c1 and cof_mean3 > c1.get("mean_3ch", c1["mean"]):
                print(f"    ⚠  CoF exceeds even label-shuffle UB — coherence loss introduces")
                print(f"       spurious coupling stronger than the real data signal.")
        else:
            print(f"\n  ~ C3 INCONCLUSIVE (old-CoF): Insufficient baselines.")

    # ── CFG-CoF C3 check (separate, uses 4ch) ────────────────────────────────
    if cof_cfg:
        print()
        print("── CFG-CoF C3 Check (class-conditional, 4ch) ──")
        cfg_mean = cof_cfg["mean"]
        cfg_ci_lo = cof_cfg["ci_lo"]
        cfg_ci_hi = cof_cfg["ci_hi"]
        print(f"  CFG-CoF 4ch mean: {cfg_mean:.5f}  CI=[{cfg_ci_lo:.5f},{cfg_ci_hi:.5f}]")
        if c1:
            print(f"  C1 UB (4ch):      {c1['mean']:.5f}")
        print()

        cfg_any_go = False
        cfg_n_nogo = 0
        for key, stats in row_baselines.items():
            gen, asm = key
            base_mean = stats["mean"]   # 4ch for CTGAN/TVAE/row_shuffle
            delta = base_mean - cfg_mean
            delta_ci_lo = stats["ci_lo"] - cfg_ci_hi
            delta_ci_hi = stats["ci_hi"] - cfg_ci_lo
            if delta > 0 and delta_ci_lo > 0:
                v = "✓ GO (CI>0)"
                cfg_any_go = True
            elif delta > 0:
                v = "~ Directional (CI∋0)"
            else:
                v = "✗ NO-GO"
                cfg_n_nogo += 1
            print(f"  {gen:>12}/{asm:<3}: Δgap(4ch)={delta:+.5f}  "
                  f"approx CI=[{delta_ci_lo:+.5f},{delta_ci_hi:+.5f}]  {v}")

        print()
        if cfg_any_go:
            print(f"  ✓ C3 CONFIRMED for CFG-CoF: 4ch < row-baselines with CI>0.")
            print(f"    CFG-CoF preserves behavior→fraud coupling better than row-level generators.")
            if c1 and cfg_mean <= c1["mean"]:
                print(f"    CFG-CoF ({cfg_mean:.5f}) ≤ C1 shuffle UB ({c1['mean']:.5f}) — near floor.")
        elif cfg_n_nogo == 0:
            print(f"  ~ CFG-CoF C3 DIRECTIONAL: all Δgap>0 but CIs straddle 0.")
        else:
            print(f"  ✗ CFG-CoF C3 NOT CONFIRMED: some baselines ≤ CFG-CoF.")

    print()


if __name__ == "__main__":
    main()
