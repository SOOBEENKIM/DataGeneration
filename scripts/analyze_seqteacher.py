"""
Interim / post-hoc analysis for amlsim_seqteacher.csv — 3-arm attribution.

3 arms:
  amount_only : use_time=False, use_recv=False  (pure amount coherence)
  amount_time : use_time=True,  use_recv=False  (amount + timing)
  full        : use_time=True,  use_recv=True   (full window)

Attribution (delta_SYNTH only — delta_aug saturated at AMLSim base≈0.9995):
  Δtiming  = amount_time − amount_only   (timing channel contribution)
  Δrecv    = full − amount_time          (receiver channel contribution)
  Δwindow  = full − amount_only          (total window, = Δtiming + Δrecv)

Judgment:
  Δwindow > 0.02  → C2b supported (window/temporal above amount)
  Δwindow ≤ 0.02  → C2a only (amount drives gain); headline narrows
  Δtiming > Δrecv → timing is dominant C2b channel (consistent with AMLSim diag +0.061)
  Δrecv   > Δtiming → receiver is dominant (less expected for AMLSim)

Early-stop: if halfway through and Δwindow consistently ≤ 0, consider stopping.
"""

import sys, csv, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent
TOTAL_RUNS = 5 * 3 * 3  # lambdas × arms × seeds = 45


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default=str(ROOT / "results" / "amlsim_seqteacher.csv"))
    p.add_argument("--n_total", type=int, default=TOTAL_RUNS)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    n_done = len(rows)
    print(f"Progress: {n_done}/{args.n_total} runs ({100*n_done/args.n_total:.0f}%)\n")

    if n_done == 0:
        print("No data yet.")
        return

    # Group by (teacher_type, lambda)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["teacher_type"], float(r["coh_lambda"]))].append(r)

    lambdas = sorted(set(float(r["coh_lambda"]) for r in rows))
    ttypes  = sorted(set(r["teacher_type"] for r in rows))

    def _m(g, c):
        vals = [float(r[c]) for r in g]
        return np.mean(vals), np.std(vals)

    # ── Per-group table ───────────────────────────────────────────────────────
    print(f"{'teacher':>12} {'λ':>5} {'n':>2}  "
          f"{'delta_SYNTH':>12}  {'coh_gap_fanout':>14}  "
          f"{'coh_gap_amt':>11}  {'teacher_ap':>10}  {'fraud_synth':>11}")
    print("─" * 95)

    by_lam_type = {}
    for ttype in ["amount_only", "amount_time", "full"]:
        printed_any = False
        for lam in lambdas:
            key = (ttype, lam)
            g = groups.get(key, [])
            if not g:
                continue
            printed_any = True
            ds_m, ds_s = _m(g, "delta_synth")
            cf_m, cf_s = _m(g, "coh_gap_fanout")
            ca_m, ca_s = _m(g, "coh_gap_amt")
            ta_m, _    = _m(g, "teacher_auprc")
            fs_m, _    = _m(g, "fraud_rate_synth")
            print(f"{ttype:>12} {lam:>5.1f} {len(g):>2}  "
                  f"{ds_m:>+10.4f}±{ds_s:.4f}  "
                  f"{cf_m:>12.6f}±{cf_s:.6f}  "
                  f"{ca_m:>9.6f}±{ca_s:.6f}  "
                  f"{ta_m:>10.4f}  "
                  f"{fs_m:>11.4f}")
            by_lam_type[key] = {
                "delta_synth_mean":   ds_m, "delta_synth_std": ds_s,
                "coh_gap_fanout_mean": cf_m, "coh_gap_amt_mean": ca_m,
                "teacher_auprc_mean": ta_m, "n": len(g),
            }
        if printed_any:
            print()

    # ── 3-arm attribution table ───────────────────────────────────────────────
    any_full    = any(k[0] == "full"         for k in by_lam_type)
    any_time    = any(k[0] == "amount_time"  for k in by_lam_type)
    any_amtonly = any(k[0] == "amount_only"  for k in by_lam_type)

    if any_full and any_time and any_amtonly:
        print("── 3-Arm Attribution (delta_SYNTH, C2b isolation) ──")
        print("  Δtiming = amount_time − amount_only")
        print("  Δrecv   = full − amount_time")
        print("  Δwindow = full − amount_only  (= Δtiming + Δrecv)")
        print()
        print(f'  {"λ":>5}  {"full":>8}  {"amt_time":>10}  {"amt_only":>10}  '
              f'{"Δtiming":>9}  {"Δrecv":>8}  {"Δwindow":>9}  {"C2b?":>18}')
        print("  " + "─" * 90)

        w_diffs, t_diffs, r_diffs = [], [], []
        for lam in lambdas:
            fk = ("full",         lam)
            tk = ("amount_time",  lam)
            ak = ("amount_only",  lam)
            missing = [k[0] for k in [fk, tk, ak] if k not in by_lam_type]
            if missing:
                print(f'  {lam:>5.1f}  (missing: {missing})')
                continue
            fd = by_lam_type[fk]["delta_synth_mean"]
            td = by_lam_type[tk]["delta_synth_mean"]
            ad = by_lam_type[ak]["delta_synth_mean"]
            dt = td - ad
            dr = fd - td
            dw = fd - ad
            if lam > 0:
                w_diffs.append(dw); t_diffs.append(dt); r_diffs.append(dr)
            if lam == 0.0:
                c2b = "(baseline)"
            elif dw > 0.03:
                c2b = "✓ C2b (window>amt)"
            elif dw > 0.01:
                c2b = "~ weak C2b"
            elif dw < -0.02:
                c2b = "✗ window<amtonly"
            else:
                c2b = "C2a only (tied)"
            print(f'  {lam:>5.1f}  {fd:>+8.4f}  {td:>+10.4f}  {ad:>+10.4f}  '
                  f'{dt:>+9.4f}  {dr:>+8.4f}  {dw:>+9.4f}  {c2b:>18}')

        if w_diffs:
            mw = np.mean(w_diffs); mt = np.mean(t_diffs); mr = np.mean(r_diffs)
            print(f'\n  Mean across λ>0:  Δtiming={mt:+.4f}  Δrecv={mr:+.4f}  Δwindow={mw:+.4f}')

            # Dominant channel
            if mt > 0.01 and mt > mr:
                channel = f"TIMING dominates (Δtiming={mt:+.4f})"
            elif mr > 0.01 and mr > mt:
                channel = f"RECEIVER dominates (Δrecv={mr:+.4f})"
            elif mt > 0.01 and mr > 0.01:
                channel = f"BOTH channels (timing={mt:+.4f}, recv={mr:+.4f})"
            else:
                channel = "no clear channel"

            print()
            print("  ── JUDGMENT ──")
            if mw > 0.03:
                print(f"  C2b SUPPORTED: window adds {mw:+.4f} above amount-only")
                print(f"    Channel: {channel}")
                print("    → thesis C2b headline valid. Run to completion.")
            elif mw > 0.01:
                print(f"  WEAK C2b: window adds {mw:+.4f} (marginal, may not be significant)")
                print(f"    Channel: {channel}")
                print("    → run to completion; check if pattern holds across all seeds")
            else:
                print(f"  C2a ONLY: Δwindow={mw:+.4f} ≤ 0.01 — window NOT above amount-only")
                print("    → timing/receiver coherence doesn't improve downstream beyond amount")
                print("    → C2b weak/absent; consider EARLY STOP and headline narrowing")
                if n_done >= args.n_total // 2:
                    print("    [EARLY STOP RECOMMENDED]")

    elif any_amtonly and not any_full and not any_time:
        # Only amount_only done so far
        # Full grid: 5 lambdas × 3 arms × 3 seeds = 45 runs; amount_only = first 15
        print("── Partial: only 'amount_only' arm in CSV so far ──")
        print("  Grid order: amount_only(runs 1-15) → amount_time(16-30) → full(31-45)")
        print(f"  amount_only runs done: {n_done}/15")
        print()
        print("  amount_only delta_synth by lambda (pre-attribution preview):")
        base_ds = by_lam_type.get(("amount_only", 0.0), {}).get("delta_synth_mean")
        for lam in lambdas:
            ak = ("amount_only", lam)
            if ak not in by_lam_type: continue
            ad = by_lam_type[ak]["delta_synth_mean"]
            vs = f"vs λ=0: {ad - base_ds:+.4f}" if base_ds is not None and lam > 0 else "(λ=0 base)"
            print(f"  λ={lam:>4.1f}  Δsynth={ad:>+7.4f}  {vs}")
    else:
        have = sorted(set(k[0] for k in by_lam_type))
        print(f"  Arms available: {have} — need all 3 for attribution.")


if __name__ == "__main__":
    main()
