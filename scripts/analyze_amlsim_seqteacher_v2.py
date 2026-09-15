"""
AMLSim v2 analyzer — strict protocol matching Sparkov (10 seed, bootstrap CI, floor+collapse).

Differences from v1 (analyze_seqteacher.py):
  - Bootstrap CI (2000 resamples) on per-λ Δwindow
  - λ=0 noise floor explicitly reported and separated
  - Collapse flagging post-hoc: amount_only Δsynth < COLLAPSE_THRESH
  - Floor-corrected Δwindow = observed − Δwindow(λ=0)
  - Side-by-side comparison section vs Sparkov null result

Why rerun with 10 seeds:
  AMLSim v1 = 3 seeds, no CI, +0.174 mean (floor/collapse uncorrected)
  Sparkov    = 10 seeds, bootstrap CI, floor+collapse corrected → null
  → Asymmetric protocols make "AMLSim positive / Sparkov null" unjustifiable.
  This script applies identical rigor to both.

Judgment:
  AMLSim CI>0 (floor-corrected, collapse-excluded) → timing-dominant regime C2b
  AMLSim CI∋0 → C2b weak/null in both → diagnostic paper framing
"""

import sys, csv, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent
COLLAPSE_THRESH = -0.3
TOTAL_RUNS_10   = 5 * 3 * 10   # 150 with 10 seeds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default=str(ROOT / "results" / "amlsim_seqteacher.csv"))
    p.add_argument("--sparkov_csv", default=str(ROOT / "results" / "sparkov_seqteacher.csv"))
    p.add_argument("--n_total",     type=int, default=TOTAL_RUNS_10)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    return p.parse_args()


def bootstrap_ci(vals, n=2000, rng=None, alpha=0.05):
    if rng is None:
        rng = np.random.default_rng(42)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))


def main():
    args     = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    n_done   = len(rows)
    n_seeds  = len(set(r["seed"] for r in rows))
    print(f"Progress: {n_done}/{args.n_total} runs  ({n_seeds} unique seeds so far)\n")

    rng    = np.random.default_rng(42)
    groups = defaultdict(list)
    raw    = defaultdict(list)
    for r in rows:
        key = (r["teacher_type"], float(r["coh_lambda"]))
        groups[key].append(r)
        raw[key].append(float(r["delta_synth"]))

    # Post-hoc collapse flag: amount_only Δsynth < COLLAPSE_THRESH
    def is_collapse(r):
        return r["teacher_type"] == "amount_only" and float(r["delta_synth"]) < COLLAPSE_THRESH

    lambdas = sorted(set(float(r["coh_lambda"]) for r in rows))

    # ── Per-group table ───────────────────────────────────────────────────────
    print(f"{'teacher':>12} {'λ':>5} {'n':>3}  "
          f"{'delta_SYNTH':>8}±std  {'95% CI':>18}  {'teacher_ap':>10}  {'collapse':>8}")
    print("─" * 90)

    by_lam_type = {}
    for ttype in ["amount_only", "amount_time", "full"]:
        printed = False
        for lam in lambdas:
            key = (ttype, lam)
            g = groups.get(key, [])
            if not g: continue
            printed = True
            ds    = np.array([float(r["delta_synth"]) for r in g])
            ta    = float(np.mean([float(r["teacher_auprc"]) for r in g]))
            n_col = sum(1 for r in g if is_collapse(r))
            ci_lo, ci_hi = bootstrap_ci(ds, n=args.n_bootstrap, rng=rng)
            ci_str = f'[{ci_lo:+.4f},{ci_hi:+.4f}]'
            col_tag = f"{n_col}/{len(g)}" if n_col > 0 else "—"
            print(f"{ttype:>12} {lam:>5.1f} {len(g):>3}  "
                  f"{ds.mean():>+7.4f}±{ds.std():.4f}  "
                  f"{ci_str:>18}  {ta:>10.4f}  {col_tag:>8}")
            by_lam_type[key] = {
                "ds_mean": ds.mean(), "ds_std": ds.std(),
                "ds_vals": ds,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "teacher_auprc": ta, "n": len(g), "n_col": n_col,
            }
        if printed:
            print()

    # ── λ=0 noise floor ───────────────────────────────────────────────────────
    print("── λ=0 Noise Floor ──")
    for ttype in ["amount_only", "amount_time", "full"]:
        k = (ttype, 0.0)
        if k in by_lam_type:
            d = by_lam_type[k]
            print(f"  {ttype:>12} λ=0: Δsynth={d['ds_mean']:+.4f}±{d['ds_std']:.4f}")
    k_f0 = ("full", 0.0); k_a0 = ("amount_only", 0.0)
    floor_val = 0.0
    if k_f0 in by_lam_type and k_a0 in by_lam_type:
        floor_val = by_lam_type[k_f0]["ds_mean"] - by_lam_type[k_a0]["ds_mean"]
        print(f"  Δwindow(λ=0) = {floor_val:+.4f}  ← noise floor to subtract from observed Δwindow")
    print()

    # ── 3-arm attribution: raw + floor-corrected + collapse-excluded ─────────
    any_f = any(k[0] == "full"         for k in by_lam_type)
    any_t = any(k[0] == "amount_time"  for k in by_lam_type)
    any_a = any(k[0] == "amount_only"  for k in by_lam_type)

    if not (any_f and any_t and any_a):
        print(f"Arms so far: {sorted(set(k[0] for k in by_lam_type))} — need all 3 for attribution.")
        return

    print("── 3-Arm Attribution: AMLSim (strict protocol) ──")
    print("  Δtiming = amount_time − amount_only")
    print("  Δrecv   = full − amount_time")
    print("  Δwindow = full − amount_only")
    print(f"  Floor correction: subtract Δwindow(λ=0)={floor_val:+.4f} from each λ>0")
    print(f"  Collapse exclusion: amount_only runs with Δsynth < {COLLAPSE_THRESH} excluded")
    print()

    print(f'  {"λ":>5}  {"Δwindow(raw)":>12}  {"Δwindow(floor-adj)":>18}  '
          f'{"Δwin 95% CI(raw)":>18}  {"n_col_a":>7}  {"C2b?":>15}')
    print("  " + "─" * 95)

    w_raw, w_floor_adj, w_excl = [], [], []
    for lam in lambdas:
        fk = ("full",        lam)
        tk = ("amount_time", lam)
        ak = ("amount_only", lam)
        if fk not in by_lam_type or tk not in by_lam_type or ak not in by_lam_type:
            continue

        fd = by_lam_type[fk]["ds_mean"]
        ad = by_lam_type[ak]["ds_mean"]
        dw = fd - ad
        dw_adj = dw - floor_val

        # Bootstrap CI on raw Δwindow
        f_raw = np.array(raw[fk])
        a_raw = np.array(raw[ak])
        min_n = min(len(f_raw), len(a_raw))
        boots = []
        for _ in range(args.n_bootstrap):
            fi = rng.choice(len(f_raw), size=min_n, replace=True)
            ai = rng.choice(len(a_raw), size=min_n, replace=True)
            boots.append(f_raw[fi].mean() - a_raw[ai].mean())
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))

        # Collapse-excluded Δwindow: remove amount_only collapse runs
        a_excl = np.array([float(r["delta_synth"]) for r in groups[ak]
                           if not is_collapse(r)])
        dw_excl = fd - (a_excl.mean() if len(a_excl) > 0 else ad)

        n_col_a = by_lam_type[ak]["n_col"]

        if lam > 0:
            w_raw.append(dw)
            w_floor_adj.append(dw_adj)
            if len(a_excl) > 0:
                w_excl.append(dw_excl)

        if lam == 0.0:
            c2b = "(noise floor)"
        elif (ci_lo - floor_val) > 0:
            c2b = "✓ C2b (adj CI>0)"
        elif ci_lo > 0:
            c2b = "~ C2b (raw CI>0, adj?)"
        elif dw > 0.03:
            c2b = "~ C2b (CI∋0)"
        elif dw < -0.02:
            c2b = "✗ window<amtonly"
        else:
            c2b = "~ C2a (tied)"

        print(f'  {lam:>5.1f}  {dw:>+12.4f}  {dw_adj:>+18.4f}  '
              f'[{ci_lo:+.4f},{ci_hi:+.4f}]  {n_col_a:>7}  {c2b:>15}')

    if w_raw:
        mw_raw  = np.mean(w_raw)
        mw_adj  = np.mean(w_floor_adj)
        mw_excl = np.mean(w_excl) if w_excl else float("nan")

        # Overall bootstrap (raw)
        per_lam = [
            np.array(raw[("full", lam)]).mean() - np.array(raw[("amount_only", lam)]).mean()
            for lam in lambdas if lam > 0
            if ("full", lam) in raw and ("amount_only", lam) in raw
        ]
        boots_all = [rng.choice(per_lam, size=len(per_lam), replace=True).mean()
                     for _ in range(args.n_bootstrap)]
        oci_lo = float(np.percentile(boots_all, 2.5))
        oci_hi = float(np.percentile(boots_all, 97.5))

        print(f'\n  Mean Δwindow (raw):          {mw_raw:+.4f}')
        print(f'  Mean Δwindow (floor-adj):    {mw_adj:+.4f}  (− floor {floor_val:+.4f})')
        print(f'  Mean Δwindow (collapse-excl):{mw_excl:+.4f}')
        print(f'  Overall 95% CI (raw, λ>0):  [{oci_lo:+.4f}, {oci_hi:+.4f}]')
        oci_adj_lo = oci_lo - floor_val
        oci_adj_hi = oci_hi - floor_val
        print(f'  Overall 95% CI (floor-adj): [{oci_adj_lo:+.4f}, {oci_adj_hi:+.4f}]')

        print()
        print("  ── AMLSim JUDGMENT (strict protocol) ──")
        if oci_adj_lo > 0:
            print(f"  ✓ C2b CONFIRMED (AMLSim): floor-adj Δwindow={mw_adj:+.4f}, CI [{oci_adj_lo:+.4f},{oci_adj_hi:+.4f}] > 0")
            print("    → timing-dominant regime: window coupling → downstream gain")
        elif oci_lo > 0:
            print(f"  ~ C2b LIKELY (AMLSim): raw CI>0 [{oci_lo:+.4f},{oci_hi:+.4f}] but floor-adj CI [{oci_adj_lo:+.4f},{oci_adj_hi:+.4f}] straddles 0")
            print("    → floor correction marginalizes signal; check collapse zone sensitivity")
        else:
            print(f"  ✗ C2b WEAK (AMLSim): raw CI [{oci_lo:+.4f},{oci_hi:+.4f}] straddles 0")
            print("    → both datasets null → diagnostic paper framing")

    # ── Side-by-side vs Sparkov ───────────────────────────────────────────────
    sparkov_path = Path(args.sparkov_csv)
    if sparkov_path.exists():
        sp_rows = []
        with open(sparkov_path) as f:
            for row in csv.DictReader(f):
                sp_rows.append(row)
        sp_raw = defaultdict(list)
        for r in sp_rows:
            sp_raw[(r["teacher_type"], float(r["coh_lambda"]))].append(float(r["delta_synth"]))

        sp_per_lam = [
            np.array(sp_raw[("full", lam)]).mean() - np.array(sp_raw[("amount_only", lam)]).mean()
            for lam in lambdas if lam > 0
            if ("full", lam) in sp_raw and ("amount_only", lam) in sp_raw
        ]
        sp_boots = [rng.choice(sp_per_lam, size=len(sp_per_lam), replace=True).mean()
                    for _ in range(args.n_bootstrap)]
        sp_oci_lo = float(np.percentile(sp_boots, 2.5))
        sp_oci_hi = float(np.percentile(sp_boots, 97.5))
        sp_mw = np.mean(sp_per_lam)

        # Sparkov floor
        sp_f0 = np.array(sp_raw.get(("full", 0.0), [0.0])).mean()
        sp_a0 = np.array(sp_raw.get(("amount_only", 0.0), [0.0])).mean()
        sp_floor = sp_f0 - sp_a0

        print(f"\n── AMLSim vs Sparkov Comparison (same strict protocol) ──")
        print(f"  {'':>20}  {'AMLSim':>12}  {'Sparkov':>12}")
        print(f"  {'n_seeds':>20}  {n_seeds:>12}  {len(set(r['seed'] for r in sp_rows)):>12}")
        print(f"  {'Δwindow mean(λ>0)':>20}  {mw_raw:>+12.4f}  {sp_mw:>+12.4f}")
        print(f"  {'λ=0 floor':>20}  {floor_val:>+12.4f}  {sp_floor:>+12.4f}")
        print(f"  {'Δwindow floor-adj':>20}  {mw_adj:>+12.4f}  {sp_mw-sp_floor:>+12.4f}")
        print(f"  {'Overall CI (raw)':>20}  [{oci_lo:+.3f},{oci_hi:+.3f}]  [{sp_oci_lo:+.3f},{sp_oci_hi:+.3f}]")
        print(f"  {'Overall CI (adj)':>20}  [{oci_adj_lo:+.3f},{oci_adj_hi:+.3f}]  [{sp_oci_lo-sp_floor:+.3f},{sp_oci_hi-sp_floor:+.3f}]")
        print()
        print("  ── REGIME JUDGMENT ──")
        if oci_adj_lo > 0 and (sp_oci_lo - sp_floor) <= 0:
            print("  → REGIME-DEPENDENT C2b confirmed:")
            print("    AMLSim (timing-dominant, Δtiming=+0.061 at teacher): window → downstream gain")
            print("    Sparkov (category-dominant, Δcat=+0.058 at teacher): downstream gain null")
            print("    Framing: 'temporal-window coupling beneficial when timing is primary fraud channel'")
        elif oci_adj_lo > 0 and (sp_oci_lo - sp_floor) > 0:
            print("  → C2b GENERALISES: both datasets CI>0 (floor-adjusted)")
        else:
            print("  → C2b WEAK in both → diagnostic paper (C1·C2a·C4) framing recommended")
            print("    Window note: 'teacher-level Δwindow≈+0.075 in both datasets (teacher AUPRC)")
            print("    but downstream transfer is dataset/regime-dependent'")


if __name__ == "__main__":
    main()
