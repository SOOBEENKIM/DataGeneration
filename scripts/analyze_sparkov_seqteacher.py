"""
Interim / post-hoc analysis for sparkov_seqteacher.csv — 3-arm attribution with bootstrap CI.

3 arms:
  amount_only : use_time=False, use_recv=False  (pure amount)
  amount_time : use_time=True,  use_recv=False  (amount + timing)
  full        : use_time=True,  use_recv=True   (amount + timing + category)

Attribution (delta_SYNTH only — delta_aug saturated at Sparkov base≈0.99):
  Δtiming  = amount_time − amount_only   (timing, expected ~small +0.017 from diagnostic)
  Δcat     = full − amount_time          (category, expected ~dominant +0.058)
  Δwindow  = full − amount_only          (total, ~+0.075 expected from diagnostic)

Bootstrap CI: 2000 resamples on per-λ seed differences.

λ=0 noise floor: Δwindow at λ=0 should ≈ 0 (no coherence loss, differences = init noise).
collapse zone: amount_only Δsynth < −0.3 = over-constraint artifact, flagged separately.

Judgment (C2b generality):
  Δwindow CI lower bound > 0 → C2b CONFIRMED (Sparkov)
  Δwindow mean > 0 but CI straddles 0 → directional, inconclusive
  Δwindow mean ≤ 0 → C2b AMLSim-specific, scope accordingly
"""

import sys, csv, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent
TOTAL_RUNS = 5 * 3 * 10   # lambdas × arms × seeds = 150


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default=str(ROOT / "results" / "sparkov_seqteacher.csv"))
    p.add_argument("--n_total",     type=int, default=TOTAL_RUNS)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--collapse_thresh", type=float, default=-0.3)
    return p.parse_args()


def bootstrap_ci(vals, n=2000, rng=None, alpha=0.05):
    if rng is None:
        rng = np.random.default_rng(42)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


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

    rng = np.random.default_rng(42)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["teacher_type"], float(r["coh_lambda"]))].append(r)

    lambdas = sorted(set(float(r["coh_lambda"]) for r in rows))
    raw     = defaultdict(list)   # (ttype, lam) → [delta_synth, ...]
    for r in rows:
        raw[(r["teacher_type"], float(r["coh_lambda"]))].append(float(r["delta_synth"]))

    # ── Per-group table ───────────────────────────────────────────────────────
    print(f"{'teacher':>12} {'λ':>5} {'n':>3}  "
          f"{'delta_SYNTH':>8}±std  {'95% CI':>18}  "
          f"{'teacher_ap':>10}  {'collapse':>8}")
    print("─" * 90)

    by_lam_type = {}
    for ttype in ["amount_only", "amount_time", "full"]:
        printed = False
        for lam in lambdas:
            key = (ttype, lam)
            g = groups.get(key, [])
            if not g:
                continue
            printed = True
            ds = np.array([float(r["delta_synth"]) for r in g])
            ta = float(np.mean([float(r["teacher_auprc"]) for r in g]))
            n_col = sum(int(r.get("collapse_flag", 0)) for r in g)
            ci_lo, ci_hi = bootstrap_ci(ds, n=args.n_bootstrap, rng=rng)
            ci_str = f'[{ci_lo:+.4f},{ci_hi:+.4f}]'
            col_tag = f"{n_col}/{len(g)}" if n_col > 0 else "—"
            print(f"{ttype:>12} {lam:>5.1f} {len(g):>3}  "
                  f"{ds.mean():>+7.4f}±{ds.std():.4f}  "
                  f"{ci_str:>18}  "
                  f"{ta:>10.4f}  "
                  f"{col_tag:>8}")
            by_lam_type[key] = {
                "ds_mean": ds.mean(), "ds_std": ds.std(),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "teacher_auprc": ta, "n": len(g), "n_col": n_col,
            }
        if printed:
            print()

    # ── λ=0 noise floor check ────────────────────────────────────────────────
    print("── λ=0 Noise Floor (no coherence loss — differences = init noise) ──")
    for ttype in ["amount_only", "amount_time", "full"]:
        k = (ttype, 0.0)
        if k in by_lam_type:
            d = by_lam_type[k]
            print(f"  {ttype:>12} λ=0: Δsynth={d['ds_mean']:+.4f}±{d['ds_std']:.4f}")
    k_f0 = ("full", 0.0); k_a0 = ("amount_only", 0.0)
    if k_f0 in by_lam_type and k_a0 in by_lam_type:
        dw0 = by_lam_type[k_f0]["ds_mean"] - by_lam_type[k_a0]["ds_mean"]
        print(f"  Δwindow at λ=0: {dw0:+.4f}  (should be ≈ 0; any nonzero = init noise)")
    print()

    # ── 3-arm attribution ────────────────────────────────────────────────────
    any_f = any(k[0] == "full"         for k in by_lam_type)
    any_t = any(k[0] == "amount_time"  for k in by_lam_type)
    any_a = any(k[0] == "amount_only"  for k in by_lam_type)

    if any_f and any_t and any_a:
        print("── 3-Arm Attribution (delta_SYNTH, C2b generality) ──")
        print("  Δtiming = amount_time − amount_only   (Sparkov: expected small ~+0.017)")
        print("  Δcat    = full − amount_time           (Sparkov: expected dominant ~+0.058)")
        print("  Δwindow = full − amount_only           (total, = Δtiming + Δcat)")
        print()
        print(f'  {"λ":>5}  {"full":>8}  {"amt_time":>10}  {"amt_only":>10}  '
              f'{"Δtiming":>8}  {"Δcat":>7}  {"Δwindow":>8}  {"Δwin 95% CI":>20}  {"C2b?":>15}')
        print("  " + "─" * 105)

        w_diffs, t_diffs, c_diffs = [], [], []
        for lam in lambdas:
            fk = ("full",         lam)
            tk = ("amount_time",  lam)
            ak = ("amount_only",  lam)
            if fk not in by_lam_type or tk not in by_lam_type or ak not in by_lam_type:
                missing = [k[0] for k in [fk,tk,ak] if k not in by_lam_type]
                print(f'  {lam:>5.1f}  (missing: {missing})')
                continue
            fd = by_lam_type[fk]["ds_mean"]
            td = by_lam_type[tk]["ds_mean"]
            ad = by_lam_type[ak]["ds_mean"]
            dt = td - ad
            dc = fd - td
            dw = fd - ad

            # bootstrap CI on Δwindow
            f_raw = np.array(raw[fk])
            a_raw = np.array(raw[ak])
            min_n = min(len(f_raw), len(a_raw))
            boots_w = []
            for _ in range(args.n_bootstrap):
                fi = rng.choice(len(f_raw), size=min_n, replace=True)
                ai = rng.choice(len(a_raw), size=min_n, replace=True)
                boots_w.append(f_raw[fi].mean() - a_raw[ai].mean())
            ci_lo_w = float(np.percentile(boots_w, 2.5))
            ci_hi_w = float(np.percentile(boots_w, 97.5))
            ci_str  = f'[{ci_lo_w:+.4f},{ci_hi_w:+.4f}]'

            n_col_a = by_lam_type[ak]["n_col"]
            col_tag = f"[col={n_col_a}]" if n_col_a > 0 else ""

            if lam > 0:
                w_diffs.append(dw); t_diffs.append(dt); c_diffs.append(dc)

            if lam == 0.0:
                c2b = "(noise floor)"
            elif ci_lo_w > 0:
                c2b = "✓ C2b (CI>0)"
            elif dw > 0.03:
                c2b = "~ C2b (CI∋0)"
            elif dw < -0.02:
                c2b = "✗ window<amt"
            else:
                c2b = "~ C2a (tied)"

            print(f'  {lam:>5.1f}  {fd:>+8.4f}  {td:>+10.4f}  {ad:>+10.4f} {col_tag:<10} '
                  f'{dt:>+8.4f}  {dc:>+7.4f}  {dw:>+8.4f}  {ci_str:>20}  {c2b:>15}')

        if w_diffs:
            mw = np.mean(w_diffs); mt = np.mean(t_diffs); mc = np.mean(c_diffs)
            print(f'\n  Mean across λ>0:  Δtiming={mt:+.4f}  Δcat={mc:+.4f}  Δwindow={mw:+.4f}')

            # Overall CI: bootstrap across λ>0 per-λ mean diffs
            per_lam_diffs = [
                np.array(raw[("full", lam)]).mean() - np.array(raw[("amount_only", lam)]).mean()
                for lam in lambdas if lam > 0
                if ("full", lam) in raw and ("amount_only", lam) in raw
            ]
            boots_all = [rng.choice(per_lam_diffs, size=len(per_lam_diffs), replace=True).mean()
                         for _ in range(args.n_bootstrap)]
            oci_lo = float(np.percentile(boots_all, 2.5))
            oci_hi = float(np.percentile(boots_all, 97.5))
            print(f'  Overall Δwindow 95% CI [across λ>0]: [{oci_lo:+.4f}, {oci_hi:+.4f}]')

            # channel
            if mt > 0.01 and mt > mc:
                ch = f"TIMING dominant (Δtiming={mt:+.4f})"
            elif mc > 0.01 and mc > mt:
                ch = f"CATEGORY dominant (Δcat={mc:+.4f})"
            elif mt > 0.01 and mc > 0.01:
                ch = f"BOTH channels (timing={mt:+.4f}, cat={mc:+.4f})"
            else:
                ch = "no clear channel"

            print()
            print("  ── JUDGMENT ──")
            if oci_lo > 0:
                print(f"  ✓ C2b CONFIRMED (Sparkov): Δwindow={mw:+.4f}, CI [{oci_lo:+.4f},{oci_hi:+.4f}] > 0")
                print(f"    Channel: {ch}")
                print("    → C2b generality: AMLSim(timing-dominant) + Sparkov(category-dominant)")
                print("    → Next: external baseline comparison (C3)")
            elif mw > 0.03:
                print(f"  ~ C2b LIKELY (Sparkov): Δwindow={mw:+.4f} but CI [{oci_lo:+.4f},{oci_hi:+.4f}] straddles 0")
                print(f"    Channel: {ch}")
                print("    → directional; generality tentative — check collapse zones, consider more seeds")
            else:
                print(f"  ✗ C2b NOT CONFIRMED (Sparkov): Δwindow={mw:+.4f}, CI [{oci_lo:+.4f},{oci_hi:+.4f}]")
                print("    → C2b may be AMLSim-specific or regime-dependent")
                print("    → scope C2b to 'window-conditional regimes' or narrow to AMLSim")

    elif any_a and not any_f:
        print("── Partial: only 'amount_only' arm so far ──")
        print(f"  {n_done}/{args.n_total} runs. amount_only preview:")
        base = by_lam_type.get(("amount_only", 0.0), {}).get("ds_mean")
        for lam in lambdas:
            k = ("amount_only", lam)
            if k not in by_lam_type: continue
            d = by_lam_type[k]
            vs = f"vs λ=0: {d['ds_mean']-base:+.4f}" if base is not None and lam > 0 else "(λ=0 base)"
            col = f" [col={d['n_col']}]" if d["n_col"] > 0 else ""
            print(f"  λ={lam}  Δsynth={d['ds_mean']:+.4f}±{d['ds_std']:.4f}  {vs}{col}")
    else:
        have = sorted(set(k[0] for k in by_lam_type))
        print(f"  Arms so far: {have}")


if __name__ == "__main__":
    main()
