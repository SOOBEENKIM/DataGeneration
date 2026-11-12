"""Exact empirical-CDF diagnosis; this does not train or select any model."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from diagnose_argn_residual import ROOT, SOURCE, decorate, generation_path

DOCS = ROOT / 'docs/argn_phase_gap_v1'
PHASES = ['onset', 'continuation', 'left_fraud']


def cancellation(real, generated):
    """Separate conditional CDF cancellation from changed phase prevalence.

    Sum of weighted conditional W1 is NOT a decomposition of pooled W1.
    Their difference, under identical real phase weights, is exactly the
    nonnegative amount hidden by conditional CDF errors of opposing signs.
    """
    r = real[real.label.eq(1) & real.gap.notna()]
    s = generated[generated.label.eq(1) & generated.gap.notna()]
    grid = np.unique(np.log1p(np.r_[r.gap, s.gap]))
    widths = np.diff(grid)
    contributions, rows, weights = [], [], np.zeros(len(s))
    for phase in PHASES:
        a = np.sort(np.log1p(r.loc[r.phase.eq(phase), 'gap']))
        b = np.sort(np.log1p(s.loc[s.phase.eq(phase), 'gap']))
        assert len(a) and len(b), 'Missing phase support must be reported explicitly'
        share = len(a) / len(r)
        fa = np.searchsorted(a, grid[:-1], side='right') / len(a)
        fb = np.searchsorted(b, grid[:-1], side='right') / len(b)
        delta = share * (fb - fa)
        contributions.append(delta)
        dist = float(np.dot(np.abs(fb - fa), widths))
        assert abs(dist - wasserstein_distance(a, b)) < 1e-10
        weights[s.phase.to_numpy() == phase] = share / len(b)
        rows.append(dict(phase=phase, real_events=len(a), generated_events=len(b),
                         real_share=share, generated_share=len(b)/len(s),
                         conditional_w1=dist, weighted_conditional_w1=share*dist,
                         mean_log_gap_shift=float(b.mean()-a.mean())))
    c = np.asarray(contributions)
    within = float(np.dot(np.abs(c).sum(axis=0), widths))
    standardized = float(np.dot(np.abs(c.sum(axis=0)), widths))
    direct = wasserstein_distance(np.log1p(r.gap), np.log1p(s.gap), v_weights=weights)
    assert abs(standardized-direct) < 1e-10
    hidden = within-standardized
    assert hidden >= -1e-10
    return dict(weighted_conditional_w1=within, standardized_pooled_w1=standardized,
                cdf_cancellation=hidden, cancellation_fraction=hidden/within if within else 0,
                actual_pooled_w1=wasserstein_distance(np.log1p(r.gap), np.log1p(s.gap))), rows


def main():
    real = decorate(pd.read_parquet(SOURCE/'prepared/validation.parquet'))
    totals, cells = [], []
    for fs in [20260930, 20261001]:
        for gs in [20261011, 20261012]:
            for arm in ['frozen', 'all_label_fit', 'boundary_fit', 'phase_mixture']:
                if arm == 'frozen':
                    path = generation_path('onset_fit', fs, gs)
                else:
                    family = 'argn_phase_gap_v1' if arm == 'phase_mixture' else 'argn_boundary_gap_v1'
                    path = ROOT/f'artifacts/{family}/runs/{arm}_{fs}/generated_validation_{gs}.parquet'
                info = dict(arm=arm, fit_seed=fs, generation_seed=gs)
                total, rows = cancellation(real, decorate(pd.read_parquet(path)))
                totals.append(dict(**info, **total))
                cells.extend(dict(**info, **row) for row in rows)
    pd.DataFrame(totals).to_csv(DOCS/'cdf_cancellation.csv', index=False)
    pd.DataFrame(cells).to_csv(DOCS/'cdf_phase_contributions.csv', index=False)
    print(pd.DataFrame(totals).groupby('arm').mean(numeric_only=True).to_string())
    (DOCS/'CDF_DIAGNOSTIC.json').write_text(json.dumps(dict(
        posthoc_diagnostic=True, test_events_read=False, model_selection=False,
        exact_cdf_and_scipy_comparisons=len(totals)*4,
        definition='real phase-weighted conditional W1 minus W1 of phase-standardized mixture',
        caution='This is a mathematical cancellation diagnostic, not a causal rollout intervention.'
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
