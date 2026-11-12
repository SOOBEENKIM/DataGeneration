"""Post-hoc description of saved development outputs; never fit or resample."""
import numpy as np
import pandas as pd
from report_initial import DOCS, CFG, locate, report, SOURCE


def main():
    real = report.times(SOURCE / 'prepared/validation.parquet')
    rows = []
    tails = []
    arms = ['current', 'init_only', 'handoff_only', 'init_handoff']
    bins = [(6, 50), (50, 100), (100, 200), (200, 500), (500, 1024), (1024, np.inf)]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            baseline = report.times(locate('current', fs, gs)[0])
            support = pd.concat([real.groupby('entity_id').size(), baseline.groupby('entity_id').size()], axis=1).min(axis=1)
            for arm in arms:
                generated = baseline if arm == 'current' else report.times(locate(arm, fs, gs)[0])
                tail = generated.event_index.ge(1024)
                tails.append(dict(arm=arm, fit_seed=fs, generation_seed=gs,
                    tail_gap_exact=np.array_equal(generated.loc[tail, 'gap'].to_numpy(), baseline.loc[tail, 'gap'].to_numpy()),
                    tail_events=int(tail.sum())))
                a = real[real.event_index.lt(real.entity_id.map(support))]
                b = generated[generated.event_index.lt(generated.entity_id.map(support))]
                for lo, hi in bins + [(1, np.inf)]:
                    # Whole-series row below has the original full support,
                    # while position bins use the registered matched support.
                    x, y = (real, generated) if lo == 1 else (a, b)
                    x = x[x.label.eq(0) & x.event_index.ge(lo) & x.event_index.lt(hi) & x.gap.notna()]
                    y = y[y.label.eq(0) & y.event_index.ge(lo) & y.event_index.lt(hi) & y.gap.notna()]
                    if lo != 1:
                        x, y = x[x.clock.gt(0)], y[y.clock.gt(0)]
                    logx, logy = np.log1p(x.gap.to_numpy()), np.log1p(y.gap.to_numpy())
                    row = dict(arm=arm, fit_seed=fs, generation_seed=gs, start=lo,
                        end=-1 if not np.isfinite(hi) else int(hi), real_events=len(x), generated_events=len(y),
                        gap_log_mean_bias=float(logy.mean()-logx.mean()),
                        gap_log_w1=report.wd(x, y, 'gap'))
                    for q in [.1, .5, .9]:
                        row[f'gap_log_q{int(100*q)}_bias'] = float(np.quantile(logy, q)-np.quantile(logx, q))
                    if lo != 1:
                        row['clock_log_mean_bias'] = float(np.log1p(y.clock).mean()-np.log1p(x.clock).mean())
                    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DOCS / 'posthoc_gap_direction.csv', index=False)
    out.groupby(['arm', 'start', 'end']).mean(numeric_only=True).to_csv(DOCS / 'posthoc_gap_direction_means.csv')
    pd.DataFrame(tails).to_csv(DOCS / 'posthoc_tail_parity.csv', index=False)
    print('POSTHOC_DIRECTION_AND_TAIL_PARITY_WRITTEN', flush=True)


if __name__ == '__main__':
    main()
