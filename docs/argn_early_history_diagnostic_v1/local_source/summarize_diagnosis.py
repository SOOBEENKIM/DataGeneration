"""Descriptive paired summaries; no tuning, extra simulation, or test access."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
P = OUT / 'report'
a = pd.read_csv(P / 'all_diagnostics.csv')
m = pd.read_csv(P / 'mean_diagnostics.csv')
s = pd.read_csv(P / 'saved_generation_positions.csv')
sm = s.groupby(['model', 'start', 'end']).mean(numeric_only=True).reset_index()
sm.to_csv(P / 'saved_generation_means.csv', index=False)

main = m[(m.model == 'refined_personal') & (m['group'] == 'normal') & (m.start == 6) & (m.end == 50)]
main.to_csv(P / 'early_interventions_summary.csv', index=False)
keys = ['split', 'path_source', 'fit_seed', 'generation_seed', 'noise_seed', 'start', 'end', 'group']
cur = a[(a.model == 'refined_personal') & (a['group'] == 'normal')]
paired = cur.pivot(index=keys, columns='mode', values='gap_w1').reset_index()
rows = []
for (split, path, start, end), g in paired.groupby(['split', 'path_source', 'start', 'end']):
    for mode in ['RR', 'RG', 'GR', 'seed5', 'reset21', 'prefix_delta_first5']:
        delta = (g[mode] - g.GG).dropna()
        if len(delta):
            rows.append(dict(split=split, path_source=path, start=start, end=end, mode=mode,
                pairs=len(delta), improved_pairs=int((delta < 0).sum()), mean_change=delta.mean(),
                min_change=delta.min(), max_change=delta.max()))
pd.DataFrame(rows).to_csv(P / 'paired_intervention_changes.csv', index=False)
cur.groupby(['split', 'path_source', 'mode', 'start', 'end']).agg(
    events_min=('events', 'min'), events_max=('events', 'max'),
    customers_min=('customers', 'min'), customers_max=('customers', 'max'),
    noise_observations=('noise_seed', 'count')).to_csv(P / 'support_counts.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
bins = [(1, 6), (6, 21), (21, 50), (50, 100), (100, 200)]
labels = ['1-5', '6-20', '21-49', '50-99', '100-199']
for model, label, color in [('clock_prefix', 'Preserved reference', '#62758b'),
                            ('coarse_personal', 'Before loss refinement', '#b7842b'),
                            ('refined_personal', 'Current candidate', '#226cbd')]:
    q = sm[sm.model == model].set_index(['start', 'end']).loc[bins]
    axes[0, 0].plot(labels, q.gap_w1, 'o-', label=label, color=color)
axes[0, 0].set(title='A. Saved full generations: normal time error', ylabel='W1(log(1 + gap))', xlabel='Event index')
axes[0, 0].legend(fontsize=8)

mode_order = ['GG', 'RR', 'RG', 'GR', 'seed5', 'prefix_delta_first5']
mode_labels = ['Generated\nhistory', 'Real clock\nboth paths', 'Real clock\nGMR only',
               'Real clock\ncorrection only', 'Real first\n5 intervals', 'Old correction\nfirst 5 only']
q = main[(main.split == 'development') & (main.path_source == 'generated_labels')].set_index('mode').loc[mode_order]
axes[0, 1].bar(np.arange(len(q)), q.gap_w1, color=['#226cbd', '#369b77', '#369b77', '#c77843', '#369b77', '#9b89ba'])
axes[0, 1].set_xticks(np.arange(len(q)), mode_labels, fontsize=8)
axes[0, 1].set(title='B. Controlled replay: event 6-49', ylabel='W1(log(1 + gap))')
for i, v in enumerate(q.gap_w1):
    axes[0, 1].text(i, v + .006, f'{v:.3f}', ha='center', fontsize=8)
axes[0, 1].set_ylim(0, .4)

for mode, label, color in [('GG', 'Current candidate', '#226cbd'),
                           ('seed5', 'Real first 5 intervals (diagnosis)', '#369b77'),
                           ('prefix_delta_first5', 'Old correction for first 5 only', '#9b89ba')]:
    q = m[(m.model == 'refined_personal') & (m['group'] == 'normal') &
          (m.split == 'development') & (m.path_source == 'generated_labels') & (m['mode'] == mode)]
    q = q.set_index(['start', 'end']).loc[bins[1:]]
    axes[1, 0].plot(labels[1:], q.gap_w1, 'o-', label=label, color=color)
axes[1, 0].set(title='C. Better initialization also changes later errors', ylabel='W1(log(1 + gap))', xlabel='Event index')
axes[1, 0].legend(fontsize=8)

for model, label, color in [('clock_prefix', 'Preserved reference', '#62758b'),
                           ('refined_personal', 'Current candidate', '#226cbd')]:
    q = sm[sm.model == model].set_index(['start', 'end']).loc[bins[1:]]
    axes[1, 1].plot(labels[1:], q.log_clock_bias, 'o-', label=label, color=color)
axes[1, 1].axhline(0, color='black', lw=.7)
axes[1, 1].set(title='D. Saved generations: bias in past-gap median', ylabel='Mean log-clock difference (generated - real)', xlabel='Event index')
axes[1, 1].legend(fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.2, axis='y')
fig.suptitle('Frozen-head early-history diagnosis | oracle interventions are not deployable performance', fontsize=12)
fig.savefig(P / 'early_history_diagnosis.png', dpi=180)
fig.savefig(P / 'early_history_diagnosis.pdf')
plt.close(fig)
print('SUMMARY_WRITTEN', len(a), 'metric rows', len(rows), 'paired summaries')
