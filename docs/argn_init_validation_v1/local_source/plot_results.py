"""Descriptive figure from the completed independent evaluation; no fitting."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parents[1]
report = out / 'report'
assert (report / 'FINAL_AUDIT.json').exists()
frame = pd.read_csv(report / 'FIT_MEANS.csv')
arms = ['init_only', 'no_initial', 'no_calibration', 'gmr', 'phase_gmm',
        'history_only', 'history_init_cal']
labels = ['init_only', 'Without initial law', 'Without calibration', 'GMR',
          'Phase mixture', 'History mixture', 'History + initial + calibration']
seeds = sorted(frame.time_fit_seed.unique())
metrics = [('normal_gap_seconds_log_w1', 'Normal timing'),
           ('onset_personal_gap_w1', 'Personal timing at fraud onset'),
           ('normal_stay_personal_gap_w1', 'Personal timing: normal stays'),
           ('early_time', 'Normal timing: positions 6–49'),
           ('late1024_time', 'Normal timing: positions 1024+')]
fig, axes = plt.subplots(1, 5, figsize=(17.2, 4.3), sharey=True)
colors = ['#2c6fbb', '#c76a26', '#36815a']
for axis, (metric, title) in zip(axes, metrics):
    for i, seed in enumerate(seeds):
        values = frame[frame.time_fit_seed.eq(seed)].set_index('arm').reindex(arms)
        axis.scatter(values[metric], np.arange(len(arms)) + (i - 1) * .16,
                     color=colors[i], s=29, label=f'Time fit {seed}', zorder=3)
    axis.set_title(title, fontsize=10)
    axis.set_xlabel('Wasserstein-1 error (lower is better)', fontsize=8)
    axis.grid(axis='x', color='#dddddd', linewidth=.7)
    axis.axhspan(-.42, .42, color='#eef3fa', zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(labelsize=8)
axes[0].set_yticks(np.arange(len(arms)), labels)
axes[0].invert_yaxis()
axes[-1].legend(loc='lower right', fontsize=7, frameon=False)
fig.suptitle('Independent time-module refits: all registered arms', y=.99, fontsize=12)
fig.text(.28, .025,
         'Each point averages 2 frozen ARGN parents × 2 paired generation draws.\n'
         'Three time-fit seeds; reused development data; these are not confidence intervals.',
         ha='left', fontsize=8)
fig.tight_layout(rect=(0, .11, 1, .94))
for suffix in ['png', 'pdf']:
    fig.savefig(report / f'REFIT_COMPARISON.{suffix}', dpi=180, bbox_inches='tight')
plt.close(fig)
print(report / 'REFIT_COMPARISON.png')
