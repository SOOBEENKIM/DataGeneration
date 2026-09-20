"""Presentation of registered aggregate results; no additional model evaluation."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    specs = [
        ('berka', 'repeat', 'Berka: same transaction operation', ['0–2', '2–4', '4–6', '6–11', '>11'], 'Gap (days)'),
        ('sparkov', 'secondary_repeat', 'Sparkov: same merchant category', ['0–1.29', '1.29–3.25', '3.25–6.60', '6.60–13.79', '>13.79'], 'Gap (hours; rounded labels)'),
    ]
    for ax, (name, metric, title, labels, xlabel) in zip(axes[:2], specs):
        data = pd.read_csv(args.results/f'{name}_profiles.csv')
        for split, color in [('train', '#286dad'), ('validation', '#ca6200')]:
            sub = data.loc[data.grouping.eq('gap_bin') & data.metric.eq(metric) & data.split.eq(split)].sort_values('gap_bin')
            ax.errorbar(range(5), sub.value*100,
                        yerr=np.vstack([sub.value-sub.ci_low, sub.ci_high-sub.value])*100,
                        marker='o', capsize=3, color=color, label=split)
        ax.set_xticks(range(5), labels, rotation=25, ha='right')
        ax.set(title=title, xlabel=xlabel, ylabel='Same operation/category (%)')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    ax = axes[2]
    data = pd.read_csv(args.results/'berka_tie_sensitivity.csv')
    sub = data.loc[data.split.eq('validation') & data.cell.eq('all')].set_index('order')
    orders = ['canonical', 'reverse', 'shuffle_101', 'shuffle_102', 'shuffle_103']
    values = sub.loc[orders, 'repeat'].to_numpy()*100
    bars = ax.bar(range(5), values, color=['#286dad', '#ca6200', '#8a99a8', '#8a99a8', '#8a99a8'])
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], fontsize=9)
    ax.set_xticks(range(5), ['Original', 'Reverse', 'Shuffle 1', 'Shuffle 2', 'Shuffle 3'], rotation=25, ha='right')
    ax.set(title='Berka: unknown within-day order', ylabel='Same operation (%)', ylim=(0, 30), xlabel='All adjacent pairs, validation')
    ax.grid(axis='y', alpha=.2)
    fig.suptitle('Observed external data: relations to preserve and ordering uncertainty', fontsize=13)
    fig.text(.01, .005, 'Left / middle: unambiguous pairs; 95% entity-bootstrap intervals. These are observed-data relations, not generated-model scores.', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .97])
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
