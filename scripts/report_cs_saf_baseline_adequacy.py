"""Descriptive Stage 1 contrasts and figures from all registered results; no fits."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continue_cs_saf_argn_adequacy import CONFIG, OUT
from scripts.run_cs_saf_external_audit_v1 import write


def main():
    report = ROOT / 'docs/cs_saf/baseline_adequacy_v1'
    c = json.loads(CONFIG.read_text())
    summary = json.loads((report / 'execution_summary.json').read_text())
    assert summary['status'] == 'COMPLETE'
    conditional = pd.read_csv(report / 'conditional_screens.csv', dtype={'group': str})
    generated = pd.read_csv(report / 'generation_metrics.csv', dtype={'group': str})
    screens = pd.read_csv(report / 'generation_screens.csv', dtype={'group': str})
    keys = ['family', 'kappa', 'index', 'variant', 'group']
    # Generation tapes are nested within saved models, not new training replicates.
    parent = generated.groupby(keys).agg(
        repeat_curve_mean=('short_gap_repeat_curve_l1', 'mean'),
        repeat_curve_tape_min=('short_gap_repeat_curve_l1', 'min'),
        repeat_curve_tape_max=('short_gap_repeat_curve_l1', 'max'),
        null_shape_mean=('null_centered_shape_l1', 'mean'),
        null_shape_tape_min=('null_centered_shape_l1', 'min'),
        null_shape_tape_max=('null_centered_shape_l1', 'max'),
        mark_tv_mean=('mark_sparse_tv', 'mean'),
    ).reset_index()
    parent.merge(screens, on=keys, validate='one_to_one').to_csv(report / 'generation_by_parent.csv', index=False)

    raw = conditional[conditional.variant == 'raw']
    original = raw[raw.family == 'argn_original'].set_index(['kappa', 'index', 'group'])
    continued = raw[raw.family == 'argn_continued'].set_index(['kappa', 'index', 'group'])
    budget = []
    gate = c['budget_contribution_screen']
    for key, row in continued.iterrows():
        base = original.loc[key]
        brier = float(base.repeat_brier - row.repeat_brier)
        nll = float(base.mark_nll - row.mark_nll)
        budget.append(dict(kappa=key[0], index=key[1], group=key[2],
            brier_improvement=brier, mark_nll_improvement=nll,
            passes_budget_contribution_screen=(brier >= gate['min_brier_improvement'] and
                                               nll >= gate['min_mark_nll_improvement'])))
    pd.DataFrame(budget).to_csv(report / 'budget_contrasts.csv', index=False)
    runs = []
    for k in c['kappas']:
        for seed in c['argn_seeds']:
            folder = OUT / f'continuations/kappa_{k}/seed_{seed}'
            start = json.loads((folder / 'start.json').read_text())
            done = json.loads((folder / 'DONE.json').read_text())
            runs.append(dict(kappa=k, seed=seed, physical_gpu=start['physical_gpu'], **done))
    pd.DataFrame(runs).to_csv(report / 'continuation_runs.csv', index=False)

    counts = []
    for (family, variant), cell in conditional.groupby(['family', 'variant']):
        generation_cell = screens[(screens.family == family) & (screens.variant == variant)]
        # Four parents / family; primary screen requires each observed group, not pooled averaging.
        cp = cell[cell.group != 'pooled'].groupby(['kappa', 'index']).passes_conditional_screen.all()
        gp = generation_cell[generation_cell.group != 'pooled'].groupby(['kappa', 'index']).passes_generation_screen.all()
        counts.append(dict(family=family, variant=variant, parent_count=len(cp),
            parents_passing_conditional=int(cp.sum()), parents_passing_generation=int(gp.sum()),
            parents_passing_both=int((cp & gp).sum())))
    pd.DataFrame(counts).to_csv(report / 'screen_counts.csv', index=False)

    contrasts = []
    for (family, kappa, index, group), cell in conditional.groupby(['family', 'kappa', 'index', 'group']):
        cell = cell.set_index('variant')
        gen = parent[(parent.family == family) & (parent.kappa == kappa) &
                     (parent['index'] == index) & (parent.group == group)].set_index('variant')
        for treatment, baseline in [('level', 'raw'), ('gap', 'raw'), ('direct', 'raw'), ('direct', 'gap')]:
            contrasts.append(dict(family=family, kappa=kappa, index=index, group=group,
                contrast=f'{treatment}_minus_{baseline}',
                conditional_brier=float(cell.loc[treatment, 'repeat_brier']-cell.loc[baseline, 'repeat_brier']),
                conditional_mark_nll=float(cell.loc[treatment, 'mark_nll']-cell.loc[baseline, 'mark_nll']),
                conditional_curve_l1=float(cell.loc[treatment, 'curve_l1']-cell.loc[baseline, 'curve_l1']),
                generated_curve_l1=float(gen.loc[treatment, 'repeat_curve_mean']-gen.loc[baseline, 'repeat_curve_mean'])))
    pd.DataFrame(contrasts).to_csv(report / 'within_parent_contrasts.csv', index=False)

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
            ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

    active_cond = conditional[(conditional.kappa == 1) & (conditional.group == '1')]
    active_gen = parent[(parent.kappa == 1) & (parent.group == '1')]
    active_rows = []
    for family in ['argn_original', 'argn_continued', 'U']:
        for variant in c['variants']:
            cr = active_cond[(active_cond.family == family) & (active_cond.variant == variant)]
            gr = active_gen[(active_gen.family == family) & (active_gen.variant == variant)]
            active_rows.append([family, variant, f'{cr.curve_l1.mean():.5f}',
                f'{gr.repeat_curve_mean.mean():.5f} [{gr.repeat_curve_mean.min():.5f}, {gr.repeat_curve_mean.max():.5f}]',
                f'{cr.repeat_brier.mean():.5f}', f'{cr.mark_nll.mean():.5f}'])
    tables = '# 전체 집계 표\n\n이 파일은 완료된 저장물에서 자동 작성한다. 범위는 두 학습 모델의 값이며 신뢰구간이 아니다.\n\n'
    tables += '## 관계가 있는 10% 집단\n\n' + table(
        ['모델', '보정', '실제 이력 곡선 L1', '생성 곡선 L1: 평균 [모델 범위]', '반복 Brier', '행동 NLL'], active_rows)
    tables += '\n\n## 사전등록 screen\n\n각 수는 4개 부모 모델 중 두 집단을 모두 통과한 모델 수다. 생성 screen은 세 생성 난수의 평균에 적용한다.\n\n'
    tables += table(['모델', '보정', '조건부 예측', '생성 및 비용', '둘 다'],
        [[r['family'], r['variant'], f"{r['parents_passing_conditional']}/4",
          f"{r['parents_passing_generation']}/4", f"{r['parents_passing_both']}/4"] for r in counts])
    tables += '\n\n## 추가 학습\n\n' + table(['조건 κ', '시드', '시작 epoch', '마지막 epoch', '선택 epoch', '추가 실행 초'],
        [[r['kappa'], r['seed'], r['start_epoch'], r['last_epoch'], r['best_epoch'], round(r['seconds'], 1)] for r in runs])
    tables += '\n\n전체 시드·집단·생성 난수와 13개 기본 지표는 같은 폴더의 CSV에 보존했다.\n'
    (report / 'result_tables.md').write_text(tables)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'font.family': 'DejaVu Sans', 'savefig.dpi': 180})
    families = ['argn_original', 'argn_continued', 'U']
    labels = ['ARGN original', 'ARGN continued', 'U (existing)']
    colors = ['#9a8f85', '#3b6ea8', '#18836d']
    variants = c['variants']
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9), layout='constrained')
    panels = [
        (conditional, 'curve_l1', 'A  Real-history repeat-curve error', 'Weighted L1 (lower is better)', .03),
        (parent, 'repeat_curve_mean', 'B  Freely generated repeat-curve error', 'Weighted L1 (lower is better)', .03),
        (conditional, 'repeat_brier', 'C  Event-level repeat prediction', 'Brier score (lower is better)', None),
        (parent, 'mark_tv_mean', 'D  Generated marginal mark distribution', 'Sparse TV: top 20 + other (lower is better)', None),
    ]
    for ax, (frame, metric, title, ylabel, threshold) in zip(axes.flat, panels):
        active = frame[(frame.kappa == 1) & (frame.group == '1')]
        for family, label, color, offset in zip(families, labels, colors, [-.1, 0, .1]):
            stats = active[active.family == family].groupby('variant')[metric].agg(['mean', 'min', 'max']).loc[variants]
            mean = stats['mean'].to_numpy()
            ax.errorbar(np.arange(4) + offset, mean,
                yerr=np.stack([mean - stats['min'].to_numpy(), stats['max'].to_numpy() - mean]),
                color=color, label=label, marker='o', capsize=3, linewidth=1.6)
        if threshold is not None:
            ax.axhline(threshold, ls='--', color='#666', lw=1, label='Preregistered screen: 0.03')
        ax.set(title=title, ylabel=ylabel, xticks=np.arange(4), xticklabels=['Raw', 'Level', 'Gap bins', 'Direct'])
        ax.grid(axis='y', alpha=.18)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Active 10% group: simple controls before a new architecture', fontsize=16, fontweight='bold')
    fig.supxlabel('Two trained parents per family; bars show their range, not a confidence interval.\n'
        'Generation: 3 tapes averaged within each parent. One explored synthetic dataset; unequal historical training budgets.', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(report / f'active_comparison.{ext}', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout='constrained')
    for ax, (kappa, group, title) in zip(axes, [(0, '0', 'No-relation data: group 0'),
            (0, '1', 'No-relation data: group 1'), (1, '0', 'Relation data: inactive group 0')]):
        cell = parent[(parent.kappa == kappa) & (parent.group == group)]
        for family, label, color, offset in zip(families, labels, colors, [-.1, 0, .1]):
            stats = cell[cell.family == family].groupby('variant').null_shape_mean.agg(['mean', 'min', 'max']).loc[variants]
            mean = stats['mean'].to_numpy()
            ax.errorbar(np.arange(4) + offset, mean,
                yerr=np.stack([mean - stats['min'].to_numpy(), stats['max'].to_numpy() - mean]),
                marker='o', color=color, label=label, capsize=3)
        ax.axhline(.02, ls='--', color='#666', lw=1)
        ax.set(title=title, ylabel='Centered repeat-curve L1', xticks=np.arange(4),
            xticklabels=['Raw', 'Level', 'Gap bins', 'Direct'])
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('All three null cells: curve shape after removing the average level', fontsize=15, fontweight='bold')
    fig.supxlabel('Empirical group curves, not a causal intervention or fixed-history sensitivity. Dashed line: 0.02 screening limit.\n'
        'Bars: range of 2 parent means, each averaging 3 generation tapes.', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(report / f'null_shape_comparison.{ext}', bbox_inches='tight')
    plt.close(fig)
    write(report / 'report_scope.json', dict(training_replicates_per_family_per_condition=2,
        generation_tapes_per_parent=3, independent_dataset_replicates=0,
        thresholds_changed=False, new_fits_from_reporting=False,
        error_bars='range_of_two_parent_values_not_confidence_intervals'))
    print(pd.DataFrame(counts).to_string(index=False))
    print(pd.DataFrame(budget).to_string(index=False))


if __name__ == '__main__':
    main()
