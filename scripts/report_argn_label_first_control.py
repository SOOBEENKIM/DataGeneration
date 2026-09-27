"""All-draw matched comparisons; no automatic novelty or superiority verdict."""
import argparse
import json
import numpy as np
import pandas as pd
from run_argn_label_first_control import ROOT,OUT,DOCS,CONTROL,check_registered,digest,write


def main(partial=False):
    check_registered(); dest=DOCS/'evaluation'; dest.mkdir(exist_ok=True)
    expected=[f'{arm}_{fs}' for arm in CONTROL['arms'] for fs in CONTROL['fit_seeds']]
    complete=[name for name in expected if (OUT/'runs'/name/'COMPLETE.json').exists()]
    if not partial: assert len(complete)==len(expected), f'incomplete: {complete}'
    old=ROOT/'docs/argn_state_first_v1'
    b=pd.read_csv(old/'evaluation/metrics.csv')
    real=b[b.arm.eq('real_validation')].iloc[0]
    stored=[b[b.arm.eq('B')],*[pd.read_csv(old/'event_weight_control'/f'metrics_{s}.csv') for s in CONTROL['fit_seeds']]]
    new=[pd.read_csv(dest/f'metrics_{name}.csv') for name in complete]
    metrics=pd.concat(stored+new,ignore_index=True)
    metrics.to_csv(dest/'all_draws.csv',index=False)
    numeric=['fraud_rate','mean_run_length','termination_rate','onset_rate','run_length_log_w1',
        'fraud_ratio_log_w1','normal_amount_log_w1','normal_gap_seconds_log_w1',
        'merchant_category_tv','class_1_gap_category_amount_history_tv',
        'customer_length_log_w1','unique_merchants','median_unique_merchants_per_customer']
    means=metrics.groupby(['arm','fit_seed'])[numeric].mean().reset_index()
    means.to_csv(dest/'fit_means.csv',index=False)
    pairs=[]
    target_metrics={'fraud_rate','mean_run_length','termination_rate','onset_rate',
                    'unique_merchants','median_unique_merchants_per_customer'}
    for arm,base in CONTROL['paired_existing_arms'].items():
        for fs in CONTROL['fit_seeds']:
            n=means[means.arm.eq(arm)&means.fit_seed.eq(fs)]
            r=means[means.arm.eq(base)&means.fit_seed.eq(fs)]
            if n.empty: continue
            for metric in numeric:
                a,c=float(r.iloc[0][metric]),float(n.iloc[0][metric])
                target=float(real[metric]) if metric in target_metrics else 0.
                pairs.append(dict(arm=arm,baseline=base,fit_seed=fs,metric=metric,real=target,
                    baseline_value=a,new_value=c,absolute_error_change=abs(c-target)-abs(a-target)))
    pd.DataFrame(pairs).to_csv(dest/'paired_fit_differences.csv',index=False)
    teachers=[pd.read_csv(old/'followup/conditional_summary.csv')]
    teachers.extend(pd.read_csv(dest/f'teacher_{name}.csv') for name in complete)
    teacher=pd.concat(teachers,ignore_index=True)
    teacher=teacher[teacher.regime.eq('teacher')&teacher['case'].eq('all_development')&~teacher.arm.eq('B_S')]
    teacher.to_csv(dest/'all_teacher_summaries.csv',index=False)
    lines=['# ARGN label-first order control', '', f'Completed fits: {len(complete)}/{len(expected)}.',
        'Two generation draws per completed fit. This is a development diagnostic, not a final test result.', '',
        '| Arm | Fit seed | Fraud % | Mean run | Termination % | Personal amount W1 | Normal amount W1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| Real development | — | {real.fraud_rate*100:.4f} | {real.mean_run_length:.3f} | {real.termination_rate*100:.3f} | 0 | 0 |"]
    for _,r in means.iterrows():
        lines.append(f'| {r.arm} | {int(r.fit_seed)} | {r.fraud_rate*100:.4f} | {r.mean_run_length:.3f} | '
            f'{r.termination_rate*100:.3f} | {r.fraud_ratio_log_w1:.4f} | {r.normal_amount_log_w1:.4f} |')
    lines += ['', 'Teacher diagnostic uses real past and real length/position tokens. Free generation uses generated length.',
        'Label-first and label-last probabilities have different current-field conditioning sets; NLL is not a standalone ranking.', '',
        '| Arm | Fit seed | Condition | Events | Actual fraud % | Predicted fraud % |',
        '|---|---:|---|---:|---:|---:|']
    focus=teacher[(teacher.grouping.eq('previous_label')&teacher.group.astype(str).eq('1'))|
        (teacher.grouping.eq('transition')&teacher.group.isin(['first','onset','termination']))]
    for _,r in focus.iterrows():
        lines.append(f'| {r.arm} | {int(r.fit_seed)} | {r.grouping}/{r.group} | {int(r.prefixes)} | '
                     f'{r.actual_fraud_rate*100:.4f} | {r.mean_fraud_probability*100:.4f} |')
    lines += ['', 'See `evaluation/all_draws.csv`, `fit_means.csv`, `paired_fit_differences.csv`, '
        '`all_teacher_summaries.csv` and per-fit hazard/age-amount/numeric files for all results.', '',
        'Negative absolute_error_change means closer to the development reference. Do not select the best draw.',
        'Order changes are an existing ARGN capability. Improvement is baseline refinement, not new architectural novelty.',
        'Joint prevalence, transition, amount, normal-quality and diversity assessment is required before the next structure.',
        'CPAR adequacy and task-matched TabDiT/other recent generator comparisons remain outstanding.']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for path,sha in {**manifest['paired_weights'],**manifest['paired_generations']}.items():
        assert digest(path)==sha
    write(OUT/'REPORT_STATUS.json',dict(complete_fits=complete,expected_fits=expected,
        final=not partial,test_events_read=False,earlier_checkpoints_and_generations_unchanged=True))
    print('ORDER_REPORT',len(complete),'of',len(expected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');main(p.parse_args().partial)
