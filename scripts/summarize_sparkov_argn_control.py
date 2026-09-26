"""Compact descriptive summaries; generation draws are not independent fit seeds."""
from pathlib import Path
import argparse
import pandas as pd
from plot_sparkov_argn_control import group,MODEL_ORDER

DOCS=Path(__file__).resolve().parents[1]/'docs/sparkov_argn_control_v2'

def summarize(supported=False):
    prefix='supported_' if supported else ''
    metrics=pd.read_csv(DOCS/f'{prefix}metrics.csv')
    primary=metrics[metrics.run.str.contains('/generated_(?:supported_)?validation_')].copy()
    rates=pd.read_csv(DOCS/f'{prefix}conditional_rates_customer_bootstrap.csv')
    numeric=pd.read_csv(DOCS/f'{prefix}numeric_metrics.csv')
    for condition in ['previous_fraud','previous_normal','positive_gap_le_5s','positive_gap_le_300s','short_sequence_le_100_RETROSPECTIVE']:
        selected=rates[rates.condition.eq(condition)].set_index('run')
        primary[condition+'_fraud_rate']=primary.run.map(selected.fraud_rate)
    for field in ['amount','gap_seconds','amount_history_ratio']:
        selected=numeric[numeric.field.eq(field)&numeric.label.eq('fraud')].set_index('run')
        for stat in ['generated_mean','ks_distance','wasserstein_log1p']:
            primary['fraud_'+field+'_'+stat]=primary.run.map(selected[stat])
    primary['model']=primary.run.map(group)
    primary['model']=pd.Categorical(primary.model,categories=MODEL_ORDER,ordered=True)
    primary=primary.sort_values(['model','run'])
    keys=['generated_fraud_rate','merchant_category_tv','class_1_category_amount_tv','class_1_history_amount_tv','class_1_merchant_tv','class_1_gap_category_amount_history_tv','previous_fraud_fraud_rate','previous_normal_fraud_rate','positive_gap_le_5s_fraud_rate','positive_gap_le_300s_fraud_rate','short_sequence_le_100_RETROSPECTIVE_fraud_rate','fraud_amount_generated_mean','fraud_amount_ks_distance','fraud_amount_wasserstein_log1p','fraud_gap_seconds_wasserstein_log1p','fraud_amount_history_ratio_wasserstein_log1p']
    primary.to_csv(DOCS/f'{prefix}primary_draw_summary.csv',index=False)
    rows=[]
    for name,g in primary.groupby('model',sort=False,observed=True):
        for metric in keys:
            x=g[metric]
            rows.append(dict(model=name,metric=metric,draws=len(g),nonempty=int(x.notna().sum()),mean=x.mean(),minimum=x.min(),maximum=x.max()))
    pd.DataFrame(rows).to_csv(DOCS/f'{prefix}primary_group_summary.csv',index=False)
    def interval(g,key,scale=1):
        x=g[key].dropna()*scale
        return 'NA' if not len(x) else f'{x.min():.3f}–{x.max():.3f}'
    count=int(metrics.loc[metrics.run.eq('real_validation'),'real_customers'].iloc[0])
    prevalence=float(metrics.loc[metrics.run.eq('real_validation'),'generated_fraud_rate'].iloc[0])
    mean=float(numeric.loc[numeric.run.eq('real_validation')&numeric.field.eq('amount')&numeric.label.eq('fraud'),'generated_mean'].iloc[0])
    lines=['# '+('Common-support' if supported else 'Primary')+' comparison: descriptive ranges across generation draws','',
        f'Results use the same {count}-customer validation cohort. Neural generators condition on static attributes; resampling controls assign customer keys but ignore those attributes. ARGN: two independent fits and two draws per fit. '+('CPAR: one fit and two draws; four unsupported contexts excluded from EVERY model. ' if supported else 'CPAR is reported separately on the 143-customer common-support cohort. ')+ 'Ranges below are descriptive, not confidence intervals. Resampling controls are diagnostics, not privacy-preserving generators.','',
        '| Setting | Fraud (%) | Merchant/category TV | Fraud category/amount TV | Fraud history TV | P(fraud after fraud) (%) | Fraud amount mean | Fraud amount log-W1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,g in primary.groupby('model',sort=False,observed=True):
        values=[interval(g,'generated_fraud_rate',100),interval(g,'merchant_category_tv'),interval(g,'class_1_category_amount_tv'),interval(g,'class_1_history_amount_tv'),interval(g,'previous_fraud_fraud_rate',100),interval(g,'fraud_amount_generated_mean'),interval(g,'fraud_amount_wasserstein_log1p')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['',f'Validation reference: fraud {prevalence*100:.6f}%; fraud amount mean {mean:.6f}. TV and Wasserstein distances are lower-is-better but must be interpreted against training-customer resampling variation, support and raw-scale checks.','',
        f'Full metrics, generated event counts and invalid values: `{prefix}metrics.csv`, `{prefix}numeric_metrics.csv`. Conditional support and customer-bootstrap intervals: `{prefix}conditional_rates_customer_bootstrap.csv`.','']
    (DOCS/('SUPPORTED_COMPARISON.md' if supported else 'COMPARISON.md')).write_text('\n'.join(lines))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--supported',action='store_true');summarize(parser.parse_args().supported)
