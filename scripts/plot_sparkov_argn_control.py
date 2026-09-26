"""Export the diagnostic comparison as standalone scientific figures."""
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];DOCS=ROOT/'docs/sparkov_argn_control_v2'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/sparkov_argn_control_v2/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_ORDER=['ARGN AUTO, native cap','ARGN AUTO, cap relaxed','ARGN AUTO + category first','ARGN AUTO + fraud first','ARGN gap DIGIT, cap relaxed','ARGN gap DIGIT + category first','ARGN gap+amount DIGIT, cap relaxed','ARGN gap+amount DIGIT + category first','CPAR, pinned implementation','Row resampling','First-order resampling']

def group(name):
    run=name.split('/')[0]
    if run.startswith('order_category_numeric_digit_relaxed'):return 'ARGN gap+amount DIGIT + category first'
    if run.startswith('numeric_digit_relaxed'):return 'ARGN gap+amount DIGIT, cap relaxed'
    if run.startswith('order_category_digit_relaxed'):return 'ARGN gap DIGIT + category first'
    if run.startswith('digit_relaxed'):return 'ARGN gap DIGIT, cap relaxed'
    if run.startswith('order_category_relaxed'):return 'ARGN AUTO + category first'
    if run.startswith('order_event_is_fraud_relaxed'):return 'ARGN AUTO + fraud first'
    if run.startswith('native_'):return 'ARGN AUTO, native cap'
    if run.startswith('relaxed_'):return 'ARGN AUTO, cap relaxed'
    if run.startswith('cpar_'):return 'CPAR, pinned implementation'
    if run=='row_resampling':return 'Row resampling'
    if run=='transition_resampling':return 'First-order resampling'
    return run

def plot(supported=False):
    prefix='supported_' if supported else ''
    metrics=pd.read_csv(DOCS/f'{prefix}metrics.csv');rates=pd.read_csv(DOCS/f'{prefix}conditional_rates_customer_bootstrap.csv')
    pattern='/generated_(?:supported_)?validation_'
    data=metrics[metrics.run.str.contains(pattern)].copy();data['model']=data.run.map(group)
    rates['model']=rates.run.map(group)
    order=MODEL_ORDER if supported else [x for x in MODEL_ORDER if not x.startswith('CPAR')]
    assert set(order)<=set(data.model)
    palette=dict(zip(MODEL_ORDER,['#426a9e','#6e91bb','#22836c','#aa6d3b','#7c68a6','#465985','#9e765a','#ab426d','#8d5663','#777777','#555555']))
    colors=[palette[x] for x in order]
    fig,axes=plt.subplots(1,3,figsize=(17,7),sharey=True)
    prevalence=rates[(rates.run=='real_validation')&(rates.condition=='all')].iloc[0]
    continuation=rates[(rates.run=='real_validation')&(rates.condition=='previous_fraud')].iloc[0]
    reference=pd.read_csv(DOCS/f'{prefix}training_customer_resampling_reference.csv')
    axes[0].axvspan(prevalence.ci_low*100,prevalence.ci_high*100,color='#dddddd',alpha=.65)
    axes[0].axvline(prevalence.fraud_rate*100,color='black',linestyle='--',linewidth=1)
    axes[1].axvspan(reference.merchant_category_tv.min(),reference.merchant_category_tv.max(),color='#dddddd',alpha=.65)
    axes[2].axvspan(continuation.ci_low*100,continuation.ci_high*100,color='#dddddd',alpha=.65)
    axes[2].axvline(continuation.fraud_rate*100,color='black',linestyle='--',linewidth=1)
    for i,(name,color) in enumerate(zip(order,colors)):
        m=data[data.model==name]
        rr=rates[(rates.model==name)&(rates.condition=='previous_fraud')&rates.run.str.contains(pattern)]
        series=[m.generated_fraud_rate*100,m.merchant_category_tv,rr.fraud_rate*100]
        for ax,values in zip(axes,series):
            values=values.dropna().to_numpy();offset=np.linspace(-.11,.11,len(values))
            ax.scatter(values,i+offset,color=color,s=27,zorder=3)
            if len(values):ax.scatter([values.mean()],[i],marker='D',facecolors='white',edgecolors=color,s=40,zorder=4)
            ax.axhline(i,color='#eeeeee',linewidth=.6,zorder=0)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(order)),order);axes[0].invert_yaxis()
    axes[0].set_xlabel('Generated fraud prevalence (%)')
    axes[1].set_xlabel('Merchant-category TV (lower is better)');axes[1].set_xlim(0,1)
    axes[2].set_xlabel('P(fraud | previous fraud) (%)');axes[2].set_xlim(-1,101)
    axes[0].set_title('Overall class balance');axes[1].set_title('Within-transaction relationship');axes[2].set_title('Across-transaction relationship')
    cohort='143 customers: common context support' if supported else '147 customers: original ARGN cohort'
    fit_note='ARGN: 2 fits; CPAR: 1 fit.' if supported else 'ARGN: 2 fits; CPAR is evaluated separately on common support.'
    fig.suptitle(f'Sparkov relationship fidelity ({cohort})',fontsize=13)
    fig.text(.02,.015,f'Dots: individual generation draws; diamonds: descriptive means. {fit_note}\nShading: validation customer-bootstrap 95% intervals (left/right); training-customer resampling range (middle).',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.94])
    out=DOCS/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/f'{prefix}relationship_comparison.png',dpi=170);fig.savefig(out/f'{prefix}relationship_comparison.pdf');plt.close(fig)
    numeric=pd.read_csv(DOCS/f'{prefix}numeric_metrics.csv');numeric['model']=numeric.run.map(group)
    nref=pd.read_csv(DOCS/f'{prefix}numeric_reference.csv')
    fields=[('amount','generated_mean','Fraud amount: mean (original units)'),('amount','wasserstein_log1p','Fraud amount: log1p Wasserstein distance'),('amount_history_ratio','wasserstein_log1p','Fraud amount / past median: log1p distance')]
    fig,axes=plt.subplots(1,3,figsize=(17,7),sharey=True)
    for ax,(field,metric,title) in zip(axes,fields):
        baseline=numeric[(numeric.run=='real_validation')&(numeric.field==field)&(numeric.label=='fraud')].iloc[0]
        if metric=='generated_mean':ax.axvline(baseline[metric],color='black',linestyle='--',linewidth=1)
        ref=nref[(nref.field==field)&(nref.label=='fraud')][metric]
        ax.axvspan(ref.min(),ref.max(),color='#dddddd',alpha=.65)
        for i,(name,color) in enumerate(zip(order,colors)):
            values=numeric[(numeric.model==name)&numeric.run.str.contains(pattern)&(numeric.field==field)&(numeric.label=='fraud')][metric].dropna().to_numpy()
            ax.scatter(values,i+np.linspace(-.11,.11,len(values)),color=color,s=27,zorder=3)
            if len(values):ax.scatter([values.mean()],[i],marker='D',facecolors='white',edgecolors=color,s=40,zorder=4)
            ax.axhline(i,color='#eeeeee',linewidth=.6,zorder=0)
        ax.set_xlabel(title);ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(order)),order);axes[0].invert_yaxis()
    fig.suptitle(f'Sparkov numeric fidelity ({cohort})',fontsize=13)
    fig.text(.02,.015,f'Dots: individual generation draws; diamonds: descriptive means. Gray: 12 training-customer resampling draws.\n{fit_note} Exploratory development-set comparison; see report for execution conventions.',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.94])
    fig.savefig(out/f'{prefix}numeric_fidelity.png',dpi=170);fig.savefig(out/f'{prefix}numeric_fidelity.pdf');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--supported',action='store_true');plot(parser.parse_args().supported)
