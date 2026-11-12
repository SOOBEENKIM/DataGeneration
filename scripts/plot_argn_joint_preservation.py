"""Standalone scientific figures; individual draws are not confidence intervals."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_argn_joint_preservation import DOCS,CFG


def main():
    data=pd.read_csv(DOCS/'all_metrics.csv');arms=['current']+list(CFG['generation_arms'])
    labels=['Current','Joint count','Onset hazard','Both','Both, no context']
    panels=[('fraud_ratio_log_w1','Personal-relative fraud amount error',None),
        ('generated_fraud_rate','Overall fraud rate',float(data.real_fraud_rate.iloc[0])),
        ('fraud_gap_seconds_log_w1','Fraud inter-arrival error',None),
        ('fraud_amount_log_w1','Fraud amount error',None),
        ('customers_with_fraud','Customers with observed fraud',114),
        ('normal_gap_seconds_log_w1','Normal inter-arrival error',None),
        ('normal_amount_log_w1','Normal amount error',None),
        ('customer_length_log_w1','Customer sequence-length error',None)]
    colors=['#64748b','#0891b2','#d97706','#7c3aed','#059669']
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for ax,(key,title,truth) in zip(axes.flat,panels):
        scale=100 if key=='generated_fraud_rate' else 1
        for i,a in enumerate(arms):
            values=data[data.arm.eq(a)][key].to_numpy()*scale
            ax.scatter(i+np.linspace(-.11,.11,len(values)),values,c=colors[i],s=29,alpha=.65,zorder=3)
            ax.hlines(np.mean(values),i-.23,i+.23,color=colors[i],lw=3)
        if truth is not None:
            ax.axhline(truth*scale,color='#334155',ls='--',lw=1.4,label=f'Real: {truth*scale:.3f}' if scale==100 else f'Real: {truth}')
            ax.legend(fontsize=8)
        ax.set_title(title,fontsize=10,pad=12);ax.set_xticks(range(5),labels,rotation=38,ha='right',fontsize=8)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Fraud transactions (%)' if scale==100 else 'Customers (of 147)' if truth is not None else 'log1p W1 (lower is better)',fontsize=8)
    fig.suptitle('ARGN: joint sequence count and exposure-scaled onset controls\n2 frozen parent fits × 2 generation draws; dots = draws, bars = means; no confidence intervals',fontsize=13)
    for ext in ['png','pdf','svg']:fig.savefig(DOCS/f'joint_preservation_factorial.{ext}',dpi=180)
    svg=DOCS/'joint_preservation_factorial.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
