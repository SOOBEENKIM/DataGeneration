"""Pooled initial/fresh draws for the fixed original and successive candidate controls."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from run_argn_joint_preservation import DOCS


def main():
    d=pd.read_csv(DOCS/'end_to_end_pooled.csv');arms=['current','fixed_joint','onset_fit']
    labels=['Frozen current','Joint count + hazard','+ Onset amount expert']
    panels=[('fraud_ratio_log_w1','Personal-relative fraud amount error',None),
            ('generated_fraud_rate','Overall fraud rate',float(d.real_fraud_rate.iloc[0])),
            ('customers_with_fraud','Customers with observed fraud',114),
            ('fraud_amount_log_w1','Fraud amount error',None),
            ('fraud_gap_seconds_log_w1','Fraud inter-arrival error',None),
            ('normal_amount_log_w1','Normal amount error',None)]
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained');colors=['#64748b','#0891b2','#7c3aed']
    for ax,(k,title,truth) in zip(axes.flat,panels):
        factor=100 if k=='generated_fraud_rate' else 1
        for i,a in enumerate(arms):
            v=d[d.arm.eq(a)][k].to_numpy()*factor
            ax.scatter(i+np.linspace(-.13,.13,len(v)),v,c=colors[i],alpha=.65,s=28)
            ax.hlines(v.mean(),i-.25,i+.25,color=colors[i],lw=3)
        if truth is not None:ax.axhline(truth*factor,color='#475569',ls='--',label=f'Real {truth*factor:.3f}');ax.legend(fontsize=8)
        ax.set_xticks(range(3),labels,rotation=23,ha='right',fontsize=9);ax.set_title(title,fontsize=11)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Percent' if factor==100 else 'Customers (of 147)' if truth is not None else 'log1p W1 (lower is better)',fontsize=9)
    fig.suptitle('ARGN joint-preservation controls, including fresh generation seeds\n2 frozen parent fits × 4 draws; points = draws, bars = means; no confidence intervals',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(DOCS/f'joint_preservation_confirmed.{ext}',dpi=180)
    svg=DOCS/'joint_preservation_confirmed.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
