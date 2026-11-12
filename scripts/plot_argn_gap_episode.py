"""Export standalone scientific figures; points are draws, not confidence limits."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_argn_gap_episode import DOCS


def main():
    data=pd.read_csv(DOCS/'all_metrics.csv')
    arms=['baseline','gap_only','transition_only','gap_and_transition']
    labels=['Category + amount','+ Gap','+ Transition','+ Both']
    panels=[('fraud_gap_seconds_log_w1','Fraud inter-arrival error',None),
            ('fraud_amount_log_w1','Fraud amount error',None),
            ('fraud_ratio_log_w1','Personal-relative fraud amount error',None),
            ('customers_with_fraud','Customers with observed fraud',114),
            ('all_fraud_customers','All-fraud customers',12),
            ('normal_gap_seconds_log_w1','Normal inter-arrival error',None)]
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    colors=['#64748b','#0891b2','#d97706','#7c3aed']
    for ax,(key,title,truth) in zip(axes.flat,panels):
        for i,a in enumerate(arms):
            values=data[data.arm.eq(a)][key].to_numpy()
            ax.scatter(i+np.linspace(-.11,.11,len(values)),values,c=colors[i],s=33,alpha=.65,zorder=3)
            ax.hlines(np.mean(values),i-.23,i+.23,color=colors[i],lw=3)
        if truth is not None:ax.axhline(truth,color='#334155',ls='--',lw=1.4,label=f'Real: {truth}');ax.legend(fontsize=9)
        ax.set_title(title,fontsize=11,pad=12);ax.set_xticks(range(4),labels,rotation=24,ha='right',fontsize=9)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('log1p Wasserstein distance (lower is better)' if truth is None else 'Customers (out of 147)',fontsize=9)
    fig.suptitle('ARGN: fixed category + amount, gap and transition ablations\n2 frozen parent fits × 2 generation draws; dots = draws, bars = means (not confidence intervals)',fontsize=13)
    for ext in ['png','pdf','svg']:fig.savefig(DOCS/f'gap_episode_factorial.{ext}',dpi=180)
    svg=DOCS/'gap_episode_factorial.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
