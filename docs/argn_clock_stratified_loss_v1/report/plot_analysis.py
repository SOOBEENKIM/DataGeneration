"""Descriptive figures only; never changes or selects a fitted model."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'local_source'))
from run_stratified import ARMS, PRIOR, verify, basis_numpy, LIMIT


def main():
    verify()
    limits=json.loads((OUT/'report/CLOCK_ENDPOINTS.json').read_text())['values']
    n=np.arange(1,3106)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,arm in zip(axes.flat,ARMS):
        quantiles=[0,.5,1] if arm=='personal_six' else [.5]
        for q in quantiles:
            x=np.c_[np.full(len(n),limits[0]+q*(limits[1]-limits[0])),np.log1p(n)]
            for folder,style,loss in [(PRIOR,'--','Coarse'),(OUT,'-','Refined')]:
                theta=json.loads((folder/arm/'FIT_COMPLETE.json').read_text())['selected']['theta']
                delta=np.clip(basis_numpy(arm,x,np.ones(len(n)),limits)@theta,-LIMIT,LIMIT)
                ax.plot(n,delta,linestyle=style,color=plt.cm.viridis(.15+.7*q),
                    label=f'{loss}, clock mix {q:g}' if arm=='personal_six' else loss)
        for b in [200,500,1024,2048]:ax.axvline(b,color='.8',linewidth=.7)
        ax.axhline(0,color='.6',linewidth=.7)
        ax.set(title=arm,xlabel='Number of previous transactions',ylabel='Normal-stay log-gap offset',ylim=(-.36,.36))
        ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Identical output families; only the 200+ loss grouping changed')
    for ext in ['png','pdf']:fig.savefig(OUT/f'report/correction_comparison.{ext}',dpi=180)
    plt.close(fig)
    p=OUT/'report/paired_models.csv'
    if not p.exists():return
    d=pd.read_csv(p)
    fields=['normal_gap_seconds_log_w1','late_normal_gap_w1','onset_personal_gap_w1','fraud_ratio_log_w1']
    titles=['All-normal time','Normal time, position 1024+','Personal fraud-onset time','Personal fraud amount']
    fig,axes=plt.subplots(1,4,figsize=(15,4.3),layout='constrained')
    for ax,field,title in zip(axes,fields,titles):
        for j,arm in enumerate(ARMS):
            for loss,offset,color in [('coarse',-.15,'#888888'),('refined',.15,'#267e9b')]:
                m=d[d.arm.eq(arm)&d.loss.eq(loss)].groupby('fit_seed')[field].mean()
                ax.scatter(np.full(len(m),j+offset),m,color=color,s=19,alpha=.55)
                ax.scatter(j+offset,m.mean(),color=color,s=70,marker='D',label=loss if j==0 else None)
        ax.set(title=title,ylabel='log-W1 (lower is better)',xticks=range(4),xticklabels=['event 2','customer 2','position 6','personal 6'])
        ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Paired free generation: dots = 2 parent means, each over 4 draws; diamonds = mean')
    for ext in ['png','pdf']:fig.savefig(OUT/f'report/paired_generation_comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
