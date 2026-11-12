"""Postprocessing only: paired effects and standalone scientific figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parents[1]/'report'
d=pd.read_csv(P/'all_models.csv')
arms=['current','init_only','handoff_only','init_handoff']
names=['Current candidate','Initial output only','Handoff only','Initial + handoff']
colors=['#62758b','#268c9b','#aa8841','#5269b7']
fig,axes=plt.subplots(2,2,figsize=(11.4,7.6),layout='constrained')
for ax,key,title in zip(axes.flat,['early_time','normal_gap_seconds_log_w1','late1024_time','onset_personal_gap_w1'],
                       ['Early normal intervals (6-49)','All normal intervals','Late normal intervals (1024+)','Personal fraud-onset interval relation']):
    for i,arm in enumerate(arms):
        vals=d[d.arm.eq(arm)].sort_values(['fit_seed','generation_seed'])[key].to_numpy()
        ax.bar(i,vals.mean(),color=colors[i],alpha=.78,width=.68)
        ax.scatter(i+np.linspace(-.15,.15,len(vals)),vals,s=18,color='#222222',alpha=.65,zorder=3)
        ax.text(i,vals.mean(),f'{vals.mean():.3f}',va='bottom',ha='center',fontsize=9)
    ax.set_xticks(range(4),['Current','Initial','Handoff','Both'])
    ax.set(title=title,ylabel='W1 (lower is better)');ax.grid(axis='y',alpha=.18)
fig.suptitle('Initial output / downstream calibration | 2 frozen parents x 4 generation draws')
fig.savefig(P/'initial_coupling_comparison.png',dpi=180)
fig.savefig(P/'initial_coupling_comparison.pdf')
plt.close(fig)

# Descriptive interaction on each matched draw; a nonlinear W1 difference is not
# an additive causal contribution or a test of independently trained heads.
rows=[]
fields=['early_time','time100_199','normal_gap_seconds_log_w1','late1024_time',
        'onset_personal_gap_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1','normal_amount_log_w1']
for (fs,gs),group in d[d.arm.isin(arms)].groupby(['fit_seed','generation_seed']):
    g=group.set_index('arm')
    for field in fields:
        rows.append(dict(fit_seed=fs,generation_seed=gs,metric=field,
            initial_only_change=g.loc['init_only',field]-g.loc['current',field],
            handoff_only_change=g.loc['handoff_only',field]-g.loc['current',field],
            handoff_after_initial_change=g.loc['init_handoff',field]-g.loc['init_only',field],
            combined_change=g.loc['init_handoff',field]-g.loc['current',field],
            interaction=(g.loc['init_handoff',field]-g.loc['init_only',field])-
                        (g.loc['handoff_only',field]-g.loc['current',field])))
out=pd.DataFrame(rows);out.to_csv(P/'factorial_contrasts.csv',index=False)
out.groupby('metric').mean(numeric_only=True).to_csv(P/'factorial_mean_contrasts.csv')
print('PLOTS_AND_DESCRIPTIVE_CONTRASTS_WRITTEN',flush=True)
