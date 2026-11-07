"""Supplemental display only. Reads completed registered evaluation tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
assert (ROOT/'FINAL_AUDIT.json').exists()
f=pd.read_csv(ROOT/'FIT_MEANS.csv')
arms=['history_only','history_global','history_position3','history_position6','history_calibration_only','init_only']
labels=['No calibration','Global (1)','Position (3)','Position (6)','Personal + position (6)','GMR init_only']
metrics=['normal_gap_seconds_log_w1','onset_personal_gap_w1','normal_stay_personal_gap_w1']
titles=['Primary: normal timing','Primary: relative fraud-onset timing','Auxiliary: relative normal timing']
colors=['#7d8798','#356ca5','#45a39b','#8565b6','#cf6a29','#879044']
fig,axs=plt.subplots(1,3,figsize=(14,5.7),layout='constrained')
for ax,key,title in zip(axs,metrics,titles):
    for j,arm in enumerate(arms):
        vals=f[f.arm.eq(arm)].sort_values('time_fit_seed')[key].to_numpy();assert len(vals)==3
        ax.scatter(j+np.linspace(-.12,.12,3),vals,color=colors[j],s=35,alpha=.8)
        ax.plot([j-.22,j+.22],[vals.mean()]*2,color=colors[j],lw=3)
    ax.set_xticks(range(6),labels,rotation=48,ha='right',fontsize=9)
    ax.set_ylabel('W1 error (lower is better)');ax.set_title(title,fontsize=11)
    ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Frozen time-head calibration controls — Sparkov development',fontsize=14)
fig.supxlabel('Dots: 3 time-fit means over 2 frozen parents × 2 draws. Bars: mean, not confidence intervals.',fontsize=10)
fig.savefig(ROOT/'CALIBRATION_COMPARISON.png',dpi=180)
fig.savefig(ROOT/'CALIBRATION_COMPARISON.pdf')
plt.close(fig)
