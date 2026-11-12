"""Show normal marginal timing cost beside both improved temporal relationships."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from run_argn_clock_regression import DOCS

data=pd.read_csv(DOCS/'all_models.csv').groupby(['arm','fit_seed']).mean(numeric_only=True)
arms=['frozen','boundary_fit','phase_mixture','history_only','history_clock','history_relative','clock_joint']
labels=['Frozen candidate','Boundary head','Phase GMM','History density','History + clock','Clock-relative density','Conditional clock GMR']
colors=['#777777','#b34f62','#d29c2a','#458baa','#458266','#7750a6','#cf672c']
fields=['fraud_gap_seconds_log_w1','normal_gap_seconds_log_w1','normal_stay_personal_gap_w1']
titles=['All-fraud gap W1 (log axis)','All-normal gap W1 (log axis)','Normal personal-gap W1']
fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
for arm,label,color in zip(arms,labels,colors):
    d=data.loc[arm]
    for ax,field in zip(axes,fields):
        ax.scatter(d[field],d.onset_personal_gap_w1,color=color,s=25,alpha=.4)
        ax.scatter(d[field].mean(),d.onset_personal_gap_w1.mean(),color=color,marker='X',s=90,label=label)
for ax,title in zip(axes,titles):ax.set_xlabel(title);ax.grid(alpha=.2)
axes[0].set_xscale('log');axes[1].set_xscale('log')
axes[0].set_ylabel('Fraud-onset personal-gap W1')
handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=4,fontsize=9)
fig.suptitle('Time-output trade-offs: lower is better; dots = parent fits, X = mean of two fits')
fig.subplots_adjust(bottom=.24,top=.9,wspace=.12)
for ext in ['png','pdf','svg']:fig.savefig(DOCS/f'time_density_full_tradeoff.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
