"""Plot registered contrasts and complete active-group matched-history curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/cs_saf'
r=json.loads((DOC/'calibrated_replay_v1_result.json').read_text())
prevalences=['0.05','0.10','0.25','0.50']; labels=['5%','10%','25%','50%']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=['#31688e','#35b779','#d97706','#222222']
fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1.3,1]})
ax=axes[0]
for j,(name,label) in enumerate(zip(['predictor_F','source_H','empirical_remainder_S','native_difference'],
    ['F: predictor on matched inputs','H: source history + current gap','S: empirical-score remainder','Native Ecal - Ucal'])):
    values=[r['summary'][p]['1']['context_1'][name] for p in prevalences]
    means=np.array([v['mean'] for v in values]);ci=np.array([v['descriptive_95_percent_t_interval'] for v in values])
    ax.errorbar(np.arange(4)+(j-1.5)*.14,means,yerr=[means-ci[:,0],ci[:,1]-means],fmt='o',
        color=colors[j],capsize=3,label=label,markersize=4)
ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(np.arange(4),labels);ax.set_xlabel('Active group prevalence')
ax.set_ylabel('Difference in repeat-curve L1');ax.set_title('A. Native error = F + H + S')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=8,frameon=False,ncol=2)
ax.text(.01,.02,'Bars: descriptive 95% intervals across 5 trial means.\nEach trial first averages 5 frozen generation repeats.',transform=ax.transAxes,fontsize=8)
ax=axes[1]
mat=[]
for p in prevalences:
    m=r['matrices'][p]['1']['context_1']
    mat.append([m[k]['mean'] for k in ('Ucal_on_Ucal','Ecal_on_Ucal','Ucal_on_Ecal','Ecal_on_Ecal')])
mat=np.array(mat)
im=ax.imshow(mat,cmap='Blues',aspect='auto')
for i in range(4):
    for j in range(4):ax.text(j,i,f'{mat[i,j]:.5f}',ha='center',va='center',color='white' if mat[i,j]>(mat.min()+mat.max())/2 else 'black',fontsize=10)
ax.set_xticks(range(4),['U on U','E on U','U on E','E on E'],rotation=20);ax.set_yticks(range(4),labels)
ax.set_title('B. Predicted-probability curve L1');ax.set_xlabel('Target predictor on source paths (all calibrated)')
fig.colorbar(im,ax=ax,label='Lower is better',fraction=.045,pad=.025)
fig.suptitle('Frozen Ucal / Ecal cross-history replay | controlled active group',fontsize=13)
fig.subplots_adjust(bottom=.24,top=.85,wspace=.34)
for ext in ('png','pdf'):fig.savefig(DOC/f'calibrated_replay_v1_decomposition.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

bins=pd.read_csv(DOC/'calibrated_replay_v1_bins.csv')
fig,axes=plt.subplots(2,4,figsize=(13,6.4),sharex=True,sharey=True)
for row,source in enumerate(('Ucal','Ecal')):
    for col,(p,label) in enumerate(zip(prevalences,labels)):
        ax=axes[row,col];b=bins[(bins.prevalence==float(p))&(bins.kappa==1)&(bins.group=='context_1')&(bins.source==source)]
        # Equal weighting after averaging repeats within each trial.
        means=b.groupby(['trial','bin'])[['Ucal_repeat','Ecal_repeat','reference_repeat']].mean().groupby('bin').mean()
        for target,color in [('Ucal','#31688e'),('Ecal','#c44e52')]:
            ax.plot(np.arange(1,6),means[target+'_repeat']-means.reference_repeat,'o-',color=color,label=target,markersize=4)
        ax.axhline(0,color='gray',linestyle='--',linewidth=.8);ax.set_xticks(range(1,6))
        ax.set_title(f'{label}, {source} source paths')
        if row==1:ax.set_xlabel('Train-fitted gap bin (short to long)')
        if col==0:ax.set_ylabel('Predicted repeat - validation repeat')
axes[0,0].legend(frameon=False)
fig.suptitle('Same inputs, different predictors: signed bin bias\nMeans over 5 training trials, each averaging 5 saved generation repeats',fontsize=12)
fig.tight_layout(rect=(0,0,1,.92))
for ext in ('png','pdf'):fig.savefig(DOC/f'calibrated_replay_v1_bin_bias.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
