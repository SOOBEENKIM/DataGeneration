"""Reproducible figures for the frozen gap-calibration result; no model fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
# Optional explicit root permits preparing this script outside the clean execution tree.
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
D=a.root/'docs/cs_saf';r=json.loads((D/'gap_calibration_v1_result.json').read_text())
pis=['0.05','0.10','0.25','0.50'];x=np.arange(4);labels=['5%','10%','25%','50%']
colors={'Ucal':'#63869a','Ugap':'#007c91','Ecal':'#c79674','Egap':'#b24820'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'font.family':'DejaVu Sans'})
def setup(ax,title,ylabel):
    ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.set_xticks(x,labels);ax.set_xlabel('Active-group prevalence');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.18)
def stats(pair,metric):return [r['contrasts'][pi][pair][metric] for pi in pis]
def paired(ax,pair,metric,offset,color,label):
    st=stats(pair,metric);mean=np.array([s['mean'] for s in st]);ci=np.array([s['descriptive_95_percent_t_interval'] for s in st]);xx=x+offset
    ax.errorbar(xx,mean,yerr=np.stack([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=4,color=color,label=label,lw=1.8)
    for j,s in enumerate(st):ax.scatter(xx[j]+np.linspace(-.035,.035,5),s['values'],s=13,alpha=.5,color=color)
    ax.axhline(0,color='#555',lw=1,ls='--')
def save(fig,name):
    fig.savefig(D/(name+'.png'),bbox_inches='tight');fig.savefig(D/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
metric='active_short_gap_repeat_curve_l1'
fig,axs=plt.subplots(1,3,figsize=(16.8,5.2),layout='constrained')
for n in ['Ucal','Ugap','Ecal','Egap']:
    axs[0].plot(x,[r['summaries'][pi][n][metric]['mean'] for pi in pis],marker='o',color=colors[n],ls='--' if n.endswith('cal') else '-',label=n,lw=2)
setup(axs[0],'A  Generated repeat-relation error','Native repeat-curve L1 (lower is better)');axs[0].legend(ncol=2,frameon=False,fontsize=10)
paired(axs[1],'Ecal_minus_Ucal',metric,-.1,'#9a9a9a','Before gap correction: Ecal − Ucal')
paired(axs[1],'Egap_minus_Ugap',metric,.1,colors['Egap'],'After equal correction: Egap − Ugap')
setup(axs[1],'B  Does E add a generation benefit?','Paired difference (negative favors E)');axs[1].legend(frameon=False,fontsize=8.5,loc='best')
paired(axs[2],'Ugap_minus_Ucal',metric,-.1,colors['Ugap'],'Ugap − Ucal')
paired(axs[2],'Egap_minus_Ecal',metric,.1,colors['Egap'],'Egap − Ecal')
setup(axs[2],'C  What did gap correction change?','Paired difference (negative is improvement)');axs[2].legend(frameon=False,fontsize=9)
fig.suptitle('Equal, bounded gap-bin calibration: frozen U and E',fontsize=17,fontweight='bold')
fig.supxlabel('5 generation tapes averaged within each of 5 training trials. Dots: trial effects; bars: descriptive 95% t intervals.\nSame synthetic data seed and explored validation; exploratory, not independent confirmation.',fontsize=10)
save(fig,'gap_calibration_v1_generation')
fig,axs=plt.subplots(2,2,figsize=(13.5,9.4),layout='constrained');ax=axs.flat
for m,o,c,l in [('grid_mark_TV',-.09,'#754a9e','Full gap-grid mark TV'),('factual_mark_TV',.09,'#278b74','Observed-gap mark TV')]:
    paired(ax[0],'Egap_minus_Ugap','active_conditional_'+m,o,c,l)
setup(ax[0],'A  Active conditional prediction','Egap − Ugap: mark TV');ax[0].legend(frameon=False,fontsize=9)
paired(ax[1],'Egap_minus_Ugap','active_gap_repeat_mi_error',0,colors['Egap'],'Native MI error')
setup(ax[1],'B  Generated gap–repeat association','Egap − Ugap: absolute MI error')
for j,measure in enumerate(['copy','repeat']):
    for o,(k,g),c in zip([-.14,0,.14],[(0,'0'),(0,'1'),(1,'0')],['#476b9b','#b57b26','#6b8e4c']):
        paired(ax[j+2],'Egap_minus_Ugap',f'null_k{k}_c{g}_{measure}_range',o,c,f'kappa {k}, context {g}')
    setup(ax[j+2],f'{chr(67+j)}  Each inactive cell: {measure} response',f'Egap − Ugap: {measure} range');ax[j+2].legend(frameon=False,fontsize=9)
fig.suptitle('Joint checks after applying the same correction to both models',fontsize=16,fontweight='bold')
fig.supxlabel('Below zero favors Egap. Inactive cells are shown separately, not hidden in a pooled mean.\nConditional endpoints have 5 fixed-model values; generation repeats do not increase that sample size.',fontsize=10)
save(fig,'gap_calibration_v1_joint')
print('Saved generation and joint-check figures (PNG/PDF)')
