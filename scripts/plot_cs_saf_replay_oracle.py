"""Render registered cross-history and joint-oracle contrasts from published JSON."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs-saf-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/cs_saf'
COLORS={'predictor_F':'#2463a6','source_history_H':'#d48820'}

def main():
    data=json.loads((DOC/'replay_oracle_v1_result.json').read_text())
    bins=json.loads((DOC/'replay_oracle_v1_binwise.json').read_text())
    prevalences=list(data['replay_summary']); x=np.arange(4)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,mode in enumerate(('OBS','FULL')):
      for col,pair in enumerate(('E_minus_U','L003_minus_E')):
        ax=axes[row,col]
        for offset,(component,color) in zip((-.07,.07),COLORS.items()):
          values=[data['replay_summary'][pi]['kappa_1_context_1'][mode]['effects'][pair][component] for pi in prevalences]
          means=np.array([v['mean'] for v in values])*1000
          errors=np.array([[v['mean']-v['ci95_descriptive'][0],v['ci95_descriptive'][1]-v['mean']] for v in values]).T*1000
          ax.errorbar(x+offset,means,yerr=errors,fmt='o-',capsize=3,color=color,label=component)
        ax.axhline(0,color='0.35',lw=.8); ax.set_title(mode+': '+pair.replace('_minus_',' - '))
        ax.set_xticks(x, ['5%','10%','25%','50%']); ax.set_ylabel('Repeat L1 difference (x 0.001)')
        if row==1:ax.set_xlabel('Group prevalence')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Same-history predictor difference persists; source-history effects vary\nActive group; paired five-trial descriptive 95% t intervals',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91))
    for ext in ('png','pdf'):fig.savefig(DOC/f'replay_oracle_v1_cross_history.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,representation in zip(axes,('CONT','BIN')):
      for mode,color,offset in [('JOINT_'+representation,'#28756a',-.06),('FIX_'+representation,'#bf4b45',.06)]:
        values=[data['oracle_summary'][pi]['kappa_1_context_1'][mode]['matched_reference']['expected_repeat_L1'] for pi in prevalences]
        means=np.array([v['mean'] for v in values]); errors=np.array([[v['mean']-v['ci95_descriptive'][0],v['ci95_descriptive'][1]-v['mean']] for v in values]).T
        ax.errorbar(x+offset,means,yerr=errors,fmt='o-',capsize=3,color=color,label=mode)
      ax.set_title('Continuous gaps' if representation=='CONT' else 'Binned gaps (matched reference)')
      ax.set_xticks(x,['5%','10%','25%','50%']); ax.set_xlabel('Group prevalence'); ax.legend(frameon=False)
    axes[0].set_ylabel('Expected repeat-curve L1 (lower is better)')
    fig.suptitle('Even the exact online oracle is distorted by forced observed gaps\nKnown-DGP controls; intervals reflect sampling trials, not fitted models',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.88))
    for ext in ('png','pdf'):fig.savefig(DOC/f'replay_oracle_v1_joint_controls.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,4),sharey=True)
    for pi,ax in zip(prevalences,axes):
      cell=bins['cells'][f'pi_{pi}/kappa_1']['1']
      for target,color in [('E','#2463a6'),('L003','#d48820')]:
        values=cell['replay'][f'FULL/target_{target}/source_E']['signed_bias']
        ax.plot(np.arange(1,6),[v['mean'] for v in values],'o-',color=color,label=target)
      ax.axhline(0,color='0.35',lw=.8);ax.set_title(f'Prevalence {int(float(pi)*100)}%')
      ax.set_xlabel('Gap metric bin (short to long)');ax.set_xticks(range(1,6))
    axes[0].set_ylabel('Expected repeat minus validation curve');axes[0].legend(frameon=False)
    fig.suptitle('Identical E-generated FULL histories: L003 further flattens the repeat curve\nFive paired-trial means; all source/target combinations retained in binwise JSON',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.87))
    for ext in ('png','pdf'):fig.savefig(DOC/f'replay_oracle_v1_bin_calibration.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
