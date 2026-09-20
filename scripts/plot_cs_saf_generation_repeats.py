"""Show fresh-tape noise within each frozen model pair, preserving trial units."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    result=json.loads((root/'docs/cs_saf/generation_repeats_v1_result.json').read_text())
    keys=list(result['contrasts']);metric='active_short_gap_repeat_curve_l1'
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained',sharex=True)
    for ax,pk in zip(axes.flat,keys):
        stats=result['contrasts'][pk]['Ecal_minus_Ucal'][metric]
        tapes=np.array(stats['tape_values']);means=tapes.mean(1);xs=np.arange(5)
        for j in range(5):
            ax.scatter(j+np.linspace(-.12,.12,5),tapes[j],s=20,color='#a2a9b0',alpha=.85,
                       label='Five fresh generation tapes' if j==0 else None)
        ax.errorbar(xs,means,yerr=2.7764451051977987*tapes.std(1,ddof=1)/np.sqrt(5),
                    fmt='o',capsize=3,color='#1766a4',label='Within-trial mean / MC 95% t interval')
        old=result['historical_comparison'][pk]['Ecal_minus_Ucal']['original_single_tape']['values']
        ax.scatter(xs,old,color='#ba621f',marker='x',s=45,label='Original single tape (excluded)')
        ax.axhline(0,color='#555',ls='--',lw=1);ax.set_xticks(xs,[str(j) for j in xs])
        ax.set_title(f'Group prevalence {100*float(pk):g}%');ax.set_xlabel('Frozen training trial')
        ax.set_ylabel('Ecal - Ucal generated repeat L1');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Fixed models and plans: does the generation difference repeat?\n'
                 'Negative favors Ecal; per-trial intervals describe generation randomness only',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(root/f'docs/cs_saf/generation_repeats_v1_tapes.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    colors={'U':'#929ca6','Ucal':'#252c34','E':'#75a9d0','Ecal':'#1766a4'}
    for name in colors:
        means=[result['summaries'][p][name][metric]['mean'] for p in keys]
        axes[0].plot(range(4),means,'o-',color=colors[name],label=name)
    axes[0].set_xticks(range(4),[f'{float(p)*100:g}%' for p in keys]);axes[0].set_xlabel('Group prevalence')
    axes[0].set_title('Generated repeat-curve L1: five-tape means');axes[0].set_ylabel('Error (lower is better)')
    axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
    for j,p in enumerate(keys):
        st=result['contrasts'][p]['Ecal_minus_Ucal'][metric];mu=st['mean']
        lo,hi=st['descriptive_95_percent_t_interval'];mlo,mhi=st['conditional_MC_95_percent_interval']
        axes[1].errorbar(mu,j+.09,xerr=[[mu-lo],[hi-mu]],fmt='o',color='#1766a4',capsize=4,
                        label='Between-trial descriptive 95% interval' if j==0 else None)
        axes[1].errorbar(mu,j-.09,xerr=[[mu-mlo],[mhi-mu]],fmt='s',color='#ba621f',capsize=4,
                        label='Generation-only conditional MC interval' if j==0 else None)
    axes[1].set_yticks(range(4),[f'{float(p)*100:g}%' for p in keys]);axes[1].invert_yaxis()
    axes[1].axvline(0,color='#555',ls='--',lw=1);axes[1].grid(axis='x',alpha=.2)
    axes[1].set_title('Two distinct uncertainty questions');axes[1].set_xlabel('Ecal - Ucal repeat L1 (negative favors Ecal)')
    axes[1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17))
    fig.suptitle('Five tapes per trial are averaged first; five training trials remain',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(root/f'docs/cs_saf/generation_repeats_v1_summary.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
