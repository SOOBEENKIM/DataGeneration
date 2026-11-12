"""Publication-exportable diagnostic figure; points denote parent-fit means."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];DOCS=ROOT/'docs/argn_phase_gap_v1'


def main():
    d=pd.read_csv(DOCS/'all_metrics.csv');fits=d.groupby(['arm','fit_seed']).mean(numeric_only=True)
    names=['frozen','all_label_fit','boundary_fit','phase_mixture']
    labels=['Frozen\ncandidate','Extra training\ncontrol','Boundary\nexpert','Phase-only\nmixture']
    colors=['#64748b','#94a3b8','#2563eb','#d97706']
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.2),layout='constrained')
    for ax,(key,title) in zip(axes,[('boundary_gap_mean_w1','Start / return gap'),('fraud_gap_seconds_log_w1','All fraud gaps'),('fraud_ratio_log_w1','Personal-relative fraud amount')]):
        means=[d.loc[d.arm.eq(a),key].mean() for a in names]
        ax.bar(np.arange(4),means,color=colors,width=.62,alpha=.85)
        for i,a in enumerate(names):
            ys=fits.loc[a,key].to_numpy();ax.scatter(i+np.linspace(-.10,.10,len(ys)),ys,c='black',s=18,zorder=3)
            ax.text(i,max(ys.max(),means[i])+max(means)*.035,f'{means[i]:.3f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_title(title,fontsize=11);ax.set_ylabel('Log-Wasserstein distance (lower is better)',fontsize=9)
        ax.set_ylim(0,max(fits[key].max(),max(means))*1.25);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Development controls: improve rare transitions without hiding marginal costs',fontsize=13)
    fig.supxlabel('Bars: 2 parent fits × 2 generation seeds. Dots: parent-fit means. Identical labels and sequence lengths.',fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(DOCS/f'boundary_controls.{suffix}',dpi=180)
    plt.close(fig)
    gpu=pd.read_csv(DOCS/'personal_gap.csv');cpu=pd.read_csv(DOCS/'confirmation_cpu/personal_gap.csv')
    arms=['frozen','boundary_fit','phase_mixture'];titles=['Fraud onset / personal past gap','Normal return / personal past gap','Normal stay / personal past gap']
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.3),layout='constrained')
    for ax,phase,title in zip(axes,['onset','return_normal','normal_stay'],titles):
        for j,(data,label,color) in enumerate([(gpu,'Initial GPU draws','#2563eb'),(cpu,'Additional CPU draws','#d97706')]):
            means=data[data.phase.eq(phase)].groupby('arm').relative_gap_w1.mean()
            ax.bar(np.arange(3)+(j-.5)*.34,[means[a] for a in arms],width=.32,label=label,color=color)
        ax.set_xticks(range(3),['Frozen\ncandidate','Boundary\nexpert','Phase-only\nmixture'],fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_ylabel('Log-Wasserstein distance (lower is better)',fontsize=9)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle('Personal timing: onset advantage repeats; return advantage does not',fontsize=13)
    fig.supxlabel('Each color: 2 parent fits × 2 draws, with its own matched-device reference. Compare models within each color.',fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(DOCS/f'personal_timing_tradeoff.{suffix}',dpi=180)
    for path in [DOCS/'boundary_controls.svg',DOCS/'personal_timing_tradeoff.svg']:
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':main()
