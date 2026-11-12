"""Standalone factorial result figure; dots are draws, not confidence intervals."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/argn-output-learning-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/argn_category_amount_v1'


def main():
    d=pd.read_csv(DOCS/'all_metrics.csv')
    order=['neither','amount_only','category_only','category_and_amount']
    labels=['Existing\ncontrol','Amount\nonly','Category\nonly','Category +\namount']
    metrics=[('fraud_amount_log_w1','Fraud amount','log(1 + amount) Wasserstein'),
             ('fraud_ratio_log_w1','Personal-relative fraud amount','log-relative Wasserstein'),
             ('class_1_category_amount_tv','Fraud category–amount relation','Total variation')]
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.0),layout='constrained')
    colors=['#777777','#0072B2','#E69F00','#009E73']
    for ax,(key,title,ylabel) in zip(axes,metrics):
        for x,(arm,color) in enumerate(zip(order,colors)):
            vals=d[d.arm.eq(arm)].sort_values(['fit_seed','generation_seed'])[key].to_numpy()
            assert len(vals)==4
            ax.scatter(x+np.array([-.09,-.03,.03,.09]),vals,color=color,s=32,alpha=.9,zorder=3)
            ax.hlines(vals.mean(),x-.24,x+.24,color=color,lw=2.5)
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(title,fontsize=11)
        ax.set_ylabel(ylabel,fontsize=9);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('ARGN output-learning controls — lower is better',fontsize=13)
    fig.supxlabel('Same duration label model; dots: 2 fit seeds × 2 generation seeds; lines: means (not confidence intervals).',fontsize=8)
    for suffix in ['png','svg','pdf']:
        path=DOCS/f'output_learning_factorial.{suffix}'
        fig.savefig(path,dpi=180)
        if suffix=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
