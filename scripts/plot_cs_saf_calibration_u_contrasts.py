"""Paired uncertainty for the equally calibrated architecture comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    result=json.loads((root/'docs/cs_saf/calibration_u_control_v1_result.json').read_text())
    keys=list(result['contrasts']);ys=np.arange(len(keys))
    panels=[('active_short_gap_repeat_curve_l1','Generated repeat-curve L1'),
            ('active_conditional_grid_mark_TV','Conditional next-mark TV'),
            ('three_null_copy_range','Unnecessary gap response')]
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.1),layout='constrained')
    for ax,(metric,title) in zip(axes,panels):
        stats=[result['contrasts'][p]['Ecal_minus_Ucal'][metric] for p in keys]
        means=np.array([d['mean'] for d in stats])
        limits=np.array([d['descriptive_95_percent_t_interval'] for d in stats])
        ax.errorbar(means,ys,xerr=np.array([means-limits[:,0],limits[:,1]-means]),
                    fmt='o',capsize=4,color='#1766a4',lw=1.7)
        ax.axvline(0,color='#555',lw=1,ls='--')
        ax.set_yticks(ys,[f'{100*float(p):g}%' for p in keys]);ax.invert_yaxis()
        ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Ecal - Ucal (negative favors Ecal)',fontsize=9)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    axes[0].set_ylabel('Group prevalence')
    fig.suptitle('Equal calibration: generated relation benefit remains unproven\n'
                 'Five paired training trials; descriptive 95% t intervals; reused data',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(root/f'docs/cs_saf/calibration_u_control_v1_contrasts.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
