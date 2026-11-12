"""Descriptive plots from completed tables, outside the frozen model/evaluator."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

report=Path(__file__).resolve().parents[1]/'report'
assert (report/'FINAL_AUDIT.json').exists()
means=pd.read_csv(report/'MEANS.csv').set_index('arm')
fits=pd.read_csv(report/'FIT_MEANS.csv')
layout=[['history_only','history_initial_only'],['history_calibration_only','history_init_cal']]
metrics=[('normal_gap_seconds_log_w1','Normal timing'),('onset_personal_gap_w1','Personal timing at fraud onset'),
         ('normal_stay_personal_gap_w1','Personal timing: normal stays'),('early_time','Normal timing: positions 6–49')]
fig,axes=plt.subplots(1,4,figsize=(13.5,4.1))
for ax,(metric,title) in zip(axes,metrics):
    a=np.array([[means.loc[arm,metric] for arm in row] for row in layout])
    ax.imshow(a,cmap='Blues',vmin=0,vmax=a.max()*1.7,aspect='auto')
    for y,row in enumerate(layout):
        for x,arm in enumerate(row):
            vals=fits[fits.arm.eq(arm)][metric]
            ax.text(x,y,f'{means.loc[arm,metric]:.4f}\n({vals.min():.4f}–{vals.max():.4f})',
                    ha='center',va='center',fontsize=9)
    ax.set_xticks([0,1],['Initial off','Initial on']);ax.set_yticks([0,1],['Calibration off','Calibration on'])
    ax.set_title(title,fontsize=10);ax.tick_params(length=0,labelsize=8)
    for s in ax.spines.values():s.set_visible(False)
fig.suptitle('Frozen history model: initial law × calibration',fontsize=12)
fig.text(.05,.015,'Cells: mean error; parentheses: range of 3 existing time-fit means. Lower is better.\n'
         'Each fit averages 2 frozen ARGN parents × 2 generation draws. Reused development; no new fits or confidence intervals.',fontsize=8)
fig.tight_layout(rect=(0,.13,1,.93))
for ext in ['png','pdf']:fig.savefig(report/f'FACTORIAL_COMPARISON.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
