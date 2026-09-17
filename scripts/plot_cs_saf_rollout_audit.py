from pathlib import Path
import json
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs_saf_rollout_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'artifacts/cs_saf/rollout_audit_v1/summary.json').read_text())
if summary['publication_screen']!='PASS': raise RuntimeError('No publication without registered evidence screen')
OUT=ROOT/'artifacts/cs_saf/rollout_audit_v1/figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
pis=['0.05','0.10','0.25','0.50'];pairs=['E_minus_U','L003_minus_E'];colors=['#3179a8','#dc8b21','#359274','#a75f98']
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,constrained_layout=True)
metrics=['grid_mark_TV','grid_repeat_L1','grid_excess_TV'];labels=['Full-mark TV','Repeat absolute error','TV excess']
for ax,pair in zip(axes,pairs):
 for j,(metric,label,color) in enumerate(zip(metrics,labels,colors)):
  stats=[summary['paired_contrasts'][pair][pi]['kappa_1_context_1']['TF_all'][metric] for pi in pis]
  ax.bar(np.arange(4)+(j-1)*.23,[x['mean'] for x in stats],width=.23,label=label,color=color)
 ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(4),['5%','10%','25%','50%']);ax.set_xlabel('Active-group prevalence');ax.set_title(pair.replace('_minus_',' minus '));ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Paired error difference (candidate - comparator)');axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Observed-history endpoint decomposition | 5 paired model trials')
for ext in ['png','pdf']:fig.savefig(OUT/f'rollout_audit_v1_endpoint_decomposition.{ext}',dpi=220)
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
stages=['TF_panel','OBS','QUANT','GAP','FULL'];stage_labels=['Real history','Marks\nrecursive','Gap\nquantized','Gaps\nsampled','Values\nsampled']
for col,pair in enumerate(pairs):
 for row,metric in enumerate(['expected_repeat_L1','empirical_repeat_L1']):
  ax=axes[row,col]
  # TF empirical error is panel sampling discrepancy, not generated prediction.
  use=stages if row==0 else stages[1:]
  labels=stage_labels if row==0 else stage_labels[1:]
  for pi,color in zip(pis,colors):
   stats=[summary['paired_contrasts'][pair][pi]['kappa_1_context_1'][domain][metric] for domain in use]
   means=np.array([s['mean'] for s in stats]);bounds=np.array([s['ci95_descriptive'] for s in stats])
   ax.errorbar(np.arange(len(use)),means,yerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),color=color,marker='o',capsize=2,label=f'{float(pi)*100:g}%',alpha=.85)
  ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(len(use)),labels);ax.set_title(pair.replace('_minus_',' minus '));ax.grid(axis='y',alpha=.15)
  if col==0:ax.set_ylabel(('Expected' if row==0 else 'Empirical')+' repeat-curve L1 difference')
axes[0,0].legend(frameon=False,ncol=4,fontsize=8)
fig.suptitle('Prefix-anchored sampling sensitivity | 3 tapes averaged per model trial\nBars: descriptive paired 95% t intervals; reused data, no multiplicity adjustment')
for ext in ['png','pdf']:fig.savefig(OUT/f'rollout_audit_v1_sampling_stages.{ext}',dpi=220)
plt.close(fig)
print(str(OUT))
