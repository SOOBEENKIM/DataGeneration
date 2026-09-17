"""Render the two registered follow-up summaries from verified result JSON."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parents[1]/'docs/cs_saf'
d=json.loads((r/'followup_v1_result.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
models=['E','L003','ER','L030'];xs=np.array([0,.003,.01,.03]);pis=['0.05','0.1','0.25','0.5']
fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
for col,pi in enumerate(pis):
 for row,cell in enumerate(['null','active']):
  ax=axes[row,col]
  for sn,color,style in [('best','#22689b','-'),('epoch_9','#bf6036','--')]:
   vals=[d['internal'][pi][m][sn]['validation'][cell]['grid_mark_TV'] for m in models]
   means=[v['mean'] for v in vals];errors=[v['standard_error'] for v in vals]
   ax.errorbar(xs,means,yerr=errors,fmt='o'+style,color=color,capsize=3,label='best val NLL' if sn=='best' else 'fixed epoch 9')
   if sn=='best':
    for x,v in zip(xs,vals):ax.scatter(np.full(5,x),v['values'],color=color,alpha=.3,s=12)
  u=d['internal'][pi]['U']['best']['validation'][cell]['grid_mark_TV']['mean'];ax.axhline(u,color='black',lw=1,ls=':',label='U best')
  ax.set_xscale('symlog',linthresh=.003);ax.set_xticks(xs,['0 (E)','.003','.01','.03']);ax.set_title(f'Prevalence {float(pi):.0%} | {cell} TV')
  ax.set_xlabel('Residual penalty coefficient');ax.set_ylabel('Full-mark TV (lower is better)');ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8)
fig.suptitle('Frozen-model penalty cost curve: 5 paired trials, existing data seed 42\nBars: seed SE; dots: individual best-checkpoint trials. Descriptive, reused validation.',fontsize=12)
for ext in ('png','pdf'):fig.savefig(r/f'followup_v1_lambda_prevalence.{ext}')
plt.close(fig)
metrics=['transition_conditioned_mark_tv','short_gap_repeat_curve_l1','gap_repeat_mi_error']
labels=['Transition-conditioned mark TV','Gap-repeat curve L1','Gap-repeat MI error']
colors={'U':'#777777','E':'#22689b','L003':'#469b8e','ER':'#d67828','L030':'#a24755','CPAR':'#723e9c','empirical_copy_control':'#333333'}
fig,axes=plt.subplots(3,2,figsize=(11,11),constrained_layout=True)
for row,(metric,title) in enumerate(zip(metrics,labels)):
 for col,k in enumerate(('0','1')):
  ax=axes[row,col]
  for model,color in colors.items():
   vals=[d['external_generation'][pi][model][k]['context_1'][metric] for pi in pis]
   ax.errorbar([float(x) for x in pis],[v['mean'] for v in vals],yerr=[v['standard_error'] for v in vals],fmt='o--' if model=='empirical_copy_control' else 'o-',color=color,ms=4,capsize=2,label=model,lw=1.2)
  ax.set_title(f'{title} | context 1, kappa={k}');ax.set_xlabel('Context-1 prevalence');ax.set_ylabel('Error (lower is better)');ax.set_xticks([.05,.1,.25,.5],['5%','10%','25%','50%']);ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8,ncol=2)
fig.suptitle('Matched generated samples vs validation: 2048 entities, 5 trials\nCPAR: fixed default 128 epochs; training compute differs. No conditional-oracle comparison.',fontsize=12)
for ext in ('png','pdf'):fig.savefig(r/f'followup_v1_external_generation.{ext}')
plt.close(fig)
