"""Regenerate the completed replication scientific figures from checked-in evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];doc=root/'docs/cs_saf'
r=json.loads((doc/'replication_v1_result.json').read_text())
colors={'CS2-U1':'#666666','CS3-E1':'#3476bd','CS4-ER1':'#cb4b30'}
labels={'CS2-U1':'U: ordinary','CS3-E1':'E: history + raw','CS4-ER1':'ER: E + residual penalty'}
nulls=((0,0),(0,1),(1,0))

def value(stage,t,c,k,y,metric,cp='best',split='validation'):
 d=stage['jobs'][f'trial_{t}/kappa_{k}/{c}']['accuracy']['checkpoints']
 if cp not in d:return np.nan
 return d[cp]['splits'][split]['groups'][str(y)]['metrics'][metric]['mean']

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
for pi,stage in r['stages'].items():
 fig,axs=plt.subplots(2,2,figsize=(11,7.4),layout='constrained');x=np.arange(1,6)
 for c in colors:
  a=[max(value(stage,t,c,k,y,'copy_range') for k,y in nulls) for t in range(5)]
  axs[0,0].plot(x,a,'o-',color=colors[c],label=labels[c])
  b=[value(stage,t,c,1,1,'grid_mark_TV') for t in range(5)]
  axs[0,1].plot(x,b,'o-',color=colors[c],label=labels[c])
 axs[0,0].axhline(.05,color='#111111',ls='--',lw=1,label='Registered null bound')
 axs[0,0].set(title='A. Maximum of three null copy responses',ylabel='Response range (lower better)')
 axs[0,0].legend(fontsize=8,loc='best')
 axs[0,1].set(title='B. Active conditional distribution error',ylabel='Full-mark grid TV (lower better)')
 for c in ('CS3-E1','CS2-U1'):
  name='CS4-ER1_minus_'+c;label='ER - '+('E' if c=='CS3-E1' else 'U')
  n=stage['gate']['accuracy_screens'][c]['null']['values']
  axs[1,0].plot(x,n,'o-',color=colors[c],label=label)
  for cp,ls,marker in (('best','-','o'),('epoch_9','--','x')):
   a=[value(stage,t,'CS4-ER1',1,1,'grid_mark_TV',cp)-value(stage,t,c,1,1,'grid_mark_TV',cp) for t in range(5)]
   axs[1,1].plot(x,a,color=colors[c],ls=ls,marker=marker,label=label+' ('+cp+')')
 for ax in axs[1]:ax.axhline(0,color='#111111',lw=1);ax.legend(fontsize=8)
 axs[1,0].set(title='C. Paired change in equal-null mean TV',ylabel='TV change (negative favors ER)')
 axs[1,1].set(title='D. Paired active TV and snapshot sensitivity',ylabel='TV change (negative favors ER)')
 for ax in axs.flat:ax.set_xticks(x);ax.set_xlabel('Registered trial (five new seed triples)');ax.grid(alpha=.15)
 fig.suptitle(f'Fixed U/E/ER replication | prevalence {float(pi):.0%} | gate: {stage["gate"]["decision"]}',fontsize=14)
 stem=doc/f'replication_v1_pi_{pi}_seed_results'
 fig.savefig(str(stem)+'.png',dpi=170)
 fig.savefig(str(stem)+'.pdf')
 plt.close(fig)
 print(stem)
