"""Report all preregistered cells, including insufficient coverage and failed screens."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cs_saf_history_diagnostic_v1 import DOC, OUT, ROOT, LABELS, digest, write


def main():
    done=json.loads((OUT/'DONE.json').read_text())
    for path,h in done['files'].items():assert digest(ROOT/path)==h
    pred=pd.read_csv(DOC/'prediction_by_tape.csv');gen=pd.read_csv(DOC/'generation_by_tape.csv')
    boot=np.load(OUT/'prediction_bootstrap.npz')['bias'];assert len(boot)==len(pred)
    pk=['kappa','trial','source','predictor','group','dimension','bin'];parents=[]
    quantities=['bias','mae','squared_probability_error','brier','mark_nll','sampled_residual',
        'mean_prediction','alternate_bias','alternate_mae','oracle_information_shift']
    for key,f in pred.groupby(pk,sort=False):
        draw=boot[f.index].mean(0)
        parents.append(dict(zip(pk,key),**{v:float(f[v].mean()) for v in quantities},
            bias_lo=float(np.nanquantile(draw,.025)),bias_hi=float(np.nanquantile(draw,.975)),
            minimum_entities=int(f.entities.min()),minimum_transitions=int(f.transitions.min()),
            adequate=bool(f.adequate.all()),tapes=len(f)))
    p=pd.DataFrame(parents);p.to_csv(DOC/'prediction_by_parent.csv',index=False)
    groupkeys=[k for k in pk if k!='trial'];screens=[]
    for key,f in p.groupby(groupkeys,sort=False):
        sign=np.sign(f.bias.mean());consistent=bool((np.sign(f.bias)==sign).all())
        intervals=bool((f.bias_lo>0).all() if sign>0 else (f.bias_hi<0).all())
        adequate=bool(f.adequate.all())
        flag=adequate and consistent and intervals and abs(f.bias.mean())>=.02 and (f.bias.abs()>=.01).all()
        screens.append(dict(zip(groupkeys,key),bias=float(f.bias.mean()),mae=float(f.mae.mean()),
            squared_probability_error=float(f.squared_probability_error.mean()),
            alternate_bias=float(f.alternate_bias.mean()),information_shift=float(f.oracle_information_shift.mean()),
            min_seed_bias=float(f.bias.min()),max_seed_bias=float(f.bias.max()),
            adequate=adequate,consistent_sign=consistent,intervals_exclude_zero=intervals,
            conditional_error_candidate=bool(flag)))
    ps=pd.DataFrame(screens);ps.to_csv(DOC/'prediction_screens.csv',index=False)
    keys=['kappa','source','group','dimension','bin'];oracle=[]
    for key,f in gen[gen.source.str.startswith('JOINT')].groupby(keys,sort=False):
        f=f.sort_values('tape');assert len(f)==30
        for metric in ('continuation','occupancy','law_residual'):
            triples=f[metric].to_numpy().reshape(10,3).mean(1)
            oracle.append(dict(zip(keys,key),metric=metric,mean=float(triples.mean()),
                q05=float(np.quantile(triples,.05)),q95=float(np.quantile(triples,.95)),
                minimum=float(triples.min()),maximum=float(triples.max()),
                adequate_all=bool(f.adequate.all()),min_entities=int(f.entities.min()),
                **{f'triplet_{i}':float(v) for i,v in enumerate(triples)}))
    oracle=pd.DataFrame(oracle);oracle.to_csv(DOC/'oracle_triplets.csv',index=False)
    own=gen[gen.source.isin(['U','G'])];gp=own.groupby(['trial',*keys],sort=False)
    g=gp[['continuation','occupancy','law_residual']].mean().reset_index()
    g['adequate']=gp.adequate.all().to_numpy();g['min_entities']=gp.entities.min().to_numpy()
    g.to_csv(DOC/'generation_by_parent.csv',index=False);gs=[]
    for key,f in g.groupby(keys,sort=False):
        meta=dict(zip(keys,key))
        for metric in ('continuation','occupancy'):
            for ref in ('JOINT_BIN','JOINT_CONT'):
                q=oracle[(oracle.kappa==meta['kappa'])&(oracle.source==ref)&(oracle.group==meta['group'])&
                    (oracle.dimension==meta['dimension'])&(oracle.bin==meta['bin'])&(oracle.metric==metric)].iloc[0]
                sameoutside=bool((f[metric]>q.q95).all() or (f[metric]<q.q05).all())
                delta=float(f[metric].mean()-q['mean'])
                adequate=bool(f.adequate.all() and q.adequate_all)
                gs.append(dict(**meta,metric=metric,reference=ref,model_mean=float(f[metric].mean()),
                    oracle_mean=float(q['mean']),delta=delta,oracle_q05=q.q05,oracle_q95=q.q95,
                    adequate=adequate,all_three_outside_same_side=sameoutside,
                    generated_error_candidate=bool(adequate and sameoutside and abs(delta)>=.02)))
    gs=pd.DataFrame(gs);gs.to_csv(DOC/'generation_screens.csv',index=False)
    mainps=ps[ps.dimension!='all'];maings=gs[(gs.dimension!='all')&(gs.reference=='JOINT_BIN')]
    write(DOC/'decision.json',dict(role='exploratory_diagnostic_not_adoption_test',
        conditional_flags=mainps[mainps.conditional_error_candidate].to_dict('records'),
        generated_flags=maings[maings.generated_error_candidate].to_dict('records'),
        conditional_cells=len(mainps),generated_primary_cells=len(maings),
        conditional_low_coverage=int((~mainps.adequate).sum()),generated_low_coverage=int((~maings.adequate).sum()),
        no_model_adopted=True,previous_C_B_P_failures_unchanged=True,independent_confirmation=False))
    # Show every group and coupling condition; color scale shared across panels.
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for row,dim in enumerate(('history','run')):
        for col,(k,group) in enumerate(((0,0),(0,1),(1,0),(1,1))):
            labels=LABELS[dim];data=[];names=[];adequacy=[]
            for hist in ('real','U','G'):
                for predictor in ('U','G'):
                    f=ps[(ps.kappa==k)&(ps.group==group)&(ps.dimension==dim)&(ps.source==hist)&(ps.predictor==predictor)].set_index('bin')
                    data.append(f.loc[labels,'bias'].to_numpy());names.append(f'{predictor} on {hist}')
                    adequacy.append(f.loc[labels,'adequate'].to_numpy())
            data=np.array(data)*100;ax=axes[row,col]
            im=ax.imshow(data,cmap='RdBu_r',vmin=-8,vmax=8,aspect='auto')
            ax.set_xticks(range(4),labels);ax.set_yticks(range(6),names)
            ax.set_title(f'kappa={k}, group={group}: {dim}')
            for i in range(6):
                for j in range(4):ax.text(j,i,f'{data[i,j]:.1f}'+('' if adequacy[i][j] else '*'),ha='center',va='center',fontsize=8)
    fig.suptitle('* Low coverage: fewer than 50 sequences or 200 transitions; no error-screen conclusion',fontsize=10)
    fig.colorbar(im,ax=axes,label='Repeat probability bias (percentage points)')
    fig.savefig(DOC/'conditional_bias.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for row,metric in enumerate(('continuation','occupancy')):
        for col,(k,group) in enumerate(((0,0),(0,1),(1,0),(1,1))):
            ax=axes[row,col];labels=LABELS['run'];x=np.arange(4)
            q=oracle[(oracle.kappa==k)&(oracle.group==group)&(oracle.dimension=='run')&(oracle.source=='JOINT_BIN')&(oracle.metric==metric)].set_index('bin').loc[labels]
            ax.fill_between(x,q.q05,q.q95,color='gray',alpha=.25,label='Oracle triplet 5-95%')
            ax.plot(x,q['mean'],'k--',label='Oracle mean')
            for name in ('U','G'):
                sub=g[(g.kappa==k)&(g.group==group)&(g.dimension=='run')&(g.source==name)]
                a=sub.groupby('bin')[metric].mean().reindex(labels)
                ax.plot(x,a,'o-',label=name+'+A')
                for _,f in sub.groupby('trial'):ax.scatter(x,f.set_index('bin').loc[labels,metric],s=10,alpha=.4)
            ax.set_xticks(x,[label+('*' if group==1 and label=='8+' else '') for label in labels]);ax.set_xlabel('Prior run length');ax.set_title(f'kappa={k}, group={group}: {metric}')
    fig.suptitle('* Low coverage; oracle bands describe ten means of three tapes, not confidence intervals',fontsize=10)
    axes[0,0].legend(fontsize=7);fig.savefig(DOC/'run_continuation_occupancy.png',dpi=180);plt.close(fig)
    selected=ps[(ps.kappa==1)&(ps.group==1)&(ps.dimension=='all')]
    selected.to_csv(DOC/'active_cross_history.csv',index=False)
    print(json.dumps(dict(conditional_flags=len(mainps[mainps.conditional_error_candidate]),
        generated_flags=len(maings[maings.generated_error_candidate]),
        cross_history=selected.to_dict('records')),indent=2))


if __name__=='__main__':main()
