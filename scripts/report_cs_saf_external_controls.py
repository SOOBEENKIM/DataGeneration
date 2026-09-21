"""Apply frozen exploratory screens and export every outcome, including costs."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/cs-saf-external-controls-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=ROOT/'artifacts/cs_saf/external_controls_v1'
OLD=ROOT/'artifacts/cs_saf/external_port_v1'
DOC=ROOT/'docs/cs_saf/external_controls_v1'


def read(path):return json.loads(path.read_text())
def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    cfg=read(ROOT/'configs/cs_saf_external_controls_v1.json');threshold=cfg['screens']
    assert read(OUT/'independent_verification.json')['passed']
    assert read(OUT/'checkpoint_audit.json')['passed']
    rows=[];pred=[];budgets=[];traces={};archive=[];quantiles=[]
    for dataset in cfg['datasets']:
        for model,base in [(m,OLD) for m in cfg['reused_models']]+[(m,OUT) for m in cfg['new_models']]:
            folder=base/'runs'/dataset/model;done=read(folder/'DONE.json');start=read(folder/'START.json')
            archive.append(dict(reused=base==OLD,start=start,done=done))
            if model in ['U','G',*cfg['new_models']]:
                budgets.append(dict(dataset=dataset,model=model,reused=base==OLD,fit_seed=cfg['fit_seed'],
                    parameters=done['architecture']['parameters'],selected_epoch=done['selected_epoch'],last_epoch=done['last_epoch'],
                    fit_seconds=done['fit_seconds'],updates=done['optimizer_updates'],stop_reason=done['stop_reason'],source_commit=start['source_commit']))
                traces[dataset+'_'+model]=read(folder/'history.json')
            for variant,data in done['results'].items():
                if model=='empirical':label=variant.title();var='raw'
                else:label=model;var=variant
                if 'prediction' in data:
                    p=data['prediction'];pred.append(dict(dataset=dataset,model=label,variant=var,
                        **{k:v for k,v in p.items() if k!='components'},**{k+'_nll':v for k,v in p['components'].items()}))
                for g in data['generations']:
                    rows.append(dict(dataset=dataset,model=label,variant=var,generation_seed=g['seed'],
                                     reused=base==OLD,**g['metrics']))
                    amounts=pd.read_parquet(folder/f'generated_{variant}_{g["seed"]}.parquet',columns=['amount_or_numeric_value']).iloc[:,0]
                    quantiles.append(dict(dataset=dataset,model=label,variant=var,generation_seed=g['seed'],
                        **{'q'+str(q):float(amounts.quantile(q)) for q in (0,.01,.5,.9,.99,.999,1)}))
    frame=pd.DataFrame(rows);prediction=pd.DataFrame(pred)
    assert len(frame)==88
    metrics=[c for c in frame.columns if c not in ('dataset','model','variant','generation_seed','reused')]
    summary=frame.groupby(['dataset','model','variant'])[metrics].mean().reset_index()
    frame.to_csv(DOC/'generation_metrics.csv',index=False)
    summary.to_csv(DOC/'generation_summary.csv',index=False)
    effects=[]
    for ds in cfg['datasets']:
        relation='daily_gap_mark_joint_tv' if ds=='berka' else 'gap_root_transition_joint_tv'
        for mode in ('U','G'):
            for variant in cfg['variants']:
                cells={suffix:summary.loc[summary.dataset.eq(ds)&summary.model.eq(mode+suffix)&summary.variant.eq(variant)].iloc[0]
                       for suffix in ('','_amount','_action','_both')}
                for metric in (relation,'root_amount_joint_tv','amount_ks','mark_tv','root_tv'):
                    y00,y10,y01,y11=[float(cells[k][metric]) for k in ('','_amount','_action','_both')]
                    effects.append(dict(dataset=ds,model=mode,variant=variant,metric=metric,
                        original=y00,amount_only=y10,action_only=y01,both=y11,
                        amount_effect_at_old_action=y10-y00,amount_effect_at_new_action=y11-y01,
                        action_effect_at_old_amount=y01-y00,action_effect_at_new_amount=y11-y10,
                        descriptive_interaction=y11-y10-y01+y00))
    pd.DataFrame(effects).to_csv(DOC/'factorial_effects.csv',index=False)
    prediction.to_csv(DOC/'prediction_metrics.csv',index=False)
    pd.DataFrame(budgets).to_csv(DOC/'training_budgets.csv',index=False)
    # Descriptive tail check only; it does not replace any registered criterion.
    for dataset in cfg['datasets']:
        inp=OLD/'input'/dataset
        events=pd.read_parquet(inp/'events.parquet');roles=pd.read_parquet(inp/'roles.parquet')
        for role in ('fit','validation'):
            amounts=events.loc[events.entity_id.isin(roles.loc[roles.role.eq(role),'entity_id']),'amount_or_numeric_value']
            quantiles.append(dict(dataset=dataset,model='observed_'+role,variant='observed',generation_seed=None,
                                  **{'q'+str(q):float(amounts.quantile(q)) for q in (0,.01,.5,.9,.99,.999,1)}))
    quantile_frame=pd.DataFrame(quantiles)
    quantile_frame.to_csv(DOC/'amount_quantiles_descriptive.csv',index=False)
    tail_order=['U','U_amount','U_both','G','G_amount','G_both','D_both','Transition']
    tail_fig,tail_axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for ax,ds in zip(tail_axes,cfg['datasets']):
        sub=quantile_frame.loc[quantile_frame.dataset.eq(ds)&quantile_frame.variant.eq('raw')].groupby('model')[['q0.99','q0.999']].mean().reindex(tail_order)
        reference=quantile_frame.loc[quantile_frame.dataset.eq(ds)&quantile_frame.model.eq('observed_validation')].iloc[0]
        for i,(key,label,color) in enumerate([('q0.99','99th percentile','#5487b4'),('q0.999','99.9th percentile','#d38e42')]):
            ax.bar(np.arange(len(tail_order))+(i-.5)*.36,sub[key]/reference[key],width=.36,label=label,color=color)
        ax.axhline(1,color='black',linestyle='--',linewidth=1)
        ax.set_yscale('log');ax.set_xticks(np.arange(len(tail_order)),tail_order,rotation=40,ha='right',fontsize=8)
        ax.set_title(ds.title());ax.set_ylabel('Generated / observed quantile (log scale)');ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    tail_fig.suptitle('Descriptive amount-tail check: 1 = observed validation; not a new selection criterion')
    tail_fig.savefig(DOC/'amount_tail_descriptive.png',dpi=170);tail_fig.savefig(DOC/'amount_tail_descriptive.pdf');plt.close(tail_fig)
    write(DOC/'results.json',archive);write(DOC/'training_traces.json',traces)
    screens=[]
    def get(ds,model,variant):
        s=summary.loc[summary.dataset.eq(ds)&summary.model.eq(model)&summary.variant.eq(variant)].iloc[0]
        p=prediction.loc[prediction.dataset.eq(ds)&prediction.model.eq(model)&prediction.variant.eq(variant)]
        return s,p.iloc[0] if len(p) else None
    def compare(ds,model,base,variant,kind,amount_required=False,action_required=False,nonnegative=True):
        s,p=get(ds,model,variant);b,q=get(ds,base,variant)
        relation='daily_gap_mark_joint_tv' if ds=='berka' else 'gap_root_transition_joint_tv'
        flags={};costs={}
        for metric in ('gap_tv','mark_tv','root_tv','amount_ks'):
            costs[metric+'_cost']=float(s[metric]-b[metric]);flags[metric]=costs[metric+'_cost']<=threshold['marginal_max_cost']
        for metric,limit in [('mark_nll',threshold['mark_nll_max_cost']),('repeat_brier',threshold['repeat_brier_max_cost']),
                              ('amount_log_mae',max(threshold['amount_log_mae_abs_cost'],q.amount_log_mae*threshold['amount_log_mae_relative_cost']))]:
            costs[metric+'_cost']=float(p[metric]-q[metric]);flags[metric]=costs[metric+'_cost']<=limit
        action_gain=float(b[relation]-s[relation]);amount_gain=float(b.root_amount_joint_tv-s.root_amount_joint_tv)
        flags['action_relation']=(action_gain>=max(threshold['action_abs_improvement'],threshold['relative_improvement']*b[relation])
                                  if action_required else action_gain>=-threshold['joint_tv_max_cost'])
        flags['amount_relation']=(amount_gain>=max(threshold['amount_abs_improvement'],threshold['relative_improvement']*b.root_amount_joint_tv)
                                  if amount_required else amount_gain>=-threshold['joint_tv_max_cost'])
        flags['amount_validity']=not nonnegative or s.invalid_amount_rate==0
        flags['gap_validity']=s.invalid_gap_rate==0
        passed=bool(all(flags.values()))
        screens.append(dict(dataset=ds,variant=variant,comparison=kind,candidate=model,reference=base,passed=passed,
            action_improvement=action_gain,amount_improvement=amount_gain,
            failed_checks=','.join(k for k,v in flags.items() if not v),**costs))
        return passed
    for ds in cfg['datasets']:
        relation='daily_gap_mark_joint_tv' if ds=='berka' else 'gap_root_transition_joint_tv'
        for variant in cfg['variants']:
            for mode in ('U','G'):
                compare(ds,mode+'_amount',mode,variant,'amount_only',amount_required=True)
                compare(ds,mode+'_action',mode,variant,'action_with_legacy_amount',action_required=True,nonnegative=False)
                compare(ds,mode+'_both',mode+'_amount',variant,'action_with_new_amount',action_required=True)
                compare(ds,mode+'_both',mode+'_action',variant,'amount_with_full_action',amount_required=True)
                combined=compare(ds,mode+'_both',mode,variant,'combined_vs_original',amount_required=True,action_required=True)
                structural=compare(ds,mode+'_both','D_both',variant,'repeat_structure_vs_direct',action_required=True)
                s,_=get(ds,mode+'_both',variant);t,_=get(ds,'Transition','raw')
                close=bool(s[relation]<=t[relation]+.01 and s.root_amount_joint_tv<=t.root_amount_joint_tv+.01)
                screens.append(dict(dataset=ds,variant=variant,comparison='external_candidate',candidate=mode+'_both',
                    reference='original + D_both + Transition',passed=combined and structural and close,
                    action_improvement=float(t[relation]-s[relation]),amount_improvement=float(t.root_amount_joint_tv-s.root_amount_joint_tv),
                    failed_checks=','.join(k for k,v in [('combined',combined),('direct_comparison',structural),('transition_level',close)] if not v)))
    table=pd.DataFrame(screens);table.to_csv(DOC/'screens.csv',index=False)
    order=['U','U_amount','U_action','U_both','G','G_amount','G_action','G_both','D_both','Transition','ARGN','CPAR_tail']
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for row,ds in enumerate(cfg['datasets']):
        relation='daily_gap_mark_joint_tv' if ds=='berka' else 'gap_root_transition_joint_tv'
        for col,metric in enumerate((relation,'root_amount_joint_tv')):
            ax=axes[row,col];subset=frame.loc[frame.dataset.eq(ds)&frame.variant.eq('raw')].groupby('model')[metric].agg(['mean','min','max']).reindex(order)
            colors=['#477fa8' if m.startswith('U') else '#d38e42' if m.startswith('G') else '#589e87' if m=='Transition' else '#8675aa' for m in order]
            bars=ax.bar(np.arange(len(order)),subset['mean'],color=colors)
            ax.errorbar(np.arange(len(order)),subset['mean'],yerr=np.vstack([subset['mean']-subset['min'],subset['max']-subset['mean']]),fmt='none',ecolor='#333333',capsize=2)
            ax.set_xticks(np.arange(len(order)),order,rotation=55,ha='right',fontsize=8)
            ax.set_ylabel('TV error (lower is better)');ax.set_title(ds.title()+': '+('time-action relation' if col==0 else 'action-amount relation'))
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fixed output-head comparison: one fit seed, two generation draws\nBars: means; whiskers: draw range, not confidence intervals',fontsize=12)
    fig.savefig(DOC/'primary_results.png',dpi=170);fig.savefig(DOC/'primary_results.pdf');plt.close(fig)
    for f in ('preflight.json','gpu_preflight.json','independent_verification.json','checkpoint_audit.json','dispatch_done.json'):
        shutil.copy2(OUT/f,DOC/f)
    print(summary.loc[summary.variant.eq('raw'),['dataset','model','daily_gap_mark_joint_tv','gap_root_transition_joint_tv','root_amount_joint_tv','invalid_amount_rate']].to_string(index=False))
    print(table.loc[table.variant.eq('raw'),['dataset','comparison','candidate','passed','failed_checks']].to_string(index=False))


if __name__=='__main__':main()
