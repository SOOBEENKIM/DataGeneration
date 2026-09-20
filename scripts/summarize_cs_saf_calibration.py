"""Verify and summarize every registered calibration cell without selecting runs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from experiments.cs_saf_calibration import ROOT, OUTPUT, CONFIG_SHA, contract, verify, parent
from experiments.cs_saf_replication import seed_statistics, tensor_digest
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

MODELS=('U','E','Ecal','L003','L003cal')
PAIRS=(('Ecal','E'),('L003cal','L003'),('L003cal','Ecal'),('L003','E'),('Ecal','U'),('L003cal','U'))
PRIMARY='short_gap_repeat_curve_l1'
MI='gap_repeat_mi_error'


def get_record(pi,k,t,name):
    calibrated=name.endswith('cal');base=name[:-3] if calibrated else name
    if calibrated:
        folder=OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_{t}/{name}'
        manifest=verify(folder)
        conditional=json.loads((folder/'conditional_accuracy.json').read_text())
        generation=json.loads((folder/'comparison.json').read_text())
        fit=json.loads((folder/'fit.json').read_text())
        provenance=fit['parent'];cp=torch.load(folder/'checkpoint_calibrated.pt',map_location='cpu')
        original={n:v for n,v in cp['model_state'].items() if not n.startswith('calibration_')}
        if tensor_digest(original)!=provenance['state_sha256'] or sha256(Path(provenance['path']))!=provenance['checkpoint_sha256']:
            raise ValueError('base checkpoint mutation')
        if sha256(folder/'train_features.npz')!=fit['feature_sha256']:
            raise ValueError('fit input mismatch')
        features=np.load(folder/'train_features.npz',allow_pickle=False)
        # Independent direct-probability Bernoulli arithmetic, not the fitter.
        for code in (3,4):
            ix=features['code']==code;z=features['logit'][ix].astype(float)
            f=features['fresh_previous'][ix].astype(float);y=features['equality'][ix]
            params=fit['contexts'][str(code-3)]
            for a,b,key in [(0.,1.,'before_nll'),(params['offset'],params['slope'],'after_nll')]:
                q=1/(1+np.exp(-(a+b*z)));r=q+(1-q)*f
                loss=-np.where(y,np.log(r),np.log1p(-r)).mean()
                if not np.isfinite(loss) or abs(loss-params[key])>1e-9:
                    raise ValueError('independent observed-repeat loss differs')
        arrays=np.load(folder/'validation_accuracy_arrays.npz',allow_pickle=False)
        for label in ('0','1'):
            values=arrays[f'label_{label}_metrics']
            for j,column in enumerate(conditional['columns']):
                if abs(values[:,j].mean()-conditional['groups'][label]['metrics'][column]['mean'])>1e-10:
                    raise ValueError('independent conditional mean differs')
    else:
        folder=parent.folder_for(pi,k,t,name);manifest=parent.verify_artifacts(folder)
        conditional=json.loads((folder/'conditional_accuracy.json').read_text())['checkpoints']['best']['splits']['validation']
        gp=parent.OUTPUT/f'generation/pi_{pi:.2f}/trial_{t}/kappa_{k}/{name}'
        parent.verify_artifacts(gp)
        generation=json.loads((gp/'comparison.json').read_text())
        fit=None
    audit=json.loads((folder/'intervention_audit.json').read_text())
    if audit['sampling_plan_sha256']!=generation['sampling_plan_sha256']:
        raise ValueError('reference generation plan changed')
    return dict(generation=generation['metrics'],conditional=conditional['groups'],
        responses=audit['responses'],fit=fit,folder=str(folder),
        sampling_plan_sha256=generation['sampling_plan_sha256'],manifest=manifest)


def main():
    c=contract();terminal=json.loads((OUTPUT/'GRID_COMPLETE.json').read_text())
    if len(terminal['completed'])!=80 or terminal['config_sha256']!=CONFIG_SHA:
        raise ValueError('all 80 registered fits required')
    raw={};vectors={};fits=[]
    for pi in c['prevalences']:
        pk=f'{pi:.2f}';raw[pk]={};vectors[pk]={}
        for name in MODELS:
            vals={}
            for t in c['trials']:
                records={k:get_record(pi,k,t,name) for k in c['kappas']}
                raw[pk][f'{name}/trial_{t}']={str(k):r for k,r in records.items()}
                for r in records.values():
                    if r['fit'] is not None:fits.append(r['fit'])
                active=records[1]['generation']['context_1']['metrics']
                item={f'active_{m}':v for m,v in active.items()}
                item.update({f'active_conditional_{m}':v['mean'] for m,v in records[1]['conditional']['1']['metrics'].items()})
                for response in ('copy','repeat'):
                    item['three_null_'+response+'_range']=float(np.mean([
                        records[k]['responses'][label]['mean_'+response+'_range']
                        for k,label in ((0,'0'),(0,'1'),(1,'0'))]))
                for m,v in item.items():vals.setdefault(m,[]).append(v)
            vectors[pk][name]=vals
        for t in c['trials']:
            for k in c['kappas']:
                plans={raw[pk][f'{name}/trial_{t}'][str(k)]['sampling_plan_sha256'] for name in MODELS}
                if len(plans)!=1:raise ValueError('unpaired generation plans')
    summaries={};contrasts={}
    for pk,models in vectors.items():
        summaries[pk]={name:{m:seed_statistics(v) for m,v in vals.items()} for name,vals in models.items()}
        contrasts[pk]={}
        for a,b in PAIRS:
            contrasts[pk][a+'_minus_'+b]={m:seed_statistics(np.array(models[a][m])-models[b][m]) for m in models[a]}
        contrasts[pk]['factorial_interaction']={m:seed_statistics(
            np.array(models['L003cal'][m])-models['L003'][m]-np.array(models['Ecal'][m])+models['E'][m]) for m in models['E']}
    screens={}
    for a,b in PAIRS:
        key=a+'_minus_'+b;good=[];joint=[]
        for pk,vs in contrasts.items():
            d=vs[key];primary=d['active_'+PRIMARY]
            if primary['mean']<0 and primary['negative_count']>=4:
                good.append(pk)
                if all(d[m]['mean']<=0 for m in ['active_'+MI,'three_null_copy_range','three_null_repeat_range']):joint.append(pk)
        screens[key]=dict(primary_qualifying_prevalences=good,joint_qualifying_prevalences=joint,
            consistent_primary_signal=len(good)>=3,joint_improvement_lead=len(joint)>=3,
            confirmatory_superiority=False)
    result=dict(source_commit=terminal['source_commit'],config_sha256=CONFIG_SHA,
        new_calibration_fits=80,base_model_fits=0,reused_models=120,raw_records=raw,
        summaries=summaries,contrasts=contrasts,diagnostic_screens=screens,
        bound_hits=sum(x['bound_hit'] for fit in fits for x in fit['contexts'].values()),
        train_loss_change=[x['after_nll']-x['before_nll'] for fit in fits for x in fit['contexts'].values()],
        test_accessed=False,independent_data_confirmation=False,old_ER_FAIL_unchanged=True)
    write_json(OUTPUT/'summary.json',result)
    compact={k:v for k,v in result.items() if k!='raw_records'}
    compact['full_evidence_path']=str(OUTPUT/'summary.json');compact['full_evidence_sha256']=sha256(OUTPUT/'summary.json')
    compact['verification']=dict(output_manifests=80,reference_models=120,base_tensor_identities=80,
        train_objective_contexts=160,conditional_array_contexts=160,paired_plans=40,all_pass=True)
    write_json(ROOT/'docs/cs_saf/calibration_v1_result.json',compact)
    plot(summaries,contrasts)
    print(json.dumps(dict(screens=screens,bound_hits=result['bound_hits'],
                         full_evidence_sha256=compact['full_evidence_sha256']),indent=2))


def plot(summaries,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'U':'#637181','E':'#2873b4','Ecal':'#56a0d1','L003':'#a34a26','L003cal':'#e29455'}
    xs=np.arange(4);keys=list(summaries)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained')
    for ax,metric,title in zip(axes,['active_'+PRIMARY,'active_'+MI,'three_null_copy_range'],
            ['Active generated repeat-curve error','Active generated gap-repeat MI error','Unnecessary gap response (three-null mean)']):
        for name in MODELS:
            means=[summaries[p][name][metric]['mean'] for p in keys]
            ax.plot(xs,means,'o-',label=name,color=colors[name],lw=1.7,ms=4)
        ax.set_xticks(xs,['5%','10%','25%','50%']);ax.set_xlabel('Group prevalence')
        ax.set_title(title,fontsize=10);ax.set_ylabel('Error / response (lower is better)');ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Frozen model calibration: five paired training trials, reused data',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(ROOT/f'docs/cs_saf/calibration_v1_comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
