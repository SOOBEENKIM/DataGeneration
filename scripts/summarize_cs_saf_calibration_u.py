"""Verify and summarize every registered calibration cell without selecting runs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from experiments.cs_saf_calibration_u import ROOT, OUTPUT, CONFIG_SHA, contract, verify, parent
from experiments.cs_saf_replication import seed_statistics, tensor_digest
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

MODELS=('U','Ucal','E','Ecal')
PAIRS=(('Ucal','U'),('Ecal','Ucal'),('Ecal','E'),('E','U'),('Ecal','U'))
PRIMARY='short_gap_repeat_curve_l1'
MI='gap_repeat_mi_error'


def get_record(pi,k,t,name):
    if name != 'Ucal':
        from scripts.summarize_cs_saf_calibration import get_record as historical_record
        return historical_record(pi,k,t,name)
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
    if len(terminal['completed'])!=40 or terminal['config_sha256']!=CONFIG_SHA:
        raise ValueError('all 40 registered fits required')
    expected={(pi,k,t,'U') for pi in c['prevalences'] for k in c['kappas'] for t in c['trials']}
    if {tuple(v) for v in terminal['completed']} != expected:
        raise ValueError('completed cells differ from registration')
    raw={};vectors={};fits=[]
    for pi in c['prevalences']:
        pk=f'{pi:.2f}';raw[pk]={};vectors[pk]={}
        for name in MODELS:
            vals={}
            for t in c['trials']:
                records={k:get_record(pi,k,t,name) for k in c['kappas']}
                raw[pk][f'{name}/trial_{t}']={str(k):r for k,r in records.items()}
                for r in records.values():
                    if name=='Ucal':fits.append(r['fit'])
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
            np.array(models['Ecal'][m])-models['E'][m]-np.array(models['Ucal'][m])+models['U'][m]) for m in models['E']}
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
        new_calibration_fits=40,base_model_fits=0,reused_models=120,raw_records=raw,
        summaries=summaries,contrasts=contrasts,diagnostic_screens=screens,
        bound_hits=sum(x['bound_hit'] for fit in fits for x in fit['contexts'].values()),
        train_loss_change=[x['after_nll']-x['before_nll'] for fit in fits for x in fit['contexts'].values()],
        test_accessed=False,independent_data_confirmation=False,old_ER_FAIL_unchanged=True)
    result['new_fit_parameters']=[dict(prevalence=pk,trial=int(name.split('_')[-1]),kappa=int(k),
        offset=r['fit']['offset'],slope=r['fit']['slope'],contexts=r['fit']['contexts'],
        parent_checkpoint_sha256=r['fit']['parent']['checkpoint_sha256'],
        artifact_manifest=r['manifest'])
        for pk,cell in raw.items() for name,conditions in cell.items() if name.startswith('Ucal/')
        for k,r in conditions.items()]
    result['gates']={name:json.loads((OUTPUT/name/'gate.json').read_text())
                     for name in ('cpu_gate','gpu_gate')}
    if any(g['decision']!='PASS' or g['source_commit']!=terminal['source_commit']
           for g in result['gates'].values()):
        raise ValueError('same-source CPU/GPU gates missing')
    precision_files=list(OUTPUT.glob('pi_*/kappa_*/trial_*/*cal/response_precision_verification.json'))
    precision=[json.loads(p.read_text()) for p in precision_files]
    result['precision_verification']=dict(full_precision_verifications=len(precision),
        historical_tf32_discrepancies_over_tolerance=sum(p['historical_precision_exceeds_tolerance'] for p in precision),
        largest_full_precision_entity_delta=max(p['full_precision_max_entity_delta'] for p in precision),
        scientific_numeric_settings_unchanged=True)
    native=json.loads((OUTPUT/'native_verification.json').read_text())
    if native['status']!='PASS' or native['groups_checked']!=480 or native['max_absolute_error']>1e-12:
        raise ValueError('independent native verification missing or failed')
    result['native_verification']={k:v for k,v in native.items() if k!='records'}
    result['native_verification']['evidence_sha256']=sha256(OUTPUT/'native_verification.json')
    pairing=json.loads((OUTPUT/'pairing_verification.json').read_text())
    if pairing['status']!='PASS' or pairing['paired_cells']!=40:
        raise ValueError('independent U/E pairing verification missing')
    result['pairing_verification']={k:v for k,v in pairing.items() if k!='records'}
    result['pairing_verification']['evidence_sha256']=sha256(OUTPUT/'pairing_verification.json')
    result['scientific_source_commits']=sorted({r['manifest']['source_commit']
        for cell in raw.values() for name,conditions in cell.items() if name.split('/')[0].endswith('cal')
        for r in conditions.values()})
    write_json(OUTPUT/'summary.json',result)
    compact={k:v for k,v in result.items() if k!='raw_records'}
    compact['full_evidence_path']=str(OUTPUT/'summary.json');compact['full_evidence_sha256']=sha256(OUTPUT/'summary.json')
    compact['verification']=dict(output_manifests=40,reference_models=120,base_tensor_identities=40,
        train_objective_contexts=80,conditional_array_contexts=80,paired_plans=40,all_pass=True)
    write_json(ROOT/'docs/cs_saf/calibration_u_control_v1_result.json',compact)
    plot(summaries,contrasts)
    print(json.dumps(dict(screens=screens,bound_hits=result['bound_hits'],
                         full_evidence_sha256=compact['full_evidence_sha256']),indent=2))


def plot(summaries,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'U':'#929ca6','Ucal':'#252c34','E':'#75a9d0','Ecal':'#1766a4'}
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
    fig.suptitle('Equal-calibration U control: five paired training trials, reused data',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(ROOT/f'docs/cs_saf/calibration_u_control_v1_comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
