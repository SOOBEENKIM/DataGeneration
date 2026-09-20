"""One preregistered train-only correction; historical sources are immutable."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments import cs_saf_generation_repeats as repeats
from experiments import cs_saf_calibration as prior
from experiments.cs_saf_v3 import accuracy_audit
from experiments.cs_saf_generation_metrics import CachedGenerationMetrics
from experiments.cs_saf_followup_external import canonical_and_plan,sample_to_frame
from experiments.cs_saf_pilot import ROOT,frozen_source,state_digest,batch,subset
from experiments.cs_saf_replication import tensor_digest
from models.cs_saf_gap_calibration import GapCalibratedU,GapCalibratedE,NEW_BUFFERS,fit_gap_offsets
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONFIG=ROOT/'configs/benchmark_v2/cs_saf_gap_calibration_v1.json'
CONFIG_SHA='4d3811ceca90b31b6a7637a1ae9b5cff59a5482bbd070bfc3b1b6c9c1b5aea7a'
OUTPUT=ROOT/'artifacts/cs_saf/gap_calibration_v1'


def contract():
    if sha256(CONFIG)!=CONFIG_SHA:raise ValueError('registration changed')
    c=json.loads(CONFIG.read_text())
    for name,h in c['frozen_sources'].items():
        if sha256(ROOT/name)!=h:raise ValueError('frozen source changed: '+name)
    repeats.contract();return c


def folder_for(pi,k,trial,name):return OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_{trial}/{name}'


def finish(folder,source,**extra):
    if (folder/'COMPLETE.json').exists():raise FileExistsError('completed results immutable')
    write_json(folder/'COMPLETE.json',dict(status='COMPLETE',source_commit=source,config_sha256=CONFIG_SHA,
        artifact_sha256={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()},**extra))


def verify(folder):
    m=json.loads((folder/'COMPLETE.json').read_text())
    if m['status']!='COMPLETE' or m['config_sha256']!=CONFIG_SHA:raise ValueError('wrong manifest')
    for p,h in m['artifact_sha256'].items():
        if sha256(folder/p)!=h:raise ValueError('changed artifact '+str(folder/p))
    return m


def old_digest(model):return tensor_digest({k:v for k,v in model.state_dict().items() if k not in NEW_BUFFERS})


def adapter(base,payload,name,device):
    state=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    cls=GapCalibratedU if name=='Ucal' else GapCalibratedE
    args=(state.gap_support,) if name=='Ucal' else (state.gap_support,base.reference_probabilities.cpu())
    m=cls(*args,**state.model_config_kwargs()).to(device)
    m.load_state_dict(dict(base.state_dict(),**{k:m.state_dict()[k] for k in NEW_BUFFERS}),strict=True)
    m.eval();m.requires_grad_(False)
    if old_digest(m)!=state_digest(base):raise ValueError('adapter changed old tensors')
    return m


def train_bins(model,train):
    mask=train['valid_mask'].clone();mask[:,0]=False
    # Support coding follows native float32 arithmetic, without fitting on outcomes.
    gaps=train['gap'][mask].to(next(model.parameters()).device)
    bins=(model._support_code(gaps)-3).cpu().numpy()
    codes=train['codes'][:,None].expand_as(mask)[mask].numpy()
    representatives=np.asarray(model.support.representatives,dtype=np.float32).astype(float)
    quantized=representatives[bins];mapping=[];counts=[];weights=[];edges=[]
    for code in (3,4):
        e=np.quantile(quantized[codes==code],[.2,.4,.6,.8],method='linear')
        if len(np.unique(e))!=4:raise ValueError('duplicate coarse quantile edges')
        m=np.searchsorted(e,representatives,side='right')
        n=np.bincount(m[bins[codes==code]],minlength=5)
        if (n==0).any():raise ValueError('empty train coarse bin')
        edges.append(e.tolist());mapping.append(m.tolist());counts.append(n.tolist());weights.append((n/n.sum()).tolist())
    return dict(mapping=mapping,counts=counts,weights=weights,edges=edges,
        quantized_support_only=True,fit_split='train',transitions=int(mask.sum()))


def extract_features(base,train,bins,device):
    f=prior.extract_train_features(base,train,device,512)
    rows=f['entity_index'];cols=f['event_index']
    gaps=train['gap'][rows,cols].to(device)
    k=(base._support_code(gaps)-3).cpu().numpy();slots=f['code'].astype(int)-3
    f['support_bin']=k;f['bin']=np.asarray(bins['mapping'])[slots,k]
    np.testing.assert_array_equal(f['code'],train['codes'][rows].numpy())
    np.testing.assert_array_equal(f['equality'],(train['receiver'][rows,cols]==train['receiver'][rows,cols-1]).numpy())
    return f


@torch.no_grad()
def response_only(model,data,device):
    sums={s:{m:[] for m in ('copy','repeat')} for s in (3,4)};zero=0.
    for start in range(0,len(data['lengths']),256):
        x=batch(data,torch.arange(start,min(start+256,len(data['lengths']))),device)
        context=model.context(model.encoder(**x),x['static_categorical']);mask=x['valid_mask'].clone();mask[:,0]=False
        previous=x['receiver'].roll(1,1)[mask];codes=x['static_categorical'][0][:,None].expand_as(mask)[mask];flat=context[mask]
        parts=[]
        for j in range(0,len(flat),4096):
            sl=slice(j,j+4096);q,r=model.response_curves(flat[sl],previous[sl],static_codes=codes[sl])
            zq,zr=model.response_curves(flat[sl],previous[sl],static_codes=codes[sl],zero_gap=True)
            zero=max(zero,float((zq.max(1).values-zq.min(1).values).max()),float((zr.max(1).values-zr.min(1).values).max()))
            parts.append(torch.stack((q.max(1).values-q.min(1).values,r.max(1).values-r.min(1).values),1))
        dense=torch.zeros((*mask.shape,2),device=device);dense[mask]=torch.cat(parts)
        per_entity=dense.sum(1)/mask.sum(1)[:,None]
        for s in (3,4):
            for j,m in enumerate(('copy','repeat')):sums[s][m].extend(per_entity[x['static_categorical'][0]==s,j].cpu().tolist())
    if zero>1e-8:raise ValueError('zero-gap control changed')
    return dict(responses={str(s-3):{**{f'mean_{m}_range':float(np.mean(v)) for m,v in ms.items()},'entities':len(ms['copy'])} for s,ms in sums.items()},zero_gap_control_max_range=zero)


def fit_job(pi,k,trial,parent_name,device):
    c=contract();source=frozen_source();name=c['variants'][parent_name];out=folder_for(pi,k,trial,name);started=time.monotonic()
    if (out/'COMPLETE.json').exists():
        if verify(out)['source_commit']!=source:raise ValueError('resume source differs')
        return
    out.mkdir(parents=True,exist_ok=False)
    try:
        payload=repeats.parent.load_cache(repeats.parent.CACHE/f'pi_{pi:.2f}_kappa_{k}.pt')
        control,provenance,_=repeats.load_fixed(payload,pi,k,trial,parent_name,device)
        bins=train_bins(control,payload['train']);write_json(out/'train_bins.json',bins)
        features=extract_features(control,payload['train'],bins,device)
        np.savez_compressed(out/'train_features.npz',**features)
        fitted=fit_gap_offsets(features,bins['weights'],c);del features
        model=adapter(control,payload,parent_name,device)
        model.set_gap_correction(bins['mapping'],bins['weights'],fitted['delta'])
        fit=dict(fitted,prevalence=pi,kappa=k,trial=trial,parent_name=parent_name,name=name,
            source_commit=source,parent=provenance,train_bins_sha256=sha256(out/'train_bins.json'),
            feature_sha256=sha256(out/'train_features.npz'),train_cache_sha256=sha256(repeats.parent.CACHE/f'pi_{pi:.2f}_kappa_{k}.pt'),
            train_entity_ids_sha256=hashlib.sha256(json.dumps(payload['train']['entity_ids']).encode()).hexdigest(),
            fit_seconds=time.monotonic()-started)
        write_json(out/'fit.json',fit)
        torch.save(dict(model_state=model.state_dict(),parent=provenance,fit=fitted,bins=bins,
            source_commit=source,config_sha256=CONFIG_SHA),out/'checkpoint_gap_calibrated.pt')
        before=state_digest(model)
        # Calibration completed and checkpoint saved before any evaluation target use.
        original=json.loads((Path(provenance['reference_folder'])/'conditional_accuracy.json').read_text())
        ref=control.reference_probabilities.detach().cpu().clone() if parent_name=='Ecal' else None
        if ref is None:
            from experiments.cs_saf_route_decomposition import reference_measure
            ref,_=reference_measure(control,payload['train'])
        conditional,arrays=accuracy_audit(model,payload['validation'],ref,k,device)
        write_json(out/'conditional_accuracy.json',conditional);np.savez_compressed(out/'conditional_arrays.npz',**arrays)
        old_cond,old_arrays=accuracy_audit(control,payload['validation'],ref,k,device)
        write_json(out/'control_conditional_accuracy.json',old_cond);np.savez_compressed(out/'control_conditional_arrays.npz',**old_arrays)
        differences={label:{m:old_cond['groups'][label]['metrics'][m]['mean']-v['mean'] for m,v in original['groups'][label]['metrics'].items()} for label in ('0','1')}
        write_json(out/'historical_conditional_differences.json',differences)
        response=response_only(model,payload['validation'],device);write_json(out/'intervention_audit.json',response)
        precision=prior.verify_response_precision(model,payload['validation'],device,response,conditional)
        write_json(out/'response_precision.json',precision)
        dataset,payload2,plan,positions=canonical_and_plan(pi,k,trial)
        evaluator=CachedGenerationMetrics(dataset,plan)
        old_plan=torch.load(repeats.folder_for(pi,k,trial)/'plan.pt',map_location='cpu')
        np.testing.assert_array_equal(positions,old_plan['positions'])
        for r,seed in enumerate(c['generation_seeds'][str(trial)]):
            folder=out/f'repeat_{r}';folder.mkdir()
            sample=repeats.generate_fixed_plan(model,payload,positions,seed,device,256)
            torch.save(dict(sample=sample,sampling_seed=seed,plan_seed=old_plan['plan_seed'],model_state_sha256=before,test_accessed=False),folder/'generated_sample.pt')
            metrics=evaluator.score(sample_to_frame(sample,payload,plan,positions))
            old=repeats.folder_for(pi,k,trial)/parent_name/f'repeat_{r}';repeats.verify(old)
            previous=json.loads((old/'comparison.json').read_text())
            for g in metrics:
                if metrics[g]['train_metric_state']!=previous['metrics'][g]['train_metric_state']:raise ValueError('metric definitions changed')
            write_json(folder/'comparison.json',dict(prevalence=pi,kappa=k,trial=trial,model=name,repeat=r,
                sampling_seed=seed,sampling_plan_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),metrics=metrics,
                parent_reference_manifest_sha256=sha256(old/'COMPLETE.json'),test_accessed=False))
        if state_digest(model)!=before or old_digest(model)!=provenance['state_sha256'] or state_digest(control)!=provenance['state_sha256']:raise ValueError('frozen weights/affine changed')
        if sha256(Path(provenance['checkpoint_path']))!=provenance['checkpoint_sha256']:raise ValueError('parent checkpoint changed')
        reloaded=adapter(control,payload,parent_name,device);reloaded.load_state_dict(torch.load(out/'checkpoint_gap_calibrated.pt',map_location='cpu')['model_state'])
        if state_digest(reloaded)!=before:raise ValueError('checkpoint reload failed')
        write_json(out/'execution.json',dict(source_commit=source,device=str(device),seconds=time.monotonic()-started,
            new_fits=1,new_generated_datasets=5,all_old_tensors_unchanged=True,checkpoint_reload=True,
            architecture=model.architecture_contract(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        finish(out,source);print(f'COMPLETE {pi} {k} {trial} {name} {time.monotonic()-started:.1f}s',flush=True)
    except Exception as exc:
        write_json(out/'FAILED.json',dict(error=type(exc).__name__,message=str(exc),source_commit=source));raise


def smoke(device,gate):
    c=contract();source=frozen_source();out=OUTPUT/gate;out.mkdir(parents=True,exist_ok=False)
    payload=repeats.parent.load_cache(repeats.parent.CACHE/'pi_0.05_kappa_1.pt');train=subset(payload['train'],32);records={}
    positions=torch.load(repeats.folder_for(.05,1,0)/'plan.pt',map_location='cpu')['positions']
    seed=c['generation_seeds']['0'][0]
    for name in c['parents']:
        base,prov,_=repeats.load_fixed(payload,.05,1,0,name,device);m=adapter(base,payload,name,device)
        b=train_bins(base,train);m.set_gap_correction(b['mapping'],b['weights'],np.zeros((2,5)))
        tiny=positions[:8]
        a=repeats.generate_fixed_plan(base,payload,tiny,seed,device);z=repeats.generate_fixed_plan(m,payload,tiny,seed,device);repeats.compare_samples(a,z)
        reproduced=False
        if str(device).startswith('cuda'):
            old=torch.load(repeats.folder_for(.05,1,0)/name/'repeat_0/generated_sample.pt',map_location='cpu')['sample']
            repeats.compare_samples(old,repeats.generate_fixed_plan(base,payload,positions,seed,device))
            repeats.compare_samples(old,repeats.generate_fixed_plan(m,payload,positions,seed,device));reproduced=True
        f=extract_features(base,train,b,device);fit=fit_gap_offsets(f,b['weights'],c)
        second=fit_gap_offsets(f,b['weights'],c)
        if second!=fit:raise ValueError('nondeterministic constrained fit')
        m.set_gap_correction(b['mapping'],b['weights'],fit['delta'])
        if old_digest(m)!=prov['state_sha256']:raise ValueError('gate changed old tensors')
        torch.save(m.state_dict(),out/(name+'.pt'));reload=adapter(base,payload,name,device);reload.load_state_dict(torch.load(out/(name+'.pt'),map_location='cpu'))
        a=repeats.generate_fixed_plan(m,payload,tiny,seed,device);z=repeats.generate_fixed_plan(reload,payload,tiny,seed,device);repeats.compare_samples(a,z)
        response=response_only(m,train,device)
        records[name]=dict(identity_samples=True,original_full_generation_reproduced=reproduced,
            constrained_fit_deterministic=True,old_tensors_frozen=True,checkpoint_reload=True,
            zero_gap_max=response['zero_gap_control_max_range'],fitted_delta=fit['delta'])
    write_json(out/'gate.json',dict(decision='PASS',source_commit=source,config_sha256=CONFIG_SHA,records=records,device=str(device)));finish(out,source)
    print(json.dumps(records),flush=True)
