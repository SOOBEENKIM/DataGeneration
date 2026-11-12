"""Frozen native-token capture and history-only intervention; no training."""
import argparse, copy, json, os, pickle, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'artifacts/argn_joint_clock_density_v1/local_source')]
from residual_density import ResidualDensity,InitialClock,residual_generation
from run_argn_amount_learning import load_parent,digest,write,seed,folder
from run_argn_state_first import OUT as BASE,SOURCE
from benchmarks.argn_past_state import PastState,STATE_COLUMN
from benchmarks.argn_state_adapter import FullHistoryCollator
from benchmarks.argn_clock_evolution import progress_features
from mostlyai.engine._tabular.argn import _make_permutation_mask

OUT=ROOT/'artifacts/argn_history_representation_diagnostic_v1'
DOC=OUT/'report'
HEAD=ROOT/'artifacts/argn_joint_clock_density_v1/mixture_marginal/time_head.pt'
INITIAL=ROOT/'artifacts/argn_clock_initial_coupling_v1/init_only/time_head.pt'
SEEDS=[20261011,20261012]
NOISES=[20261501,20261502,20261503,20261504]
HSLICE=slice(256,1088)

def verify():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['sha256'].items():assert digest(p)==h,p

def actual_records():
    with (ROOT/'artifacts/argn_followup_diagnostics_v1/encoded_development.pkl').open('rb') as f:
        records=pickle.load(f)['records']
    pos=pd.read_parquet(ROOT/'artifacts/argn_followup_diagnostics_v1/positions.parquet')
    mapping=pos[['record','customer_id']].drop_duplicates().sort_values('record')
    assert mapping.record.tolist()==list(range(len(records)))
    assert mapping.customer_id.is_unique and len(mapping)==147
    return records,pos,mapping.customer_id.to_numpy()

@torch.no_grad()
def fast_base(model,records):
    """Same native context/history/current-column expressions, no output heads."""
    collate=FullHistoryCollator(True,None,model.device);parts=[];common_parts=[]
    key=next(k for k in model.tgt_cardinalities if k.startswith('tgt:t0/c0__'))
    lookup=model.tgt_sub_columns_lookup[key]
    mask=_make_permutation_mask(col_embedding_dims=model.column_embedders.dims,
        columns=model.tgt_columns,column_order=model.column_order,is_sequential=True,device=model.device)
    for lo in range(0,len(records),2):
        batch=collate(records[lo:lo+2]);embeds=model.embedders(batch)
        emb=torch.cat(list(embeds.values()),-1)
        hin=torch.nn.functional.pad(emb[:,:-1],(0,0,1,0))
        history,_=model.history_compressor(hin)
        flat,seq,masks=model.context_compressor(batch)
        assert not seq and model.history_compressor.dim_output==832
        ctx=model._repeat_flat_context(flat,history.size(1))
        current=torch.cat(list(model.column_embedders(embeds).values()),-1)*mask[lookup.col_idx,:].int()
        base=torch.cat(ctx+[history,current],-1)
        if lo==0:
            got={}
            def hook(module,args):
                if args[1]==key:got['base']=torch.cat(args[0][:2],-1)
            h=model.regressors.register_forward_pre_hook(hook)
            try:model(batch,mode='trn')
            finally:h.remove()
            torch.testing.assert_close(base,got['base'],rtol=0,atol=0)
        for j,r in enumerate(records[lo:lo+2]):
            n=len(r[key]);parts.append(base[j,:n].cpu())
    return torch.cat(parts)

def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    files=[Path(__file__),DOC/'PROTOCOL.md',HEAD,INITIAL,
        ROOT/'artifacts/argn_followup_diagnostics_v1/positions.parquet',
        ROOT/'artifacts/argn_followup_diagnostics_v1/encoded_development.pkl',
        ROOT/'artifacts/argn_residual_v1/metadata_development.parquet',
        BASE/'prepared/validation_context.parquet',SOURCE/'prepared/validation.parquet']
    old=json.loads((ROOT/'artifacts/argn_joint_clock_density_v1/MANIFEST.json').read_text())
    files += [Path(p) for p in old['hashes'] if '/cache/' not in p]
    prior=json.loads((ROOT/'artifacts/argn_clock_feedback_gain_v1/report/PRIOR_HASHES.json').read_text())
    hashes={**prior['sha256'],**{str(p):digest(p) for p in files}}
    for p,h in hashes.items():assert digest(p)==h,p
    write(OUT/'MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        sha256=hashes,device='cpu',fit_seed=20260930,generation_seeds=SEEDS,noise_seeds=NOISES,
        history_slice=[256,1088],fits=0,test_events_read=False))
    print('REGISTERED',len(hashes),flush=True)

@torch.no_grad()
def real_cache():
    verify();seed(20261500);torch.set_num_threads(2)
    records,positions,ids=actual_records();ws,model=load_parent(20260930,torch.device('cpu'))
    result=fast_base(model,records)
    m=pd.read_parquet(ROOT/'artifacts/argn_residual_v1/metadata_development.parquet')
    assert np.array_equal(m.entity_id,positions.record)
    assert np.array_equal(m.event_index,positions.event_index)
    assert len(result)==len(m)
    p=torch.load(INITIAL,map_location='cpu',weights_only=True)
    clock,valid=progress_features(m,p['fallback'])
    target=m.assign(customer_id=positions.customer_id.to_numpy())
    target['clock0']=clock[:,0];target['clock1']=clock[:,1];target['valid']=valid
    target['raw_clock']=target.groupby('customer_id',sort=False).gap.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    codec=PastState(ws.tgt_stats.read());key=codec.prefixes['event_is_fraud']+'__cat'
    assert np.array_equal(positions.label,m.label)
    target['phase_num']=(2*m.previous.clip(lower=0)+m.label).astype(int)
    (OUT/'cache').mkdir(exist_ok=True)
    torch.save(dict(base=result,ids=ids.tolist()),OUT/'cache/actual.pt')
    target.to_parquet(OUT/'cache/actual_metadata.parquet',index=False)
    write(DOC/'ACTUAL_CACHE_COMPLETE.json',dict(rows=len(m),customers=len(ids),dim=result.shape[1],
        actual_base_shape=list(result.shape),fast_forward_parity=True,test_events_read=False))
    print('ACTUAL_CACHE_COMPLETE',len(m),flush=True)

@torch.no_grad()
def capture(gs):
    verify();assert gs in SEEDS
    from run_argn_boundary_gap import generation_args
    from mostlyai.engine import generate
    from mostlyai.engine._workspace import Workspace
    import mostlyai.engine._tabular.generation as generation
    run=OUT/f'capture_{gs}';run.mkdir(exist_ok=False)
    start=time.monotonic();seed(gs);torch.set_num_threads(2)
    write(run/'START.json',dict(pid=os.getpid(),device='cpu',seed=gs,started_utc=datetime.now(timezone.utc).isoformat()))
    shutil.copytree(folder('B_event_label_first',20260930)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    ws=Workspace(run/'workspace');before=digest(ws.model_tabular_weights_path)
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet')
    args,kwargs=generation_args(ws,20260930)
    chunks=[];pending={};errors=[];step_count=0
    old_sample=ResidualDensity.sample
    def sample(head,base,phase,clock,valid,rng,limits,quantum):
        cloned=np.random.default_rng();cloned.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
        u=cloned.random((len(base),1))[:,0];z=cloned.standard_normal(len(base))
        logits,mu,sd=head(base,phase,clock,valid)
        prob=logits.softmax(-1).cpu().numpy();component=(u[:,None]>np.cumsum(prob,-1)).sum(-1).clip(max=2)
        row=np.arange(len(base));zz=mu.cpu().numpy()[row,component]+sd.cpu().numpy()[row,component]*z
        expected=np.floor(np.expm1(np.clip(zz,*np.log1p(limits)))/quantum+1e-8)*quantum
        result=old_sample(head,base,phase,clock,valid,rng,limits,quantum)
        np.testing.assert_array_equal(result[0],expected)
        pending['sample']=dict(u=u,z=z,phase=phase.cpu().numpy(),gap=result[0])
        return result
    old_continue=generation._continue_sequence_mask
    def continuation(**kw):
        nonlocal step_count
        keep=old_continue(**kw).to_numpy();ids=kw['step_keys'].to_numpy()
        b=pending.pop('base');clock,valid=pending.pop('clock');mask=pending.pop('mask')
        token=kw['step_output'].squeeze(-1).cpu().numpy()
        assert len(ids)==len(b)==len(keep)
        n=len(ids);u=np.full(n,np.nan);z=u.copy();gap=u.copy();phase=np.full(n,-1,dtype=int)
        s=pending.pop('sample',None)
        if s is not None:
            assert mask.sum()==len(s['u'])
            for dst,key in [(u,'u'),(z,'z'),(gap,'gap'),(phase,'phase')]:dst[mask]=s[key]
        positions=np.expm1(clock[:,1]).round().astype(int)
        chunks.append(dict(base=b[keep],clock=clock[keep],valid=valid[keep],
            ids=ids[keep],pos=positions[keep],tokens=token[keep],u=u[keep],z=z[keep],
            phase=phase[keep],gap=gap[keep],sample_mask=mask[keep]))
        step_count+=1
        if step_count%250==0:print('CAPTURE_PROGRESS',gs,step_count,'seconds',round(time.monotonic()-start,1),flush=True)
        return pd.Series(keep)
    with residual_generation(*args,**kwargs,time_payload=HEAD) as audit:
        Parent=generation.SequentialModel
        class Observed(Parent):
            def __init__(self,*a,**k):
                super().__init__(*a,**k)
                def take(module,inputs):
                    if inputs[1]==self.time_keys[0]:pending['base']=torch.cat(inputs[0][:2],-1).detach().cpu().numpy().reshape(-1,1322).copy()
                self.regressors.register_forward_pre_hook(take)
            def forward(self,*a,**k):
                result=super().forward(*a,**k)
                pending['clock']=tuple(v.copy() for v in self.current_clock)
                pending['mask']=self.time_mask.reshape(-1).cpu().numpy().copy()
                return result
        with patch.object(generation,'SequentialModel',Observed),patch.object(generation,'_continue_sequence_mask',continuation),patch.object(ResidualDensity,'sample',sample):
            generate(ctx_data=context,device='cpu',workspace_dir=run/'workspace')
    captured={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    del chunks
    ix=np.lexsort((captured['pos'],captured['ids']));captured={k:v[ix] for k,v in captured.items()}
    raw=pd.read_parquet(run/'workspace/SyntheticData');raw['event_index']=raw.groupby('customer_id',sort=False).cumcount()
    raw=raw.sort_values(['customer_id','event_index']).reset_index(drop=True)
    np.testing.assert_array_equal(raw.customer_id,captured['ids']);np.testing.assert_array_equal(raw.event_index,captured['pos'])
    mask=captured['sample_mask'];np.testing.assert_array_equal(raw.gap.to_numpy()[mask],captured['gap'][mask])
    assert np.array_equal(mask,captured['pos']>0)
    np.testing.assert_array_equal(captured['phase'][mask]%2,pd.to_numeric(raw.event_is_fraud).to_numpy()[mask])
    assert digest(ws.model_tabular_weights_path)==before
    raw.to_parquet(run/'native.parquet',index=False)
    torch.save({k:torch.from_numpy(v.copy()) for k,v in captured.items()},run/'captured.pt')
    write(run/'COMPLETE.json',dict(rows=len(raw),steps=step_count,seconds=time.monotonic()-start,
        native_noise_parity_exact=True,explicit_id_mapping=True,weights_unchanged=True,sampling_audit=audit,
        original_cuda_replay=False,test_events_read=False,device='cpu',sha256=digest(run/'captured.pt')))
    print('CAPTURE_COMPLETE',gs,len(raw),round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','real','capture']);p.add_argument('--seed',type=int)
    a=p.parse_args();prepare() if a.mode=='prepare' else real_cache() if a.mode=='real' else capture(a.seed)
