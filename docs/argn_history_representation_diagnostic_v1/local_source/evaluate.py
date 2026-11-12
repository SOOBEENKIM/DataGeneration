"""Paired conditional laws, changing native LSTM history only."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from diagnose import *

BINS=[(1,None),(1,6),(6,50),(50,100),(100,200),(200,500),(500,1024),(1024,None)]
GROUPS={'normal':lambda p:p%2==0,'NN':lambda p:p==0,'onset':lambda p:p==1,
        'continuation':lambda p:p==3,'return':lambda p:p==2,'fraud':lambda p:p%2==1}

def register():
    assert not (OUT/'EVALUATION_MANIFEST.json').exists()
    write(OUT/'EVALUATION_MANIFEST.json',dict(sha256={str(Path(__file__)):digest(Path(__file__))},
        created_utc=datetime.now(timezone.utc).isoformat(),post_results_changes=False))

def verify_eval():
    verify()
    for p,h in json.loads((OUT/'EVALUATION_MANIFEST.json').read_text())['sha256'].items():assert digest(p)==h

def states(head,b,phase,clock,valid):
    values=[]
    with torch.no_grad():
        for start in range(0,len(b),4096):
            sl=slice(start,start+4096)
            values.append(tuple(x.cpu().numpy() for x in head(b[sl],torch.as_tensor(phase[sl]).long(),
                torch.as_tensor(clock[sl]).float(),torch.as_tensor(valid[sl]).float())))
    return tuple(np.concatenate([v[i] for v in values]) for i in range(3))

def sample(law,u,z):
    logits,mu,sd=law
    # Match native float32 softmax before numpy cumulative sums.
    prob=torch.from_numpy(logits).softmax(-1).numpy()
    component=(u[:,None]>prob.cumsum(-1)).sum(-1).clip(max=2);j=np.arange(len(u))
    zz=mu[j,component]+sd[j,component]*z
    return np.floor(np.expm1(zz.clip(0,np.log1p(1045691.)))+1e-8).astype(np.int64)

def metric_rows(meta,gaps,anchor,head,history,gs,noise):
    rows=[];pos=meta.event_index.to_numpy();rp=meta.real_phase.to_numpy()
    gp=meta.generated_phase.to_numpy() if anchor=='generated' else rp
    rawclock=meta.generated_raw_clock.to_numpy() if anchor=='generated' else meta.real_raw_clock.to_numpy()
    rgap=meta.real_gap.to_numpy();ids=meta.customer_id.to_numpy()
    for lo,hi in BINS:
        window=(pos>=lo)&(pos<(hi if hi is not None else np.inf))
        for support in ['phase_population','same_phase']:
            common=window & ((rp==meta.generated_phase.to_numpy()) if support=='same_phase' else True)
            for group,fn in GROUPS.items():
                r=common&fn(rp)&np.isfinite(rgap);g=common&fn(gp)
                if not r.any() or not g.any():continue
                x=np.log1p(rgap[r]);y=np.log1p(gaps[g])
                rw=1/pd.Series(ids[r]).map(pd.Series(ids[r]).value_counts()).to_numpy()
                gw=1/pd.Series(ids[g]).map(pd.Series(ids[g]).value_counts()).to_numpy()
                pr=r&np.isfinite(meta.real_raw_clock)&(meta.real_raw_clock.to_numpy()>0)
                pg=g&np.isfinite(rawclock)&(rawclock>0)
                personal=(wasserstein_distance(np.log1p(rgap[pr]/meta.real_raw_clock.to_numpy()[pr]),
                    np.log1p(gaps[pg]/rawclock[pg])) if pr.any() and pg.any() else np.nan)
                rows.append(dict(generation_seed=gs,noise=noise,anchor=anchor,head=head,history=history,
                    support=support,start=lo,end=-1 if hi is None else hi,group=group,
                    actual_events=int(r.sum()),sample_events=int(g.sum()),
                    actual_customers=len(np.unique(ids[r])),sample_customers=len(np.unique(ids[g])),
                    gap_w1=wasserstein_distance(x,y),equal_customer_w1=wasserstein_distance(x,y,u_weights=rw,v_weights=gw),
                    signed_log_bias=float(y.mean()-x.mean()),personal_w1=personal,
                    personal_actual_events=int(pr.sum()),personal_sample_events=int(pg.sum())))
    return rows

@torch.no_grad()
def evaluate(gs):
    verify_eval();seed(20261500);torch.set_num_threads(2);start=time.monotonic()
    run=OUT/f'capture_{gs}';assert (run/'COMPLETE.json').exists() and (DOC/'ACTUAL_CACHE_COMPLETE.json').exists()
    assert not (run/'EVALUATION_COMPLETE.json').exists()
    cap=torch.load(run/'captured.pt',map_location='cpu',weights_only=True)
    raw=pd.read_parquet(run/'native.parquet')
    records,pos,ids=actual_records();record_by_id={int(cid):r for cid,r in zip(ids,records)}
    ws,model=load_parent(20260930,torch.device('cpu'));keys=list(model.tgt_cardinalities)
    real=torch.load(OUT/'cache/actual.pt',map_location='cpu',weights_only=True)['base']
    rmeta=pd.read_parquet(OUT/'cache/actual_metadata.parquet').reset_index(names='real_index')
    rmeta=rmeta.rename(columns={'phase_num':'real_phase','gap':'real_gap','raw_clock':'real_raw_clock'})
    gmeta=pd.DataFrame(dict(customer_id=cap['ids'].numpy(),event_index=cap['pos'].numpy(),gen_index=np.arange(len(raw)),
        generated_phase=cap['phase'].numpy()))
    gaps=raw.gap.astype(float).where(raw.event_index.gt(0))
    gmeta['generated_raw_clock']=gaps.groupby(raw.customer_id,sort=False).transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    match=gmeta.merge(rmeta,on=['customer_id','event_index'],how='inner',validate='one_to_one',sort=False)
    assert not match.duplicated(['customer_id','event_index']).any()
    # Replay the exact generated native tokens, never re-encoded decoded values.
    grecords=[];aligned=[];aligned_order=[]
    tokens=cap['tokens'].numpy()
    for cid,g in gmeta.groupby('customer_id',sort=False):
        original=record_by_id[int(cid)];ii=g.gen_index.to_numpy();n=len(ii)
        assert np.array_equal(g.event_index,np.arange(n))
        rec={k:copy.deepcopy(v) for k,v in original.items() if not k.startswith('tgt:') and k!=STATE_COLUMN}
        for j,k in enumerate(keys):rec[k]=tokens[ii,j].tolist()
        rec[STATE_COLUMN]=np.zeros((n,records[0][STATE_COLUMN].shape[-1]),dtype=np.float32)
        grecords.append(rec)
        ncommon=min(n,len(original[keys[0]]));ar=copy.deepcopy(rec)
        for k in keys:ar[k]=(rec[k][:ncommon] if k.startswith('tgt:/__') else original[k][:ncommon])
        ar[STATE_COLUMN]=np.zeros((ncommon,rec[STATE_COLUMN].shape[-1]),dtype=np.float32)
        aligned.append(ar);aligned_order += [(int(cid),t) for t in range(ncommon)]
    independent=fast_base(model,grecords)
    diff=(independent-cap['base']).abs()
    max_h=float(diff[:,HSLICE].max());max_c=float(diff[:,:256].max());max_current=float(diff[:,1088:].max())
    torch.testing.assert_close(independent,cap['base'],rtol=2e-5,atol=2e-5)
    # Native inputs are retained for the primary G branch, even within tolerance.
    del independent,diff
    aligned_base=fast_base(model,aligned)
    assert aligned_order==list(zip(match.customer_id.astype(int),match.event_index.astype(int)))
    ri=match.real_index.to_numpy();gi=match.gen_index.to_numpy()
    torch.testing.assert_close(real[ri,:256],cap['base'][gi,:256],rtol=2e-5,atol=2e-5)
    # An explicit current/future-token perturbation must not change earlier history.
    test=copy.deepcopy(grecords[0]);cut=min(40,len(test[keys[0]])//2)
    first=fast_base(model,[test])
    k=next(k for k in keys if k.startswith('tgt:t2/'))
    test[k][cut:]=[(int(v)+1)%model.tgt_cardinalities[k] for v in test[k][cut:]]
    altered=fast_base(model,[test]);torch.testing.assert_close(first[:cut+1,HSLICE],altered[:cut+1,HSLICE],rtol=0,atol=0)
    del test,first,altered,model,grecords,aligned
    keep=match.event_index.gt(0).to_numpy();meta=match.loc[keep].reset_index(drop=True)
    ri=ri[keep];gi=gi[keep];aligned_base=aligned_base[keep]
    real_base=real[ri];generated_base=cap['base'][gi]
    heads={}
    for name,path,cls in [('mixture_marginal',HEAD,ResidualDensity),('init_only',INITIAL,InitialClock)]:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        h=cls(payload['dim'],payload['arm']);h.load_state_dict(payload['state_dict']);heads[name]=h.eval().requires_grad_(False)
    histories={'generated':generated_base[:,HSLICE],'actual_aligned':aligned_base[:,HSLICE],
               'actual_literal':real_base[:,HSLICE]}
    meta.to_parquet(run/'matched_metadata.parquet',index=False)
    rngs={s:np.random.default_rng(s) for s in NOISES}
    noises={s:(r.random(len(meta)),r.standard_normal(len(meta))) for s,r in rngs.items()}
    rows=[];law_rows=[];nll_rows=[];negative=0;early=0;sample_count=0;law_save={};noise_parity={}
    scale=heads['mixture_marginal'].base_scale[HSLICE]
    distance=((histories['generated']-histories['actual_aligned'])/scale).square().mean(-1).sqrt().numpy()
    write(run/'REPRESENTATION_DISTANCE.json',dict(mean=float(distance.mean()),median=float(np.median(distance)),p95=float(np.quantile(distance,.95)),
        feature_count=832,not_a_quality_metric=True))
    for anchor,ab,ap,ac,av in [
        ('generated',generated_base,meta.generated_phase.to_numpy(),cap['clock'][gi].numpy(),cap['valid'][gi].numpy()),
        ('actual',real_base,meta.real_phase.to_numpy(),meta[['clock0','clock1']].to_numpy(),meta.valid.to_numpy())]:
        originals={};samples={}
        for name,head in heads.items():
            lawsets={}
            for source,hist in histories.items():
                base=ab.clone();base[:,HSLICE]=hist
                assert torch.equal(base[:,:256],ab[:,:256]) and torch.equal(base[:,1088:],ab[:,1088:])
                law=states(head,base,ap,ac,av);lawsets[source]=law
                lawskey=f'{anchor}__{name}__{source}'
                law_save[lawskey]=np.stack(law,1)
                if anchor=='actual':
                    losses=[]
                    for lo in range(0,len(base),4096):
                        sl=slice(lo,lo+4096)
                        losses.append(head.nll(base[sl],torch.as_tensor(ap[sl]).long(),torch.as_tensor(ac[sl]).float(),
                            torch.as_tensor(av[sl]).float(),torch.as_tensor(meta.real_gap.to_numpy()[sl]).float(),[0.,1045691.],1.).numpy())
                    loss=np.concatenate(losses);assert np.isfinite(loss).all()
                    for group,fn in GROUPS.items():
                        mask=fn(ap);nll_rows.append(dict(generation_seed=gs,anchor=anchor,head=name,history=source,
                            group=group,events=int(mask.sum()),mean_nll=float(loss[mask].mean()) if mask.any() else np.nan))
                output=[]
                for ns,(u,z) in noises.items():
                    gaps=sample(law,u,z);output.append(gaps)
                    rows.extend(metric_rows(meta,gaps,anchor,name,source,gs,ns));sample_count+=len(gaps)
                outputs=np.stack(output)
                np.savez_compressed(run/f'samples_{lawskey}.npz',gaps=outputs,noise_seeds=NOISES)
                if name=='init_only':
                    if source=='generated':samples[name]=outputs
                    else:np.testing.assert_array_equal(outputs,samples[name]);negative+=outputs.size
                if source=='generated':originals[name]=outputs
                else:
                    mask=meta.event_index.le(5).to_numpy();np.testing.assert_array_equal(outputs[:,mask],originals[name][:,mask]);early+=int(mask.sum())*len(NOISES)
                print('CONDITIONAL',gs,anchor,name,source,len(gaps),flush=True)
            first=lawsets['generated']
            for source,law in lawsets.items():
                if name=='init_only':
                    for a,b in zip(first,law):np.testing.assert_array_equal(a,b)
                for lo,hi in BINS:
                    mask=meta.event_index.ge(lo).to_numpy()&(meta.event_index.lt(hi).to_numpy() if hi else True)
                    for group,fn in GROUPS.items():
                        m=mask&fn(ap)
                        if not m.any():continue
                        law_rows.append(dict(generation_seed=gs,anchor=anchor,head=name,history=source,start=lo,end=-1 if hi is None else hi,
                            group=group,events=int(m.sum()),
                            logits_mae=float(np.abs(law[0][m]-first[0][m]).mean()),
                            location_mae=float(np.abs(law[1][m]-first[1][m]).mean()),
                            scale_mae=float(np.abs(law[2][m]-first[2][m]).mean())))
            if anchor=='generated' and name=='mixture_marginal':
                replay=sample(first,cap['u'][gi].numpy(),cap['z'][gi].numpy())
                original=cap['gap'][gi].numpy();delta=np.abs(replay-original)
                # Different GEMM batch shape can round float32 logits/means by ulps.
                assert float((delta>0).mean())<.01 and delta.max()<=5,(delta.max(),(delta>0).mean())
                noise_parity=dict(nonidentical_seconds=int((delta>0).sum()),values=len(delta),max_seconds=float(delta.max()),
                    native_in_call_parity_exact=True,float32_batch_shape_rounding_reported=True)
    pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False)
    pd.DataFrame(law_rows).to_csv(run/'law_effects.csv',index=False)
    pd.DataFrame(nll_rows).to_csv(run/'conditional_nll.csv',index=False)
    np.savez_compressed(run/'conditional_laws.npz',**law_save)
    torch.save(dict(real_base=real_base,generated_base=generated_base,aligned_history=histories['actual_aligned'],
        generated_clock=cap['clock'][gi],generated_valid=cap['valid'][gi]),run/'matched_inputs.pt')
    write(run/'EVALUATION_COMPLETE.json',dict(rows=len(meta),customers=meta.customer_id.nunique(),
        conditional_sample_values=sample_count,conditions=12,noises=4,new_fits=0,
        independent_native_base_max_abs=dict(history=max_h,static=max_c,current=max_current),
        static_and_current_frozen=True,strict_past_current_future_invariance=True,
        init_only_identical_sample_checks=negative,first5_identical_sample_checks=early,
        noise_parity=noise_parity,seconds=time.monotonic()-start,test_events_read=False))
    verify_eval();print('EVALUATION_COMPLETE',gs,sample_count,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','evaluate']);p.add_argument('--seed',type=int);a=p.parse_args()
    register() if a.mode=='register' else evaluate(a.seed)
