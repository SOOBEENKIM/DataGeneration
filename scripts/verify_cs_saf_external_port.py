"""Independent NumPy histogram/ECDF reductions of all frozen external outputs.

Does not import the experiment, evaluator, model, or generator implementations.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cs_saf/external_port_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def table(*cols):
    if len(cols[0])==0:return {}
    values,count=np.unique(np.column_stack(cols),axis=0,return_counts=True)
    return {tuple(v):float(n)/len(cols[0]) for v,n in zip(values,count)}


def distance(a,b):
    if not a or not b:return None
    return sum(abs(a.get(k,0.)-b.get(k,0.)) for k in set(a)|set(b))/2


def ks(a,b):
    a,b=np.sort(np.asarray(a)),np.sort(np.asarray(b))
    points=np.unique(np.r_[a,b])
    return float(np.max(abs(np.searchsorted(a,points,side='right')/len(a)-np.searchsorted(b,points,side='right')/len(b))))


def bins(values,edges):
    a=np.asarray(values,float)
    ans=np.digitize(a,edges,right=True)+1
    ans[a==0]=0;ans[~np.isfinite(a)]=-1;ans[a<0]=-2
    return ans


def daily(frame,rootcodes=None):
    entity=pd.factorize(frame.entity_id,sort=False)[0]
    keys=np.column_stack([entity,np.floor(frame.timestamp.to_numpy())])
    unique,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    gaps=np.r_[np.nan,np.diff(unique[:,1])]
    gaps[np.r_[True,np.diff(unique[:,0])!=0]]=np.nan
    if rootcodes is None:return gaps,counts,np.array([]),np.array([])
    per_event=gaps[inverse];has=np.isfinite(per_event)
    return gaps,counts,np.asarray(rootcodes)[has],per_event[has]


def transforms(fit,name):
    root='category' if name=='sparkov' else 'receiver_or_mark'
    roots=sorted(fit[root].fillna('<MISSING>').astype(str).unique())
    marks=sorted(fit.receiver_or_mark.fillna('<MISSING>').astype(str).unique())
    dg,dc,_,_=daily(fit)
    return dict(root=root,rootmap={v:i+1 for i,v in enumerate(roots)},markmap={v:i+1 for i,v in enumerate(marks)},
        gaps=np.unique(np.quantile(fit.loc[fit.gap.gt(0),'gap'],[.2,.4,.6,.8])),
        amounts=np.unique(np.quantile(np.log1p(fit.amount_or_numeric_value),np.arange(.1,1,.1))),
        daily_gaps=np.unique(np.quantile(dg[np.isfinite(dg)],[.2,.4,.6,.8])),
        daily_counts=np.unique(np.quantile(dc,[.2,.4,.6,.8])))


def distributions(frame,state,name):
    f=frame.sort_values(['entity_id','event_index'],kind='stable').reset_index(drop=True)
    root=f[state['root']].fillna('<MISSING>').astype(str).map(state['rootmap']).fillna(0).to_numpy(int)
    mark=f.receiver_or_mark.fillna('<MISSING>').astype(str).map(state['markmap']).fillna(0).to_numpy(int)
    previous=np.r_[-1,root[:-1]]
    has=f.event_index.gt(0).to_numpy()
    previous[~has]=-1
    g=bins(f.gap,state['gaps'])
    amounts=f.amount_or_numeric_value.to_numpy()
    good=np.isfinite(amounts)&(amounts>=0)
    ac=np.full(len(f),-1,dtype=int);ac[good]=np.digitize(np.log1p(amounts[good]),state['amounts'],right=True)
    tiecounts=Counter(zip(f.entity_id,f.timestamp))
    singleton=np.asarray([tiecounts[k]==1 for k in zip(f.entity_id,f.timestamp)])
    unambiguous=has&singleton&np.r_[False,singleton[:-1]]
    tables=dict(gap_tv=table(g[has]),mark_tv=table(mark),root_tv=table(root),
        root_amount_joint_tv=table(root,ac),merchant_amount_joint_tv=table(mark,ac),
        gap_root_transition_joint_tv=table(g[has],previous[has],root[has]),
        unambiguous_gap_root_transition_tv=table(g[unambiguous],previous[unambiguous],root[unambiguous]))
    if name=='berka':
        dg,dc,dm,dgm=daily(f,mark);finite=np.isfinite(dg)
        tables['daily_gap_mark_joint_tv']=table(bins(dgm,state['daily_gaps']),dm)
        tables['daily_gap_count_joint_tv']=table(bins(dg[finite],state['daily_gaps']),np.digitize(dc[finite],state['daily_counts'],right=True))
    lengths=f.groupby('entity_id').size().to_numpy()
    return tables,dict(amounts=amounts,lengths=lengths,generated_events=len(f),generated_entities=len(lengths),
        generated_max_length=int(max(lengths)),unknown_mark_rate=float(np.mean(mark==0)),
        unknown_root_rate=float(np.mean(root==0)),invalid_amount_rate=float(np.mean(~good)),
        invalid_gap_rate=float(np.mean((f.gap.to_numpy()[has]<0)|~np.isfinite(f.gap.to_numpy()[has]))),
        noninteger_gap_rate=float(np.mean(abs(f.gap.to_numpy()[has]-np.rint(f.gap.to_numpy()[has]))>1e-6)),
        generated_unambiguous_fraction=float(np.mean(unambiguous[has])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true',help='verify completed outputs while final registered jobs finish')
    args=parser.parse_args()
    result={'passed':True,'verifier_sha256':sha(__file__),'implementation_imported':False,
            'test_outcomes_accessed':False,'runs':[],'scalar_metrics_verified':0,'max_absolute_error':0.,'events_in_generated_datasets':0}
    basehash=sha(ROOT/'configs/cs_saf_external_port_v1.json')
    for name in ('berka','sparkov'):
        inp=OUT/'input'/name
        pre=json.loads((inp/'preflight.json').read_text())
        for f,h in pre['files'].items():assert sha(inp/f)==h
        events=pd.read_parquet(inp/'events.parquet');roles=pd.read_parquet(inp/'roles.parquet');plan=pd.read_parquet(inp/'plan.parquet')
        assert set(roles.role)=={'fit','check','validation'}
        fit=events.loc[events.entity_id.isin(roles.loc[roles.role.eq('fit'),'entity_id'])]
        val=events.loc[events.entity_id.isin(roles.loc[roles.role.eq('validation'),'entity_id'])]
        state=transforms(fit,name);reference,rinfo=distributions(val,state,name)
        for model in ('U','G','ARGN','empirical','CPAR_tail'):
            folder=OUT/'runs'/name/model
            if args.wait and not (folder/'DONE.json').exists():
                print(name,model,'waiting for registered job completion',flush=True)
                deadline=time.monotonic()+5400
                while not (folder/'DONE.json').exists():
                    assert not (folder/'FAILED.json').exists(),(name,model,'scientific job failed')
                    assert time.monotonic()<deadline,'verification wait expired'
                    time.sleep(10)
            done=json.loads((folder/'DONE.json').read_text())
            assert done['config_sha256']==basehash
            for file,expected in done['files'].items():assert sha(folder/file)==expected,(name,model,file)
            if model=='CPAR_tail':
                assert done['segmentation_coverage']['input_events']==done['segmentation_coverage']['segmented_events']==len(fit)
            for variant,data in done['results'].items():
                for generation in data['generations']:
                    gs=generation['seed'];path=folder/f'generated_{variant}_{gs}.parquet'
                    frame=pd.read_parquet(path);actual,info=distributions(frame,state,name)
                    assert set(frame.entity_id)==set(plan.entity_id)
                    expected_index=frame.groupby('entity_id',sort=False).cumcount().to_numpy()
                    np.testing.assert_array_equal(frame.event_index,expected_index)
                    rebuilt=frame.gap.fillna(0).groupby(frame.entity_id,sort=False).cumsum()
                    np.testing.assert_allclose(frame.timestamp,rebuilt,rtol=1e-7,atol=1e-5)
                    if model!='ARGN':
                        np.testing.assert_array_equal(frame.groupby('entity_id').size().reindex(plan.entity_id),plan.length)
                    computed={k:distance(reference[k],v) for k,v in actual.items()}
                    computed['amount_ks']=ks(rinfo['amounts'],info.pop('amounts'))
                    computed['length_ks']=ks(rinfo['lengths'],info.pop('lengths'))
                    computed.update(info)
                    for k,value in computed.items():
                        expected=generation['metrics'][k]
                        if value is None:assert expected is None
                        else:
                            error=abs(value-expected)
                            assert error<1e-11,(name,model,variant,gs,k,value,expected)
                            result['max_absolute_error']=max(result['max_absolute_error'],error)
                        result['scalar_metrics_verified']+=1
                    result['events_in_generated_datasets']+=len(frame)
                    result['runs'].append(dict(dataset=name,model=model,variant=variant,seed=gs,events=len(frame),
                        source_sha256=sha(path),done_sha256=sha(folder/'DONE.json')))
            print(name,model,'independent metrics PASS',flush=True)
        print(name,'independent metrics PASS',flush=True)
    frozen=['models/cs_saf.py','models/cs_saf_v2.py','models/cs_saf_structure.py','models/cof_seqgen_saf.py','data/cof_seqgen_saf_tensorizer.py']
    result['old_files_unchanged']=[]
    for file in frozen:
        old=subprocess.check_output(['git','show',f'04a8918:{file}'],cwd=ROOT)
        assert hashlib.sha256(old).hexdigest()==sha(ROOT/file)
        result['old_files_unchanged'].append(file)
    path=OUT/'independent_verification.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('ALL PASS',result['scalar_metrics_verified'],result['max_absolute_error'],flush=True)


if __name__=='__main__':main()
