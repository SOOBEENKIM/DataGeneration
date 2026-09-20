"""CPU gates and immutable common inputs; no scientific model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import torch
from data.cs_saf_external import VIEWS,load_view,fit_check_ids,fit_tensorizer,TargetWindows
from models.cs_saf_external import ExternalUG


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    for name in ('berka','sparkov'):
        started=time.monotonic();folder=a.output/name;folder.mkdir()
        path=a.data_root/'canonical'/name
        manifest=json.loads((path/'canonical_manifest.json').read_text())
        for filename,record in manifest['files'].items():assert digest(path/filename)==record['sha256']
        ds=load_view(path,name);fit_ids,check_ids=fit_check_ids(ds)
        tf=fit_tensorizer(ds,fit_ids)
        alltrain=tf.transform_split(ds,'train');val=tf.transform_split(ds,'validation')
        seqs={'fit':tuple(s for s in alltrain if s.entity_id in set(fit_ids)),
              'check':tuple(s for s in alltrain if s.entity_id in set(check_ids)), 'validation':val}
        # Check data exclusions, fit provenance, all targets and long prefixes.
        assert not set(VIEWS[name]['excluded']) & (set(ds.events)|set(ds.static_context))
        assert not set(fit_ids)&set(check_ids)
        stats={}
        for role,s in seqs.items():
            w=TargetWindows(s)
            for array in [w.static,w.flat['numeric_value']]:assert torch.isfinite(array).all()
            assert len(w)==sum(x.length for x in s)
            starts=np.cumsum(np.r_[0,w.lengths[:-1]])
            targets=np.unique(np.r_[starts,starts+w.lengths-1,starts+np.minimum(w.lengths-1,32)])
            maxdiff=0.;params={}
            for mode in ('U','G'):
                torch.manual_seed(20260921)
                model=ExternalUG(mode,tf.state.gap_support,**tf.state.model_config_kwargs())
                for b in np.array_split(targets,max(1,int(np.ceil(len(targets)/256)))):
                    inp=w.batch(b)
                    terms,out=model.terms(**inp)
                    assert all(torch.isfinite(v[0]) for v in terms.values())
                    assert torch.allclose(out['logmark'].exp().sum(-1),torch.ones(len(b)),atol=1e-5)
                    maxdiff=max(maxdiff,float(abs(out['logmark'].exp().sum(-1)-1).max()))
                params[mode]=model.architecture_contract()['parameters']
            assert params['U']==params['G']
            stats[role]=dict(entities=len(s),events=len(w),max_length=int(w.lengths.max()),
                             boundary_target_checks=len(targets),parameters=params,max_probability_error=maxdiff)
        gaps=ds.events.loc[ds.events.entity_id.isin(fit_ids),'gap'].dropna().to_numpy()
        codes=tf.state.gap_support.encode_numpy(gaps)
        if tf.state.gap_support.zero_is_explicit:
            assert np.array_equal(codes==3,gaps==0)
        torch.save(dict(state=tf.state.to_dict(),sequences=seqs),folder/'prepared.pt')
        ds.static_context.to_parquet(folder/'context.parquet',index=False)
        ds.events.to_parquet(folder/'events.parquet',index=False)
        ds.entity_splits.assign(role=ds.entity_splits.entity_id.map(
            {**dict.fromkeys(fit_ids,'fit'),**dict.fromkeys(check_ids,'check'),
             **dict.fromkeys(ds.entity_ids_for_split('validation'),'validation')})).to_parquet(folder/'roles.parquet',index=False)
        # Plan draws training-fit context/length only; generation outputs never
        # borrow a real trajectory's mark, gap, or amount values.
        ids=np.random.default_rng(20260924).choice(len(seqs['fit']),size=128,replace=True)
        plan=ds.static_context.set_index('entity_id').loc[[seqs['fit'][i].entity_id for i in ids]].reset_index()
        plan.insert(1,'source_entity_id',plan.entity_id)
        plan['entity_id']=[f'{name}-external-{i:04d}' for i in range(len(plan))]
        plan['length']=[seqs['fit'][i].length for i in ids]
        plan['fit_position']=ids
        plan.to_parquet(folder/'plan.parquet',index=False)
        write(folder/'preflight.json',dict(dataset=name,passed=True,statistics=stats,view=VIEWS[name],
            tensorizer=tf.state.to_dict(),phase='engineering_only_untrained_forward',scientific_fits=0,
            test_outcomes_accessed=False,elapsed_seconds=time.monotonic()-started,
            source_manifest_sha256=digest(path/'canonical_manifest.json'),
            files={p.name:digest(p) for p in folder.iterdir() if p.is_file()}))
        print(name,json.dumps(stats),flush=True)


if __name__=='__main__':main()
