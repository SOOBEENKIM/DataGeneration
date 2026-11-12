"""Frozen native strict-past representations; optimization/internal only."""
from pathlib import Path
import sys, json, os, time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_argn_amount_learning import inputs,load_parent,build_cache,digest,write
from benchmarks.argn_clock_evolution import progress_features

OUT=ROOT/'artifacts/argn_joint_clock_density_v1'
BASELINE=ROOT/'artifacts/argn_clock_initial_coupling_v1/init_only/time_head.pt'

def main():
    started=time.monotonic();assert torch.cuda.is_available()
    assert not (OUT/'CACHE_COMPLETE.json').exists()
    old=torch.load(BASELINE,map_location='cpu',weights_only=True)
    _,codec,frames,metas=inputs();ws,model=load_parent(20260930,torch.device('cuda:0'))
    keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes['gap']+'__')]
    records={};(OUT/'cache').mkdir(exist_ok=True)
    for split in ['optimization','internal_validation']:
        m=pd.read_parquet(ROOT/f'artifacts/argn_residual_v1/metadata_{split}.parquet')
        meta=metas[split]
        assert np.array_equal(meta.record,m.entity_id)
        assert np.array_equal(meta.event_index,m.event_index)
        assert np.array_equal(meta.label,m.label)
        cache,checks=build_cache(model,frames[split],meta,codec,keys,model.device)
        encoded=codec.numeric({k:cache['targets'][:,j].numpy() for j,k in enumerate(keys)},'gap')
        real=m.gap.to_numpy();keep=m.event_index.gt(0).to_numpy()
        differences=np.flatnonzero(keep & (encoded!=real))
        # Metadata is the existing evaluation representation. Never silently
        # rewrite it to match integer/native digits; retain a complete ledger.
        mismatch=[dict(row=int(i),entity_id=str(m.entity_id.iloc[i]),event_index=int(m.event_index.iloc[i]),
                       metadata=float(real[i]),decoded=float(encoded[i])) for i in differences]
        x,valid=progress_features(m,old['fallback'])
        codes,ids=pd.factorize(m.entity_id,sort=False)
        grid=np.full((int(m.event_index.max())+1,len(ids)),-1,dtype=np.int64)
        grid[m.event_index.to_numpy(),codes]=np.arange(len(m))
        data=dict(base=cache['base'],phase=torch.tensor((2*m.previous.clip(lower=0)+m.label).to_numpy(),dtype=torch.long),
            gap=torch.tensor(np.nan_to_num(real),dtype=torch.float32),clock=torch.tensor(x,dtype=torch.float32),
            valid=torch.tensor(valid,dtype=torch.float32),position=torch.tensor(m.event_index.to_numpy()),
            customer=torch.tensor(codes),grid=torch.tensor(grid),keys=keys,
            original_ids=[str(i) for i in ids],fallback=old['fallback'])
        path=OUT/f'cache/{split}.pt';torch.save(data,path)
        records[split]=dict(rows=len(m),customers=len(ids),base_dim=cache['base'].shape[-1],
            native_logits_checked=checks,mapping_exact=True,decoded_metadata_differences=mismatch,
            sha256=digest(path))
        print('CACHE',split,records[split],flush=True)
    write(OUT/'CACHE_COMPLETE.json',dict(records=records,parent_seed=20260930,
        seconds=time.monotonic()-started,pid=os.getpid(),test_events_read=False))

if __name__=='__main__':main()
