"""Registered Sparkov study using the version-qualified Berka ARGN runtime."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/sparkov_argn_control_v2'
DOCS = ROOT / 'docs/sparkov_argn_control_v2'
SOURCE = ROOT.parent / 'cof-seqgen-0707-2119-Version3-complete/data/cof_seqgen_saf/canonical/sparkov'
CONFIG = ROOT / 'configs/sparkov_argn_control_v2.json'
for k, v in {'HF_HOME':str(OUT/'cache/hf'), 'JOBLIB_TEMP_FOLDER':str(OUT/'cache/joblib'),
             'LOKY_MAX_CPU_COUNT':'4','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
             'TOKENIZERS_PARALLELISM':'false'}.items(): os.environ.setdefault(k,v)
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from benchmarks.argn_fraud_audit import CORE, STATIC, LABEL, MERCHANT, AMOUNT, metric_state, summaries
CFG = json.loads(CONFIG.read_text())

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n')

def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.set_num_threads(CFG['cpu_threads'])

def native_frame(raw):
    d=raw.rename(columns={'customer_id':'entity_id'}).copy()
    d['event_index']=d.groupby('entity_id',sort=False).cumcount()
    d.loc[d.event_index.eq(0),'gap']=np.nan
    return d[['entity_id','event_index',*CORE]]

def prepare():
    from mostlyai.engine import split,analyze,encode
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._encoding_types.tabular.numeric import encode_numeric,decode_numeric
    from mostlyai.engine._encoding_types.tabular.categorical import encode_categorical,decode_categorical
    base=OUT/'prepared';base.mkdir(parents=True,exist_ok=False)
    splits=pd.read_parquet(SOURCE/'entity_splits.parquet')
    roles=splits[splits.split.isin(['train','validation'])].copy().reset_index(drop=True)
    roles['customer_id']=np.arange(len(roles),dtype=np.int64)
    ids=roles.entity_id.tolist()
    events=pd.read_parquet(SOURCE/'events.parquet',columns=['entity_id','event_index','timestamp',*CORE],filters=[('entity_id','in',ids)])
    ctx=pd.read_parquet(SOURCE/'static_context.parquet',columns=['entity_id',*STATIC],filters=[('entity_id','in',ids)])
    events=events.merge(roles,on='entity_id',validate='many_to_one').sort_values(['customer_id','event_index'],kind='stable')
    np.testing.assert_array_equal(events.event_index,events.groupby('customer_id').cumcount())
    assert np.allclose(events.gap,events.groupby('customer_id').timestamp.diff(),equal_nan=True)
    assert set(pd.to_numeric(events[LABEL]).unique())=={0,1}
    events[LABEL]=pd.to_numeric(events[LABEL]).astype(int).astype(str)
    ctx=ctx.merge(roles,on='entity_id',validate='one_to_one')
    for c in ['cardholder_gender','cardholder_state']:ctx[c]=ctx[c].astype(str)
    roles.to_parquet(base/'roles.parquet',index=False)
    for role in ['train','validation']:
        events[events.split.eq(role)].drop(columns=['entity_id','split']).rename(columns={'customer_id':'entity_id'}).to_parquet(base/f'{role}.parquet',index=False)
        ctx[ctx.split.eq(role)][['customer_id',*STATIC]].to_parquet(base/f'{role}_context.parquet',index=False)
    train=pd.read_parquet(base/'train.parquet');val=pd.read_parquet(base/'validation.parquet')
    state=metric_state(train,CFG);write(base/'metric_state.json',state)
    parent=pd.read_parquet(base/'train_context.parquet')
    target=train.rename(columns={'entity_id':'customer_id'})[['customer_id',*CORE]].copy();target.gap=target.gap.fillna(0)
    types={c:'TABULAR_CATEGORICAL' if c in [MERCHANT,'category',LABEL] else 'TABULAR_NUMERIC_AUTO' for c in CORE}
    ctypes={c:'TABULAR_CATEGORICAL' if c in ['cardholder_gender','cardholder_state'] else 'TABULAR_NUMERIC_AUTO' for c in STATIC}
    seed(CFG['preparation_seed'])
    for name in ['parent','child']:
        wsdir=base/name
        if name=='parent':split(tgt_data=parent,tgt_primary_key='customer_id',tgt_encoding_types=ctypes,workspace_dir=wsdir)
        else:split(tgt_data=target,ctx_data=parent,tgt_context_key='customer_id',ctx_primary_key='customer_id',tgt_encoding_types=types,ctx_encoding_types=ctypes,workspace_dir=wsdir)
        analyze(workspace_dir=wsdir);encode(workspace_dir=wsdir)
    stats=Workspace(base/'child').tgt_stats.read()
    assert set(stats['columns'][LABEL]['codes']) >= {'0','1'}
    rt=val.copy();seed(CFG['preparation_seed'])
    for c,st in stats['columns'].items():
        x=rt[c].fillna(0) if c=='gap' else rt[c]
        if st['encoding_type']=='TABULAR_CATEGORICAL':rt[c]=decode_categorical(encode_categorical(x,st),st).to_numpy()
        else:rt[c]=decode_numeric(encode_numeric(x,st),st).to_numpy()
    rt.loc[rt.event_index.eq(0),'gap']=np.nan
    rt.to_parquet(base/'validation_roundtrip.parquet',index=False)
    write(DOCS/'preparation.json',dict(config_sha256=digest(CONFIG),protocol_sha256=digest(DOCS/'PROTOCOL.md'),
        test_events_loaded=False,train_customers=train.entity_id.nunique(),train_events=len(train),train_frauds=int(train[LABEL].eq('1').sum()),
        validation_customers=val.entity_id.nunique(),validation_events=len(val),validation_frauds=int(val[LABEL].eq('1').sum()),
        target_stats=stats,roundtrip=summaries(val,rt,state),train_vs_validation=summaries(train,val,state),
        inputs={p.name:digest(p) for p in base.glob('*.parquet')}))

@contextmanager
def relaxed_cap(folder):
    import mostlyai.engine._tabular.training as module
    source=inspect.getsource(module)
    old='max_epochs_cap = math.ceil((trn_cnt + val_cnt) / 50)'
    assert source.count(old)==1
    modified=source.replace(old,'max_epochs_cap = max_epochs  # registered cap-only control')
    path=folder/'training_cap_control.py';path.write_text(modified)
    (folder/'training_cap_control.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile='official_training.py',tofile=path.name)))
    write(folder/'source_control.json',dict(official_sha256=hashlib.sha256(source.encode()).hexdigest(),controlled_sha256=digest(path),installed_source_unchanged=True))
    namespace=dict(module.__dict__);exec(compile(modified,str(path),'exec'),namespace)
    original=module.train;module.train=namespace['train']
    try:yield
    finally:module.train=original

def fit(arm,fit_seed):
    from contextlib import nullcontext
    from mostlyai.engine import train,generate
    from mostlyai.engine._workspace import Workspace
    folder=OUT/'runs'/f'{arm}_{fit_seed}';folder.mkdir(parents=True,exist_ok=False)
    template=OUT/'codec_digit_both' if arm=='numeric_digit_relaxed' else OUT/'codec_digit_gap' if arm=='digit_relaxed' else OUT/'prepared'/('parent' if arm=='parent' else 'child')
    wsdir=folder/'workspace';shutil.copytree(template,wsdir)
    seed(fit_seed);start=time.monotonic()
    write(folder/'START.json',dict(arm=arm,fit_seed=fit_seed,engine=importlib.metadata.version('mostlyai-engine'),config_sha256=digest(CONFIG),script_sha256=digest(__file__)))
    with relaxed_cap(folder) if arm in ['relaxed','digit_relaxed','numeric_digit_relaxed'] else nullcontext():
        train(max_training_time=CFG['max_training_minutes'],device='cpu',workspace_dir=wsdir)
    ws=Workspace(wsdir)
    write(folder/'FIT.json',dict(seconds=time.monotonic()-start,weights_sha256=digest(ws.model_tabular_weights_path)))
    if arm=='parent':
        seed(CFG['generation_seeds'][0]);generate(sample_size=147,device='cpu',workspace_dir=wsdir)
        d=pd.read_parquet(wsdir/'SyntheticData');d.customer_id=np.arange(len(d),dtype=np.int64)+10000
        d.to_parquet(OUT/'prepared/synthetic_context.parquet',index=False)
    else: sample(arm,fit_seed)

def sample(arm,fit_seed):
    from mostlyai.engine import generate
    from mostlyai.engine._workspace import Workspace
    folder=OUT/'runs'/f'{arm}_{fit_seed}';wsdir=folder/'workspace';ws=Workspace(wsdir)
    weights=digest(ws.model_tabular_weights_path)
    for population in ['validation','synthetic']:
        ctx=pd.read_parquet(OUT/f'prepared/{population}_context.parquet')
        for gs in CFG['generation_seeds']:
            dest=folder/f'generated_{population}_{gs}.parquet'
            if dest.exists():continue
            seed(gs);start=time.monotonic();generate(ctx_data=ctx,device='cpu',workspace_dir=wsdir)
            raw=pd.read_parquet(wsdir/'SyntheticData');raw.to_parquet(folder/f'native_{population}_{gs}.parquet',index=False)
            d=native_frame(raw);d.to_parquet(dest,index=False)
            assert digest(ws.model_tabular_weights_path)==weights
            write(folder/f'generation_{population}_{gs}.json',dict(seconds=time.monotonic()-start,events=len(d),customers=d.entity_id.nunique(),weights_unchanged=True,sha256=digest(dest)))
            print(f'GENERATED {arm} {fit_seed} {population} {gs}: {len(d)}',flush=True)

def prepare_digit():
    from mostlyai.engine import encode
    from mostlyai.engine._workspace import Workspace
    target=OUT/'codec_digit_gap';original=OUT/'prepared/child'
    old=Workspace(original).tgt_stats.read();new=Workspace(target).tgt_stats.read()
    assert all(old['columns'][c]==new['columns'][c] for c in CORE if c!='gap')
    ctx=target/'ModelStore/ctx-stats/stats.json'
    shutil.copy2(ctx,target/'independently_analyzed_context_not_used.json')
    shutil.copy2(original/'ModelStore/ctx-stats/stats.json',ctx)
    seed(CFG['preparation_seed']);encode(workspace_dir=target)
    write(DOCS/'digit_preparation.json',dict(only_changed_target_column='gap',context_stats_sha256=digest(ctx),context_stats_equal=True,protocol_sha256=digest(DOCS/'DIGIT_FIT_PROTOCOL.md'),neural_fit_started=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','prepare-digit','fit','sample']);p.add_argument('--arm',choices=['parent','native','relaxed','digit_relaxed','numeric_digit_relaxed']);p.add_argument('--seed',type=int,default=20260928);args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    assert importlib.metadata.version('mostlyai-engine')==CFG['engine_version']
    if args.command=='prepare':prepare()
    elif args.command=='prepare-digit':prepare_digit()
    elif args.command=='fit':fit(args.arm,args.seed)
    else:sample(args.arm,args.seed)
