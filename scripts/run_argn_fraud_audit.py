"""Isolated, resumable official ARGN audit. Never reads final test outcomes."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
OUT = ROOT / 'artifacts/argn_fraud_audit_v1'
DOCS = ROOT / 'docs/argn_fraud_audit_v1'
CONFIG = ROOT / 'configs/argn_fraud_audit_v1.json'
OLD = PARENT / 'research-argn-relation-pilot'
SOURCE = PARENT / 'cof-seqgen-0707-2119-Version3-complete/data/cof_seqgen_saf/canonical/sparkov'
ROLES = PARENT / 'research-cs-saf-external-audit/artifacts/cs_saf/external_port_v1/input/sparkov/roles.parquet'
os.environ.setdefault('HF_HOME', str(OUT / 'cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(OUT / 'cache/datasets'))
os.environ.setdefault('JOBLIB_TEMP_FOLDER', str(OUT / 'cache/joblib'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from benchmarks.argn_fraud_audit import CORE, STATIC, LABEL, AMOUNT, MERCHANT, metric_state, summaries, position_curves, risk_table


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def config():
    cfg = json.loads(CONFIG.read_text())
    assert importlib.metadata.version('mostlyai-engine') == cfg['engine_version']
    torch.set_num_threads(cfg['cpu_threads'])
    return cfg


def data():
    roles = pd.read_parquet(ROLES)
    split = pd.read_parquet(SOURCE/'entity_splits.parquet')
    allowed = set(split.loc[split.split.isin(['train','validation']), 'entity_id'])
    assert set(roles.entity_id) <= allowed and not roles.entity_id.duplicated().any()
    ids = roles.entity_id.tolist()
    columns = ['entity_id','event_index','timestamp',*CORE]
    events = pd.read_parquet(SOURCE/'events.parquet', columns=columns, filters=[('entity_id','in',ids)])
    parents = pd.read_parquet(SOURCE/'static_context.parquet', columns=['entity_id',*STATIC], filters=[('entity_id','in',ids)])
    for c in ['cardholder_gender','cardholder_state']: parents[c] = parents[c].astype(str)
    assert set(pd.to_numeric(events[LABEL]).unique()) <= {0,1}
    events[LABEL] = pd.to_numeric(events[LABEL]).astype(int).astype(str)
    events = events.sort_values(['entity_id','event_index'], kind='stable').reset_index(drop=True)
    assert not events.duplicated(['entity_id','event_index']).any()
    assert np.array_equal(events.event_index.to_numpy(), events.groupby('entity_id').cumcount().to_numpy())
    actual_gap = events.groupby('entity_id').timestamp.diff()
    assert np.allclose(events.gap, actual_gap, equal_nan=True)
    frames = {r: events[events.entity_id.isin(roles.loc[roles.role.eq(r),'entity_id'])].copy() for r in ['fit','check','validation']}
    return frames, parents, roles


def codec_roundtrip(frame, stats, seed):
    from mostlyai.engine import set_random_state
    from mostlyai.engine._encoding_types.tabular.numeric import encode_numeric, decode_numeric
    from mostlyai.engine._encoding_types.tabular.categorical import encode_categorical, decode_categorical
    set_random_state(seed)
    d = frame.copy()
    for c, st in stats['columns'].items():
        x = d[c].copy()
        if c == 'gap': x = x.fillna(0)
        if st['encoding_type'] == 'TABULAR_CATEGORICAL':
            d[c] = decode_categorical(encode_categorical(x, st), st).to_numpy()
        else:
            d[c] = decode_numeric(encode_numeric(x, st), st).to_numpy()
    d.loc[d.event_index.eq(0),'gap'] = np.nan
    return d


def audit_existing():
    cfg = config(); frames, _, _ = data(); state = metric_state(frames['fit'], cfg)
    from mostlyai.engine._workspace import Workspace
    ws = Workspace(OLD/'artifacts/argn_relation_pilot_v1/A/workspace')
    roundtrip = codec_roundtrip(frames['validation'].drop(columns=[LABEL]), ws.tgt_stats.read(), 20260926)
    rows=[]; curves=[]; hashes={}
    for arm in ['A','G','R']:
        for p in sorted((OLD/f'artifacts/argn_relation_pilot_v1/{arm}').glob('generated_*.parquet')):
            d=pd.read_parquet(p); hashes[str(p.relative_to(OLD))]=digest(p)
            rows.append(dict(arm=arm,seed=p.stem,**summaries(frames['validation'],d,state)))
            curves += [dict(arm=arm,seed=p.stem,**x) for x in position_curves(frames['validation'],d,frames['fit'],state)]
    DOCS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DOCS/'existing_metrics.csv', index=False)
    pd.DataFrame(curves).to_csv(DOCS/'existing_position_curves.csv', index=False)
    result=dict(local_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=OLD,text=True).strip(),
        generated_hashes=hashes, old_outputs_include_fraud=False,
        codec_roundtrip=summaries(frames['validation'],roundtrip,state),
        codec_target_types={k:v['encoding_type'] for k,v in ws.tgt_stats.read()['columns'].items()},
        test_outcomes_accessed=False, validation_status='previously reused development split')
    write(DOCS/'existing_audit.json',result)
    print(json.dumps(result['codec_roundtrip'],indent=2),flush=True)
    print(pd.DataFrame(curves).query("arm == 'A'").to_string(index=False),flush=True)


def prepare():
    cfg=config(); frames, parents, roles=data()
    from mostlyai.engine import split,analyze,encode,set_random_state
    from mostlyai.engine._workspace import Workspace
    base=OUT/'prepared'; base.mkdir(parents=True,exist_ok=False)
    state=metric_state(frames['fit'],cfg); write(base/'metric_state.json',state)
    for role,d in frames.items(): d.to_parquet(base/f'{role}.parquet',index=False)
    parents.to_parquet(base/'context.parquet',index=False); roles.to_parquet(base/'roles.parquet',index=False)
    fit_ids=set(frames['fit'].entity_id); check_ids=set(frames['check'].entity_id)
    raw=pd.concat([frames['fit'],frames['check']]).sort_values(['entity_id','event_index'])[['entity_id',*CORE]].copy()
    raw['gap']=raw.gap.fillna(0)
    ctx=parents[parents.entity_id.isin(fit_ids|check_ids)]
    types={c:'TABULAR_CATEGORICAL' if c in [MERCHANT,'category',LABEL] else 'TABULAR_NUMERIC_AUTO' for c in CORE}
    ct={c:'TABULAR_CATEGORICAL' if c in ['cardholder_gender','cardholder_state'] else 'TABULAR_NUMERIC_AUTO' for c in STATIC}
    wsdir=base/'workspace'; set_random_state(20260926)
    split(tgt_data=raw,ctx_data=ctx,tgt_context_key='entity_id',ctx_primary_key='entity_id',
          tgt_encoding_types=types,ctx_encoding_types=ct,workspace_dir=wsdir,
          trn_val_split=lambda keys:(keys[keys.isin(fit_ids)],keys[keys.isin(check_ids)]))
    ws=Workspace(wsdir)
    # Fit codec on fit customers only. Temporarily withhold check partitions in THIS new workspace.
    parked=[]; park=base/'withheld_check'; park.mkdir()
    for kind,folder in [('tgt',ws.tgt_data_path),('ctx',ws.ctx_data_path)]:
        for p in folder.glob('*val.parquet'):
            dest=park/(kind+'-'+p.name); shutil.move(str(p),dest); parked.append((p,dest))
    assert len(parked)==2
    try:
        analyze(value_protection=cfg['value_protection'],workspace_dir=wsdir)
    finally:
        for original,p in parked: shutil.move(str(p),original)
    stats=ws.tgt_stats.read()
    assert stats['no_of_training_records']==len(fit_ids) and stats['no_of_validation_records']==0
    fit_only_columns=hashlib.sha256(json.dumps(stats['columns'],sort_keys=True).encode()).hexdigest()
    stats['no_of_validation_records']=len(check_ids)  # counts only; no check values enter statistics
    ws.tgt_stats.write(stats)
    assert set(stats['columns'][LABEL]['codes']) >= {'0','1'}
    assert set(ws.ctx_stats.read()['columns'])==set(STATIC)
    encode(workspace_dir=wsdir)
    manifest=dict(config_sha256=digest(CONFIG),roles_sha256=digest(ROLES),source_schema_sha256=digest(SOURCE/'schema.json'),
        prepared_files={p.name:digest(p) for p in base.glob('*.parquet')},
        counts={r:dict(customers=d.entity_id.nunique(),events=len(d),frauds=int(d[LABEL].eq('1').sum())) for r,d in frames.items()},
        fit_only_codec_columns_sha256=fit_only_columns,codec_types={c:s['encoding_type'] for c,s in stats['columns'].items()},
        fraud_codes=stats['columns'][LABEL]['codes'],test_outcomes_accessed=False,
        engine_version=importlib.metadata.version('mostlyai-engine'),
        official_source_hashes={name:digest(Path(__import__('mostlyai.engine',fromlist=['x']).__file__).parent/name) for name in ['_tabular/training.py','_tabular/argn.py','_tabular/generation.py']})
    write(base/'manifest.json',manifest); write(DOCS/'preparation.json',manifest)
    for r,d in frames.items(): risk_table(d,state,r).to_csv(DOCS/f'{r}_risk_support.csv',index=False)
    roundtrip=codec_roundtrip(frames['validation'],stats,20260926)
    write(DOCS/'new_codec_roundtrip.json',summaries(frames['validation'],roundtrip,state))
    print(json.dumps(manifest,indent=2),flush=True)


def load_prepared():
    base=OUT/'prepared'; m=json.loads((base/'manifest.json').read_text())
    for n,h in m['prepared_files'].items(): assert digest(base/n)==h
    assert digest(CONFIG)==m['config_sha256']
    return base,m


def fit(seed, device):
    cfg=config(); base,m=load_prepared()
    assert seed in cfg['fit_seeds']
    from mostlyai.engine import train,set_random_state
    from mostlyai.engine._workspace import Workspace
    run=OUT/f'seed_{seed}'; run.mkdir(exist_ok=False)
    wsdir=run/'workspace'; shutil.copytree(base/'workspace',wsdir)
    set_random_state(seed); start=time.monotonic()
    write(run/'START.json',dict(seed=seed,device=device,config_sha256=digest(CONFIG),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=digest(__file__),prepared=m))
    train(model=cfg['model'],max_epochs=cfg['max_epochs'],max_training_time=cfg['max_training_minutes'],
          batch_size=cfg['batch_size'],max_sequence_window=cfg['max_sequence_window'],
          enable_flexible_generation=cfg['enable_flexible_generation'],device=device,workspace_dir=wsdir)
    ws=Workspace(wsdir); progress=pd.read_csv(ws.model_progress_messages_path)
    best=progress[progress.is_checkpoint.eq(1)].iloc[-1]
    result=dict(seed=seed,device=device,seconds=time.monotonic()-start,last_epoch=float(progress.epoch.max()),
        selected_epoch=float(best.epoch),updates=int(progress.steps.max()),selected_check_loss=float(best.val_loss),
        checkpoint_sha256=digest(ws.model_tabular_weights_path),native_early_stop=True,
        budget_reached=bool(progress.epoch.max()>=cfg['max_epochs'] or time.monotonic()-start>=cfg['max_training_minutes']*60))
    progress.to_csv(DOCS/f'training_{seed}.csv',index=False); write(run/'FIT.json',result);write(DOCS/f'fit_{seed}.json',result)
    print(json.dumps(result,indent=2),flush=True)


def generate(seed, device):
    cfg=config();base,_=load_prepared();run=OUT/f'seed_{seed}';fit_result=json.loads((run/'FIT.json').read_text())
    from mostlyai.engine import generate as native_generate,set_random_state
    from mostlyai.engine._common import load_generated_data
    from mostlyai.engine._workspace import Workspace
    ws=Workspace(run/'workspace');assert digest(ws.model_tabular_weights_path)==fit_result['checkpoint_sha256']
    real=pd.read_parquet(base/'validation.parquet');parents=pd.read_parquet(base/'context.parquet')
    ctx=parents[parents.entity_id.isin(real.entity_id)].copy()
    for gs in cfg['generation_seeds']:
        target=run/f'generated_{gs}.parquet'
        if target.exists():
            if not (run/f'generation_{gs}.json').exists(): raise RuntimeError('Incomplete prior generation')
            continue
        set_random_state(gs);start=time.monotonic()
        native_generate(ctx_data=ctx,batch_size=64,device=device,workspace_dir=run/'workspace',
                        sampling_temperature=1.,sampling_top_p=1.)
        d=load_generated_data(run/'workspace')
        assert set(d.entity_id)<=set(ctx.entity_id)
        d['event_index']=d.groupby('entity_id',sort=False).cumcount()
        d.loc[d.event_index.eq(0),'gap']=np.nan
        d['timestamp']=d.gap.fillna(0).groupby(d.entity_id,sort=False).cumsum()
        d.to_parquet(target,index=False)
        write(run/f'generation_{gs}.json',dict(seed=gs,events=len(d),customers=d.entity_id.nunique(),seconds=time.monotonic()-start,
             sha256=digest(target),checkpoint_sha256=digest(ws.model_tabular_weights_path)))
        print('GENERATED',seed,gs,len(d),flush=True)


def evaluate():
    cfg=config();base,_=load_prepared();state=json.loads((base/'metric_state.json').read_text())
    real=pd.read_parquet(base/'validation.parquet');fitdata=pd.read_parquet(base/'fit.parquet')
    rows=[];curves=[];risk=[]
    for p in sorted(OUT.glob('seed_*/generated_*.parquet')):
        d=pd.read_parquet(p);name=p.parent.name+'/'+p.stem
        rows.append(dict(run=name,**summaries(real,d,state)))
        curves += [dict(run=name,**x) for x in position_curves(real,d,fitdata,state)]
        risk.append(risk_table(d,state,name))
    pd.DataFrame(rows).to_csv(DOCS/'generation_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(DOCS/'position_curves.csv',index=False)
    if risk: pd.concat([risk_table(real,state,'original_validation'),*risk]).to_csv(DOCS/'conditional_risk.csv',index=False)
    ids=real.entity_id.unique();rng=np.random.default_rng(20260926);noise=[]
    for i in range(cfg['real_real_repeats']):
        x=rng.permutation(ids);a=real[real.entity_id.isin(x[:len(x)//2])];b=real[real.entity_id.isin(x[len(x)//2:])]
        noise.append(dict(repeat=i,**summaries(a,b,state)))
    pd.DataFrame(noise).to_csv(DOCS/'real_real_reference.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit-existing','prepare','fit','generate','evaluate'])
    parser.add_argument('--seed',type=int);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.device=='cuda': raise SystemExit('GPU runner not enabled: host GPUs are currently occupied. Use bounded CPU execution.')
    {'audit-existing':audit_existing,'prepare':prepare,'fit':lambda:fit(args.seed,args.device),
     'generate':lambda:generate(args.seed,args.device),'evaluate':evaluate}[args.stage]()
