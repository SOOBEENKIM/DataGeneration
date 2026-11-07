"""Fresh independent native backbones; frozen settings, no inherited weights."""
import argparse, json, logging, os, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import torch
from importlib.metadata import version
from run_argn_state_first import ROOT, OUT as BASE, CFG, seed, digest, write, check_manifest
from mostlyai.engine._workspace import Workspace
from benchmarks.argn_label_first_control import label_first_adapter
OUT=ROOT/'artifacts/argn_whole_validation_v1'
SEEDS=[20261021,20261022,20261023]
def now():return datetime.now(timezone.utc).isoformat()
def verify():
    m=json.loads((OUT/'NATIVE_MANIFEST.json').read_text())
    assert all(digest(p)==h for p,h in m['hashes'].items()),'Frozen source drift'
    assert CFG==m['inherited_config']
    assert version('mostlyai-engine')==CFG['engine_version'] and torch.__version__==CFG['torch_version']
    return check_manifest()
def fit(fs):
    assert fs in SEEDS
    prep=verify(); run=OUT/f'backbones/{fs}';run.mkdir(parents=True,exist_ok=False)
    split=prep['encoded_splits']['part.000000-trn.parquet'];mean=split['events']/split['customers']
    shutil.copytree(BASE/'prepared/workspace',run/'workspace')
    ws=Workspace(run/'workspace');assert not ws.model_tabular_weights_path.exists()
    assert torch.cuda.is_available(),'Host GPU required'
    torch.set_num_threads(CFG['cpu_threads']);seed(fs)
    write(run/'START.json',dict(seed=fs,pid=os.getpid(),utc=now(),fresh_initialization=True,
         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),mean_training_length=mean))
    start=time.monotonic()
    with label_first_adapter(ws.tgt_stats.read(),'event',mean,run) as train:
        train(model=CFG['model'],max_training_time=CFG['max_training_minutes'],max_epochs=CFG['max_epochs'],
          batch_size=CFG['batch_size'],gradient_accumulation_steps=CFG['gradient_accumulation_steps'],
          max_sequence_window=ws.tgt_stats.read()['seq_len']['max'],enable_flexible_generation=False,
          device='cuda:0',workspace_dir=run/'workspace')
    weights=torch.load(ws.model_tabular_weights_path,map_location='cpu',weights_only=True)
    verify()
    write(run/'FIT.json',dict(seed=fs,seconds=time.monotonic()-start,weights_sha256=digest(ws.model_tabular_weights_path),
          total_parameters=sum(x.numel() for x in weights.values()),utc=now(),fresh_initialization=True,test_events_read=False))
    print('FRESH_NATIVE_FIT_COMPLETE',fs,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s',force=True)
    fit(a.seed)
