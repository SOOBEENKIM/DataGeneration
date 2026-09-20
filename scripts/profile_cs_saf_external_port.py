"""Allocation/throughput gate without optimizer updates or scientific fitting."""
import json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data.cs_saf_external import TargetWindows
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from models.cs_saf_external import ExternalUG

torch.set_num_threads(1)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
index,used,util=map(int,gpu.splitlines()[0].split(','))
assert index==0 and used<512 and util<5,'GPU no longer free; do not interrupt another job'
torch.cuda.set_per_process_memory_fraction(.5)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
report={}
for name in ('berka','sparkov'):
    p=torch.load(ROOT/f'artifacts/cs_saf/external_port_v1/input/{name}/prepared.pt',map_location='cpu')
    state=SAFTensorizerState.from_dict(p['state'])
    w=TargetWindows(p['sequences']['fit'],device='cuda')
    m=ExternalUG('U',state.gap_support,**state.model_config_kwargs()).cuda()
    batch=w.batch(np.arange(512)+500)
    timings=[]
    for i in range(4):
        torch.cuda.synchronize();start=time.monotonic()
        m.zero_grad(set_to_none=True);m.compute_loss(**batch)['loss'].backward()
        torch.cuda.synchronize();timings.append(time.monotonic()-start)
    report[name]=dict(batch_size=512,forward_backward_seconds=timings,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),optimizer_updates=0,
        cuda_device=torch.cuda.get_device_name(),estimated_fit_epoch_seconds=len(w)/512*np.mean(timings[1:]))
    del w,m,batch,p;torch.cuda.empty_cache()
path=ROOT/'artifacts/cs_saf/external_port_v1/gpu_profile.json'
assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
