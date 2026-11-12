"""Frozen history-head two-factor removal experiment. No model fitting."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
PRIOR=ROOT/'artifacts/argn_init_validation_v1'
OUT=ROOT/'artifacts/argn_history_component_ablation_v1';DOCS=OUT/'report'
sys.path[:0]=[str(PRIOR/'local_source'),str(ROOT),str(ROOT/'scripts')]
import experiment as prior
from validation_head import factory
digest,write,now=prior.digest,prior.write,prior.now
SEEDS,PARENTS,DRAWS=prior.SEEDS,prior.PARENTS,prior.DRAWS
NEW_ARMS=['history_initial_only','history_calibration_only']
ARMS=['history_only',*NEW_ARMS,'history_init_cal','init_only']
CFG=dict(time_fit_seeds=SEEDS,parents=PARENTS,draws=DRAWS,new_arms=NEW_ARMS,
    new_fits=0,new_generations=24,reused_generations=36,evaluated_outputs=60)

def head_path(arm,rs,parent):
    root=OUT if arm in NEW_ARMS else PRIOR
    return root/f'heads/{rs}/{parent if arm.startswith("history") else 0}/{arm}/time_head.pt'

def verify():
    m=json.loads((OUT/'MANIFEST.json').read_text())
    assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['head']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()

def preserve():
    frozen=json.loads((OUT/'PREVIOUS_HASHES.json').read_text())
    for p,h in frozen.items():assert digest(p)==h,p
    return len(frozen)

def load_head(payload):
    h=factory(payload['dim'],payload['arm']);h.load_state_dict(payload['state_dict']);h.eval()
    return h

@torch.no_grad()
def prepare():
    prior.verify();assert not (OUT/'MANIFEST.json').exists()
    frozen=json.loads((ROOT/'artifacts/research_process_audit_20260930/VERIFIED_HASHES.json').read_text())
    for f in (PRIOR/'runs').glob('*/GENERATION_*.json'):
        r=json.loads(f.read_text());gs=f.stem.split('_')[-1]
        frozen[str(f.parent/f'generated_validation_{gs}.parquet')]=r['sha256']
    paths=list((PRIOR/'heads').glob('**/time_head.pt'))+list((PRIOR/'local_source').glob('*.py'))
    paths+=list((PRIOR/'report').glob('*.json'))+list((PRIOR/'report').glob('*.csv'))
    paths+=[PRIOR/'MANIFEST.json',PRIOR/'EVALUATION_READY.json',PRIOR/'report/PROTOCOL.md']
    frozen.update({str(f):digest(f) for f in paths})
    write(OUT/'PREVIOUS_HASHES.json',frozen);n=preserve();records=[]
    for rs in SEEDS:
        for parent in PARENTS:
            raw=torch.load(head_path('history_only',rs,parent),map_location='cpu',weights_only=True)
            both=torch.load(head_path('history_init_cal',rs,parent),map_location='cpu',weights_only=True)
            for k,v in raw['state_dict'].items():
                if k not in ['correction','use_initial']:assert torch.equal(v,both['state_dict'][k]),k
            assert not raw['state_dict']['use_initial'] and both['state_dict']['use_initial']
            assert torch.count_nonzero(raw['state_dict']['correction'])==0
            payloads={'history_only':raw,'history_init_cal':both}
            for arm in NEW_ARMS:
                p=copy.deepcopy(both);p['arm']=arm
                if arm=='history_initial_only':p['state_dict']['correction'].zero_()
                else:p['state_dict']['use_initial'].fill_(False)
                path=head_path(arm,rs,parent);path.parent.mkdir(parents=True,exist_ok=False)
                torch.save(p,path)
                write(path.parent/'HEAD.json',dict(arm=arm,time_fit_seed=rs,parent=parent,sha256=digest(path),
                    source_sha256=digest(head_path('history_init_cal',rs,parent)),new_fit=False))
                payloads[arm]=p
            heads={a:load_head(p) for a,p in payloads.items()}
            torch.manual_seed(rs+parent)
            pos=torch.arange(400).repeat_interleave(4);phase=torch.arange(4).repeat(400)
            clock=torch.stack([torch.linspace(8,13,len(pos)),pos.float().log1p()],1)
            valid=pos.ge(6).float();h=heads['history_only'].history_head
            base=h.center+torch.randn(len(pos),h.dim)*h.scale
            laws={a:head(base,phase,clock,valid) for a,head in heads.items()}
            initial=phase.eq(0)&pos.ge(1)&pos.le(5)
            for a,b,mask in [('history_initial_only','history_only',~initial),
                             ('history_calibration_only','history_init_cal',~initial),
                             ('history_initial_only','history_init_cal',initial),
                             ('history_calibration_only','history_only',phase.ne(0))]:
                for x,y in zip(laws[a],laws[b]):torch.testing.assert_close(x[mask],y[mask],rtol=0,atol=0)
            for j in [0,2]:torch.testing.assert_close(laws['history_calibration_only'][j],laws['history_only'][j],rtol=0,atol=0)
            for a,head in heads.items():
                for k,v in head.history_head.state_dict().items():assert torch.equal(v,h.state_dict()[k])
            records.append(dict(time_fit_seed=rs,parent=parent,positions=len(pos),frozen_neural_exact=True,
                conditional_scope_exact=True,calibration_weights_scales_exact=True))
    write(DOCS/'PREFLIGHT.json',dict(checks=records,new_fits=0,previous_hashes_verified=n))
    files=list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',DOCS/'PREFLIGHT.json',OUT/'PREVIOUS_HASHES.json']
    files+=list((OUT/'heads').glob('**/time_head.pt'))+list((PRIOR/'local_source').glob('*.py'))
    files+=list((ROOT/'benchmarks').glob('argn_*.py'))
    files+=[ROOT/'scripts'/s for s in ['run_argn_time_density.py','run_argn_amount_learning.py','run_argn_state_first.py']]
    files+=list((PRIOR/'heads').glob('**/time_head.pt'))
    write(OUT/'MANIFEST.json',dict(config=CFG,created_utc=now(),hashes={str(f):digest(f) for f in files},
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),test_events_read=False))
    verify();write(OUT/'READY.json',dict(prepared_utc=now(),new_fits=0,derived_heads=12));print('READY',n,flush=True)

def generate(parent,rs,arm):
    verify()
    with patch.object(prior,'OUT',OUT),patch.object(prior,'DOCS',DOCS),patch.object(prior,'verify',verify):
        prior.generate(parent,rs,arm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','generate'])
    p.add_argument('--parent',type=int,choices=PARENTS);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--arm',choices=NEW_ARMS);a=p.parse_args()
    prepare() if a.mode=='prepare' else generate(a.parent,a.seed,a.arm)
