"""Complete init_only on the frozen latest whole-pipeline parents."""
import argparse,json,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/argn_matched_init_comparison_v1';DOCS=OUT/'report'
OLD=ROOT/'artifacts/argn_whole_validation_v1'
sys.path[:0]=[str(OLD/'local_source'),str(ROOT),str(ROOT/'scripts')]
import stages as whole
prior=whole.prior
from initial_clock import InitialClock,initial_generation
import torch
import numpy as np
SEEDS=whole.SEEDS;DRAWS=whole.DRAWS
digest,write,now=whole.digest,whole.write,whole.now
old_verify=whole.verify

def verify():
    with patch.object(whole,'OUT',OLD):old_verify()
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():
        assert digest(p)==h,p

def head_path(arm,rs,parent=0):return OUT/f'heads/{rs}/{arm}/time_head.pt'
def save_head(head,arm,rs,parent=0):
    path=head_path(arm,rs);path.parent.mkdir(parents=True,exist_ok=False)
    payload={**prior.parameters(),'dim':head.dim,'arm':arm,'state_dict':{k:v.cpu() for k,v in head.state_dict().items()}}
    torch.save(payload,path)
    write(path.parent/'HEAD.json',dict(arm=arm,fit_seed=rs,sha256=digest(path)))
    return path

def fit(fs):
    verify();torch.set_num_threads(2)
    with patch.object(prior,'OUT',OUT),patch.object(prior,'DOCS',DOCS),patch.object(prior,'verify',verify),patch.object(prior,'save_head',save_head):
        prior.fit_statistics(fs)
    # Independent conditional-law checks: initial override and finite output.
    p=torch.load(head_path('init_only',fs),map_location='cpu',weights_only=True)
    h=InitialClock(p['dim'],p['arm']);h.load_state_dict(p['state_dict']);h.eval()
    n=torch.tensor([1,2,5,6,25,50,125,350,1024,2500]).repeat_interleave(4)
    ph=torch.arange(4).repeat(len(n)//4);clock=torch.stack([torch.linspace(7,14,len(n)),n.float().log1p()],1)
    valid=n.ge(6).float();base=torch.zeros(len(n),p['dim'])
    with torch.no_grad():a=h(base,ph,clock,valid);b=h(base+17,ph,clock,valid)
    for x,y in zip(a,b):assert torch.equal(x,y) and torch.isfinite(x).all()
    assert (a[2]>0).all()
    mask=ph.eq(0)&n.le(5)
    torch.testing.assert_close(a[0][mask],h.initial_weights.log()[None].expand(mask.sum(),-1),rtol=0,atol=0)
    torch.testing.assert_close(a[1][mask],h.initial_location[None].expand(mask.sum(),-1),rtol=0,atol=0)
    torch.testing.assert_close(a[2][mask],h.initial_scale[None].expand(mask.sum(),-1),rtol=0,atol=0)
    assert not h.handoff.any()
    write(OUT/f'fits/statistics_{fs}/HEAD_PREFLIGHT.json',dict(finite=True,base_input_ignored_as_registered=True,
      first_five_override_exact=True,handoff_zero=True,head_sha256=digest(head_path('init_only',fs))))
    verify()

def folder(arm,fs):
    assert arm=='B_event_label_first' and fs in SEEDS
    return OLD/f'backbones/{fs}'

def generation_args(ws,fs):
    f=OLD/f'foundation/{fs}';joint=whole.joint
    return (ws.tgt_stats.read(),json.loads(joint.OLD.read_text()),f/'amount_head.pt',
      json.loads((joint.PRIOR/'episode_parameters.json').read_text()),
      json.loads((joint.OUT/'hazard_parameters.json').read_text()),f/'count/count_head.pt',True),dict(onset_payload=f/'onset/onset_head.pt')

def generate(fs):
    verify();assert (OUT/f'fits/statistics_{fs}/HEAD_PREFLIGHT.json').exists()
    with patch.object(whole,'OUT',OUT),patch.object(whole,'DOCS',DOCS),patch.object(whole,'verify',verify),\
         patch.object(whole,'folder',folder),patch.object(whole,'head_path',head_path),\
         patch.object(whole,'generation_args',generation_args),patch.object(whole,'generation',initial_generation):
        whole.generate(fs,'init_only')
    verify()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','generate']);p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    a=p.parse_args();globals()[a.mode](a.seed)
