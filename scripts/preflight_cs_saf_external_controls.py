"""Validate real development inputs without accessing test or fitting a model."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from data.cs_saf_external import TargetWindows
from scripts.run_cs_saf_external_controls import OUT, OLD, CONFIG, amount_state, build_model
from scripts.run_cs_saf_external_port import inputs, seed, write, digest


def main():
    torch.set_num_threads(1)
    cfg=json.loads(CONFIG.read_text());result={}
    for name in cfg['datasets']:
        inp,parents,ids,frames,plan,metric_state=inputs(name)
        data=torch.load(inp/'prepared.pt',map_location='cpu')
        state=SAFTensorizerState.from_dict(data['state']);ast=amount_state(frames['fit'],state)
        models={};checks={}
        for mid in cfg['new_models']:
            reference=OLD/'runs'/name/('G' if mid[0]=='D' else mid[0])/'initial.pt'
            seed(cfg['fit_seed']);m,shared=build_model(mid,state,ast,torch.load(reference,map_location='cpu'))
            models[mid]=m
            checks[mid]=dict(**m.architecture_contract(),shared_keys=len(shared),reference_sha256=digest(reference))
        assert abs(checks['D_both']['parameters']/checks['G_both']['parameters']-1)<=cfg['screens']['direct_parameter_relative_tolerance']
        for role,sequences in data['sequences'].items():
            assert role in ('fit','check','validation')
            w=TargetWindows(sequences)
            values=torch.cat([torch.as_tensor(seq.numeric_value) for seq in sequences])
            m=models['G_amount'];n=len(values)
            # Every observed amount, including zero targets, must have finite density.
            for start in range(0,n,65536):
                v=values[start:start+65536];k=len(v)
                params=(torch.full((k,3),-np.log(3)),torch.zeros(k,3),torch.ones(k,3),torch.full((k,),-5.))
                assert torch.isfinite(m.amount_nll(v,params)).all()
            indices=[];offset=0
            for seq in sequences:
                length=len(seq.receiver)
                indices.extend(offset+p for p in sorted(set([0,min(1,length-1),min(31,length-1),min(32,length-1),length-1])))
                offset+=length
            assert n==len(w)==len(frames[role])
            for mid,m in models.items():
                for start in range(0,len(indices),256):
                    with torch.no_grad():
                        terms,o=m.terms(**w.batch(indices[start:start+256]))
                    assert all(torch.isfinite(v) for v,c in terms.values())
                    assert torch.allclose(o['logmark'].exp().sum(-1),torch.ones(len(o['mark'])),atol=2e-6)
            checks[role]=dict(events=n,boundary_targets_checked=len(indices),all_amount_targets_checked=True)
        result[name]=dict(input_sha256=digest(inp/'preflight.json'),plan_sha256=digest(inp/'plan.parquet'),checks=checks)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'preflight.json';assert not path.exists()
    write(path,dict(config_sha256=digest(CONFIG),datasets=result,test_accessed=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
