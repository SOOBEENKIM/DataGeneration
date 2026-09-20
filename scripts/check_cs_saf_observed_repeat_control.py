"""CPU gates for a real probability transformation, before scientific execution."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cs_saf_observed_repeat_control import fit_controls, apply_numpy, apply_torch, redistribute
from scripts.run_cs_saf_external_audit_v1 import write


def main():
    torch.set_num_threads(1)
    c=json.loads((ROOT/'configs/benchmark_v2/cs_saf_baseline_adequacy_v1.json').read_text())
    rng=np.random.default_rng(1902)
    n=4000; code=np.tile(np.arange(100),40);group=np.repeat([0,1],n//2)
    y=rng.random(n)<(.2+.3*(code<40)*(group==1))
    f=pd.DataFrame(dict(p=np.full(n,.04),gap_code=code,group=group,y=y.astype(int)))
    controls=fit_controls(f,c);errors=[]
    for control in controls.values():
        expected=apply_numpy(f.p.to_numpy(),code,group,control)
        actual=apply_torch(torch.tensor(f.p.to_numpy()),torch.tensor(code),torch.tensor(group),control).numpy()
        errors.append(float(np.max(abs(actual-expected))))
    p=torch.tensor(rng.dirichlet(np.ones(64),size=20),dtype=torch.float64)
    previous=torch.tensor(rng.integers(64,size=20));r=torch.linspace(.01,.8,20,dtype=torch.float64)
    out=redistribute(p,previous,r)
    assert torch.allclose(out.sum(-1),torch.ones(20,dtype=torch.float64),atol=1e-14,rtol=0)
    assert torch.equal(out.gather(1,previous[:,None])[:,0],r)
    native_r=p.gather(1,previous[:,None])[:,0]
    identity=float((redistribute(p,previous,native_r)-p).abs().max())
    mask=torch.ones_like(p,dtype=torch.bool).scatter(1,previous[:,None],False)
    ratio=(out/p)[mask].reshape(20,63)
    preserved=float((ratio-ratio[:,:1]).abs().max())
    assert max(errors+[identity,preserved])<1e-12
    report=dict(passed=True,numpy_torch_max_error=max(errors),identity_probability_max_error=identity,
                nonrepeat_relative_probability_max_error=preserved,not_a_scientific_fit=True)
    write(ROOT/'docs/cs_saf/baseline_adequacy_v1/probability_cpu_gate.json',report);print(report)


if __name__=='__main__':main()
