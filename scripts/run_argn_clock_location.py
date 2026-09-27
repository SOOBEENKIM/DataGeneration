"""Bounded location-only ablation based on internal validation, not dev outputs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import run_argn_clock_evolution as study
from run_argn_clock_evolution import ROOT,digest,write,ConditionalSimulation,objective,NumpyDensity,verify_prior
from benchmarks.argn_clock_location import ClockLocation,location_generation

OUT=ROOT/'artifacts/argn_clock_location_v1'; DOCS=ROOT/'docs/argn_clock_location_v1'
CFG={**study.CFG,'arms':['clock_prefix','clock_rollout']}


def verify():
    verify_prior()
    for path in [ROOT/'artifacts/argn_clock_evolution_v1/MANIFEST.json',OUT/'MANIFEST.json']:
        m=json.loads(path.read_text())
        for p,h in m['hashes'].items(): assert digest(p)==h,p
    assert json.loads((OUT/'MANIFEST.json').read_text())['config']==CFG


def prepare():
    study.verify(); assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),ROOT/'benchmarks/argn_clock_location.py',ROOT/'tests/test_argn_clock_location.py',
        DOCS/'PROTOCOL.md',study.OUT/'MANIFEST.json',study.DOCS/'internal_conditional_diagnostics.csv']
    record=dict(config=CFG,created_utc=datetime.now(timezone.utc).isoformat(),test_events_read=False,
        hashes={str(p):digest(p) for p in paths},development_generations_not_consulted=True)
    write(OUT/'MANIFEST.json',record);write(DOCS/'MANIFEST.json',record)
    old=torch.load(ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',map_location='cpu',weights_only=True)
    simulator=ConditionalSimulation('optimization',old['fallback']); density=NumpyDensity(old)
    for arm,source in [('clock_prefix','actual'),('clock_rollout','recursive')]:
        reference=simulator.run(density,[0.,0.],source,CFG['fit_noise_seed']);curve=[]
        def evaluate(theta):
            metrics=simulator.run(density,theta,source,CFG['fit_noise_seed']);score=objective(metrics,reference)
            curve.append(dict(evaluation=len(curve),a=float(theta[0]),b=float(theta[1]),objective=score,**metrics))
            print('LOCATION_FIT',arm,len(curve),np.round(theta,5),round(score,6),flush=True);return score
        evaluate([0.,0.])
        result=minimize(evaluate,np.zeros(2),method='Powell',bounds=CFG['bounds'],options=dict(maxfev=CFG['maxfev']-1,xtol=.001,ftol=.001))
        best=min(curve,key=lambda r:r['objective']);head=ClockLocation(1,arm)
        head.load_state_dict({**old['state_dict'],'correction':torch.tensor([best['a'],best['b']],dtype=torch.float32)})
        dest=OUT/arm;dest.mkdir();torch.save({**old,'arm':arm,'state_dict':head.state_dict()},dest/'time_head.pt')
        for k,v in old['state_dict'].items():torch.testing.assert_close(head.state_dict()[k],v,rtol=0,atol=0)
        pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv',index=False)
        result=dict(selected=best,initial=curve[0],source=source,optimizer_success=bool(result.success),
            optimizer_message=str(result.message),evaluations=len(curve),reference=reference,head_sha256=digest(dest/'time_head.pt'),
            original_gmr_buffers_unchanged=True)
        write(dest/'FIT_COMPLETE.json',result);write(DOCS/f'{arm}_fit.json',result)
    rows=[];simulator=ConditionalSimulation('internal_validation',old['fallback'])
    for gs in CFG['diagnostic_seeds']:
        for arm in CFG['arms']:
            p=torch.load(OUT/arm/'time_head.pt',map_location='cpu',weights_only=True);theta=p['state_dict']['correction'].tolist()
            for source in ['actual','recursive']:
                rows.append(dict(arm=arm,source=source,noise_seed=gs,**simulator.run(NumpyDensity(p),theta,source,gs)))
    pd.DataFrame(rows).to_csv(DOCS/'internal_conditional_diagnostics.csv',index=False)
    verify();write(OUT/'FIT_COMPLETE.json',dict(statistical_phase_fits=0,calibration_fits=2,native_argn_fits=0,
        test_events_read=False,heads={arm:digest(OUT/arm/'time_head.pt') for arm in CFG['arms']}))
    print('LOCATION_PREPARED',flush=True)


def execute(mode,fs,arm):
    verify()
    with patch.object(study,'OUT',OUT),patch.object(study,'DOCS',DOCS),patch.object(study,'CFG',CFG), \
         patch.object(study,'verify',verify),patch.object(study,'evolution_generation',location_generation), \
         patch.object(study,'__file__',str(Path(__file__).resolve())):
        study.worker(fs,arm) if mode=='worker' else study.dispatch()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','dispatch'])
    p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--arm',choices=CFG['arms']);a=p.parse_args()
    prepare() if a.mode=='prepare' else execute(a.mode,a.seed,a.arm)
