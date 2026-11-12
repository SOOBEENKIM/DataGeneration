"""Freeze inherited sources, selected pipelines and the new matched protocol."""
import json,sys
from pathlib import Path
from matched_study import ROOT,OUT,OLD,DOCS,SEEDS,DRAWS,whole,prior,old_verify,generation_args,digest,write,now,verify
import pipeline,evaluate_matched

assert not (OUT/'MANIFEST.json').exists()
old_verify()
assert json.loads((OUT/'EVALUATION_READY.json').read_text())['source_sha256']==digest(evaluate_matched.__file__)
hashes=json.loads((ROOT/'artifacts/argn_calibration_specificity_v1/PREVIOUS_HASHES.json').read_text())
hashes.update(json.loads((ROOT/'artifacts/comparison_completion_v1/FROZEN_CURRENT.json').read_text())['hashes'])
preserved=len(hashes)
for p,h in hashes.items():assert digest(p)==h,p
paths=list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',OUT/'EVALUATION_READY.json',
    OLD/'NATIVE_MANIFEST.json',OLD/'STAGES_MANIFEST.json',OLD/'report/ALL_DRAWS.csv']
for module in list(sys.modules.values()):
    f=getattr(module,'__file__',None)
    if f:
        p=Path(f).resolve()
        if ROOT in p.parents and p.suffix=='.py' and p.is_file():paths.append(p)
from run_argn_state_first import SOURCE
paths.extend([SOURCE/'prepared/validation.parquet',prior.META/'metadata_optimization.parquet',
    ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',
    ROOT/'artifacts/argn_clock_stratified_loss_v1/report/CLOCK_ENDPOINTS.json'])
checks=[]
for fs in SEEDS:
    ws=whole.amount.Workspace(OLD/f'backbones/{fs}/workspace')
    a=whole.generation_args(ws,fs);b=generation_args(ws,fs)
    assert json.dumps(a,sort_keys=True,default=str)==json.dumps(b,sort_keys=True,default=str)
    paths.extend([ws.model_tabular_weights_path,OLD/f'backbones/{fs}/FIT.json'])
    for p in (OLD/f'foundation/{fs}').rglob('*'):
        if p.suffix in ['.pt','.json']:paths.append(p)
    for arm in whole.ARMS:
        paths.append(whole.head_path(arm,fs))
        for gs in DRAWS:
            run=OLD/f'runs/{arm}_{fs}'
            paths.extend([run/f'generated_validation_{gs}.parquet',run/f'GENERATION_{gs}.json',run/'COMPLETE.json'])
    checks.append(dict(seed=fs,foundation_arguments_identical=True,backbone_frozen=True))
for p in set(paths):hashes[str(p)]=digest(p)
write(OUT/'MANIFEST.json',dict(utc=now(),hashes=hashes,preserved_previous_hashes=preserved,
    protocol_sha256=digest(DOCS/'PROTOCOL.md'),seeds=SEEDS,draws=DRAWS,new_generations=6,
    unchanged_generations=18,native_argn_fits=0,statistical_mixtures=27,calibrations=3,
    matching_checks=checks,test_reads=0))
(OUT/'logs').mkdir()
verify()
print('REGISTERED',len(hashes),'frozen files;',preserved,'previous hashes;',checks,flush=True)
