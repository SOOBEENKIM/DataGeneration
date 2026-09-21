"""Recompute all new generated metrics with independent histogram/ECDF code."""
import json
from pathlib import Path
import subprocess
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.verify_cs_saf_external_port import sha,transforms,distributions,distance,ks

OUT=ROOT/'artifacts/cs_saf/external_controls_v1'
OLD=ROOT/'artifacts/cs_saf/external_port_v1'


def main():
    config=ROOT/'configs/cs_saf_external_controls_v1.json'
    cfg=json.loads(config.read_text())
    result=dict(passed=True,scalar_metrics_verified=0,max_absolute_error=0.,events=0,runs=[],
                evaluator_imported=False,test_outcomes_accessed=False,verifier_sha256=sha(__file__))
    for name in cfg['datasets']:
        inp=OLD/'input'/name;pre=json.loads((inp/'preflight.json').read_text())
        for f,h in pre['files'].items():assert sha(inp/f)==h
        events=pd.read_parquet(inp/'events.parquet');roles=pd.read_parquet(inp/'roles.parquet');plan=pd.read_parquet(inp/'plan.parquet')
        assert set(roles.role)=={'fit','check','validation'}
        fit=events.loc[events.entity_id.isin(roles.loc[roles.role.eq('fit'),'entity_id'])]
        val=events.loc[events.entity_id.isin(roles.loc[roles.role.eq('validation'),'entity_id'])]
        state=transforms(fit,name);reference,rinfo=distributions(val,state,name)
        for model in cfg['new_models']:
            folder=OUT/'runs'/name/model;done=json.loads((folder/'DONE.json').read_text())
            assert done['config_sha256']==sha(config)
            for f,h in done['files'].items():assert sha(folder/f)==h
            for variant,data in done['results'].items():
                for generation in data['generations']:
                    gs=generation['seed'];path=folder/f'generated_{variant}_{gs}.parquet'
                    assert sha(path)==generation['sha256']
                    frame=pd.read_parquet(path);actual,info=distributions(frame,state,name)
                    assert set(frame.entity_id)==set(plan.entity_id)
                    np.testing.assert_array_equal(frame.event_index,frame.groupby('entity_id',sort=False).cumcount())
                    np.testing.assert_array_equal(frame.groupby('entity_id').size().reindex(plan.entity_id),plan.length)
                    rebuilt=frame.gap.fillna(0).groupby(frame.entity_id,sort=False).cumsum()
                    np.testing.assert_allclose(frame.timestamp,rebuilt,rtol=1e-7,atol=1e-5)
                    computed={k:distance(reference[k],v) for k,v in actual.items()}
                    computed['amount_ks']=ks(rinfo['amounts'],info.pop('amounts'))
                    computed['length_ks']=ks(rinfo['lengths'],info.pop('lengths'));computed.update(info)
                    for k,v in computed.items():
                        expected=generation['metrics'][k]
                        if v is None:assert expected is None
                        else:
                            error=abs(v-expected);assert error<1e-11,(name,model,variant,gs,k,error)
                            result['max_absolute_error']=max(result['max_absolute_error'],error)
                        result['scalar_metrics_verified']+=1
                    if not model.endswith('_action'):assert computed['invalid_amount_rate']==0
                    result['events']+=len(frame)
                    result['runs'].append(dict(dataset=name,model=model,variant=variant,seed=gs,events=len(frame),sha256=sha(path)))
            print(name,model,'independent metrics PASS',flush=True)
    result['old_files_unchanged']=[]
    for file in ('models/cs_saf_external.py','models/cof_seqgen_saf.py','data/cof_seqgen_saf_tensorizer.py',
                 'data/cs_saf_external.py','benchmarks/cs_saf_external.py','scripts/run_cs_saf_external_port.py'):
        blob=subprocess.check_output(['git','show',cfg['parent_commit']+':'+file],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==sha(ROOT/file)
        result['old_files_unchanged'].append(file)
    assert len(result['runs'])==56
    path=OUT/'independent_verification.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('ALL PASS',result['scalar_metrics_verified'],result['max_absolute_error'],flush=True)


if __name__=='__main__':main()
