"""Independent saved-row evaluation; no fitting or source mutation."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
from stages import ROOT,OUT,DOCS,SEEDS,ARMS,DRAWS,verify,digest,write,now,head_path
from run_argn_state_first import SOURCE
p=ROOT/'artifacts/argn_init_validation_v1/local_source/evaluate.py'
spec=importlib.util.spec_from_file_location('old_independent_evaluator',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def check():
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet')
    path=ROOT/'artifacts/argn_init_validation_v1/runs/history_only_73001_20260930/generated_validation_20261011.parquet'
    values,_=old.metrics(real,pd.read_parquet(path));table=pd.read_csv(ROOT/'artifacts/argn_init_validation_v1/report/ALL_DRAWS.csv')
    r=table[(table.arm=='history_only')&(table.time_fit_seed==73001)&(table.parent==20260930)&(table.generation_seed==20261011)].iloc[0]
    errors={k:abs(float(v)-float(r[k])) for k,v in values.items() if k in r}
    assert max(errors.values())<1e-10
    write(OUT/'EVALUATION_READY.json',dict(utc=now(),checks=errors,max_error=max(errors.values()),source_sha256=digest(__file__),test_reads=0))

def main():
    verify();assert (OUT/'STAGES_COMPLETE.json').exists()
    assert digest(__file__)==json.loads((OUT/'EVALUATION_READY.json').read_text())['source_sha256']
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rows=[];checks=[];supports=[];total=0
    for fs in SEEDS:
        for gs in DRAWS:
            base=pd.read_parquet(OUT/f'runs/history_only_{fs}/generated_validation_{gs}.parquet')
            for arm in ARMS:
                run=OUT/f'runs/{arm}_{fs}';assert (run/'COMPLETE.json').exists()
                path=run/f'generated_validation_{gs}.parquet';rec=json.loads((run/f'GENERATION_{gs}.json').read_text())
                assert digest(path)==rec['sha256'] and digest(head_path(arm,fs))==rec['head_sha256']
                raw=pd.read_parquet(path);cols=['entity_id','event_index','event_is_fraud'];assert raw[cols].equals(base[cols]);total+=len(raw)
                r=pd.read_csv(DOCS/f'evaluation/metrics_{arm}_{fs}.csv');r=r[r.generation_seed==gs].iloc[0].to_dict()
                vals,support=old.metrics(real,raw);errors={k:abs(float(v)-float(r[k])) for k,v in vals.items() if k in r};assert max(errors.values())<1e-10
                info=dict(arm=arm,fit_seed=fs,generation_seed=gs);r.update(vals);r.update(info);r['events']=len(raw);rows.append(r)
                checks.append({**info,'checks':len(errors),'max_error':max(errors.values()),'exact_labels_lengths':True})
                supports.extend([{**info,**s} for s in support]);print('AUDITED',info,len(raw),flush=True)
    d=pd.DataFrame(rows);assert len(d)==18;d.to_csv(DOCS/'ALL_DRAWS.csv',index=False)
    pd.DataFrame(checks).to_csv(DOCS/'INDEPENDENT_CHECKS.csv',index=False);pd.DataFrame(supports).to_csv(DOCS/'SUPPORT.csv',index=False)
    metrics=old.KEYS;f=d.groupby(['arm','fit_seed'])[metrics].mean();f.to_csv(DOCS/'FIT_MEANS.csv');m=f.groupby('arm').mean();m.to_csv(DOCS/'MEANS.csv')
    paired=[]
    for arm in ARMS[:2]:
        for fs in SEEDS:
            for k in metrics:paired.append(dict(comparator=arm,fit_seed=fs,metric=k,delta_C_minus_control=f.loc[('history_calibration_only',fs),k]-f.loc[(arm,fs),k]))
    pd.DataFrame(paired).to_csv(DOCS/'PAIRED_FIT_EFFECTS.csv',index=False)
    n,o='normal_gap_seconds_log_w1','onset_personal_gap_w1';C=m.loc['history_calibration_only'];P=m.loc['history_position6']
    stable=int(sum(f.loc[('history_calibration_only',fs),n]<f.loc[('history_position6',fs),n] for fs in SEEDS))
    decision=dict(normal_relative_improvement=float(1-C[n]/P[n]),onset_delta=float(C[o]-P[o]),lower_normal_independent_fits=stable,
      primary_descriptive_screen=bool(C[n]<=.9*P[n] and C[o]<=P[o]+.01 and stable==3),promotion=False,novelty_established=False)
    write(DOCS/'MODEL_DECISION.json',decision)
    prev=json.loads((ROOT/'artifacts/argn_calibration_specificity_v1/PREVIOUS_HASHES.json').read_text())
    for p,h in prev.items():assert digest(p)==h,p
    native=[json.loads((OUT/f'backbones/{fs}/FIT.json').read_text()) for fs in SEEDS]
    assert len({x['weights_sha256'] for x in native})==3
    fits=dict(native_argn=3,category_amount_gap=9,onset_amount=3,count_initial=3,history_neural=3,phase_mixtures=12,time_calibration=6)
    write(DOCS/'FINAL_AUDIT.json',dict(utc=now(),fits=fits,full_generations=18,rows=total,prior_hashes=len(prev),independent_metric_checks=sum(x['checks'] for x in checks),max_error=max(x['max_error'] for x in checks),test_reads=0,push=False,model_replacement=False))
    lines=['# Fresh whole-pipeline validation','',m.to_markdown(),'','Independent fits: 3 complete pipelines; 2 draws per arm. Same existing development cohort; test untouched.',
      '',json.dumps(decision,ensure_ascii=False,indent=2),'','Costs remain explicit in FIT_MEANS / PAIRED_FIT_EFFECTS. This does not establish newest-generator superiority or novelty. CPAR and MeLD have separate completion records.']
    (DOCS/'INTERPRETATION.md').write_text('\n'.join(lines))
    print('WHOLE_VALIDATION_COMPLETE',total,decision,flush=True)
if __name__=='__main__':
    import sys
    check() if '--check' in sys.argv else main()
