"""Frozen metric reuse; internal previews cannot become a development comparison."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/meld_500_pilot_v1'
sys.path.insert(0,str(ROOT/'artifacts/argn_whole_validation_v1/local_source'))
from compare_cpar import metrics
import pandas as pd
import numpy as np
import hashlib

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')

def evaluate_files(dst,split,files,fs,step,kind):
    dst.mkdir(parents=True,exist_ok=True)
    from run_argn_state_first import SOURCE
    if split=='development':
        real=pd.read_parquet(SOURCE/'prepared/validation.parquet')
        shared=set(json.loads((ROOT/'artifacts/cpar_adequacy_v1/budget_128/development_SUPPORT.json').read_text())['supported'])
        cohorts=[('full147',set(real.entity_id)),('common143',shared)]
    else:
        assert split=='internal'
        # Same genuine internal set used by the existing CPAR/internal evaluator.
        from run_sparkov_argn_control import native_frame
        raw=pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/workspace/OriginalData/tgt-data/part.000000-val.parquet')
        real=native_frame(raw)
        cohorts=[('internal69',set(real.entity_id))]
    rows=[];support=[];hashes={}
    for gs,path in files:
        hashes[str(path)]=digest(path);g=pd.read_parquet(path)
        assert set(g.entity_id)==set(real.entity_id)
        assert np.isfinite(pd.to_numeric(g['amount_or_numeric_value'])).all()
        assert np.isfinite(pd.to_numeric(g.loc[g.event_index.gt(0),'gap'])).all()
        assert set(pd.to_numeric(g.event_is_fraud).unique()) <= {0,1}
        for name,ids in cohorts:
            a=real[real.entity_id.isin(ids)];b=g[g.entity_id.isin(ids)]
            values,counts=metrics(a,b)
            info=dict(model='MeLD_500_pilot',fit_seed=fs,generation_seed=gs,step=step,kind=kind,cohort=name,preliminary=True)
            rows.append({**info,**values});support.extend([{**info,**c} for c in counts])
    pd.DataFrame(rows).to_csv(dst/'METRICS.csv',index=False)
    pd.DataFrame(support).to_csv(dst/'SUPPORT.csv',index=False)
    write(dst/'EVALUATED.json',dict(complete=True,split=split,hashes=hashes,
       preliminary=True,adequate_latest_baseline=False,final_comparison_complete=False,test_reads=0))
    print('PILOT_EVALUATED',fs,step,kind,split,len(files),flush=True)

def summary():
    rows=[]
    for p in OUT.glob('runs/*/evaluation_development/METRICS.csv'):
        rows.append(pd.read_csv(p))
    if rows:
        data=pd.concat(rows,ignore_index=True);data.to_csv(OUT/'report/DEVELOPMENT_DRAWS.csv',index=False)
        cols=[c for c in data.select_dtypes('number').columns if c not in ['fit_seed','generation_seed','step']]
        fit=data.groupby(['model','cohort','fit_seed'])[cols].mean()
        fit.to_csv(OUT/'report/FIT_MEANS.csv');fit.groupby(['model','cohort']).mean().to_csv(OUT/'report/MEANS.csv')
        prior=pd.read_csv(ROOT/'artifacts/comparison_completion_v1/report/cpar_interim_20261001/COMPARATOR_DRAWS.csv')
        prior['cohort']=prior.cohort.replace({'common_supported':'common143'})
        prior['preliminary']=False  # This column qualifies pilot budget, not SOTA/generalization.
        allrows=pd.concat([prior,data],ignore_index=True)
        allrows.to_csv(OUT/'report/PRELIMINARY_COMPARISON_DRAWS.csv',index=False)
        values=[c for c in cols if c in prior.columns]
        allfit=allrows.groupby(['model','cohort','fit_seed'])[values].mean()
        allfit.groupby(['model','cohort']).mean().to_csv(OUT/'report/PRELIMINARY_COMPARISON_MEANS.csv')
        (OUT/'report/INTERPRETATION.md').write_text(
          '# MeLD 500-epoch preliminary result\n\n'
          'This is a user-requested 500-VAE-epoch / 2000-DiT-update pilot. '
          'It does not complete the adequate-budget MeLD comparison. '
          'VAE and generator checkpoints were selected on INTERNAL losses only.\n\n'
          'Read PRELIMINARY_COMPARISON_MEANS.csv together with per-fit and per-draw '
          'metrics and SUPPORT.csv. The 143-customer common cohort and full147 '
          'are separate. Missing fraud/onset/tail support is retained. '
          'Internal EMA/online previews are diagnostics and not added as fits.\n\n'
          'Neither a low score nor poor early synthesis establishes superiority '
          'to fully trained MeLD. Existing candidates and all original long-budget '
          'checkpoints are preserved; no promotion, push, or sealed-test reads.\n')
    write(OUT/'report/PILOT_EVALUATION.json',dict(completed_fits=len(rows),planned_fits=2,
      preliminary=True,final_comparison_complete=False,test_reads=0))

if __name__=='__main__':
    if '--request' in sys.argv:
        r=json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_text())
        evaluate_files(Path(r['dst']),r['split'],[(s,Path(p)) for s,p in r['files']],r['fs'],r['step'],r['kind'])
    else:summary()
