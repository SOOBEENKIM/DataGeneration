"""Supplemental evaluation of completed CPAR/ARGN outputs while MeLD continues; not final comparison."""
import sys,json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/comparison_completion_v1';OLD=ROOT/'artifacts/argn_whole_validation_v1'
sys.path.insert(0,str(OLD/'local_source'))
from compare_cpar import metrics
from run_argn_state_first import SOURCE
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=json.loads((OUT/'EVALUATOR_MANIFEST.json').read_text())
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');cases=[];support_sets=[];status=[]
    for fs in [20261021,20261022,20261023]:
        original=ROOT/'artifacts/cpar_adequacy_v1' if fs==20261021 else OUT/f'cpar/{fs}'
        if not (original/'COMPLETE.json').exists():
            status.append(dict(model='CPAR_native',fit=fs,status='incomplete'));continue
        selected=json.loads((original/'SELECTION.json').read_text())['chosen_epochs'];directory=original/f'budget_{selected}'
        ids=json.loads((directory/'development_SUPPORT.json').read_text())['supported'];support_sets.append(set(ids))
        for gs in [20261031,20261032]:
            cases.append(('CPAR_native',fs,gs,directory/f'generated_development_{gs}.parquet'))
            length=OUT/f'cpar_length/{fs}'
            if (length/'COMPLETE.json').exists():cases.append(('CPAR_shared_synthetic_count',fs,gs,length/f'generated_development_{gs}.parquet'))
    common=set(real.entity_id).intersection(*support_sets);assert common
    for fs in [20261021,20261022,20261023]:
        for arm in ['history_only','history_position6','history_calibration_only']:
            for gs in [20261031,20261032]:cases.append((arm,fs,gs,OLD/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet'))
    for fs in [20261021,20261022]:
        directory=OUT/f'meld/fit_{fs}'
        if (directory/'COMPLETE.json').exists():
            for gs in [20261031,20261032]:cases.append(('MeLD_public_Sparkov',fs,gs,directory/f'generated_development_{gs}.parquet'))
        else:status.append(dict(model='MeLD_public_Sparkov',fit=fs,status='adequacy_block' if (directory/'ADEQUACY_BLOCK.json').exists() else 'incomplete'))
    rows=[];supports=[];hashes={}
    for model,fs,gs,path in cases:
        hashes[str(path)]=digest(path);raw=pd.read_parquet(path)
        if model.startswith('history_'):
            assert hashes[str(path)]==json.loads((path.parent/f'GENERATION_{gs}.json').read_text())['sha256']
        for cohort,ids in [('common_supported',common),('full147',set(real.entity_id))]:
            if cohort=='full147' and set(raw.entity_id)!=ids:continue
            a=real[real.entity_id.isin(ids)];b=raw[raw.entity_id.isin(ids)];assert set(b.entity_id)==ids
            v,s=metrics(a,b);info=dict(model=model,fit_seed=fs,generation_seed=gs,cohort=cohort,cohort_customers=len(ids));rows.append({**info,**v});supports.extend([{**info,**x} for x in s])
        print('COMPARATOR_EVALUATED',model,fs,gs,len(raw),flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'report/cpar_interim_20261001/COMPARATOR_DRAWS.csv',index=False)
    pd.DataFrame(supports).to_csv(OUT/'report/cpar_interim_20261001/COMPARATOR_SUPPORT.csv',index=False)
    keys=[k for k in rows[0] if k not in ['model','fit_seed','generation_seed','cohort','cohort_customers']]
    f=d.groupby(['model','cohort','fit_seed'])[keys].mean();f.to_csv(OUT/'report/cpar_interim_20261001/COMPARATOR_FIT_MEANS.csv');f.groupby(['model','cohort']).mean().to_csv(OUT/'report/cpar_interim_20261001/COMPARATOR_MEANS.csv')
    lengths=[]
    for model,fs,gs,path in cases:
        x=pd.read_parquet(path,columns=['entity_id']);n=x.groupby('entity_id').size()
        lengths.append(dict(model=model,fit_seed=fs,generation_seed=gs,customers=len(n),events=int(n.sum()),minimum=int(n.min()),median=float(n.median()),q90=float(n.quantile(.9)),maximum=int(n.max()),customers_ge1024=int((n>=1024).sum())))
    pd.DataFrame(lengths).to_csv(OUT/'report/cpar_interim_20261001/LENGTHS.csv',index=False)
    prior=json.loads((ROOT/'artifacts/argn_calibration_specificity_v1/PREVIOUS_HASHES.json').read_text())
    for p,h in prior.items():assert digest(p)==h,p
    for p,h in hashes.items():assert digest(p)==h,p
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    report=dict(interim=True,final_comparison_complete=False,complete=True,available_datasets=len(cases),common_customers=len(common),cohort_sizes_separate=True,status=status,prior_hashes=len(prior),test_reads=0,automatic_promotion=False,novelty_established=False)
    (OUT/'report/cpar_interim_20261001/COMPARISON_COMPLETE.json').write_text(json.dumps(report,indent=2))
    print('COMPARISON_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
