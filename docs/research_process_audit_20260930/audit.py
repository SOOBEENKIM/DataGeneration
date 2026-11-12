"""Read-only historical artifact and raw-output audit; no fitting/generation."""
import hashlib,json,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_argn_state_first import SOURCE
OUT=Path(__file__).parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n')
def data(p):
    d=pd.read_parquet(p,columns=['entity_id','event_index','event_is_fraud','gap','amount_or_numeric_value'])
    d=d.sort_values(['entity_id','event_index']).reset_index(drop=True)
    d['label']=pd.to_numeric(d.event_is_fraud).astype(int)
    d['gap']=pd.to_numeric(d.gap).astype(float).where(d.event_index>0)
    d['past_gap']=d.groupby('entity_id').gap.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    d['past_amount']=d.groupby('entity_id').amount_or_numeric_value.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    d['previous']=d.groupby('entity_id').label.shift()
    return d
def w(x,y):
    x=np.asarray(x);y=np.asarray(y);x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    return wasserstein_distance(np.log1p(x),np.log1p(y)) if len(x) and len(y) else np.nan
def numbers(real,d,common):
    out={}
    for label,name in [(0,'normal'),(1,'fraud')]:
        a=real[real.label==label];b=d[d.label==label]
        out[f'{name}_gap_seconds_log_w1']=w(a.gap,b.gap)
    a=real[(real.label==1)&real.past_amount.gt(0)];b=d[(d.label==1)&d.past_amount.gt(0)]
    out['fraud_ratio_log_w1']=w(a.amount_or_numeric_value/a.past_amount,b.amount_or_numeric_value/b.past_amount)
    a=real[(real.label==0)];b=d[(d.label==0)]
    out['normal_amount_log_w1']=w(a.amount_or_numeric_value,b.amount_or_numeric_value)
    for onset,name in [(True,'onset'),(False,'normal_stay')]:
        a=real[real.previous.eq(0)&real.label.eq(int(onset))&real.past_gap.gt(0)]
        b=d[d.previous.eq(0)&d.label.eq(int(onset))&d.past_gap.gt(0)]
        out[f'{name}_personal_gap_w1']=w(a.gap/a.past_gap,b.gap/b.past_gap)
    for lo,hi,name in [(6,50,'early_time'),(100,200,'time100_199'),(1024,np.inf,'late1024_time')]:
        def keep(f):return f[(f.label==0)&f.event_index.ge(lo)&f.event_index.lt(hi)&f.event_index.lt(f.entity_id.map(common))&f.past_gap.gt(0)]
        a=keep(real);b=keep(d);out[name]=w(a.gap,b.gap)
    out['fraud_rate']=float(d.label.mean());out['events']=len(d)
    return out

def main():
    previous=ROOT/'artifacts/argn_history_representation_diagnostic_v1'
    manifest=json.loads((previous/'MANIFEST.json').read_text())['sha256']
    checked={}
    for p,h in manifest.items():
        assert sha(p)==h,p;checked[p]=h
    old= pd.read_csv(ROOT/'artifacts/argn_clock_feedback_gain_v1/report/all_models.csv')
    latest=pd.read_csv(ROOT/'artifacts/argn_joint_clock_density_v1/report/all_models.csv')
    location={'clock_prefix':('argn_clock_location_v1','clock_prefix'),
        'current':('argn_clock_stratified_loss_v1','personal_six'),
        'init_only':('argn_clock_initial_coupling_v1','init_only'),
        'init_handoff':('argn_clock_initial_coupling_v1','init_handoff')}
    specs=[(a,fs,gs,group,folder,old) for a,(group,folder) in location.items()
        for fs in [20260930,20261001] for gs in [20261011,20261012,20261021,20261022]]
    specs += [(a,20260930,gs,'argn_joint_clock_density_v1',a,latest)
        for a in ['mean_marginal','mean_joint','mixture_marginal','mixture_joint'] for gs in [20261011,20261012]]
    real=data(SOURCE/'prepared/validation.parquet');rl=real.groupby('entity_id').size()
    rows=[];errors=[];labels={};preserved_head_checks=[]
    for arm,fs,gs,group,folder,table in specs:
        run=ROOT/f'artifacts/{group}/runs/{folder}_{fs}';p=run/f'generated_validation_{gs}.parquet'
        record=json.loads((run/f'GENERATION_{gs}.json').read_text())
        assert sha(p)==record['sha256'];checked[str(p)]=record['sha256']
        d=data(p);dl=d.groupby('entity_id').size();common=pd.concat([rl,dl],axis=1).min(axis=1)
        values=numbers(real,d,common)
        key=(fs,gs);frame=d[['entity_id','event_index','label']]
        if key in labels:assert labels[key].equals(frame),(arm,key)
        else:labels[key]=frame.copy()
        expected=table[(table.arm==arm)&(table.fit_seed==fs)&(table.generation_seed==gs)]
        assert len(expected)==1
        for metric,value in values.items():
            if metric not in expected:continue
            error=abs(value-float(expected[metric].iloc[0]));assert error<1e-10,(arm,fs,gs,metric,value,float(expected[metric].iloc[0]))
            errors.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,metric=metric,error=error))
        rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,**values))
        print('RAW_AUDITED',arm,fs,gs,len(d),flush=True)
    # Selected identity branches remain identical in all stored output fields.
    identities=0
    for gs in [20261011,20261012]:
        base=pd.read_parquet(ROOT/f'artifacts/argn_clock_initial_coupling_v1/runs/init_only_20260930/generated_validation_{gs}.parquet')
        for arm in ['mean_marginal','mean_joint','mixture_joint']:
            candidate=pd.read_parquet(ROOT/f'artifacts/argn_joint_clock_density_v1/runs/{arm}_20260930/generated_validation_{gs}.parquet')
            assert base.equals(candidate);identities+=1
    curves=[]
    for arm in ['mean_marginal','mean_joint','mixture_marginal','mixture_joint']:
        p=ROOT/f'artifacts/argn_joint_clock_density_v1/{arm}'
        fit=json.loads((p/'FIT_COMPLETE.json').read_text());assert sha(p/'time_head.pt')==fit['sha256']
        for row in json.loads((p/'LEARNING_CURVE.json').read_text()):
            curves.append(dict(arm=arm,selected_step=fit['selected_step'],step=row['step'],stage=row['stage'],score=row['score']))
    allrows=pd.DataFrame(rows);allrows.to_csv(OUT/'RAW_RECOMPUTATION.csv',index=False)
    pd.DataFrame(errors).to_csv(OUT/'METRIC_PARITY.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'CHECKPOINT_SELECTION.csv',index=False)
    # Seed identifiers are cohort metadata, never numeric outcomes to average.
    def cohort(frame,name):
        metrics=[c for c in frame if c not in ['arm','fit_seed','generation_seed']]
        table=frame.groupby('arm')[metrics].mean()
        table['parent_count']=frame.groupby('arm').fit_seed.nunique()
        table['draw_count']=frame.groupby('arm').size()
        table.to_csv(OUT/name)
    cohort(allrows[allrows.arm.isin(location)],'PRESERVED_2PARENTS_4DRAWS.csv')
    cohort(allrows[(allrows.fit_seed==20260930)&allrows.generation_seed.isin([20261011,20261012])],'LATEST_MATCHED_1PARENT_2DRAWS.csv')
    # A chronology of all relevant result/protocol records; no old files edited.
    ledger=[]
    groups=['sparkov_argn_control_v2','argn_state_first_v1','argn_label_first_control_v1','research_reaudit_20260927',
      'argn_amount_learning_v1','argn_category_amount_v1','argn_gap_episode_v1','argn_joint_preservation_v1',
      'argn_residual_v1','argn_boundary_gap_v1','argn_phase_gap_v1','argn_time_density_v1','argn_clock_regression_v1',
      'argn_clock_evolution_v1','argn_clock_location_v1','argn_clock_full_length_v1','argn_clock_adaptive_v1',
      'argn_clock_stratified_loss_v1','argn_early_history_diagnostic_v1','argn_clock_initial_coupling_v1',
      'argn_mid_history_diagnostic_v1','argn_clock_feedback_gain_v1','research_reassessment_20260929',
      'argn_joint_clock_density_v1','lbdtpp_sparkov_v1','argn_history_representation_diagnostic_v1']
    for i,group in enumerate(groups):
        for folder in [ROOT/'docs'/group,ROOT/'artifacts'/group/'report']:
            for name in ['INTERPRETATION.md','PROTOCOL.md','RESULTS.md','MODEL_DECISION.json','FINAL_AUDIT.json','AUDIT_REPORT.md','COMPARISON.md']:
                p=folder/name
                if p.exists():ledger.append(dict(stage=i,group=group,file=str(p),sha256=sha(p),bytes=p.stat().st_size))
    pd.DataFrame(ledger).to_csv(OUT/'STUDY_EVIDENCE_INDEX.csv',index=False)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    write(OUT/'AUDIT_COMPLETE.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        checked_frozen_hashes=len(checked),previous_manifest_entries=len(manifest),raw_outputs=len(rows),
        raw_rows=int(allrows.events.sum()),metrics_recomputed=len(errors),max_metric_difference=max(e['error'] for e in errors),
        identity_all_fields_comparisons=identities,study_groups=len(groups),indexed_documents=len(ledger),
        new_training=0,new_generation=0,test_events_read=False,push=False,head_unchanged=True))
    write(OUT/'VERIFIED_HASHES.json',checked)
    print((OUT/'AUDIT_COMPLETE.json').read_text(),flush=True)
if __name__=='__main__':main()
