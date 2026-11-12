"""Independent raw metrics, seed-level comparisons, and preservation audit."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from experiment import ROOT,OUT,DOCS,ARMS,SEEDS,PARENTS,DRAWS,verify,write,digest,head_path,now
from run_argn_state_first import SOURCE
from report_argn_joint_preservation import independent_numbers
from diagnose_argn_residual import decorate,generation_path
from report_argn_time_density import markdown_table

BINS=[(6,50,'early_time'),(50,100,'time50_99'),(100,200,'time100_199'),
      (200,500,'time200_499'),(500,1024,'time500_1023'),(1024,np.inf,'late1024_time')]
KEYS=['normal_gap_seconds_log_w1','onset_personal_gap_w1','fraud_gap_seconds_log_w1',
      'normal_stay_personal_gap_w1','fraud_ratio_log_w1','normal_amount_log_w1',
      'equal_customer_normal_gap_w1','class_0_merchant_tv','median_unique_merchants_per_customer']+[x[2] for x in BINS]

def wd(a,b,equal=False):
    x,y=a.gap.to_numpy(),b.gap.to_numpy()
    wa=1/a.groupby('entity_id').entity_id.transform('size').to_numpy() if equal else None
    wb=1/b.groupby('entity_id').entity_id.transform('size').to_numpy() if equal else None
    return float(wasserstein_distance(np.log1p(x),np.log1p(y),u_weights=wa,v_weights=wb))

def metrics(real,raw):
    d=decorate(raw);r=decorate(real);out=independent_numbers(real,raw);support=[]
    for phase in ['onset','normal_stay','return_normal','continuation','left_fraud']:
        a=r[r.phase.eq(phase)&r.past_gap.gt(0)];b=d[d.phase.eq(phase)&d.past_gap.gt(0)]
        value=float(wasserstein_distance(np.log1p(a.gap/a.past_gap),np.log1p(b.gap/b.past_gap))) if len(a) and len(b) else np.nan
        out[f'{phase}_personal_gap_w1']=value
        support.append(dict(group=phase,real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique()))
    a=r[r.label.eq(0)&r.event_index.gt(0)&r.gap.notna()];b=d[d.label.eq(0)&d.event_index.gt(0)&d.gap.notna()]
    out['equal_customer_normal_gap_w1']=wd(a,b,True)
    common=pd.concat([r.groupby('entity_id').size(),d.groupby('entity_id').size()],axis=1).dropna().min(axis=1)
    for lo,hi,name in BINS:
        def subset(f):return f[f.label.eq(0)&f.event_index.ge(lo)&f.event_index.lt(hi)&f.event_index.lt(f.entity_id.map(common))&f.past_gap.gt(0)]
        a,b=subset(r),subset(d);out[name]=wd(a,b)
        support.append(dict(group=name,real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique()))
    return out,support

def check():
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet')
    path=ROOT/f'artifacts/argn_clock_initial_coupling_v1/runs/init_only_{PARENTS[0]}/generated_validation_{DRAWS[0]}.parquet'
    values,_=metrics(real,pd.read_parquet(path))
    old=pd.read_csv(ROOT/'artifacts/argn_joint_clock_density_v1/report/all_models.csv')
    old=old[(old.arm=='init_only')&(old.fit_seed==PARENTS[0])&(old.generation_seed==DRAWS[0])].iloc[0]
    errors={k:abs(float(v)-float(old[k])) for k,v in values.items() if k in old}
    assert errors and max(errors.values())<1e-10,errors
    write(OUT/'EVALUATION_READY.json',dict(source_sha256=digest(__file__),registered_utc=now(),
        prior_frozen_output_checks=errors,new_output_reads=0,test_events_read=False))
    print('EVALUATION_READY',len(errors),max(errors.values()),flush=True)

def main():
    verify();assert (OUT/'EXECUTION_COMPLETE.json').exists()
    registration=json.loads((OUT/'EVALUATION_READY.json').read_text());assert digest(__file__)==registration['source_sha256']
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rows=[];checks=[];supports=[];total=0
    for rs in SEEDS:
        for parent in PARENTS:
            for gs in DRAWS:
                baseline=pd.read_parquet(generation_path('onset_fit',parent,gs));cols=['entity_id','event_index','event_is_fraud']
                for arm in ARMS:
                    run=OUT/f'runs/{arm}_{rs}_{parent}';assert (run/'COMPLETE.json').exists()
                    path=run/f'generated_validation_{gs}.parquet';record=json.loads((run/f'GENERATION_{gs}.json').read_text())
                    assert digest(path)==record['sha256'];assert digest(head_path(arm,rs,parent))==record['head_sha256']
                    raw=pd.read_parquet(path);assert raw[cols].equals(baseline[cols]);total+=len(raw)
                    prior=pd.read_csv(DOCS/f'evaluation/metrics_{arm}_{rs}_{parent}.csv')
                    row=prior[prior.generation_seed==gs].iloc[0].to_dict()
                    values,support=metrics(real,raw)
                    common=[k for k in values if k in row];errors={k:abs(float(values[k])-float(row[k])) for k in common}
                    assert max(errors.values())<1e-10,errors
                    info=dict(arm=arm,time_fit_seed=rs,parent=parent,generation_seed=gs)
                    row.update(info);row.update(values);row['events']=len(raw)
                    rows.append(row);supports.extend([{**info,**s} for s in support])
                    checks.append(dict(**info,rows=len(raw),sha256=record['sha256'],metric_checks=len(errors),max_metric_error=max(errors.values()),labels_lengths_exact=True))
                    print('AUDITED',arm,rs,parent,gs,len(raw),flush=True)
    d=pd.DataFrame(rows);assert len(d)==84
    d.to_csv(DOCS/'ALL_DRAWS.csv',index=False)
    pd.DataFrame(checks).to_csv(DOCS/'INDEPENDENT_CHECKS.csv',index=False)
    pd.DataFrame(supports).to_csv(DOCS/'SUPPORT.csv',index=False)
    d.groupby(['arm','time_fit_seed','parent'])[KEYS].mean().to_csv(DOCS/'PARENT_FIT_MEANS.csv')
    fm=d.groupby(['arm','time_fit_seed'])[KEYS].mean();fm.to_csv(DOCS/'FIT_MEANS.csv')
    grand=d.groupby('arm')[KEYS].mean().reindex(ARMS);grand.to_csv(DOCS/'MEANS.csv')
    effects=[];costs=[];init=d[d.arm=='init_only'].set_index(['time_fit_seed','parent','generation_seed'])
    for arm in ARMS[1:]:
        alt=d[d.arm==arm].set_index(['time_fit_seed','parent','generation_seed'])
        delta=init[KEYS]-alt[KEYS]
        for rs in SEEDS:
            for key in KEYS:
                vals=delta.loc[rs,key]
                effects.append(dict(comparator=arm,time_fit_seed=rs,metric=key,delta_init_minus_comparator=float(vals.mean()),
                    lower_draws=int((vals<0).sum()),draws=len(vals),lower_parent_means=int((vals.groupby(level=0).mean()<0).sum())))
        a,b=grand.loc['init_only'],grand.loc[arm];limitations=[]
        for _,_,key in BINS:
            if a[key]>b[key]+.02:limitations.append(key)
        for key in ['fraud_ratio_log_w1','normal_amount_log_w1','class_0_merchant_tv']:
            if not np.isfinite(a[key]) or a[key]>b[key]+.01:limitations.append(key)
        if a.median_unique_merchants_per_customer<.95*b.median_unique_merchants_per_customer:limitations.append('merchant_diversity')
        n,o='normal_gap_seconds_log_w1','onset_personal_gap_w1'
        normal=bool(a[n]<=.9*b[n] and a[o]<=b[o]+.01)
        onset=bool(a[o]<=.9*b[o] and a[n]<=b[n]+.01)
        costs.append(dict(comparator=arm,normal_target=normal,onset_target=onset,costs=','.join(limitations),
            lower_normal_fits=int((fm.loc['init_only',n]<fm.loc[arm,n]).sum()),
            lower_onset_fits=int((fm.loc['init_only',o]<fm.loc[arm,o]).sum()),
            descriptive_screen=bool((normal or onset) and not limitations)))
    pd.DataFrame(effects).to_csv(DOCS/'PAIRED_EFFECTS.csv',index=False)
    pd.DataFrame(costs).to_csv(DOCS/'TRADEOFF_SCREEN.csv',index=False)
    fits=[]
    for rs in SEEDS:
        for folder in [OUT/f'fits/statistics_{rs}']+[OUT/f'fits/history_{p}_{rs}' for p in PARENTS]:
            c=json.loads((folder/'CALIBRATION.json').read_text())
            fit=dict(run=folder.name,evaluations=c['evaluations'],calibration_success=c['optimizer_success'],
                initial_objective=c['initial']['objective'],selected_objective=c['selected']['objective'])
            if (folder/'FIT_COMPLETE.json').exists():
                n=json.loads((folder/'FIT_COMPLETE.json').read_text());fit.update(neural_selected_step=n['selected_step'],neural_steps=n['steps'],neural_initial_nll=n['initial_nll'],neural_best_nll=n['best_nll'])
            fits.append(fit)
    pd.DataFrame(fits).to_csv(DOCS/'FIT_AUDIT.csv',index=False)
    prior_hashes=json.loads((ROOT/'artifacts/research_process_audit_20260930/VERIFIED_HASHES.json').read_text())
    for p,h in prior_hashes.items():assert digest(p)==h,p
    verify()
    audit=dict(completed_utc=now(),full_generations=84,rows=total,independent_metric_checks=sum(x['metric_checks'] for x in checks),
        max_metric_error=max(x['max_metric_error'] for x in checks),preserved_previous_hashes=len(prior_hashes),
        time_fit_seeds=SEEDS,statistical_mixture_fits=27,neural_history_fits=6,calibration_fits=9,
        new_native_argn_fits=0,new_cpar_fits=0,latest_generator_fits=0,test_events_read=False,push=False,
        model_replacement=False)
    write(DOCS/'FINAL_AUDIT.json',audit)
    write(DOCS/'MODEL_DECISION.json',dict(active='clock_prefix',preserved_candidate='init_only',
        automatic_promotion=False,comparisons=costs,independent_native_argn_refits=False,
        same_task_whole_generator_adequacy_complete=False))
    display=['normal_gap_seconds_log_w1','onset_personal_gap_w1','early_time','time100_199','late1024_time','fraud_ratio_log_w1']
    lines=['# init_only 시간 모듈 독립 재학습 검증','',
        '학습 난수3개 × 기존 부모2개 × 생성 난수2개. 각 팔12생성. 새 ARGN 본체 재학습은 아니다.',
        '모든 결과는 반복 사용한 development이며 최종 test는 읽지 않았다. 낮을수록 좋은 W1.', '',
        markdown_table(grand[display].round(6).reset_index()),'',markdown_table(pd.DataFrame(costs)), '',
        '강한 이력 대조 history_init_cal에도 같은 초기 간격 분포와 같은6계수/목적/예산 보정을 제공했다.',
        '각 시간 fit의 차이는 PAIRED_EFFECTS.csv, 부모별 값은 PARENT_FIT_MEANS.csv에 기록했다.',
        '희귀 사건의 표본 수는 SUPPORT.csv, 수렴/선택 정보는 FIT_AUDIT.csv에 보존했다.',
        '이 결과를 최신 전체 생성기 대비 우월성이나 새로운 방법의 기여 확정으로 해석하지 않는다.',
        '기존 모델과 실패 결과 모두 보존; 자동 교체 및 push 없음.']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');a=parser.parse_args()
    check() if a.check_only else main()
