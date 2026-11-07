"""Common time/mark task, all-customer and CPAR-common-support tables."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'artifacts/argn_joint_clock_density_v1/local_source')]
from evaluate_density import locate,ARMS
from run_argn_state_first import SOURCE
from run_argn_amount_learning import digest,write
OUT=ROOT/'artifacts/lbdtpp_sparkov_v1';DOCS=OUT/'report'
OLD=ROOT.parent/'research-argn-fraud-audit/artifacts/sparkov_argn_control_v2'

def features(path,ids=None):
    d=pd.read_parquet(path,columns=['entity_id','event_index','event_is_fraud','gap','category'])
    if ids is not None:d=d[d.entity_id.isin(ids)]
    d=d.sort_values(['entity_id','event_index']).reset_index(drop=True)
    assert not d.duplicated(['entity_id','event_index']).any()
    d['label']=pd.to_numeric(d.event_is_fraud).astype(int);assert set(d.label.unique())<={0,1}
    d['gap']=pd.to_numeric(d.gap).astype(float);d.loc[d.event_index.eq(0),'gap']=np.nan
    d['previous']=d.groupby('entity_id').label.shift().fillna(-1).astype(int)
    d['clock']=d.groupby('entity_id').gap.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    return d

def wd(a,b,relative=False,equal=False):
    if not len(a) or not len(b):return np.nan
    x=a.gap.to_numpy();y=b.gap.to_numpy()
    if relative:x=x/a.clock.to_numpy();y=y/b.clock.to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all() or min(x.min(),y.min())<0:return np.nan
    weights=lambda d:1/d.groupby('entity_id').entity_id.transform('size').to_numpy() if equal else None
    return wasserstein_distance(np.log1p(x),np.log1p(y),u_weights=weights(a),v_weights=weights(b))

def score(real,d):
    post=d.event_index.gt(0);bad=~np.isfinite(d.gap)|d.gap.lt(0)
    row=dict(customers=d.entity_id.nunique(),events=len(d),fraud_rate=d.label.mean(),
             fraud_customers=int(d.groupby('entity_id').label.max().sum()),
             invalid_intervals=int((post&bad).sum()),zero_intervals=int((post&d.gap.eq(0)).sum()),
             over_argn_support_intervals=int((post&d.gap.gt(1045691)).sum()))
    for label,name in [(0,'normal'),(1,'fraud')]:
        a=real[real.label.eq(label)&real.event_index.gt(0)];b=d[d.label.eq(label)&post]
        row[name+'_gap_w1']=wd(a,b);row[name+'_customer_equal_gap_w1']=wd(a,b,equal=True)
        x=a.category.value_counts(normalize=True);y=b.category.value_counts(normalize=True)
        cats=x.index.union(y.index);row[name+'_category_tv']=float(.5*(x.reindex(cats,fill_value=0)-y.reindex(cats,fill_value=0)).abs().sum()) if len(b) else np.nan
    for previous,label,name in [(0,1,'onset'),(0,0,'normal_stay'),(1,1,'continuation'),(1,0,'return_normal')]:
        a=real[real.previous.eq(previous)&real.label.eq(label)&real.clock.gt(0)]
        b=d[d.previous.eq(previous)&d.label.eq(label)&d.clock.gt(0)]
        row[name+'_personal_gap_w1']=wd(a,b,relative=True)
        row[name+'_real_events']=len(a);row[name+'_generated_events']=len(b)
    for lo,hi in [(6,50),(100,200),(1024,np.inf)]:
        a=real[real.event_index.ge(lo)&real.event_index.lt(hi)&real.label.eq(0)&real.clock.gt(0)]
        b=d[d.event_index.ge(lo)&d.event_index.lt(hi)&d.label.eq(0)&d.clock.gt(0)]
        row[f'position_{lo}_normal_w1']=wd(a,b)
        row[f'position_{lo}_customers']=b.entity_id.nunique()
    return row

def main():
    assert all((OUT/a/'COMPLETE.json').exists() for a in ['official_scale','median_scale'])
    assert (ROOT/'artifacts/argn_joint_clock_density_v1/report/COMPLETE.json').exists()
    supported=set(pd.read_parquet(OLD/'prepared/supported_validation_context.parquet').customer_id)
    rows=[];provenance={};real_path=SOURCE/'prepared/validation.parquet'
    for gs in [20261011,20261012]:
        models={a:locate(a,20260930,gs)[0] for a in ['init_only','history_only','clock_prefix','current']+ARMS}
        models.update({f'LBDTPP_{a}':OUT/a/f'generated_validation_{gs}.parquet' for a in ['official_scale','median_scale']})
        for fs in [20260928,20260929]:
            for arm in ['native','numeric_digit_relaxed','order_category_numeric_digit_relaxed']:
                models[f'ARGN_{arm}_parent{fs}']=OLD/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet'
        models['CPAR_128_CPU']=OLD/f'runs/cpar_20260928/generated_supported_validation_{gs}.parquet'
        for cohort,ids in [('all147',None),('common143',supported)]:
            real=features(real_path,ids)
            for arm,path in models.items():
                if cohort=='all147' and arm.startswith('CPAR'):continue
                assert path.exists(),path
                d=features(path,ids);assert set(d.entity_id)==set(real.entity_id),(arm,cohort)
                row=dict(model=arm,cohort=cohort,generation_seed=gs,**score(real,d));rows.append(row)
                provenance[str(path)]=digest(path)
    data=pd.DataFrame(rows);data.to_csv(DOCS/'COMMON_TASK_ALL_DRAWS.csv',index=False)
    keys=['normal_gap_w1','onset_personal_gap_w1','fraud_gap_w1','normal_stay_personal_gap_w1',
          'fraud_rate','fraud_customers','normal_category_tv','fraud_category_tv','events',
          'position_6_normal_w1','position_100_normal_w1','position_1024_normal_w1',
          'invalid_intervals','over_argn_support_intervals','onset_generated_events']
    summary=data.groupby(['cohort','model'])[keys].mean();summary.to_csv(DOCS/'COMMON_TASK_SUMMARY.csv')
    # Independent agreement with the new full native evaluator on comparable
    # definitions. The position support is explicitly different (all rows vs
    # matched positions), so do not assert or combine those columns.
    native=pd.read_csv(ROOT/'artifacts/argn_joint_clock_density_v1/report/all_models.csv')
    for r in data[(data.cohort=='all147')&data.model.isin(['init_only','history_only','clock_prefix','current']+ARMS)].itertuples():
        old=native[native.arm.eq(r.model)&native.generation_seed.eq(r.generation_seed)].iloc[0]
        for a,b in [('normal_gap_w1','normal_gap_seconds_log_w1'),('fraud_gap_w1','fraud_gap_seconds_log_w1'),('onset_personal_gap_w1','onset_personal_gap_w1')]:
            np.testing.assert_allclose(getattr(r,a),old[b],atol=1e-12,rtol=0)
    from report_argn_time_density import markdown_table
    text=['# 현재 생성기와 시간·상태 공통 과제 비교','',
      'LBDTPP 새 학습2설정×2draw; ARGN/CPAR는 명시된 이전 체크포인트·출력을 재평가. 최종 test 미사용.',
      'LBDTPP는 init_only가 생성했던 길이 스케줄을 공통 조건으로 받고 시간·업종·사기 라벨을 생성한다. 금액/가맹점ID/정적 고객 조건은 모델링하지 않는다.',
      'ARGN/CPAR는 자체 길이 생성 결과다. 전체 자유 길이·전체필드 동등 과제 우위로 해석하지 않는다. CPAR는 이전128update 단일 fit이며 최적 튜닝 증거가 아니다.',
      'common143에서는 모든 모델에서 동일4고객을 제외했다. all147과 평균을 섞지 않는다. NaN은 해당 생성 사건이 없어 관계오차를 정의할 수 없다는 뜻이며 좋은 점수가 아니다.',
      '원형 ARGN들은2개의 부모를 각각 표기했다. 행들을 독립 head 재학습 반복처럼 평균하지 않는다.',
      '위치 지표는 각 모델의 해당 위치 모든 지원 고객을 사용한다. 새 head 제거 실험의 matched-position 표와 정의가 다르다.','']
    for cohort in ['all147','common143']:
        text += [cohort,'',markdown_table(summary.loc[cohort].round(6).reset_index()),'']
    (DOCS/'COMPARISON.md').write_text('\n'.join(text))
    write(DOCS/'COMPARISON_AUDIT.json',dict(raw_output_hashes=provenance,new_lbdtpp_fits=2,new_native_argn_fits=0,
        new_cpar_fits=0,test_events_read=False,independent_native_metric_parity=True,automatic_promotion=False))
    print('COMMON_TASK_COMPLETE',len(data),flush=True)

if __name__=='__main__':main()
