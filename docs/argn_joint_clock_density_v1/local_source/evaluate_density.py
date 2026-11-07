"""Independent exact native-output metrics; no fitting or checkpoint selection."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from run_density import ROOT,OUT,DOCS,CFG,ARMS,BINS,verify,digest,write
sys.path[:0]=[str(ROOT/'artifacts/argn_clock_initial_coupling_v1/local_source'),
             str(ROOT/'artifacts/argn_clock_stratified_loss_v1/local_source'),
             str(ROOT/'artifacts/argn_clock_adaptive_v1/local_source')]
import report_initial as initial
import report_argn_time_density as audit
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate

BASE=['init_only','history_only','clock_joint','clock_prefix','current','init_handoff']

def locate(arm,fs,gs):
    if arm in ARMS:return OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet',DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    return initial.locate(arm,fs,gs)

def times(path):
    d=pd.read_parquet(path,columns=['entity_id','event_index','event_is_fraud','gap'])
    d=d.sort_values(['entity_id','event_index']).reset_index(drop=True)
    d['label']=pd.to_numeric(d.event_is_fraud).astype(int);d['gap']=pd.to_numeric(d.gap).astype(float)
    d.loc[d.event_index.eq(0),'gap']=np.nan
    d['clock']=d.groupby('entity_id').gap.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    return d

def distance(a,b,column='gap',relative=False,equal=False):
    if not len(a) or not len(b):return float('nan')
    x=a[column].to_numpy();y=b[column].to_numpy()
    if relative:x=x/a.clock.to_numpy();y=y/b.clock.to_numpy()
    def weights(d):return 1/d.groupby('entity_id').entity_id.transform('size').to_numpy() if equal else None
    return wasserstein_distance(np.log1p(x),np.log1p(y),u_weights=weights(a),v_weights=weights(b))

def horizons(arms):
    real=times(SOURCE/'prepared/validation.parquet');lengths=real.groupby('entity_id').size();rows=[];customers=[]
    for gs in CFG['generation_seeds']:
        fs=CFG['parent_seed'];base=times(locate('init_only',fs,gs)[0]);lens=base.groupby('entity_id').size()
        common=pd.concat([lengths,lens],axis=1).dropna().min(axis=1)
        r=real[real.event_index.lt(real.entity_id.map(common))]
        for arm in arms:
            raw=times(locate(arm,fs,gs)[0]);d=raw[raw.event_index.lt(raw.entity_id.map(common))]
            a=real[real.label.eq(0)&real.event_index.gt(0)&real.gap.notna()]
            b=raw[raw.label.eq(0)&raw.event_index.gt(0)&raw.gap.notna()]
            customers.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,equal_customer_normal_gap_w1=distance(a,b,equal=True)))
            for lo,hi in BINS+[(1024,None)]:
                high=np.inf if hi is None else hi
                a=r[r.event_index.ge(lo)&r.event_index.lt(high)&r.label.eq(0)&r.clock.gt(0)]
                b=d[d.event_index.ge(lo)&d.event_index.lt(high)&d.label.eq(0)&d.clock.gt(0)]
                rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=lo,end=-1 if hi is None else hi,
                    real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique(),
                    gap_w1=distance(a,b),personal_gap_w1=distance(a,b,relative=True),clock_w1=distance(a,b,column='clock')))
            if arm in ARMS:
                assert raw[['entity_id','event_index','label']].equals(base[['entity_id','event_index','label']])
                keep=raw.event_index.between(1,5)
                np.testing.assert_array_equal(raw.loc[keep,'gap'],base.loc[keep,'gap'])
    return pd.DataFrame(rows),pd.DataFrame(customers)

def main():
    verify();assert all((OUT/a/'NATIVE_COMPLETE.json').exists() for a in ARMS)
    with patch.object(audit,'CFG',{**CFG,'arms':[]}),patch.object(audit,'locate',locate):
        new,phases,personal,checks,sampling,_=audit.collect(ARMS)
    for name,frame in zip(['new_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],[new,phases,personal,checks,sampling]):
        frame.to_csv(DOCS/f'{name}.csv',index=False)
    old=pd.read_csv(ROOT/'artifacts/argn_clock_feedback_gain_v1/report/all_models.csv')
    old=old[old.arm.isin(BASE)&old.fit_seed.eq(CFG['parent_seed'])&old.generation_seed.isin(CFG['generation_seeds'])].copy()
    prior_checks={}
    for arm in BASE:
        for gs in CFG['generation_seeds']:
            p,_=locate(arm,CFG['parent_seed'],gs);record=json.loads((p.parent/f'GENERATION_{gs}.json').read_text())
            assert digest(p)==record['sha256'];prior_checks[str(p)]=record['sha256']
    for r in checks.itertuples():
        p,_=locate(r.arm,r.fit_seed,r.generation_seed)
        assert json.loads((p.parent/f'GENERATION_{r.generation_seed}.json').read_text())['head_sha256']==digest(OUT/r.arm/'time_head.pt')
    enrich=['early_time','time100_199','late1024_time','equal_customer_normal_gap_w1','mid_personal_gap_w1']
    data=pd.concat([old.drop(columns=enrich,errors='ignore'),new],ignore_index=True)
    h,c=horizons(BASE+ARMS);h.to_csv(DOCS/'horizon_diagnostic.csv',index=False);c.to_csv(DOCS/'customer_metrics.csv',index=False)
    keys=['arm','fit_seed','generation_seed']
    for lo,hi,name,col in [(6,50,'early_time','gap_w1'),(100,200,'time100_199','gap_w1'),(1024,-1,'late1024_time','gap_w1'),(100,200,'mid_personal_gap_w1','personal_gap_w1')]:
        data=data.merge(h[(h.start==lo)&(h.end==hi)][keys+[col]].rename(columns={col:name}),on=keys,validate='one_to_one')
    data=data.merge(c,on=keys,validate='one_to_one');data.to_csv(DOCS/'all_models.csv',index=False)
    fields=['normal_gap_seconds_log_w1','onset_personal_gap_w1','fraud_gap_seconds_log_w1','normal_stay_personal_gap_w1',
            'fraud_ratio_log_w1','normal_amount_log_w1','class_0_merchant_tv','median_unique_merchants_per_customer']+enrich
    means=data.groupby('arm')[fields].mean().reindex(BASE+ARMS);means.to_csv(DOCS/'SUMMARY_METRICS.csv')
    rows=[];base=means.loc['init_only']
    hm=h.groupby(['arm','start','end']).gap_w1.mean()
    for arm in ARMS:
        m=means.loc[arm];cost=[]
        for lo,hi in BINS+[(1024,None)]:
            hi=-1 if hi is None else hi
            if hm.loc[(arm,lo,hi)]>hm.loc[('init_only',lo,hi)]+.02:cost.append(f'position_{lo}_{hi}')
        for k in ['fraud_ratio_log_w1','normal_amount_log_w1','class_0_merchant_tv']:
            if not np.isfinite(m[k]) or m[k]>base[k]+.01:cost.append(k)
        if m.median_unique_merchants_per_customer<.95*base.median_unique_merchants_per_customer:cost.append('merchant_diversity')
        wins={}
        for metric in ['onset_personal_gap_w1','normal_gap_seconds_log_w1']:
            paired=data[data.arm.isin(['init_only',arm])].pivot(index='generation_seed',columns='arm',values=metric)
            wins[metric]=int((paired[arm]<paired.init_only).sum())
        onset=m.onset_personal_gap_w1<=.9*base.onset_personal_gap_w1 and m.normal_gap_seconds_log_w1<=base.normal_gap_seconds_log_w1+.01 and wins['onset_personal_gap_w1']==2
        normal=m.normal_gap_seconds_log_w1<=.9*base.normal_gap_seconds_log_w1 and m.onset_personal_gap_w1<=base.onset_personal_gap_w1+.01 and wins['normal_gap_seconds_log_w1']==2
        rows.append(dict(arm=arm,onset_target=bool(onset),normal_target=bool(normal),costs=','.join(cost),
            promising=bool((onset or normal) and not cost),selected_step=json.loads((OUT/arm/'FIT_COMPLETE.json').read_text())['selected_step']))
    screen=pd.DataFrame(rows);screen.to_csv(DOCS/'SCREEN.csv',index=False)
    report=['# 시간 출력 2×2 native 생성 결과','',
        '1개 동결 ARGN 부모, 각 조건1회 head 학습×2회 전체 생성. 독립 학습 반복에 의한 일반화 결론은 아님.',
        '체크포인트는 internal 조건부 시간 재생으로 선택. 표는 전체 native 자유 생성의 개발 평가. test 미사용.','',
        audit.markdown_table(means.round(6).reset_index()),'',audit.markdown_table(screen),'',
        '초기5개 간격과 라벨·길이 일치 검사 통과. 활성 clock_prefix, init_only 및 모든 이전 출력 보존. 자동 채택/push 없음.',
        'LBDTPP/ARGN/CPAR 전체 생성 비교는 별도 common-task 표에 기록하며 서로 다른 지원집합을 섞지 않음.']
    (DOCS/'RESULTS.md').write_text('\n'.join(report)+'\n')
    decision=dict(new_head_fits=4,new_native_argn_fits=0,full_generations=len(new),generated_rows=int(checks.rows.sum()),
        promising=screen.loc[screen.promising,'arm'].tolist(),active='clock_prefix',init_only_preserved=True,
        test_events_read=False,pushed=False,prior_output_hashes=prior_checks)
    write(DOCS/'MODEL_DECISION.json',decision);verify();write(DOCS/'COMPLETE.json',decision)
    print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
