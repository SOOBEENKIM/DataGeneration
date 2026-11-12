"""Independent raw-output checks and predeclared preservation decisions."""
import argparse
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from run_argn_joint_preservation import ROOT,OUT,DOCS,CFG,PRIOR,foundation,verify,digest,write
from run_argn_state_first import SOURCE
from report_argn_gap_episode import customer_summary

KEYS=['fraud_gap_seconds_log_w1','fraud_amount_log_w1','fraud_ratio_log_w1',
      'generated_fraud_rate','fraud_rate_absolute_error','customers_with_fraud','all_fraud_customers',
      'normal_gap_seconds_log_w1','normal_amount_log_w1','class_0_merchant_tv',
      'class_1_gap_category_amount_history_tv','customer_length_log_w1','run_length_log_w1',
      'mean_run_length','termination_rate','unique_merchants','median_unique_merchants_per_customer',
      'seen_merchant_rate','customer_fraction_w1','episodes_per_customer_w1','repeated_episode_customers',
      'fraud_customer_count_error','all_fraud_customer_count_error']


def independent_numbers(real,raw):
    results={}
    for label,prefix in [(0,'normal'),(1,'fraud')]:
        for field,column,minimum in [('amount','amount_or_numeric_value',0),('gap_seconds','gap',1),('ratio','relative',5)]:
            arrays=[]
            for frame in [real,raw]:
                d=frame.sort_values(['entity_id','event_index'],kind='stable')
                values=pd.to_numeric(d[column]) if column!='relative' else pd.to_numeric(d.amount_or_numeric_value)/d.groupby('entity_id').amount_or_numeric_value.transform(lambda s:pd.to_numeric(s).shift().rolling(20,min_periods=5).median()).replace(0,np.nan)
                mask=pd.to_numeric(d.event_is_fraud).eq(label)&d.event_index.ge(minimum)&np.isfinite(values)&values.ge(0)
                arrays.append(np.log1p(values[mask]))
            results[f'{prefix}_{field}_log_w1']=wasserstein_distance(*arrays)
    return results


def main(partial=False):
    verify();real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rc=customer_summary(real)
    real_rate=float(pd.to_numeric(real.event_is_fraud).mean());frames=[];checks=[];extra=[];fits=[]
    for fs in CFG['fit_seeds']:
        b=pd.read_csv(ROOT/f'docs/argn_gap_episode_v1/evaluation/metrics_gap_and_transition_{fs}.csv');b['arm']='current';frames.append(b)
        for name in ['conditional','unconditional']:
            p=OUT/f'worker_{fs}/{name}/FIT_COMPLETE.json'
            if not p.exists():continue
            fit=json.loads(p.read_text());assert digest(p.parent/'count_head.pt')==fit['head_sha256'];fits.append(fit)
            write(DOCS/f'fit_{name}_{fs}.json',fit)
            pd.read_csv(p.parent/'learning_curve.csv').to_csv(DOCS/f'learning_{name}_{fs}.csv',index=False)
        for arm in CFG['generation_arms']:
            p=DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
            if p.exists():frames.append(pd.read_csv(p))
        for arm in ['current']+list(CFG['generation_arms']):
            run=foundation(fs) if arm=='current' else OUT/f'runs/{arm}_{fs}'
            if not (run/'COMPLETE.json').exists():continue
            assert digest(run/'amount_head.pt')==digest(foundation(fs)/'amount_head.pt')
            for gs in CFG['generation_seeds']:
                raw=pd.read_parquet(run/f'generated_validation_{gs}.parquet')
                g=json.loads((run/f'GENERATION_{gs}.json').read_text())
                assert digest(run/f'generated_validation_{gs}.parquet')==g['sha256']
                assert digest(run/'amount_head.pt')==g['head_sha256']
                if arm!='current' and CFG['generation_arms'][arm]['count']:
                    p=OUT/f"worker_{fs}/{CFG['generation_arms'][arm]['count']}/count_head.pt"
                    assert digest(p)==g['count_sha256']
                c=customer_summary(raw)
                assert len(c)==len(rc)==147
                check=dict(arm=arm,fit_seed=fs,generation_seed=gs,rows=len(raw),frozen_outputs_exact=True,
                           **independent_numbers(real,raw))
                if arm=='hazard_only':
                    native=pd.read_parquet(foundation(fs)/f'generated_validation_{gs}.parquet')
                    pd.testing.assert_series_equal(raw.groupby('entity_id').size(),native.groupby('entity_id').size())
                    check['native_length_equal']=True
                checks.append(check)
                extra.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,
                    customer_fraction_w1=wasserstein_distance(rc.share,c.share),
                    episodes_per_customer_w1=wasserstein_distance(rc.episodes,c.episodes),
                    repeated_episode_customers=int(c.episodes.gt(1).sum()),
                    fraud_customer_count_error=abs(c.frauds.gt(0).sum()-rc.frauds.gt(0).sum()),
                    all_fraud_customer_count_error=abs(c.frauds.eq(c.length).sum()-rc.frauds.eq(rc.length).sum())))
    data=pd.concat(frames,ignore_index=True)
    if not partial:assert len(checks)==20 and len(fits)==4 and len(data)==20
    independent=pd.DataFrame(checks)
    for _,c in independent.iterrows():
        r=data[(data.arm==c.arm)&(data.fit_seed==c.fit_seed)&(data.generation_seed==c.generation_seed)]
        if not len(r):continue
        for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']:
            assert abs(float(r.iloc[0][k])-c[k])<1e-12,(c.arm,k)
    data=data.merge(pd.DataFrame(extra),on=['arm','fit_seed','generation_seed'],how='left')
    data['fraud_rate_absolute_error']=(data.generated_fraud_rate-real_rate).abs()
    data.to_csv(DOCS/'all_metrics.csv',index=False);independent.to_csv(DOCS/'independent_checks.csv',index=False)
    means=data.groupby(['arm','fit_seed']).mean(numeric_only=True).reset_index();means.to_csv(DOCS/'fit_means.csv',index=False)
    ranges=data.groupby('arm')[KEYS].agg(['min','max','mean']);ranges.columns=['_'.join(k) for k in ranges.columns];ranges.to_csv(DOCS/'ranges.csv')
    gates=[];effects=[]
    for fs in CFG['fit_seeds']:
        cells=means[means.fit_seed.eq(fs)].set_index('arm');base=cells.loc['current']
        for arm in CFG['generation_arms']:
            if arm not in cells.index:continue
            s=cells.loc[arm];costs=[]
            for k in ['fraud_gap_seconds_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','normal_amount_log_w1','class_0_merchant_tv']:
                if s[k]>base[k]+.05:costs.append(k)
            for k in ['fraud_customer_count_error','all_fraud_customer_count_error']:
                if s[k]>base[k]+5:costs.append(k)
            for k in ['median_unique_merchants_per_customer','seen_merchant_rate']:
                if s[k]<base[k]*.95:costs.append(k)
            primary=bool(s.fraud_ratio_log_w1<base.fraud_ratio_log_w1 and s.fraud_rate_absolute_error<base.fraud_rate_absolute_error)
            gates.append(dict(fit_seed=fs,arm=arm,both_target_errors_improved=primary,large_costs=','.join(costs),
                passes_preregistered_screen=primary and not costs))
            for k in KEYS:effects.append(dict(fit_seed=fs,arm=arm,metric=k,current=base[k],proposed=s[k],difference=s[k]-base[k]))
    pd.DataFrame(gates).to_csv(DOCS/'preservation_screen.csv',index=False);pd.DataFrame(effects).to_csv(DOCS/'paired_effects.csv',index=False)
    lines=['# 길이·초기 상태·발생 위험 대조','',
        f'완료: count head {len(fits)}/4학습, 새 전체 거래열 {max(0,len(checks)-4)}/16생성.',
        '고정 현재 기준 대비2부모×2생성 평균. W1은 작을수록 좋다. 실제 사기율0.6438%, 사기 경험114명, 사기 전용12명.','',
        '| 변경 | 개인 대비 금액 W1 | 사기율 % | 사기 간격 W1 | 사기 금액 W1 | 사기 경험 고객 | 사기 전용 고객 | 정상 간격 W1 | 정상 금액 W1 | 길이 W1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    table=['fraud_ratio_log_w1','generated_fraud_rate','fraud_gap_seconds_log_w1','fraud_amount_log_w1',
           'customers_with_fraud','all_fraud_customers','normal_gap_seconds_log_w1','normal_amount_log_w1','customer_length_log_w1']
    for arm in ['current']+list(CFG['generation_arms']):
        if arm not in ranges.index:continue
        vals=[ranges.loc[arm,k+'_mean']*(100 if k=='generated_fraud_rate' else 1) for k in table]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','사전 판정은 preservation_screen.csv, 모든 비용은 paired_effects.csv 및 all_metrics.csv에 보존한다.',
        '두 부모 학습과 각 두 생성은 작은 개발 실험이다. 통계적 우월성이나 최신 모델 대비 기여를 확정하지 않는다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines))
    write(DOCS/'completion_evidence.json',dict(complete=not partial,new_count_fits=len(fits),new_full_generations=max(0,len(checks)-4),
        new_generated_rows=sum(c['rows'] for c in checks if c['arm']!='current'),test_events_read=False,
        source_sha256=digest(__file__),created_utc=datetime.now(timezone.utc).isoformat()))
    print('\n'.join(lines));print(pd.DataFrame(gates).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
