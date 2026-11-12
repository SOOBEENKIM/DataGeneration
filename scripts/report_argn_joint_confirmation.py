"""Keep fresh-draw results separate from the initial candidate screen."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import run_argn_joint_preservation as study
from run_argn_joint_confirmation import ROOT,OUT,DOCS,CFG,verify,write,digest,foundation
from run_argn_state_first import SOURCE
from report_argn_joint_preservation import independent_numbers,KEYS
from report_argn_gap_episode import customer_summary
from diagnose_argn_joint_preservation import augment,w1


def main():
    m=verify();candidate=m['selected_arm'];real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rc=customer_summary(real)
    state=json.loads((SOURCE/'prepared/metric_state.json').read_text());r,_=augment(real,state)
    real_rate=pd.to_numeric(real.event_is_fraud).mean();data=[];checks=[];phase=[]
    for fs in CFG['fit_seeds']:
        for arm in ['current',candidate]:
            run=OUT/f'runs/{arm}_{fs}';assert (run/'COMPLETE.json').exists()
            metrics=pd.read_csv(DOCS/f'evaluation/metrics_{arm}_{fs}.csv')
            for gs in CFG['generation_seeds']:
                raw=pd.read_parquet(run/f'generated_validation_{gs}.parquet');g=json.loads((run/f'GENERATION_{gs}.json').read_text())
                assert digest(run/f'generated_validation_{gs}.parquet')==g['sha256']
                assert digest(run/'amount_head.pt')==g['head_sha256']==digest(foundation(fs)/'amount_head.pt')
                independent=independent_numbers(real,raw);row=metrics[metrics.generation_seed.eq(gs)].iloc[0].to_dict()
                for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']:
                    assert abs(row[k]-independent[k])<1e-12
                c=customer_summary(raw)
                row.update(customer_fraction_w1=wasserstein_distance(rc.share,c.share),
                    episodes_per_customer_w1=wasserstein_distance(rc.episodes,c.episodes),
                    repeated_episode_customers=int(c.episodes.gt(1).sum()),
                    fraud_customer_count_error=abs(c.frauds.gt(0).sum()-rc.frauds.gt(0).sum()),
                    all_fraud_customer_count_error=abs(c.frauds.eq(c.length).sum()-rc.frauds.eq(rc.length).sum()),
                    fraud_rate_absolute_error=abs(row['generated_fraud_rate']-real_rate),stage='fresh')
                data.append(row);checks.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,rows=len(raw),**independent))
                s,_=augment(raw,state)
                for part in ['onset','continuation','left_boundary']:
                    a=r[r.fraud.eq(1)&r.phase.eq(part)&r.event_index.ge(5)]
                    b=s[s.fraud.eq(1)&s.phase.eq(part)&s.event_index.ge(5)]
                    phase.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,phase=part,real_events=len(a),generated_events=len(b),
                        ratio_w1=w1(a.amount_history_ratio_raw,b.amount_history_ratio_raw),amount_w1=w1(a.amount_or_numeric_value,b.amount_or_numeric_value)))
    fresh=pd.DataFrame(data);fresh.to_csv(DOCS/'fresh_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(DOCS/'independent_checks.csv',index=False);pd.DataFrame(phase).to_csv(DOCS/'fresh_phase.csv',index=False)
    initial=pd.read_csv(study.DOCS/'all_metrics.csv');initial=initial[initial.arm.isin(['current',candidate])].copy();initial['stage']='initial'
    combined=pd.concat([initial,fresh],ignore_index=True);combined.to_csv(DOCS/'all_metrics.csv',index=False)
    results=[];deltas=[]
    for stage,frame in [('initial',initial),('fresh',fresh),('pooled',combined)]:
        for fs in CFG['fit_seeds']:
            mean=frame[frame.fit_seed.eq(fs)].groupby('arm').mean(numeric_only=True);b,s=mean.loc['current'],mean.loc[candidate]
            primary=s.fraud_ratio_log_w1<b.fraud_ratio_log_w1 and s.fraud_rate_absolute_error<b.fraud_rate_absolute_error
            costs=[]
            for k in ['fraud_gap_seconds_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','normal_amount_log_w1','class_0_merchant_tv']:
                if s[k]>b[k]+.05:costs.append(k)
            for k in ['fraud_customer_count_error','all_fraud_customer_count_error']:
                if s[k]>b[k]+5:costs.append(k)
            for k in ['median_unique_merchants_per_customer','seen_merchant_rate']:
                if s[k]<b[k]*.95:costs.append(k)
            results.append(dict(stage=stage,fit_seed=fs,both_target_errors_improved=bool(primary),large_costs=','.join(costs),screen_passed=bool(primary and not costs)))
            for k in KEYS:deltas.append(dict(stage=stage,fit_seed=fs,metric=k,current=b[k],candidate=s[k],difference=s[k]-b[k]))
    pd.DataFrame(results).to_csv(DOCS/'screen.csv',index=False);pd.DataFrame(deltas).to_csv(DOCS/'paired_effects.csv',index=False)
    lines=['# 새 생성 seed 확인','',f'고정 후보: {candidate}. 추가8생성 완료, 새 학습0.','',
        '| 범위 | 모델 | 개인 대비 금액 W1 | 사기율 % | 사기 간격 W1 | 사기 금액 W1 | 사기 경험 고객 | 정상 금액 W1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for stage,frame in [('initial',initial),('fresh',fresh),('pooled',combined)]:
        for arm,g in frame.groupby('arm'):
            n=g.mean(numeric_only=True)
            lines.append(f'| {stage} | {arm} | {n.fraud_ratio_log_w1:.4f} | {100*n.generated_fraud_rate:.4f} | {n.fraud_gap_seconds_log_w1:.4f} | {n.fraud_amount_log_w1:.4f} | {n.customers_with_fraud:.2f} | {n.normal_amount_log_w1:.4f} |')
    lines+=['','독립 생성 seed 재확인이며, 새로운 고객 test split이나 새 부모 학습 seed는 아니다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines));print(pd.DataFrame(results).to_string(index=False))
    write(DOCS/'COMPLETE.json',dict(new_generations=8,new_rows=sum(c['rows'] for c in checks),new_neural_fits=0,test_events_read=False,
                                  source_sha256=digest(__file__)))


if __name__=='__main__':main()
