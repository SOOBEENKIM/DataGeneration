"""Check targeted onset output, every preservation cost and exact fixed labels."""
import json
import numpy as np
import pandas as pd
import torch
import run_argn_joint_preservation as study
from run_argn_onset_output import ROOT,OUT,DOCS,CFG,verify,digest,write
from run_argn_state_first import SOURCE
from report_argn_joint_preservation import independent_numbers
from diagnose_argn_joint_preservation import augment,w1


def main():
    verify();real=pd.read_parquet(SOURCE/'prepared/validation.parquet');state=json.loads((SOURCE/'prepared/metric_state.json').read_text())
    r,_=augment(real,state);metrics=[];checks=[];phases=[];fits=[]
    for fs in CFG['fit_seeds']:
        for arm in ['fixed_joint']+CFG['arms']:
            run=study.OUT/f'runs/count_hazard_no_context_{fs}' if arm=='fixed_joint' else OUT/f'runs/{arm}_{fs}'
            source=study.DOCS/f'evaluation/metrics_count_hazard_no_context_{fs}.csv' if arm=='fixed_joint' else DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
            assert (run/'COMPLETE.json').exists();rows=pd.read_csv(source);rows['arm']=arm;metrics.append(rows)
            if arm!='fixed_joint':
                fit_dir=OUT/f'worker_{fs}/{arm}';fit=json.loads((fit_dir/'FIT_COMPLETE.json').read_text());fits.append(fit)
                assert fit['head_sha256']==digest(fit_dir/'onset_head.pt');write(DOCS/f'fit_{arm}_{fs}.json',fit)
                pd.read_csv(fit_dir/'learning_curve.csv').to_csv(DOCS/f'learning_{arm}_{fs}.csv',index=False)
            for gs in CFG['generation_seeds']:
                raw=pd.read_parquet(run/f'generated_validation_{gs}.parquet');baseline=pd.read_parquet(study.OUT/f'runs/count_hazard_no_context_{fs}/generated_validation_{gs}.parquet')
                g=json.loads((run/f'GENERATION_{gs}.json').read_text());assert digest(run/f'generated_validation_{gs}.parquet')==g['sha256']
                assert digest(run/'amount_head.pt')==g['head_sha256']==digest(study.foundation(fs)/'amount_head.pt')
                cols=['entity_id','event_index','event_is_fraud'];assert raw[cols].equals(baseline[cols])
                independent=independent_numbers(real,raw)
                row=rows[rows.generation_seed.eq(gs)].iloc[0]
                for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']:
                    assert abs(row[k]-independent[k])<1e-12
                checks.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,rows=len(raw),exact_labels_lengths=True,**independent))
                s,_=augment(raw,state)
                for part in ['onset','continuation','left_boundary']:
                    a=r[r.fraud.eq(1)&r.phase.eq(part)&r.event_index.ge(5)];b=s[s.fraud.eq(1)&s.phase.eq(part)&s.event_index.ge(5)]
                    phases.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,phase=part,real_events=len(a),generated_events=len(b),
                        ratio_w1=w1(a.amount_history_ratio_raw,b.amount_history_ratio_raw),amount_w1=w1(a.amount_or_numeric_value,b.amount_or_numeric_value)))
    data=pd.concat(metrics,ignore_index=True);data.to_csv(DOCS/'all_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(DOCS/'independent_checks.csv',index=False);p=pd.DataFrame(phases);p.to_csv(DOCS/'phase_metrics.csv',index=False)
    data.groupby(['arm','fit_seed']).mean(numeric_only=True).to_csv(DOCS/'fit_means.csv')
    p.groupby(['arm','fit_seed','phase']).mean(numeric_only=True).to_csv(DOCS/'phase_fit_means.csv')
    lines=['# 사기 시작 금액 출력 대조 결과','',f'{len(fits)}head 학습, 8새 전체 생성 완료. 길이·라벨열은 기준과 정확히 동일.','',
        '| 모델 | 전체 개인 대비 금액 W1 | 시작 거래 개인 대비 W1 | 시작 거래 금액 W1 | 사기 간격 W1 | 사기 금액 W1 | 정상 금액 W1 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['fixed_joint']+CFG['arms']:
        a=data[data.arm.eq(arm)].mean(numeric_only=True);b=p[p.arm.eq(arm)&p.phase.eq('onset')].mean(numeric_only=True)
        lines.append(f'| {arm} | {a.fraud_ratio_log_w1:.4f} | {b.ratio_w1:.4f} | {b.amount_w1:.4f} | {a.fraud_gap_seconds_log_w1:.4f} | {a.fraud_amount_log_w1:.4f} | {a.normal_amount_log_w1:.4f} |')
    lines+=['','사기 시작 시점만 실제 과거를 제공한 진단은 conditional_*.csv; seed별 모든 비용은 fit_means.csv 및 phase_fit_means.csv.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines))
    write(DOCS/'COMPLETE.json',dict(new_head_fits=len(fits),new_generations=8,new_rows=sum(c['rows'] for c in checks if c['arm']!='fixed_joint'),
        test_events_read=False,exact_labels_and_lengths=True,source_sha256=digest(__file__)))


if __name__=='__main__':main()
