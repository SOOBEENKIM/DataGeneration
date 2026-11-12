"""Collect every registered diagnostic/control and independently check run counts."""
import json
import pathlib
import numpy as np
import pandas as pd
from run_argn_state_first import ROOT, digest, write

D=ROOT/'docs/research_reaudit_20260927'
A=ROOT/'artifacts/research_reaudit_20260927'


def main():
    fits=[]
    for fs in [20260930,20261001]:
        assert (A/f'COMPLETE_{fs}.json').exists()
        fits.append(pd.read_csv(D/f'fit_generalization_{fs}.csv'))
    fit=pd.concat(fits,ignore_index=True);fit.to_csv(D/'fit_generalization_all.csv',index=False)
    parents=pd.read_csv(ROOT/'docs/argn_label_first_control_v1/evaluation/all_draws.csv')
    parents=parents[parents.arm.eq('B_event_label_first')]
    parts=[parents];evidence=[]
    for mode in ['markov','duration']:
        for fs in [20260930,20261001]:
            run=A/'transition_reference'/f'{mode}_{fs}'
            assert (run/'COMPLETE.json').exists()
            table=pd.read_csv(D/'transition_reference/evaluation'/f'metrics_reference_{mode}_{fs}.csv')
            for gs in [20261011,20261012]:
                path=run/f'generated_validation_{gs}.parquet'
                g=json.loads((run/f'GENERATION_{gs}.json').read_text())
                assert digest(path)==g['sha256']
                raw=pd.read_parquet(path).sort_values(['entity_id','event_index'])
                fraud=0;lengths=[];p1=0;end=0;customers_f=0;all_f=0
                for _,group in raw.groupby('entity_id'):
                    y=group.event_is_fraud.astype(int).to_numpy();fraud+=y.sum()
                    customers_f+=int(y.any());all_f+=int(y.all())
                    start=None
                    for t,label in enumerate(y):
                        if label and start is None:start=t
                        if not label and start is not None:lengths.append(t-start);start=None
                        if t<len(y)-1 and label:p1+=1;end+=int(y[t+1]==0)
                    if start is not None:lengths.append(len(y)-start)
                row=table[table.generation_seed.eq(gs)].iloc[0]
                for key,value in dict(fraud_rate=fraud/len(raw),mean_run_length=np.mean(lengths),
                    termination_rate=end/p1,customers_with_fraud=customers_f,all_fraud_customers=all_f).items():
                    np.testing.assert_allclose(row[key],value,atol=1e-12,rtol=0)
                evidence.append(dict(mode=mode,fit_seed=fs,generation_seed=gs,rows=len(raw),
                    customers=raw.entity_id.nunique(),sha256=digest(path),independent_metrics_checked=5))
            parts.append(table)
    results=pd.concat(parts,ignore_index=True)
    results.to_csv(D/'transition_reference/all_draws_with_parents.csv',index=False)
    cols=['fraud_rate','mean_run_length','termination_rate','onset_rate','fraud_ratio_log_w1',
          'normal_amount_log_w1','normal_gap_seconds_log_w1','customers_with_fraud','all_fraud_customers',
          'class_1_gap_category_amount_history_tv','merchant_category_tv','customer_length_log_w1']
    results.groupby(['arm','fit_seed'])[cols].mean().reset_index().to_csv(D/'transition_reference/fit_means.csv',index=False)
    results.groupby('arm')[cols].agg(['min','max']).to_csv(D/'transition_reference/ranges.csv')
    real=pd.read_csv(ROOT/'docs/argn_state_first_v1/evaluation/metrics.csv')
    real=real[real.arm.eq('real_validation')].iloc[0]
    lines=['# 완료된 생성 대조', '', '실제 개발 고객 147명. 두 부모 checkpoint × 두 생성 seed의 모든 결과 범위.', '',
        '| 지표 | 실제 | 기존 거래별 손실·라벨 우선 ARGN | ARGN + Markov | ARGN + Duration |',
        '|---|---:|---:|---:|---:|']
    for key,label,scale in [('fraud_rate','사기 비율 (%)',100),('mean_run_length','평균 사기 구간 (건)',1),
        ('termination_rate','정상 복귀율 (%)',100),('customers_with_fraud','사기 경험 고객 (명)',1),
        ('all_fraud_customers','사기 전용 고객 (명)',1),('fraud_ratio_log_w1','개인 금액 비율 log-W1',1)]:
        vals=[f'{real[key]*scale:.4f}']
        for arm in ['B_event_label_first','reference_markov','reference_duration']:
            v=results.loc[results.arm.eq(arm),key]*scale;vals.append(f'{v.min():.4f}–{v.max():.4f}')
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines+=['','단순 전환 대조는 사기 비율·구간을 회복했지만 개인 금액과 고객 구성을 해결하지 못했다.',
        '새 방법의 우월성이나 통계적 유의성 판정이 아니다. 변경한 것은 label sampling 확률이며 신경망 재학습은 없다.']
    (D/'TRANSITION_REFERENCE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    weight_hashes={}
    original=json.loads((A/'DIAGNOSTIC_MANIFEST.json').read_text())
    for p,h in original['files'].items():
        if 'model-weights' in p or p.endswith('.pt') or p.endswith('.safetensors'):
            assert digest(p)==h;weight_hashes[p]=h
    assert len(weight_hashes)==10
    amount_checks=[]
    for fs in [20260930,20261001]:
        path=D/f'amount_probe/COMPLETE_{fs}.json'
        if path.exists():
            item=json.loads(path.read_text())
            for split,h in item['rows'].items():
                assert digest(A/f'amount_probe/amount_{fs}_{split}.parquet')==h
            amount_checks.append(item)
    if len(amount_checks)==2:
        assert amount_checks[0]['selection_sha256']==amount_checks[1]['selection_sha256']
    write(D/'completion_evidence.json',dict(saved_model_split_audits=10,splits_per_model=3,
        new_generations=len(evidence),generated_rows=sum(x['rows'] for x in evidence),
        generation_checks=evidence,original_weight_hashes=weight_hashes,test_events_read=False,
        neural_training=False,artifacts_local=str(A),conditional_amount_probes=amount_checks,
        first_digit_teacher_generation_equivalence_checks=sum(
            x['first_digit_teacher_generation_equivalence_checks'] for x in amount_checks)))
    print('\n'.join(lines));print('REAUDIT_COMPLETE',len(evidence),sum(x['rows'] for x in evidence))


if __name__=='__main__':main()
