"""Raw-output cross-checks, boundary metrics, and registered preservation screen."""
import json
import numpy as np
import pandas as pd
from run_argn_boundary_gap import ROOT,OUT,DOCS,CFG,verify,write,digest
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate,summary,generation_path
from report_argn_joint_preservation import independent_numbers


def main():
    verify();real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rd=decorate(real)
    data=[];phase=[];checks=[];total=0
    base=pd.read_csv(ROOT/'docs/argn_joint_preservation_v1/end_to_end_pooled.csv')
    base=base[base.arm.eq('onset_fit')&base.generation_seed.isin(CFG['generation_seeds'])].copy();base['arm']='frozen';data.append(base)
    for fs in CFG['fit_seeds']:
        for arm in ['frozen']+CFG['arms']:
            run=OUT/f'runs/{arm}_{fs}'
            if arm!='frozen':
                assert (run/'COMPLETE.json').exists()
                data.append(pd.read_csv(DOCS/f'evaluation/metrics_{arm}_{fs}.csv'))
            for gs in CFG['generation_seeds']:
                path=generation_path('onset_fit',fs,gs) if arm=='frozen' else run/f'generated_validation_{gs}.parquet'
                raw=pd.read_parquet(path);info=dict(arm=arm,fit_seed=fs,generation_seed=gs)
                phase+=summary(rd,decorate(raw),info)
                checks.append(dict(**info,**independent_numbers(real,raw)))
                if arm!='frozen':
                    evidence=json.loads((run/f'GENERATION_{gs}.json').read_text());assert digest(path)==evidence['sha256'];total+=len(raw)
                    original=pd.read_parquet(generation_path('onset_fit',fs,gs));cols=['entity_id','event_index','event_is_fraud']
                    assert raw[cols].equals(original[cols])
    data=pd.concat(data,ignore_index=True);phase=pd.DataFrame(phase)
    for r in checks:
        x=data[data.arm.eq(r['arm'])&data.fit_seed.eq(r['fit_seed'])&data.generation_seed.eq(r['generation_seed'])].iloc[0]
        for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']:
            assert abs(x[k]-r[k])<1e-12,(r['arm'],k)
    p=phase[phase.field.eq('gap')&phase.group.isin(['onset','return_normal'])]
    targets=p.pivot(index=['arm','fit_seed','generation_seed'],columns='group',values='log_w1').reset_index()
    targets['boundary_gap_mean_w1']=targets[['onset','return_normal']].mean(axis=1)
    data=data.merge(targets,on=['arm','fit_seed','generation_seed'],validate='one_to_one')
    data.to_csv(DOCS/'all_metrics.csv',index=False);phase.to_csv(DOCS/'phase_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(DOCS/'independent_checks.csv',index=False)
    means=data.groupby(['fit_seed','arm']).mean(numeric_only=True);means.to_csv(DOCS/'fit_means.csv')
    gates=[]
    for fs in CFG['fit_seeds']:
        b=means.loc[(fs,'frozen')];c=means.loc[(fs,'all_label_fit')]
        for arm in CFG['arms']:
            a=means.loc[(fs,arm)];cost=[]
            for k,delta in [('fraud_ratio_log_w1',.02),('normal_amount_log_w1',.02),('normal_gap_seconds_log_w1',.02),
                            ('fraud_gap_seconds_log_w1',.05),('class_0_merchant_tv',.02)]:
                if a[k]>b[k]+delta:cost.append(k)
            if a.median_unique_merchants_per_customer<b.median_unique_merchants_per_customer*.95:cost.append('merchant_diversity')
            improves=a.boundary_gap_mean_w1<b.boundary_gap_mean_w1
            gates.append(dict(fit_seed=fs,arm=arm,boundary_improved=bool(improves),
                better_than_equal_capacity_control=bool(a.boundary_gap_mean_w1<c.boundary_gap_mean_w1) if arm!='all_label_fit' else None,
                costs=','.join(cost),passes_screen=bool(improves and not cost)))
    pd.DataFrame(gates).to_csv(DOCS/'screen.csv',index=False)
    keys=['onset','return_normal','boundary_gap_mean_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1','normal_amount_log_w1','normal_gap_seconds_log_w1']
    lines=['# 사기 구간 경계 간격 대조 결과','',
        '작은 gap head4학습, 자유 생성8회. 두 부모×두 생성 seed 평균. 최종 개선본이 frozen 기준이다.',
        '두 학습 팔은 같은 추가 용량과 routing을 사용하며 학습 대상만 다르다. W1은 작을수록 좋다.','',
        '| 팔 | 시작 gap | 복귀 gap | 경계 평균 | 전체 사기 gap | 개인 대비 사기 금액 | 정상 금액 | 정상 gap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['frozen']+CFG['arms']:
        m=data[data.arm.eq(arm)].mean(numeric_only=True);lines.append('| '+arm+' | '+' | '.join(f'{m[k]:.4f}' for k in keys)+' |')
    lines+=['','사전 screen은 screen.csv, 전체 비용은 all_metrics.csv에 보존한다. 모든 새 거래열의 label/길이는 대응 기준과 정확히 일치한다.',
            '최종 test 미사용. 작은 개발 대조이며 최신 생성기 대비 우월성·새 구조의 신규성은 아직 검증하지 않았다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines));print(pd.DataFrame(gates).to_string(index=False))
    write(DOCS/'COMPLETE.json',dict(fits=4,new_generations=8,new_rows=total,test_events_read=False,
        source_sha256=digest(__file__),exact_label_length_comparisons=8,independent_w1_comparisons=60))


if __name__=='__main__':main()
