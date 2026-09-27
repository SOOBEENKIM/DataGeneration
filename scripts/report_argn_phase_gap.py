"""Compare the simple temporal control with fixed and neural boundary outputs."""
import json
import numpy as np
import pandas as pd
from run_argn_phase_gap import ROOT,OUT,DOCS,CFG,verify,write,digest
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate,summary,generation_path,w1
from report_argn_joint_preservation import independent_numbers


def main():
    verify();real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rd=decorate(real)
    data=[pd.read_csv(ROOT/'docs/argn_boundary_gap_v1/all_metrics.csv')];phases=[pd.read_csv(ROOT/'docs/argn_boundary_gap_v1/phase_metrics.csv')]
    checks=[];personal=[];sampling=[];total=0
    for fs in CFG['fit_seeds']:
        run=OUT/f'runs/phase_mixture_{fs}';assert (run/'COMPLETE.json').exists()
        d=pd.read_csv(DOCS/f'evaluation/metrics_phase_mixture_{fs}.csv');rows=[]
        for gs in CFG['generation_seeds']:
            path=run/f'generated_validation_{gs}.parquet';raw=pd.read_parquet(path);e=json.loads((run/f'GENERATION_{gs}.json').read_text());assert digest(path)==e['sha256']
            baseline=pd.read_parquet(generation_path('onset_fit',fs,gs));cols=['entity_id','event_index','event_is_fraud'];assert raw[cols].equals(baseline[cols])
            nums=independent_numbers(real,raw);row=d[d.generation_seed.eq(gs)].iloc[0]
            for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']:assert abs(nums[k]-row[k])<1e-12
            info=dict(arm='phase_mixture',fit_seed=fs,generation_seed=gs);rows+=summary(rd,decorate(raw),info);checks.append(dict(**info,**nums));total+=len(raw)
            sampling.append(dict(**info,clipped_values=sum(x['clipped_values'] for x in e['sampling_audit']),sampled_values=sum(x['sampled_values'] for x in e['sampling_audit'])))
        rows=pd.DataFrame(rows);phases.append(rows)
        target=rows[rows.field.eq('gap')&rows.group.isin(['onset','return_normal'])].pivot(index=['arm','fit_seed','generation_seed'],columns='group',values='log_w1').reset_index()
        target['boundary_gap_mean_w1']=target[['onset','return_normal']].mean(axis=1);data.append(d.merge(target,on=['arm','fit_seed','generation_seed'],validate='one_to_one'))
    data=pd.concat(data,ignore_index=True);pd.concat(phases,ignore_index=True).to_csv(DOCS/'phase_metrics.csv',index=False)
    data.to_csv(DOCS/'all_metrics.csv',index=False);pd.DataFrame(checks).to_csv(DOCS/'independent_checks.csv',index=False);pd.DataFrame(sampling).to_csv(DOCS/'sampling_audit.csv',index=False)
    # A direct personal timing relationship, distinct from unconditional gap W1.
    for fs in CFG['fit_seeds']:
        for arm in ['frozen','all_label_fit','boundary_fit','phase_mixture']:
            for gs in CFG['generation_seeds']:
                if arm=='frozen':p=generation_path('onset_fit',fs,gs)
                elif arm=='phase_mixture':p=OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet'
                else:p=ROOT/f'artifacts/argn_boundary_gap_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet'
                s=decorate(pd.read_parquet(p));r=rd
                for phase in ['onset','return_normal','continuation','normal_stay']:
                    a=r[r.phase.eq(phase)&r.past_gap.gt(0)];b=s[s.phase.eq(phase)&s.past_gap.gt(0)]
                    personal.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,phase=phase,real_events=len(a),generated_events=len(b),
                        relative_gap_w1=w1(a.gap/a.past_gap,b.gap/b.past_gap)))
    pd.DataFrame(personal).to_csv(DOCS/'personal_gap.csv',index=False)
    means=data.groupby(['fit_seed','arm']).mean(numeric_only=True);means.to_csv(DOCS/'fit_means.csv');gates=[]
    for fs in CFG['fit_seeds']:
        b=means.loc[(fs,'frozen')];a=means.loc[(fs,'phase_mixture')];cost=[]
        for k,delta in [('fraud_ratio_log_w1',.02),('normal_amount_log_w1',.02),('normal_gap_seconds_log_w1',.02),('fraud_gap_seconds_log_w1',.05),('class_0_merchant_tv',.02)]:
            if a[k]>b[k]+delta:cost.append(k)
        if a.median_unique_merchants_per_customer<b.median_unique_merchants_per_customer*.95:cost.append('merchant_diversity')
        gates.append(dict(fit_seed=fs,arm='phase_mixture',boundary_improved=bool(a.boundary_gap_mean_w1<b.boundary_gap_mean_w1),costs=','.join(cost),passes_screen=bool(a.boundary_gap_mean_w1<b.boundary_gap_mean_w1 and not cost)))
    pd.DataFrame(gates).to_csv(DOCS/'screen.csv',index=False)
    keys=['onset','return_normal','boundary_gap_mean_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1','normal_amount_log_w1','normal_gap_seconds_log_w1']
    lines=['# 단순 조건부 시간 분포와 신경 경계 출력의 비교','',
        '모든 값은 같은2부모×2생성의 평균이다. phase_mixture는 고객 이력을 쓰지 않는 통계 대조다.','',
        '| 팔 | 시작 gap | 복귀 gap | 경계 평균 | 전체 사기 gap | 개인 대비 사기 금액 | 정상 금액 | 정상 gap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['frozen','all_label_fit','boundary_fit','phase_mixture']:
        m=data[data.arm.eq(arm)].mean(numeric_only=True);lines.append('| '+arm+' | '+' | '.join(f'{m[k]:.4f}' for k in keys)+' |')
    lines+=['','추가 신경망 학습0,4개 새 거래열. 현재/과거label 조건만의 GMM으로 각 gap을 직접 샘플링한다.',
            '모든 label열/길이는 기준과 일치한다. 금액/merchant/개인 gap 비용은 all_metrics.csv와 personal_gap.csv에 보고한다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines));print(pd.DataFrame(gates).to_string(index=False))
    write(DOCS/'COMPLETE.json',dict(new_full_generations=4,new_rows=total,new_neural_training=0,test_events_read=False,
        exact_label_length_comparisons=4,source_sha256=digest(__file__)))


if __name__=='__main__':main()
