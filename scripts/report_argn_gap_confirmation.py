"""Keep the additional CPU draws separate from the original GPU comparison."""
import json
import pandas as pd
from confirm_argn_gap_cpu import ROOT,OUT,DOCS,CFG,verify,write,digest
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate,summary,w1
from report_argn_joint_preservation import independent_numbers
from diagnose_argn_gap_cancellation import cancellation


def main():
    verify();real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rd=decorate(real)
    frames=[];phases=[];personal=[];checks=[];cdf=[];total=0
    for fs in CFG['fits']:
        for arm in CFG['arms']:
            run=OUT/f'runs/{arm}_{fs}';assert (run/'COMPLETE.json').exists()
            frame=pd.read_csv(DOCS/f'evaluation/metrics_{arm}_{fs}.csv');frames.append(frame)
            for gs in CFG['draws']:
                path=run/f'generated_validation_{gs}.parquet';e=json.loads((run/f'GENERATION_{gs}.json').read_text());assert digest(path)==e['sha256']
                raw=pd.read_parquet(path);syn=decorate(raw);total+=len(raw)
                cols=['entity_id','event_index','event_is_fraud'];base=pd.read_parquet(OUT/f'runs/frozen_{fs}/generated_validation_{gs}.parquet');assert raw[cols].equals(base[cols])
                info=dict(arm=arm,fit_seed=fs,generation_seed=gs);phases+=summary(rd,syn,info)
                nums=independent_numbers(real,raw);row=frame[frame.generation_seed.eq(gs)].iloc[0]
                for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']:assert abs(nums[k]-row[k])<1e-12
                checks.append(dict(**info,**nums));c,_=cancellation(rd,syn);cdf.append(dict(**info,**c))
                for phase in ['onset','return_normal','continuation','normal_stay']:
                    a=rd[rd.phase.eq(phase)&rd.past_gap.gt(0)];b=syn[syn.phase.eq(phase)&syn.past_gap.gt(0)]
                    personal.append(dict(**info,phase=phase,real_events=len(a),generated_events=len(b),relative_gap_w1=w1(a.gap/a.past_gap,b.gap/b.past_gap)))
    phase=pd.DataFrame(phases);phase.to_csv(DOCS/'phase_metrics.csv',index=False)
    targets=phase[phase.field.eq('gap')&phase.group.isin(['onset','return_normal'])].pivot(index=['arm','fit_seed','generation_seed'],columns='group',values='log_w1').reset_index()
    targets['boundary_gap_mean_w1']=targets[['onset','return_normal']].mean(axis=1)
    data=pd.concat(frames,ignore_index=True).merge(targets,on=['arm','fit_seed','generation_seed'],validate='one_to_one')
    data.to_csv(DOCS/'all_metrics.csv',index=False);pd.DataFrame(checks).to_csv(DOCS/'independent_checks.csv',index=False)
    personal=pd.DataFrame(personal);personal.to_csv(DOCS/'personal_gap.csv',index=False);pd.DataFrame(cdf).to_csv(DOCS/'cdf_cancellation.csv',index=False)
    means=data.groupby(['fit_seed','arm']).mean(numeric_only=True);means.to_csv(DOCS/'fit_means.csv');gates=[]
    for fs in CFG['fits']:
        b=means.loc[(fs,'frozen')]
        for arm in ['boundary_fit','phase_mixture']:
            a=means.loc[(fs,arm)];cost=[]
            for k,delta in [('fraud_ratio_log_w1',.02),('normal_amount_log_w1',.02),('normal_gap_seconds_log_w1',.02),('fraud_gap_seconds_log_w1',.05),('class_0_merchant_tv',.02)]:
                if a[k]>b[k]+delta:cost.append(k)
            if a.median_unique_merchants_per_customer<b.median_unique_merchants_per_customer*.95:cost.append('merchant_diversity')
            gates.append(dict(fit_seed=fs,arm=arm,costs=','.join(cost),passes_screen=bool(a.boundary_gap_mean_w1<b.boundary_gap_mean_w1 and not cost)))
    pd.DataFrame(gates).to_csv(DOCS/'screen.csv',index=False)
    keys=['onset','return_normal','boundary_gap_mean_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1','normal_amount_log_w1','normal_gap_seconds_log_w1']
    lines=['# CPU 추가 생성 확인','',
        '새 학습0. 같은 CPU에서 다시 생성한 frozen 기준과 비교한다.2부모×2추가 seed. 원래 GPU 결과와 평균을 합치지 않는다.','',
        '| 팔 | 시작 gap | 복귀 gap | 경계 평균 | 전체 사기 gap | 개인 대비 사기 금액 | 정상 금액 | 정상 gap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in CFG['arms']:
        m=data[data.arm.eq(arm)].mean(numeric_only=True);lines.append('| '+arm+' | '+' | '.join(f'{m[k]:.4f}' for k in keys)+' |')
    lines+=['','| 팔 | 개인 대비 시작 gap | 개인 대비 복귀 gap | 개인 대비 정상 gap |','|---|---:|---:|---:|']
    p=personal.groupby(['arm','phase']).relative_gap_w1.mean()
    for arm in CFG['arms']:lines.append('| '+arm+' | '+' | '.join(f'{p.loc[(arm,k)]:.4f}' for k in ['onset','return_normal','normal_stay'])+' |')
    lines+=['','모든 출력의 hash,8개 수정 거래열의 전체 label/길이,60개 W1을 독립 확인했다.','최종 test는 읽지 않았다. 이는 같은 개발 데이터에서의 추가 draw/장치 강건성 확인이다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines));print(pd.DataFrame(gates).to_string(index=False))
    write(DOCS/'COMPLETE.json',dict(new_full_generations=12,new_rows=total,new_training=0,
        test_events_read=False,matched_reference_device='cpu',source_sha256=digest(__file__)))


if __name__=='__main__':main()
