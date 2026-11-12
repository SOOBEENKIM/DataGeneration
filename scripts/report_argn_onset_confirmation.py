"""Final fresh-draw and pooled accounting without replacing initial results."""
import json
import pandas as pd
import run_argn_joint_preservation as study
import run_argn_onset_output as onset
from confirm_argn_onset_output import OUT,DOCS,CFG,verify,write,digest
from run_argn_state_first import SOURCE
from report_argn_joint_preservation import independent_numbers
from diagnose_argn_joint_preservation import augment,w1


def main():
    verify();raw_real=pd.read_parquet(SOURCE/'prepared/validation.parquet');state=json.loads((SOURCE/'prepared/metric_state.json').read_text());real,_=augment(raw_real,state)
    fresh=[];checks=[];phases=[]
    for fs in CFG['fit_seeds']:
        for arm in ['fixed_joint','onset_fit']:
            root=study.OUT/'confirmation' if arm=='fixed_joint' else OUT
            docs=study.DOCS/'confirmation' if arm=='fixed_joint' else DOCS
            name='count_hazard_no_context' if arm=='fixed_joint' else arm
            run=root/f'runs/{name}_{fs}';assert (run/'COMPLETE.json').exists()
            data=pd.read_csv(docs/f'evaluation/metrics_{name}_{fs}.csv');data['arm']=arm;data['stage']='fresh';fresh.append(data)
            for gs in CFG['generation_seeds']:
                raw=pd.read_parquet(run/f'generated_validation_{gs}.parquet');g=json.loads((run/f'GENERATION_{gs}.json').read_text())
                assert digest(run/f'generated_validation_{gs}.parquet')==g['sha256']
                numbers=independent_numbers(raw_real,raw);row=data[data.generation_seed.eq(gs)].iloc[0]
                for k in ['fraud_ratio_log_w1','fraud_amount_log_w1','fraud_gap_seconds_log_w1','normal_amount_log_w1','normal_gap_seconds_log_w1']:assert abs(row[k]-numbers[k])<1e-12
                checks.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,rows=len(raw),**numbers))
                syn,_=augment(raw,state)
                for part in ['onset','continuation','left_boundary']:
                    a=real[real.fraud.eq(1)&real.phase.eq(part)&real.event_index.ge(5)];b=syn[syn.fraud.eq(1)&syn.phase.eq(part)&syn.event_index.ge(5)]
                    phases.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,phase=part,ratio_w1=w1(a.amount_history_ratio_raw,b.amount_history_ratio_raw),amount_w1=w1(a.amount_or_numeric_value,b.amount_or_numeric_value)))
    fresh=pd.concat(fresh,ignore_index=True);fresh.to_csv(DOCS/'fresh_metrics.csv',index=False);pd.DataFrame(phases).to_csv(DOCS/'fresh_phase.csv',index=False)
    pd.DataFrame(checks).to_csv(DOCS/'independent_checks.csv',index=False)
    initial=pd.read_csv(onset.DOCS/'all_metrics.csv');initial=initial[initial.arm.isin(['fixed_joint','onset_fit'])].copy();initial['stage']='initial'
    data=pd.concat([initial,fresh],ignore_index=True);data.to_csv(DOCS/'all_metrics.csv',index=False)
    original=pd.read_csv(study.DOCS/'confirmation/all_metrics.csv');original=original[original.arm.eq('current')]
    end=pd.concat([original,data],ignore_index=True);end.to_csv(study.DOCS/'end_to_end_pooled.csv',index=False)
    a=pd.read_csv(onset.DOCS/'phase_metrics.csv');a=a[a.arm.isin(['fixed_joint','onset_fit'])].copy();a['stage']='initial'
    b=pd.DataFrame(phases);b['stage']='fresh';p=pd.concat([a,b],ignore_index=True);p.to_csv(DOCS/'all_phase.csv',index=False)
    data.groupby(['stage','arm','fit_seed']).mean(numeric_only=True).to_csv(DOCS/'fit_means.csv')
    p.groupby(['stage','arm','fit_seed','phase']).mean(numeric_only=True).to_csv(DOCS/'phase_fit_means.csv')
    lines=['# 시작 거래 출력 새 seed 확인','', '새4생성, 추가 학습0. 초기/추가 각각2부모×2draw, 합산2부모×4draw.','',
        '| 범위 | 모델 | 전체 개인 대비 W1 | 시작 거래 개인 대비 W1 | 시작 거래 금액 W1 | 사기 간격 W1 | 사기 금액 W1 | 정상 금액 W1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for stage in ['initial','fresh','pooled']:
        for arm in ['fixed_joint','onset_fit']:
            d=data[data.arm.eq(arm)];q=p[p.arm.eq(arm)&p.phase.eq('onset')]
            if stage!='pooled':d=d[d.stage.eq(stage)];q=q[q.stage.eq(stage)]
            m=d.mean(numeric_only=True);n=q.mean(numeric_only=True)
            lines.append(f'| {stage} | {arm} | {m.fraud_ratio_log_w1:.4f} | {n.ratio_w1:.4f} | {n.amount_w1:.4f} | {m.fraud_gap_seconds_log_w1:.4f} | {m.fraud_amount_log_w1:.4f} | {m.normal_amount_log_w1:.4f} |')
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
    write(DOCS/'COMPLETE.json',dict(new_training=0,new_generations=4,new_rows=sum(c['rows'] for c in checks if c['arm']=='onset_fit'),
        source_sha256=digest(__file__),test_events_read=False))


if __name__=='__main__':main()
