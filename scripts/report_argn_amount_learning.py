"""Aggregate all registered amount controls without choosing favorable draws."""
import argparse
from datetime import datetime, timezone
import json

import pandas as pd

from run_argn_amount_learning import ROOT, OUT, DOCS, CFG, digest, write, verify


def main(partial=False):
    verify();fits=[];generations=[];evidence={};conditional=[];cache_checks={};label_checks=[]
    mirror=DOCS/'runs';mirror.mkdir(exist_ok=True)
    for fs in CFG['fit_seeds']:
        cache=OUT/f'worker_{fs}/CACHE_COMPLETE.json'
        if cache.exists():
            record=json.loads(cache.read_text());cache_checks[str(fs)]=record['replayed_digit_positions']
            write(DOCS/f'CACHE_COMPLETE_{fs}.json',record)
        probe=DOCS/f'conditional_{fs}.csv'
        if probe.exists():conditional.append(pd.read_csv(probe))
        for arm in CFG['arms']:
            run=OUT/'runs'/f'{arm}_{fs}';fp=run/'FIT_COMPLETE.json'
            if fp.exists():
                fit=json.loads(fp.read_text());assert digest(run/'amount_head.pt')==fit['head_sha256']
                write(mirror/f'{arm}_{fs}_FIT_COMPLETE.json',fit)
                pd.read_csv(run/'learning_curve.csv').to_csv(mirror/f'{arm}_{fs}_learning_curve.csv',index=False)
                fits.append(dict(arm=arm,fit_seed=fs,selected_step=fit['selected_step'],last_step=fit['last_step'],
                    trainable_parameters=fit['trainable_parameters'],seconds=fit['seconds'],
                    **{f'validation_{k}':v for k,v in fit['validation'].items()},
                    **{f'optimization_{k}':v for k,v in fit['optimization'].items()}))
            ep=DOCS/'evaluation'/f'metrics_{arm}_{fs}.csv'
            if ep.exists():generations.append(pd.read_csv(ep))
            if (run/'COMPLETE.json').exists():
                evidence[f'{arm}_{fs}']=dict(head_sha256=digest(run/'amount_head.pt'),generations={})
                for gs in CFG['generation_seeds']:
                    path=run/f'generated_validation_{gs}.parquet'
                    meta=json.loads((run/f'GENERATION_{gs}.json').read_text())
                    assert digest(path)==meta['sha256']
                    evidence[f'{arm}_{fs}']['generations'][str(gs)]=meta
                    parent=ROOT/f'artifacts/research_reaudit_20260927/transition_reference/duration_{fs}/generated_validation_{gs}.parquet'
                    columns=['entity_id','event_index','event_is_fraud']
                    new=pd.read_parquet(path,columns=columns);old=pd.read_parquet(parent,columns=columns)
                    label_checks.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,
                        rows=len(new),parent_rows=len(old),same_label_sequences=new.equals(old)))
    if not partial:
        assert len(fits)==8 and len(evidence)==8 and sum(len(x) for x in generations)==16
    if fits:pd.DataFrame(fits).to_csv(DOCS/'fits.csv',index=False)
    if not conditional or not generations:
        print('AMOUNT_REPORT_WAITING',len(fits),len(evidence));return
    probe=pd.concat(conditional,ignore_index=True);probe.to_csv(DOCS/'conditional_all.csv',index=False)
    gen=pd.concat(generations,ignore_index=True)
    parent=pd.concat([pd.read_csv(ROOT/f'docs/research_reaudit_20260927/transition_reference/evaluation/metrics_reference_duration_{fs}.csv') for fs in CFG['fit_seeds']],ignore_index=True)
    parent['arm']='frozen_duration_parent'
    all_gen=pd.concat([parent,gen],ignore_index=True);all_gen.to_csv(DOCS/'all_generation_metrics.csv',index=False)
    fit_means=all_gen.groupby(['arm','fit_seed']).mean(numeric_only=True).reset_index()
    fit_means.to_csv(DOCS/'generation_fit_means.csv',index=False)
    pd.DataFrame(label_checks).to_csv(DOCS/'paired_label_sequences.csv',index=False)
    screens=[]
    for fs in CFG['fit_seeds']:
        p=probe[(probe.arm=='parent')&(probe.fit_seed==fs)&(probe.split=='development')].set_index('label')
        if len(p)!=2:continue
        pg=fit_means[(fit_means.arm=='frozen_duration_parent')&(fit_means.fit_seed==fs)].iloc[0]
        for arm in CFG['arms']:
            q=probe[(probe.arm==arm)&(probe.fit_seed==fs)&(probe.split=='development')].set_index('label')
            g=fit_means[(fit_means.arm==arm)&(fit_means.fit_seed==fs)]
            if len(q)!=2 or len(g)!=1:continue
            g=g.iloc[0]
            screens.append(dict(arm=arm,fit_seed=fs,
                conditional_fraud_w1_reduction=1-q.loc[1,'log_w1']/p.loc[1,'log_w1'],
                conditional_normal_w1_delta=q.loc[0,'log_w1']-p.loc[0,'log_w1'],
                free_fraud_w1_delta=g.fraud_amount_log_w1-pg.fraud_amount_log_w1,
                free_personal_ratio_w1_delta=g.fraud_ratio_log_w1-pg.fraud_ratio_log_w1,
                free_normal_w1_delta=g.normal_amount_log_w1-pg.normal_amount_log_w1))
    screen=pd.DataFrame(screens)
    if len(screen):
        screen['conditional_pass']=(screen.conditional_fraud_w1_reduction>=.2)&(screen.conditional_normal_w1_delta<=.05)
        screen['free_generation_pass']=(screen.free_fraud_w1_delta<0)&(screen.free_personal_ratio_w1_delta<0)&(screen.free_normal_w1_delta<=.05)
        screen.to_csv(DOCS/'screen.csv',index=False)
    keys=['fraud_amount_log_w1','fraud_ratio_log_w1','normal_amount_log_w1','mean_run_length','customers_with_fraud',
          'class_0_merchant_tv','normal_gap_seconds_log_w1','class_1_gap_category_amount_history_tv']
    ranges=[]
    for arm,group in all_gen.groupby('arm',sort=False):
        ranges.append(dict(arm=arm,**{f'{k}_{agg}':float(getattr(group[k],agg)()) for k in keys for agg in ['min','max','mean']}))
    pd.DataFrame(ranges).to_csv(DOCS/'generation_ranges.csv',index=False)
    lines=['# ARGN 금액 출력 학습 결과', '',
        f'완료 학습 {len(fits)}/8, 완료 자유 생성 {len(gen)}/16. 최종 test 미접근.', '',
        '모든 값은 두 부모 fit × 두 생성 seed의 범위다. 작을수록 좋은 W1 지표.', '',
        '| 모델 | 사기 금액 log-W1 | 개인 대비 금액 log-W1 | 정상 금액 log-W1 |',
        '|---|---:|---:|---:|']
    for row in ranges:
        vals=[f"{row[k+'_min']:.4f}–{row[k+'_max']:.4f}" for k in keys[:3]]
        lines.append('| '+row['arm']+' | '+' | '.join(vals)+' |')
    lines+=['','사전 screen은 다음 단계 선택용이며 통계적 유의성/논문 기여 판정이 아니다.',
        '조건부 probe, 모든 세부 지표, 실제 선택 step은 각각 conditional_all.csv,',
        'all_generation_metrics.csv, fits.csv에 보존했다. 개인 고객 구성은 이번 학습의 변경 대상이 아니다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines))
    write(DOCS/'completion_evidence.json',dict(complete=not partial,fits=evidence,
        replayed_digit_positions=cache_checks,label_sequence_checks=label_checks,
        report_source_sha256=digest(__file__),test_events_read=False,
        created_utc=datetime.now(timezone.utc).isoformat()))
    print('\n'.join(lines));print('AMOUNT_REPORT',len(fits),len(gen),int(gen.generated_events.sum()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
