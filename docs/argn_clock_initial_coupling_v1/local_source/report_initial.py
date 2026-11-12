"""Frozen paired development evaluation, independent raw-data checks and costs."""
from datetime import datetime, timezone
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
from run_initial import ROOT,OUT,DOCS,PRIOR,CFG,ARMS,verify,digest,write
import report_adaptive as report
import report_argn_clock_evolution as audit
from run_argn_state_first import SOURCE
from report_argn_time_density import markdown_table

BASE = ['clock_joint','history_only','clock_prefix','current']
ALL = BASE+ARMS
BINS = report.BINS+[(6,21),(21,50),(50,100),(100,200)]
old_locate=report.locate


def locate(arm,fs,gs):
    if arm in ARMS:
        return OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet',DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    if arm=='current':
        return PRIOR/f'runs/personal_six_{fs}/generated_validation_{gs}.parquet',PRIOR/f'report/evaluation/metrics_personal_six_{fs}.csv'
    return old_locate(arm,fs,gs)


def gate_current(data,h,c):
    mean=data.groupby(['fit_seed','arm']).mean(numeric_only=True)
    hm=h.groupby(['fit_seed','arm','start','end']).gap_w1.mean()
    cm=c.groupby(['fit_seed','arm']).equal_customer_normal_gap_w1.mean()
    rows=[]
    for fs in CFG['fit_seeds']:
        baseline=mean.loc[(fs,'current')]
        for arm in ARMS:
            new=mean.loc[(fs,arm)];costs=[]
            paired=h[(h.fit_seed==fs)&h.arm.isin(['current',arm])&(h.start==6)&(h.end==50)].pivot(index='generation_seed',columns='arm',values='gap_w1')
            wins=int((paired[arm]<paired['current']).sum())
            early=paired[arm].mean()<=.9*paired['current'].mean() and wins>=3
            if new.normal_gap_seconds_log_w1>baseline.normal_gap_seconds_log_w1+.005:costs.append('whole_normal_time')
            if hm.loc[(fs,arm,1024,-1)]>hm.loc[(fs,'current',1024,-1)]+.02:costs.append('late1024_time')
            if cm.loc[(fs,arm)]>cm.loc[(fs,'current')]+.01:costs.append('equal_customer_time')
            for lo,hi in [(50,200),(100,200),(200,500),(500,1024)]:
                if hm.loc[(fs,arm,lo,hi)]>hm.loc[(fs,'current',lo,hi)]+.03:costs.append(f'position_{lo}_{hi}')
            for key,tol in [('onset_personal_gap_w1',.01),('normal_stay_personal_gap_w1',.005),
                            ('fraud_gap_seconds_log_w1',.01),('fraud_ratio_log_w1',.01),
                            ('normal_amount_log_w1',.01),('class_0_merchant_tv',.01)]:
                if not np.isfinite(new[key]) or new[key]>baseline[key]+tol:costs.append(key)
            if new.median_unique_merchants_per_customer<.95*baseline.median_unique_merchants_per_customer:costs.append('merchant_diversity')
            rows.append(dict(arm=arm,fit_seed=fs,early_pass=bool(early),early_improved_draws=wins,costs=','.join(costs),passes=bool(early and not costs)))
    return pd.DataFrame(rows)


def main():
    verify();assert json.loads((DOCS/'DISPATCH_COMPLETE.json').read_text())['success']
    with patch.object(audit,'DOCS',DOCS),patch.object(audit,'CFG',CFG),patch.object(audit,'locate',locate):
        result=audit.collect_cached(ARMS)
    new,phases,personal,checks,sampling=result
    for name,d in zip(['new_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],result):
        d.to_csv(DOCS/f'{name}.csv',index=False)
    for row in checks.itertuples():
        p,_=locate(row.arm,row.fit_seed,row.generation_seed)
        rec=json.loads((p.parent/f'GENERATION_{row.generation_seed}.json').read_text())
        assert rec['head_sha256']==digest(OUT/row.arm/'time_head.pt')
    old=pd.read_csv(PRIOR/'report/all_models.csv')
    old=old[old.arm.isin(BASE[:3]+['personal_six'])].copy()
    old.loc[old.arm=='personal_six','arm']='current'; assert len(old)==32
    prior_hashes=json.loads((PRIOR/'report/ARTIFACT_HASHES.json').read_text())['sha256']
    for relative,sha in prior_hashes.items():assert digest(PRIOR/relative)==sha,relative
    data=pd.concat([old,new],ignore_index=True)
    with patch.object(report,'OUT',OUT),patch.object(report,'DOCS',DOCS),patch.object(report,'CFG',CFG), \
         patch.object(report,'ARMS',ARMS),patch.object(report,'ALL',ALL),patch.object(report,'BINS',BINS),patch.object(report,'locate',locate):
        h,c=report.horizons_and_customers()
        active_gate=report.screen(data,h,c)
    active_gate.to_csv(DOCS/'screen_vs_active.csv',index=False)
    current_gate=gate_current(data,h,c);current_gate.to_csv(DOCS/'screen_vs_current.csv',index=False)
    # The original evaluator's raw metric checks cover amounts, labels, merchants and diversity.
    initial=[]; labels=[]; real=report.times(SOURCE/'prepared/validation.parquet')
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            original=report.times(locate('current',fs,gs)[0])
            lengths=pd.concat([real.groupby('entity_id').size(),original.groupby('entity_id').size()],axis=1).min(axis=1)
            for arm in ALL:
                raw=report.times(locate(arm,fs,gs)[0])
                assert raw[['entity_id','event_index','label']].equals(original[['entity_id','event_index','label']])
                labels.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,rows=len(raw),fraud_rate=raw.label.mean()))
                a=real[real.label.eq(0)&real.event_index.between(1,5)&real.event_index.lt(real.entity_id.map(lengths))]
                b=raw[raw.label.eq(0)&raw.event_index.between(1,5)&raw.event_index.lt(raw.entity_id.map(lengths))]
                initial.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,gap_w1=report.wd(a,b,'gap'),
                    real_median=a.gap.median(),generated_median=b.gap.median(),real_events=len(a),generated_events=len(b)))
    pd.DataFrame(initial).to_csv(DOCS/'initial_positions.csv',index=False)
    pd.DataFrame(labels).to_csv(DOCS/'labels_lengths_audit.csv',index=False)
    keys=['arm','fit_seed','generation_seed']
    for lo,hi,name in [(6,50,'early_time'),(100,200,'time100_199'),(1024,-1,'late1024_time')]:
        temp=h[(h.start==lo)&(h.end==hi)][keys+['gap_w1']].rename(columns={'gap_w1':name})
        data=data.merge(temp,on=keys,validate='one_to_one')
    data=data.merge(c[keys+['equal_customer_normal_gap_w1']],on=keys,validate='one_to_one')
    data.to_csv(DOCS/'all_models.csv',index=False)
    fields=audit.KEYS+['early_time','time100_199','late1024_time','equal_customer_normal_gap_w1']
    summary=data.groupby('arm')[fields].mean().reindex(ALL)
    summary.to_csv(DOCS/'SUMMARY_METRICS.csv')
    paired=[]
    for arm in ARMS:
        p=data[data.arm=='current'].merge(data[data.arm==arm],on=['fit_seed','generation_seed'],suffixes=('_base','_new'),validate='one_to_one')
        for _,r in p.iterrows():
            for metric in fields:paired.append(dict(arm=arm,fit_seed=r.fit_seed,generation_seed=r.generation_seed,metric=metric,
                current=r[metric+'_base'],new=r[metric+'_new'],change=r[metric+'_new']-r[metric+'_base']))
    pd.DataFrame(paired).to_csv(DOCS/'paired_changes.csv',index=False)
    qualified=[a for a in ARMS if current_gate[current_gate.arm==a].passes.all()]
    eligible=[a for a in qualified if active_gate[active_gate.arm==a].passes.all()]
    decision=dict(current_screen_passed=qualified,also_active_screen_passed=eligible,
        active_model='clock_prefix',automatic_promotion=False,pushed=False,new_native_argn_fits=0,
        initial_mixture_fits=1,calibration_fits=2,new_full_generations=len(new),generated_rows=int(checks.rows.sum()),
        original_heads_and_outputs_unchanged=True,test_events_read=False,completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'MODEL_DECISION.json',decision)
    lines=['# 초기 출력·후속 보정 2×2 대조 결과','',
        '새 초기 통계 mixture1개+2계수 보정2개,24개 전체 생성. 기존 ARGN 재학습0회. 낮을수록 좋은 W1.',
        '',markdown_table(summary.reset_index().round(6)),'',markdown_table(current_gate),'',markdown_table(active_gate),'',
        '전체 사기 라벨/길이는 matched draw와 정확히 같음. 원본 생성/모델 보존. 아직 자동교체나 push 없음.']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    verify();write(DOCS/'COMPLETE.json',decision);print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()
