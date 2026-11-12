"""Independent native-output evaluation with frozen paired baselines."""
from datetime import datetime, timezone
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
from run_feedback import ROOT,OUT,DOCS,PRIOR,CFG,ARMS,verify,digest,write
import report_initial as initial
import report_adaptive as report
import report_argn_clock_evolution as audit
from report_argn_time_density import markdown_table

BASE=['clock_joint','history_only','clock_prefix','current','init_only','init_handoff']
ALL=BASE+ARMS
BINS=initial.BINS


def locate(arm,fs,gs):
    if arm in ARMS:
        return OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet',DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    return initial.locate(arm,fs,gs)


def screen(data,h,c):
    mean=data.groupby(['fit_seed','arm']).mean(numeric_only=True)
    hm=h.groupby(['fit_seed','arm','start','end']).mean(numeric_only=True)
    cm=c.groupby(['fit_seed','arm']).equal_customer_normal_gap_w1.mean()
    rows=[]
    for fs in CFG['fit_seeds']:
        base=mean.loc[(fs,'init_only')]
        for arm in ARMS:
            new=mean.loc[(fs,arm)];costs=[]
            mid=h[(h.fit_seed==fs)&h.arm.isin(['init_only',arm])&(h.start==100)&(h.end==200)].pivot(index='generation_seed',columns='arm',values='gap_w1')
            whole=data[(data.fit_seed==fs)&data.arm.isin(['init_only',arm])].pivot(index='generation_seed',columns='arm',values='normal_gap_seconds_log_w1')
            mid_wins=int((mid[arm]<mid.init_only).sum());whole_wins=int((whole[arm]<whole.init_only).sum())
            mid_pass=bool(mid[arm].mean()<=.9*mid.init_only.mean() and mid_wins>=3)
            whole_pass=bool(whole[arm].mean()<=.95*whole.init_only.mean() and whole_wins>=3)
            for lo,hi,tol in [(6,50,.005),(50,100,.03),(200,500,.03),(500,1024,.03),(1024,-1,.02)]:
                if hm.loc[(fs,arm,lo,hi),'gap_w1']>hm.loc[(fs,'init_only',lo,hi),'gap_w1']+tol:costs.append(f'position_{lo}_{hi}')
            if hm.loc[(fs,arm,100,200),'personal_gap_w1']>hm.loc[(fs,'init_only',100,200),'personal_gap_w1']+.005:costs.append('mid_personal')
            if cm.loc[(fs,arm)]>cm.loc[(fs,'init_only')]+.01:costs.append('equal_customer_normal')
            for key,tol in [('onset_personal_gap_w1',.01),('normal_stay_personal_gap_w1',.005),('fraud_gap_seconds_log_w1',.01),('fraud_ratio_log_w1',.01),('normal_amount_log_w1',.01),('class_0_merchant_tv',.01)]:
                if not np.isfinite(new[key]) or new[key]>base[key]+tol:costs.append(key)
            if new.median_unique_merchants_per_customer<.95*base.median_unique_merchants_per_customer:costs.append('merchant_diversity')
            rows.append(dict(arm=arm,fit_seed=fs,mid_pass=mid_pass,mid_wins=mid_wins,whole_pass=whole_pass,whole_wins=whole_wins,costs=','.join(costs),passes=bool(mid_pass and whole_pass and not costs)))
    return pd.DataFrame(rows)


def main():
    verify();assert json.loads((DOCS/'DISPATCH_COMPLETE.json').read_text())['success']
    with patch.object(audit,'DOCS',DOCS),patch.object(audit,'CFG',CFG),patch.object(audit,'locate',locate):
        result=audit.collect_cached(ARMS)
    new,phases,personal,checks,sampling=result
    for name,d in zip(['new_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],result):d.to_csv(DOCS/f'{name}.csv',index=False)
    for row in checks.itertuples():
        p,_=locate(row.arm,row.fit_seed,row.generation_seed)
        record=json.loads((p.parent/f'GENERATION_{row.generation_seed}.json').read_text())
        assert record['head_sha256']==digest(OUT/row.arm/'time_head.pt')
    old=pd.read_csv(PRIOR/'report/all_models.csv');old=old[old.arm.isin(BASE)].copy();assert len(old)==48
    enriched=['early_time','time100_199','late1024_time','equal_customer_normal_gap_w1','mid_personal_gap_w1']
    data=pd.concat([old.drop(columns=enriched,errors='ignore'),new],ignore_index=True)
    with patch.object(report,'OUT',OUT),patch.object(report,'DOCS',DOCS),patch.object(report,'CFG',CFG),patch.object(report,'ARMS',ARMS),patch.object(report,'ALL',ALL),patch.object(report,'BINS',BINS),patch.object(report,'locate',locate):
        h,c=report.horizons_and_customers();active=report.screen(data,h,c)
    with patch.object(initial,'CFG',CFG),patch.object(initial,'ARMS',ARMS):current=initial.gate_current(data,h,c)
    gates=screen(data,h,c)
    for name,d in [('screen_vs_init_only',gates),('screen_vs_active',active),('screen_vs_current',current)]:d.to_csv(DOCS/f'{name}.csv',index=False)
    labels=[];prefix=[]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            baseline=report.times(locate('init_only',fs,gs)[0])
            for arm in ARMS:
                raw=report.times(locate(arm,fs,gs)[0])
                assert raw[['entity_id','event_index','label']].equals(baseline[['entity_id','event_index','label']])
                mask=raw.event_index.between(1,50)
                np.testing.assert_array_equal(raw.loc[mask,'gap'],baseline.loc[mask,'gap'])
                labels.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,rows=len(raw),fraud_rate=raw.label.mean(),labels_lengths_exact=True))
                prefix.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,events=int(mask.sum()),first50_time_exact=True))
    pd.DataFrame(labels).to_csv(DOCS/'labels_lengths_audit.csv',index=False)
    pd.DataFrame(prefix).to_csv(DOCS/'prefix_audit.csv',index=False)
    keys=['arm','fit_seed','generation_seed']
    for lo,hi,name,col in [(6,50,'early_time','gap_w1'),(100,200,'time100_199','gap_w1'),(1024,-1,'late1024_time','gap_w1'),(100,200,'mid_personal_gap_w1','personal_gap_w1')]:
        data=data.merge(h[(h.start==lo)&(h.end==hi)][keys+[col]].rename(columns={col:name}),on=keys,validate='one_to_one')
    data=data.merge(c[keys+['equal_customer_normal_gap_w1']],on=keys,validate='one_to_one')
    data.to_csv(DOCS/'all_models.csv',index=False)
    fields=audit.KEYS+enriched
    summary=data.groupby('arm')[fields].mean().reindex(ALL);summary.to_csv(DOCS/'SUMMARY_METRICS.csv')
    data.groupby(['fit_seed','arm'])[fields].mean().to_csv(DOCS/'parent_means.csv')
    paired=[]
    for base in ['init_only','current','clock_prefix','correction_gain','regression_gain','output_gain_shift']:
        for arm in ARMS:
            if arm==base:continue
            joined=data[data.arm==base].merge(data[data.arm==arm],on=['fit_seed','generation_seed'],suffixes=('_base','_new'),validate='one_to_one')
            for r in joined.to_dict('records'):
                for metric in fields:paired.append(dict(baseline=base,arm=arm,fit_seed=r['fit_seed'],generation_seed=r['generation_seed'],metric=metric,base=r[metric+'_base'],new=r[metric+'_new'],change=r[metric+'_new']-r[metric+'_base']))
    pd.DataFrame(paired).to_csv(DOCS/'paired_changes.csv',index=False)
    hashes=json.loads((OUT/'report/PRIOR_HASHES.json').read_text())['sha256']
    for p,sha in hashes.items():assert digest(p)==sha,p
    qualified=[a for a in ARMS if gates[gates.arm==a].passes.all()]
    decision=dict(qualified_vs_init_only=qualified,active_model='clock_prefix',automatic_promotion=False,pushed=False,
        new_native_argn_fits=0,calibration_fits=4,new_full_generations=len(new),generated_rows=int(checks.rows.sum()),
        prior_artifact_hashes_preserved=len(hashes),matched_generations_compared=len(data),
        first50_time_exact=True,labels_lengths_exact=True,test_events_read=False,completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'MODEL_DECISION.json',decision)
    lines=['# 개인 속도 반영 강도/보정 강도 결과','',
        '작은 계수4회 학습,2개 기존 부모×4draw×4팔=32개 전체 생성. native ARGN 신규 학습0회. 새 head은 공유하므로2개 독립 학습 seed가 아니다.',
        '',markdown_table(summary.reset_index().round(6)),'',markdown_table(gates),'',markdown_table(active),'',
        '초반50개 거래간격, labels/lengths 정확 일치. 기존 모델 보존. 자동 교체/push 없음.']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    verify();write(DOCS/'COMPLETE.json',decision);print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()
