"""Predeclared matched assessment; unsuccessful outputs remain local only."""
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from run_full_length import ROOT,OUT,DOCS,CFG,verify,digest,write
import report_argn_clock_evolution as audit
import report_argn_clock_location as previous
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate,w1
from report_argn_time_density import markdown_table

ARMS=['clock_joint','history_only','clock_prefix','clock_full_prefix']


def locate(arm,fs,gs):
    if arm=='clock_full_prefix':
        return OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet',DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    return previous.locate(arm,fs,gs)


def horizons():
    real=decorate(pd.read_parquet(SOURCE/'prepared/validation.parquet'))
    lengths=real.groupby('entity_id').size(); rows=[]
    bins=[(6,50),(50,200),(200,500),(500,1024),(1024,2048),(2048,np.inf),(1024,np.inf)]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            base=pd.read_parquet(locate('clock_joint',fs,gs)[0]); lens=base.groupby('entity_id').size()
            common=pd.concat([lengths.rename('real'),lens.rename('generated')],axis=1).dropna().min(axis=1)
            r=real[real.event_index.lt(real.entity_id.map(common))]
            for arm in ARMS:
                d=decorate(pd.read_parquet(locate(arm,fs,gs)[0]))
                d=d[d.event_index.lt(d.entity_id.map(common))]
                for low,high in bins:
                    a=r[r.event_index.ge(low)&r.event_index.lt(high)&r.label.eq(0)&r.past_gap.gt(0)]
                    b=d[d.event_index.ge(low)&d.event_index.lt(high)&d.label.eq(0)&d.past_gap.gt(0)]
                    assert len(a) and len(b)
                    rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=low,
                        end=-1 if not np.isfinite(high) else int(high),real_events=len(a),generated_events=len(b),
                        real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique(),
                        gap_w1=w1(a.gap,b.gap),personal_gap_w1=w1(a.gap/a.past_gap,b.gap/b.past_gap),
                        clock_w1=w1(a.past_gap,b.past_gap)))
    frame=pd.DataFrame(rows);frame.to_csv(DOCS/'horizon_diagnostic.csv',index=False)
    frame.groupby(['arm','start','end']).mean(numeric_only=True).to_csv(DOCS/'horizon_means.csv')
    # Recompute the old post-hoc metrics instead of trusting a cached table.
    old=pd.read_csv(previous.DOCS/'horizon_diagnostic.csv'); old['end']=old['end'].fillna(-1).astype(int)
    joint=frame.merge(old,on=['arm','fit_seed','generation_seed','start','end'],suffixes=['','_old'])
    assert len(joint)==144,len(joint)
    for k in ['gap_w1','personal_gap_w1','clock_w1']:
        np.testing.assert_allclose(joint[k],joint[k+'_old'],rtol=0,atol=1e-12)
    return frame


def assess(data,horizon):
    means=data.groupby(['fit_seed','arm']).mean(numeric_only=True)
    hmeans=horizon.groupby(['fit_seed','arm','start','end']).mean(numeric_only=True)
    rows=[]
    for fs in CFG['fit_seeds']:
        prior=means.loc[(fs,'clock_prefix')]; new=means.loc[(fs,'clock_full_prefix')]
        before=hmeans.loc[(fs,'clock_prefix',1024,-1)].gap_w1
        after=hmeans.loc[(fs,'clock_full_prefix',1024,-1)].gap_w1
        draws=horizon[horizon.fit_seed.eq(fs)&horizon.start.eq(1024)&horizon.end.eq(-1)].pivot(index='generation_seed',columns='arm',values='gap_w1')
        wins=int((draws.clock_full_prefix<draws.clock_prefix).sum())
        primary=bool(after<=.95*before and wins>=3);costs=[]
        if new.normal_gap_seconds_log_w1>prior.normal_gap_seconds_log_w1:costs.append('normal_gap')
        for lo,hi,tolerance in [(6,50,.03),(50,200,.03),(500,1024,.01),(1024,2048,.01),(2048,-1,.01)]:
            a=hmeans.loc[(fs,'clock_full_prefix',lo,hi)].gap_w1
            b=hmeans.loc[(fs,'clock_prefix',lo,hi)].gap_w1
            if a>b+tolerance:costs.append(f'horizon_{lo}_{hi}')
        for k,tolerance in [('onset_personal_gap_w1',.01),('normal_stay_personal_gap_w1',.005),
             ('fraud_gap_seconds_log_w1',.01),('fraud_ratio_log_w1',.01),
             ('normal_amount_log_w1',.01),('class_0_merchant_tv',.01)]:
            if not np.isfinite(new[k]) or new[k]>prior[k]+tolerance:costs.append(k)
        if new.median_unique_merchants_per_customer<.95*prior.median_unique_merchants_per_customer:
            costs.append('merchant_diversity')
        rows.append(dict(fit_seed=fs,late_before=before,late_after=after,late_relative_improvement=1-after/before,
            improved_draws=wins,primary_passed=primary,costs=','.join(costs),passes=primary and not costs))
    return pd.DataFrame(rows)


def main():
    verify();assert json.loads((DOCS/'DISPATCH_COMPLETE.json').read_text())['success']
    manifest=json.loads((OUT/'EVALUATION_MANIFEST.json').read_text())
    assert digest(__file__)==manifest['source_sha256']
    with patch.object(audit,'DOCS',DOCS),patch.object(audit,'CFG',CFG),patch.object(audit,'locate',locate):
        result=audit.collect_cached(CFG['arms'])
    new,phases,personal,checks,sampling=result
    for name,frame in zip(['new_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],result):
        frame.to_csv(DOCS/f'{name}.csv',index=False)
    for row in checks.itertuples():
        path=locate(row.arm,row.fit_seed,row.generation_seed)[0]
        rec=json.loads((path.parent/f'GENERATION_{row.generation_seed}.json').read_text())
        assert digest(OUT/row.arm/'time_head.pt')==rec['head_sha256']
    old=pd.read_csv(previous.DOCS/'all_models.csv'); old=old[old.arm.isin(ARMS)]
    assert len(old)==24
    for arm in ARMS[:-1]:
        folder=ROOT/'docs/argn_clock_location_v1' if arm=='clock_prefix' else ROOT/'docs/argn_clock_regression_v1' if arm=='clock_joint' else ROOT/'docs/argn_time_density_v1'
        cache=pd.read_csv(folder/'independent_checks.csv')
        for row in cache[cache.arm.eq(arm)].itertuples():
            assert digest(locate(arm,row.fit_seed,row.generation_seed)[0])==row.output_sha256
    data=pd.concat([old,new],ignore_index=True);data.to_csv(DOCS/'all_models.csv',index=False)
    data.groupby(['fit_seed','arm'])[audit.KEYS].mean().to_csv(DOCS/'fit_means.csv')
    h=horizons();screen=assess(data,h);screen.to_csv(DOCS/'screen.csv',index=False)
    table=data.groupby('arm')[audit.KEYS].mean().reindex(ARMS).round(5).reset_index()
    ht=h.groupby(['arm','start','end']).gap_w1.mean().unstack('arm')[ARMS].round(5).reset_index()
    accepted=bool(screen.passes.all())
    decision=dict(status='passes_full_length_screen_pending_review' if accepted else 'rejected_keep_previous',
        screen_passed=accepted,active_candidate='clock_prefix',previous_files_unchanged=True,
        push_permitted_by_screen=accepted,pushed=False,new_calibration_fits=1,reproduction_fits=1,
        new_native_argn_fits=0,new_full_generations=len(new),generated_rows=int(checks.rows.sum()),
        test_events_read=False,completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'MODEL_DECISION.json',decision)
    lines=['# 전체 거래 길이 학습 대조 결과','',
        '같은2부모×4draw. 전체 길이 두 계수 보정1회,1024재현1회,새 native ARGN학습0회.',
        '개발용 평가이며 최종 test 미사용. 모든 값은 낮을수록 좋은 W1(merchant 다양성 제외).','',
        markdown_table(table),'','## 같은 위치 지원집합의 정상 시간 오류','',markdown_table(ht),'',
        '## 실행 전 고정한 교체 판단','',markdown_table(screen),'',
        '전체 길이 후보가 사전 기준을 통과했다. 최종 검토 전 활성 모델은 아직 직전 모델이다.' if accepted else
        '전체 길이 후보는 교체 기준을 통과하지 못했다. 직전 clock_prefix를 유지하고 push하지 않는다.',
        '', '이는 전체 길이 학습의 이번 제한된 대조이며, 다른 보정식이나 충분히 최적화한 모든 방법에 대한 부정이 아니다.',
        '48평가 예산의 optimizer 종료 상태는 fit 기록에 보존한다. 비용 screen은 논문 성립 조건이 아니다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines))
    verify();write(DOCS/'COMPLETE.json',decision)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
