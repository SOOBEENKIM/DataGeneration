"""Independent matched audits and prespecified costs. No test-set reads."""
from datetime import datetime, timezone
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from run_adaptive import ROOT, OUT, DOCS, CFG, ARMS, verify, digest, write, equal_weights
import report_argn_clock_evolution as audit
import report_argn_clock_location as previous
from run_argn_state_first import SOURCE
from report_argn_time_density import markdown_table

BASELINES = ['clock_joint', 'history_only', 'clock_prefix']
ALL = BASELINES + ARMS
BINS = [(6,50), (50,200), (200,500), (500,1024), (1024,2048), (2048,np.inf), (1024,np.inf)]


def locate(arm, fs, gs):
    if arm in ARMS:
        return OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet', DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    return previous.locate(arm, fs, gs)


def times(path):
    d = pd.read_parquet(path, columns=['entity_id','event_index','event_is_fraud','gap'])
    d = d.sort_values(['entity_id','event_index']).reset_index(drop=True)
    d['label'] = pd.to_numeric(d.event_is_fraud).astype(int)
    d['gap'] = pd.to_numeric(d.gap).astype(float)
    d.loc[d.event_index.eq(0), 'gap'] = np.nan
    d['clock'] = d.groupby('entity_id').gap.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    return d


def wd(a, b, column, personal=False, equal=False):
    x = a[column].to_numpy(); y = b[column].to_numpy()
    if personal:
        x = x/a.clock.to_numpy(); y = y/b.clock.to_numpy()
    wa = equal_weights(a.entity_id.to_numpy()) if equal else None
    wb = equal_weights(b.entity_id.to_numpy()) if equal else None
    return wasserstein_distance(np.log1p(x), np.log1p(y), u_weights=wa, v_weights=wb)


def horizons_and_customers():
    real = times(SOURCE/'prepared/validation.parquet')
    lengths = real.groupby('entity_id').size()
    rows = []; customers = []; cohorts = []
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            base = times(locate('clock_prefix', fs, gs)[0]); lens = base.groupby('entity_id').size()
            common = pd.concat([lengths, lens], axis=1).dropna().min(axis=1)
            r = real[real.event_index.lt(real.entity_id.map(common))]
            for arm in ALL:
                raw = base if arm == 'clock_prefix' else times(locate(arm, fs, gs)[0])
                a = real[real.label.eq(0)&real.event_index.gt(0)&real.gap.notna()]
                b = raw[raw.label.eq(0)&raw.event_index.gt(0)&raw.gap.notna()]
                customers.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,
                    equal_customer_normal_gap_w1=wd(a,b,'gap',equal=True),
                    real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique()))
                d = raw[raw.event_index.lt(raw.entity_id.map(common))]
                for lo, hi in BINS:
                    a = r[r.event_index.ge(lo)&r.event_index.lt(hi)&r.label.eq(0)&r.clock.gt(0)]
                    b = d[d.event_index.ge(lo)&d.event_index.lt(hi)&d.label.eq(0)&d.clock.gt(0)]
                    assert len(a) and len(b)
                    rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=lo,
                        end=-1 if not np.isfinite(hi) else int(hi),
                        real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),
                        generated_customers=b.entity_id.nunique(),gap_w1=wd(a,b,'gap'),
                        equal_customer_gap_w1=wd(a,b,'gap',equal=True),
                        personal_gap_w1=wd(a,b,'gap',personal=True),clock_w1=wd(a,b,'clock')))
                long_ids = common.index[common.ge(1024)]
                for lo, hi in BINS[:4]:
                    a = r[r.entity_id.isin(long_ids)&r.event_index.ge(lo)&r.event_index.lt(hi)&r.label.eq(0)&r.clock.gt(0)]
                    b = d[d.entity_id.isin(long_ids)&d.event_index.ge(lo)&d.event_index.lt(hi)&d.label.eq(0)&d.clock.gt(0)]
                    cohorts.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=lo,end=int(hi),
                        real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique(),
                        gap_w1=wd(a,b,'gap'),equal_customer_gap_w1=wd(a,b,'gap',equal=True),
                        log_clock_bias=float(np.log1p(b.clock).mean()-np.log1p(a.clock).mean())))
            print('HORIZON_AUDITED', fs, gs, flush=True)
    h = pd.DataFrame(rows); c = pd.DataFrame(customers); cohort = pd.DataFrame(cohorts)
    for name, data in [('horizon_diagnostic',h),('customer_metrics',c),('cohort1024',cohort)]:
        data.to_csv(DOCS/f'{name}.csv',index=False)
    h.groupby(['arm','start','end']).mean(numeric_only=True).to_csv(DOCS/'horizon_means.csv')
    c.groupby('arm').mean(numeric_only=True).to_csv(DOCS/'customer_means.csv')
    cohort.groupby(['arm','start','end']).mean(numeric_only=True).to_csv(DOCS/'cohort1024_means.csv')
    old = pd.read_csv(ROOT/'artifacts/argn_clock_full_length_v1/report/horizon_diagnostic.csv')
    joint = h.merge(old,on=['arm','fit_seed','generation_seed','start','end'],suffixes=['','_old'])
    assert len(joint) == 168, len(joint)
    for k in ['gap_w1','personal_gap_w1','clock_w1']:
        np.testing.assert_allclose(joint[k],joint[k+'_old'],rtol=0,atol=1e-12)
    return h,c


def screen(data, h, customers):
    mean = data.groupby(['fit_seed','arm']).mean(numeric_only=True)
    hm = h.groupby(['fit_seed','arm','start','end']).gap_w1.mean()
    cm = customers.groupby(['fit_seed','arm']).equal_customer_normal_gap_w1.mean()
    rows = []
    for fs in CFG['fit_seeds']:
        p = mean.loc[(fs,'clock_prefix')]
        for arm in ARMS:
            new = mean.loc[(fs,arm)]
            draws = data[data.fit_seed.eq(fs)&data.arm.isin(['clock_prefix',arm])].pivot(
                index='generation_seed',columns='arm',values='normal_gap_seconds_log_w1')
            wins = int((draws[arm]<draws.clock_prefix).sum())
            normal = new.normal_gap_seconds_log_w1 <= .95*p.normal_gap_seconds_log_w1 and wins >= 3
            late = hm.loc[(fs,arm,1024,-1)] <= hm.loc[(fs,'clock_prefix',1024,-1)]
            customer = cm.loc[(fs,arm)] <= cm.loc[(fs,'clock_prefix')]
            costs=[]
            for lo,hi in BINS[:4]:
                if hm.loc[(fs,arm,lo,hi)] > hm.loc[(fs,'clock_prefix',lo,hi)]+.03:
                    costs.append(f'position_{lo}_{hi}')
            for key,tol in [('onset_personal_gap_w1',.01),('normal_stay_personal_gap_w1',.005),
                            ('fraud_gap_seconds_log_w1',.01),('fraud_ratio_log_w1',.01),
                            ('normal_amount_log_w1',.01),('class_0_merchant_tv',.01)]:
                if not np.isfinite(new[key]) or new[key]>p[key]+tol:
                    costs.append(key)
            if new.median_unique_merchants_per_customer < .95*p.median_unique_merchants_per_customer:
                costs.append('merchant_diversity')
            rows.append(dict(arm=arm,fit_seed=fs,normal_pass=bool(normal),improved_draws=wins,
                late_pass=bool(late),customer_pass=bool(customer),costs=','.join(costs),
                passes=bool(normal and late and customer and not costs)))
    return pd.DataFrame(rows)


def main():
    verify(); assert json.loads((DOCS/'DISPATCH_COMPLETE.json').read_text())['success']
    assert digest(__file__) == json.loads((OUT/'EVALUATION_MANIFEST.json').read_text())['source_sha256']
    with patch.object(audit,'DOCS',DOCS),patch.object(audit,'CFG',CFG),patch.object(audit,'locate',locate):
        result = audit.collect_cached(ARMS)
    new,phases,personal,checks,sampling = result
    for name, frame in zip(['new_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],result):
        frame.to_csv(DOCS/f'{name}.csv',index=False)
    for row in checks.itertuples():
        path=locate(row.arm,row.fit_seed,row.generation_seed)[0]
        rec=json.loads((path.parent/f'GENERATION_{row.generation_seed}.json').read_text())
        assert rec['head_sha256']==digest(OUT/row.arm/'time_head.pt')
    old=pd.read_csv(previous.DOCS/'all_models.csv');old=old[old.arm.isin(BASELINES)]
    assert len(old)==24
    for arm in BASELINES:
        folder = 'argn_clock_location_v1' if arm=='clock_prefix' else 'argn_clock_regression_v1' if arm=='clock_joint' else 'argn_time_density_v1'
        cache=pd.read_csv(ROOT/f'docs/{folder}/independent_checks.csv')
        for row in cache[cache.arm.eq(arm)].itertuples():
            assert digest(locate(arm,row.fit_seed,row.generation_seed)[0])==row.output_sha256
    data=pd.concat([old,new],ignore_index=True);data.to_csv(DOCS/'all_models.csv',index=False)
    data.groupby(['fit_seed','arm'])[audit.KEYS].mean().to_csv(DOCS/'fit_means.csv')
    h,c=horizons_and_customers(); gates=screen(data,h,c);gates.to_csv(DOCS/'screen.csv',index=False)
    table=data.groupby('arm')[audit.KEYS].mean().reindex(ALL).round(5).reset_index()
    ht=h.groupby(['start','end','arm']).gap_w1.mean().unstack('arm')[ALL].round(5).reset_index()
    qualified=[a for a in ARMS if gates[gates.arm.eq(a)].passes.all()]
    decision=dict(status='screen_passed_pending_review' if qualified else 'rejected_keep_previous',
        qualified_arms=qualified,active_candidate='clock_prefix',pushed=False,
        previous_files_unchanged=True,new_calibration_fits=4,new_native_argn_fits=0,
        new_full_generations=len(new),generated_rows=int(checks.rows.sum()),
        test_events_read=False,completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'MODEL_DECISION.json',decision)
    lines=['# 위치·개인 속도 시간 출력 결과','',
        '시간 출력4회 적합, 기존2부모×4draw×4팔=32개 자유 생성. native ARGN 재학습0회.',
        '개발 평가이며 test 미사용. W1은 낮을수록 좋다. 고객 균등은 고객별 평균 W1과 다른 marginal 지표다.','',
        markdown_table(table),'','## 동일 위치 지원집합의 정상 gap W1','',markdown_table(ht),'',
        '## 사전 등록한 보수적 교체 판단','',markdown_table(gates),'',
        f'기준 통과 팔: {qualified}. 활성 모델은 최종 검토 전까지 clock_prefix. push 없음.',
        '위치6계수와 위치3×속도2계수는 같은 크기의 서로 다른 basis이며 완전 nested 제거실험이 아니다.',
        '이번 결과만으로 최신 생성기 대비 우위나 신규 기여가 확정되지 않는다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines))
    verify();write(DOCS/'COMPLETE.json',decision)
    print('\n'.join(lines),flush=True)


if __name__=='__main__': main()
