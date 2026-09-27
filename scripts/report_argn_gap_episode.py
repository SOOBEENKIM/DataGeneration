"""Audit four factorial cells plus the initial-length removal control."""
import argparse
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch

from run_argn_gap_episode import ROOT, OUT, DOCS, CFG, foundation, verify, digest, write
from run_argn_state_first import SOURCE

KEYS=['fraud_gap_seconds_log_w1','fraud_amount_log_w1','fraud_ratio_log_w1',
      'customers_with_fraud','all_fraud_customers','normal_gap_seconds_log_w1','normal_amount_log_w1',
      'class_0_merchant_tv','class_1_category_amount_tv','class_1_gap_category_amount_history_tv',
      'run_length_log_w1','completed_run_length_log_w1','generated_fraud_rate','unique_merchants',
      'median_unique_merchants_per_customer','seen_merchant_rate']


def customer_summary(raw):
    d=raw.sort_values(['entity_id','event_index'],kind='stable').copy()
    y=pd.to_numeric(d.event_is_fraud).eq(1)
    prev=y.groupby(d.entity_id).shift(1,fill_value=False)
    d['frauds']=y.astype(int);d['episode_start']=(y&~prev).astype(int)
    c=d.groupby('entity_id').agg(length=('event_index','size'),frauds=('frauds','sum'),episodes=('episode_start','sum'))
    c['share']=c.frauds/c.length
    return c


def main(partial=False):
    verify();frames=[];checks=[];fits=[];bands=[];extra=[]
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');ry=pd.to_numeric(real.event_is_fraud)
    rc=customer_summary(real)
    for fs in CFG['fit_seeds']:
        base=pd.read_csv(ROOT/f'docs/argn_category_amount_v1/evaluation/metrics_category_and_amount_{fs}.csv')
        base['arm']='baseline';frames.append(base)
        for name in CFG['arms']:
            p=OUT/f'worker_{fs}/{name}/FIT_COMPLETE.json'
            if p.exists():
                fit=json.loads(p.read_text());assert fit['fit_seed']==fs;fits.append(fit)
                assert digest(p.parent/'amount_head.pt')==fit['head_sha256']
                write(DOCS/f'fit_{name}_{fs}.json',fit)
                pd.read_csv(p.parent/'learning_curve.csv').to_csv(DOCS/f'learning_{name}_{fs}.csv',index=False)
        for arm in CFG['generation_arms']:
            p=DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
            if p.exists():frames.append(pd.read_csv(p))
        for arm in ['baseline']+list(CFG['generation_arms']):
            run=foundation(fs) if arm=='baseline' else OUT/f'runs/{arm}_{fs}'
            if not (run/'COMPLETE.json').exists():continue
            base_head=torch.load(foundation(fs)/'amount_head.pt',map_location='cpu',weights_only=True)
            head=torch.load(run/'amount_head.pt',map_location='cpu',weights_only=True)
            for k,v in base_head['state_dict'].items():assert torch.equal(v,head['state_dict'][k]),k
            if arm in ['gap_only','gap_and_transition']:
                gap_path=OUT/f'worker_{fs}/balanced_gap/amount_head.pt'
                assert head['gap_sha256']==digest(gap_path)
                gap_head=torch.load(gap_path,map_location='cpu',weights_only=True)
                for k,v in gap_head['state_dict'].items():assert torch.equal(v,head['state_dict'][k]),k
            for gs in CFG['generation_seeds']:
                raw=pd.read_parquet(run/f'generated_validation_{gs}.parquet')
                generation=json.loads((run/f'GENERATION_{gs}.json').read_text())
                assert digest(run/f'generated_validation_{gs}.parquet')==generation['sha256']
                assert digest(run/'amount_head.pt')==generation['head_sha256']
                c=customer_summary(raw);sy=pd.to_numeric(raw.event_is_fraud)
                b=pd.read_parquet(foundation(fs)/f'generated_validation_{gs}.parquet')
                pd.testing.assert_series_equal(raw.groupby('entity_id').size(),b.groupby('entity_id').size())
                same=raw[['entity_id','event_index','event_is_fraud']].equals(b[['entity_id','event_index','event_is_fraud']])
                if arm in ['baseline','gap_only']:assert same
                if arm=='gap_and_transition':
                    t=pd.read_parquet(OUT/f'runs/transition_only_{fs}/generated_validation_{gs}.parquet')
                    assert raw[['entity_id','event_index','event_is_fraud']].equals(t[['entity_id','event_index','event_is_fraud']])
                check=dict(arm=arm,fit_seed=fs,generation_seed=gs,rows=len(raw),same_lengths=True,same_baseline_labels=same,
                    frozen_category_amount_exact=True)
                for label,prefix in [(0,'normal'),(1,'fraud')]:
                    for field,column in [('amount','amount_or_numeric_value'),('gap_seconds','gap')]:
                        r=pd.to_numeric(real.loc[ry.eq(label),column]).dropna();s=pd.to_numeric(raw.loc[sy.eq(label),column]).dropna()
                        if field=='gap_seconds':
                            r=pd.to_numeric(real.loc[ry.eq(label)&real.event_index.gt(0),column]).dropna()
                            s=pd.to_numeric(raw.loc[sy.eq(label)&raw.event_index.gt(0),column]).dropna()
                        check[f'{prefix}_{field}_log_w1']=wasserstein_distance(np.log1p(r),np.log1p(s))
                checks.append(check)
                extra.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,
                    customer_fraction_w1=wasserstein_distance(rc.share,c.share),
                    episodes_per_customer_w1=wasserstein_distance(rc.episodes,c.episodes),
                    repeated_episode_customers=int(c.episodes.gt(1).sum()),
                    fraud_customer_count_error=abs(c.frauds.gt(0).sum()-rc.frauds.gt(0).sum()),
                    all_fraud_customer_count_error=abs(c.frauds.eq(c.length).sum()-rc.frauds.eq(rc.length).sum())))
                for band,select in [('short_le100',lambda x:x.length.le(100)),('long_gt100',lambda x:x.length.gt(100))]:
                    rr,ss=rc[select(rc)],c[select(c)]
                    bands.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,band=band,real_customers=len(rr),generated_customers=len(ss),
                        real_with_fraud=int(rr.frauds.gt(0).sum()),generated_with_fraud=int(ss.frauds.gt(0).sum()),
                        real_all_fraud=int(rr.frauds.eq(rr.length).sum()),generated_all_fraud=int(ss.frauds.eq(ss.length).sum())))
    data=pd.concat(frames,ignore_index=True)
    if not partial:assert len(checks)==20 and len(fits)==4 and len(data)==20
    independent=pd.DataFrame(checks)
    for _,c in independent.iterrows():
        r=data[(data.arm==c.arm)&(data.fit_seed==c.fit_seed)&(data.generation_seed==c.generation_seed)]
        if not len(r):continue
        for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1']:
            assert abs(float(r.iloc[0][k])-c[k])<1e-12,(c.arm,k)
    if extra:data=data.merge(pd.DataFrame(extra),on=['arm','fit_seed','generation_seed'],how='left')
    data.to_csv(DOCS/'all_metrics.csv',index=False);independent.to_csv(DOCS/'independent_checks.csv',index=False)
    pd.DataFrame(bands).to_csv(DOCS/'customer_bands.csv',index=False)
    means=data.groupby(['arm','fit_seed']).mean(numeric_only=True).reset_index();means.to_csv(DOCS/'fit_means.csv',index=False)
    keys=KEYS+['customer_fraction_w1','episodes_per_customer_w1','repeated_episode_customers','fraud_customer_count_error','all_fraud_customer_count_error']
    ranges=data.groupby('arm')[keys].agg(['min','max','mean']);ranges.columns=['_'.join(k) for k in ranges.columns]
    ranges.to_csv(DOCS/'ranges.csv')
    paired=[]
    for fs in CFG['fit_seeds']:
        cells=means[means.fit_seed.eq(fs)].set_index('arm')
        if not all(x in cells.index for x in ['baseline','gap_only','transition_only','gap_and_transition']):continue
        for k in keys:
            b,g,t,j=[cells.loc[a,k] for a in ['baseline','gap_only','transition_only','gap_and_transition']]
            paired.append(dict(fit_seed=fs,metric=k,baseline=b,gap_only=g,transition_only=t,joint=j,
                               joint_vs_baseline=j-b,interaction_on_metric=j-g-t+b))
    pd.DataFrame(paired).to_csv(DOCS/'paired_effects.csv',index=False)
    lines=['# 간격·사기 전환 대조 결과','',f'새 gap 학습 {len(fits)}/4, 새 전체 거래열 생성 {max(0,len(checks)-4)}/16 완료.',
        '기준은 고정 업종+금액 ARGN + 기존 duration 전환. 표는2fit×2draw 평균.','',
        '| 변경 | 사기 간격 W1 | 사기 금액 W1 | 개인 대비 금액 W1 | 사기 경험 고객 | 사기 전용 고객 | 정상 간격 W1 | 정상 금액 W1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    cols=KEYS[:7]
    for a in ['baseline','gap_only','episode_only','transition_only','gap_and_transition']:
        if a not in ranges.index:continue
        lines.append('| '+a+' | '+' | '.join(f'{ranges.loc[a,k+"_mean"]:.4f}' for k in cols)+' |')
    lines+=['','W1는 log1p 공간에서 계산하고 작을수록 좋다. 실제 사기 경험 고객114/147, 사기 전용12.',
            'observed-length teacher 진단은 online 예측이 아니다. 새 label은 post-hoc 재배정하지 않았다.',
            '두 fit만으로 통계적 우월성/새 방법의 기여를 확정하지 않는다. 모든 비용은 all_metrics.csv에 포함.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines))
    write(DOCS/'completion_evidence.json',dict(complete=not partial,new_gap_fits=len(fits),new_full_generations=max(0,len(checks)-4),
        new_generated_rows=sum(c['rows'] for c in checks if c['arm']!='baseline'),test_events_read=False,
        source_sha256=digest(__file__),created_utc=datetime.now(timezone.utc).isoformat()))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
