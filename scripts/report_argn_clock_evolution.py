"""Audit every new full generation against immutable matched baselines."""
from datetime import datetime, timezone
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import report_argn_time_density as prior
from run_argn_clock_evolution import ROOT,OUT,DOCS,CFG,verify,digest,write
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate

BASELINES=['frozen','phase_mixture','history_only','clock_joint']
original_locate=prior.locate
KEYS=['normal_gap_seconds_log_w1','onset_personal_gap_w1','fraud_gap_seconds_log_w1',
      'normal_stay_personal_gap_w1','onset_gap_w1','fraud_ratio_log_w1','normal_amount_log_w1',
      'class_0_merchant_tv','median_unique_merchants_per_customer']


def locate(arm,fs,gs):
    if arm in CFG['arms']:
        return OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet',DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    if arm=='clock_joint':
        base=ROOT/'artifacts/argn_clock_regression_v1'
        return base/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet',ROOT/f'docs/argn_clock_regression_v1/evaluation/metrics_{arm}_{fs}.csv'
    return original_locate(arm,fs,gs)


def screen(data):
    means=data.groupby(['fit_seed','arm']).mean(numeric_only=True); rows=[]
    for fs in CFG['fit_seeds']:
        gmr=means.loc[(fs,'clock_joint')]; hist=means.loc[(fs,'history_only')]; simple=means.loc[(fs,'phase_mixture')]
        for arm in CFG['arms']:
            a=means.loc[(fs,arm)]; costs=[]
            for key,tolerance in [('onset_personal_gap_w1',.01),('normal_stay_personal_gap_w1',.01),
                ('fraud_gap_seconds_log_w1',.02),('fraud_ratio_log_w1',.02),('normal_amount_log_w1',.02),('class_0_merchant_tv',.02)]:
                if not np.isfinite(a[key]) or a[key]>gmr[key]+tolerance: costs.append('GMR:'+key)
            if a.median_unique_merchants_per_customer<.95*gmr.median_unique_merchants_per_customer: costs.append('GMR:merchant_diversity')
            for key in ['normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1']:
                if not np.isfinite(a[key]) or a[key]>simple[key]+.03: costs.append('phase_GMM:'+key)
            normal=bool(a.normal_gap_seconds_log_w1<gmr.normal_gap_seconds_log_w1 and a.normal_gap_seconds_log_w1<hist.normal_gap_seconds_log_w1)
            extra=arm=='progress_rollout' and all(a.normal_gap_seconds_log_w1<means.loc[(fs,b)].normal_gap_seconds_log_w1 for b in ['progress_joint','progress_prefix'])
            rows.append(dict(fit_seed=fs,arm=arm,normal_better_than_GMR_and_history=normal,costs=','.join(costs),
                passes_screen=normal and not costs,rollout_added_effect=extra and normal and not costs))
    return pd.DataFrame(rows)


def trajectory(data):
    real=decorate(pd.read_parquet(SOURCE/'prepared/validation.parquet')); lengths=real.groupby('entity_id').size(); rows=[]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            p,_=locate('frozen',fs,gs); raw=pd.read_parquet(p); lens=raw.groupby('entity_id').size()
            common=lengths.index[lengths.ge(1000)&lens.reindex(lengths.index).ge(1000)]
            for arm in ['real']+BASELINES+CFG['arms']:
                d=real if arm=='real' else decorate(pd.read_parquet(locate(arm,fs,gs)[0]))
                d=d[d.entity_id.isin(common)&d.event_index.ge(6)&d.event_index.lt(1000)&d.past_gap.gt(0)].copy()
                d['clock']=np.log1p(d.past_gap); anchor=d[d.event_index.lt(50)].groupby('entity_id').clock.median()
                d['delta']=d.clock-d.entity_id.map(anchor)
                for low,high in [(6,50),(50,200),(200,500),(500,1000)]:
                    a=d[d.event_index.ge(low)&d.event_index.lt(high)]; c=a.groupby('entity_id')[['clock','delta']].mean()
                    rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=low,end=high,customers=len(c),
                        mean_log_clock=float(c.clock.mean()),mean_clock_change=float(c.delta.mean()),gap_median=float(a.gap.median())))
    frame=pd.DataFrame(rows); frame.to_csv(DOCS/'clock_trajectories.csv',index=False)
    frame.groupby(['arm','start'])[['customers','mean_log_clock','mean_clock_change','gap_median']].mean().to_csv(DOCS/'clock_trajectory_means.csv')


def figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'frozen':'Frozen ARGN extension','phase_mixture':'Phase GMM','history_only':'History density',
        'clock_joint':'Clock GMR','progress_joint':'Progress GMR','progress_prefix':'Actual-prefix calibration',
        'progress_rollout':'Recursive calibration'}
    means=data.groupby(['arm','fit_seed']).mean(numeric_only=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for arm,color in zip(BASELINES+CFG['arms'],['#888888','#d29922','#287ca6','#c06522','#73509b','#427e6d','#ad3551']):
        x=means.loc[arm]
        for ax,key,title in zip(axes,['onset_personal_gap_w1','fraud_gap_seconds_log_w1','normal_stay_personal_gap_w1'],
            ['Personal fraud-onset relation','All-fraud timing','Personal normal relation']):
            ax.scatter(x.normal_gap_seconds_log_w1,x[key],s=27,color=color,alpha=.45)
            ax.scatter(x.normal_gap_seconds_log_w1.mean(),x[key].mean(),s=105,marker='X',color=color,label=names[arm])
            ax.set(xlabel='All-normal gap log-W1',ylabel=title+' W1'); ax.grid(alpha=.18)
    axes[2].legend(fontsize=7.5,loc='upper right')
    fig.suptitle('Normal timing versus relationship costs — lower is better; 2 parents × 4 draws')
    for ext in ['png','pdf','svg']:
        p=DOCS/f'clock_evolution_tradeoff.{ext}'; fig.savefig(p,dpi=180)
        if ext=='svg': p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    verify(); assert json.loads((DOCS/'DISPATCH_COMPLETE.json').read_text())['success']
    # Prior collect rechecks raw metrics with an independent implementation. Its old
    # per-worker head convention is disabled; our shared heads are checked below.
    result=collect_cached(CFG['arms'])
    new,phases,personal,checks,sampling=result[:5]
    for row in checks.itertuples():
        path=locate(row.arm,row.fit_seed,row.generation_seed)[0]
        g=json.loads((path.parent/f'GENERATION_{row.generation_seed}.json').read_text())
        assert digest(OUT/row.arm/'time_head.pt')==g['head_sha256']
    for name,frame in zip(['new_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],result[:5]):
        frame.to_csv(DOCS/f'{name}.csv',index=False)
    old=pd.read_csv(ROOT/'docs/argn_clock_regression_v1/all_models.csv')
    # Revalidate all reused raw output hashes, not just an old aggregate CSV.
    for doc in ['argn_time_density_v1','argn_clock_regression_v1']:
        cache=pd.read_csv(ROOT/f'docs/{doc}/independent_checks.csv')
        for row in cache[cache.arm.isin(BASELINES)].itertuples():
            assert digest(locate(row.arm,row.fit_seed,row.generation_seed)[0])==row.output_sha256
    all_data=pd.concat([old,new],ignore_index=True); all_data.to_csv(DOCS/'all_models.csv',index=False)
    combined=all_data[all_data.arm.isin(BASELINES+CFG['arms'])]
    combined.groupby(['fit_seed','arm'])[KEYS].mean().to_csv(DOCS/'fit_means.csv')
    combined.groupby('arm')[KEYS].agg(['mean','min','max']).to_csv(DOCS/'ranges.csv')
    gates=screen(combined); gates.to_csv(DOCS/'screen.csv',index=False)
    fields=KEYS[:7]; table=combined.groupby('arm')[fields].mean().reindex(BASELINES+CFG['arms']).round(5).reset_index()
    lines=['# 개인 속도 진행과 반복 입력 보정 결과','',
        '같은2개 frozen ARGN 부모×4개 GPU sampling draw. 새 GMM4개와2계수 보정2개는 공유 적합이며 독립 ARGN6학습이 아니다.',
        'development 평가이며 최종 test는 사용하지 않았다. W1은 낮을수록 좋다.','',prior.markdown_table(table),
        '', '사전에 고정한 부모별 비용 판정:', '',prior.markdown_table(gates),'']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)); print('\n'.join(lines),flush=True)
    trajectory(combined); figure(combined)
    write(DOCS/'COMPLETE.json',dict(new_full_generations=len(new),generated_rows=int(checks.rows.sum()),
        statistical_phase_fits=4,calibration_fits=2,new_native_argn_fits=0,matched_compared_generations=len(combined),
        all_preserved_comparisons=len(all_data),test_events_read=False,source_sha256=digest(__file__),
        heads=json.loads((OUT/'FIT_COMPLETE.json').read_text())['heads'],completed_utc=datetime.now(timezone.utc).isoformat()))


def collect_cached(arms):
    names=['metrics','phases','personal','checks','sampling'];parts=[]
    for arm in arms:
        dest=DOCS/'arm_audits'/arm
        if (dest/'COMPLETE.json').exists():
            record=json.loads((dest/'COMPLETE.json').read_text())
            for name in names:assert digest(dest/f'{name}.csv')==record['sha256'][name]
            frames=[pd.read_csv(dest/f'{name}.csv') for name in names]
            for row in frames[3].itertuples():assert digest(locate(arm,row.fit_seed,row.generation_seed)[0])==row.output_sha256
        else:
            with patch.object(prior,'locate',locate),patch.object(prior,'CFG',{**prior.CFG,'arms':[]}):
                result=prior.collect([arm])
            frames=list(result[:5]);dest.mkdir(parents=True,exist_ok=True)
            for name,frame in zip(names,frames):frame.to_csv(dest/f'{name}.csv',index=False)
            write(dest/'COMPLETE.json',dict(raw_independent_audit=True,sha256={name:digest(dest/f'{name}.csv') for name in names}))
        parts.append(frames)
    return tuple(pd.concat([p[i] for p in parts],ignore_index=True) for i in range(5))


if __name__=='__main__': main()
