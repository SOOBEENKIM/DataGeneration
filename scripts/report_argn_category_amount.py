"""All four factorial cells, paired differences, and completion verification."""
import argparse
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from run_argn_category_amount import ROOT, OUT, DOCS, CFG, verify, digest, write
from run_argn_state_first import SOURCE


def main(partial=False):
    verify();frames=[];proof={};fit_rows=[]
    for fs in CFG['fit_seeds']:
        parent=pd.read_csv(ROOT/f'docs/research_reaudit_20260927/transition_reference/evaluation/metrics_reference_duration_{fs}.csv')
        parent['arm']='neither';frames.append(parent)
        amount=pd.read_csv(ROOT/f'docs/argn_amount_learning_v1/evaluation/metrics_balanced_shared_{fs}.csv')
        amount['arm']='amount_only';frames.append(amount)
        wd=OUT/f'worker_{fs}';fit=wd/'category_fit/FIT_COMPLETE.json'
        if fit.exists():
            record=json.loads(fit.read_text());assert digest(wd/'category_fit/amount_head.pt')==record['head_sha256']
            write(DOCS/f'FIT_COMPLETE_{fs}.json',record)
            pd.read_csv(wd/'category_fit/learning_curve.csv').to_csv(DOCS/f'learning_curve_{fs}.csv',index=False)
            fit_rows.append(dict(fit_seed=fs,selected_step=record['selected_step'],last_step=record['last_step'],
                trainable_parameters=record['trainable_parameters'],**record['validation']))
        for arm in CFG['generation_arms']:
            p=DOCS/'evaluation'/f'metrics_{arm}_{fs}.csv'
            if p.exists():frames.append(pd.read_csv(p))
            run=OUT/'runs'/f'{arm}_{fs}'
            if (run/'COMPLETE.json').exists():
                head=digest(run/'amount_head.pt');proof[f'{arm}_{fs}']=dict(head_sha256=head,generations={})
                for gs in CFG['generation_seeds']:
                    g=json.loads((run/f'GENERATION_{gs}.json').read_text())
                    assert digest(run/f'generated_validation_{gs}.parquet')==g['sha256'] and g['head_sha256']==head
                    proof[f'{arm}_{fs}']['generations'][str(gs)]=g
    if not partial:assert len(proof)==4 and len(fit_rows)==2
    data=pd.concat(frames,ignore_index=True);data.to_csv(DOCS/'all_metrics.csv',index=False)
    if fit_rows:pd.DataFrame(fit_rows).to_csv(DOCS/'fits.csv',index=False)
    means=data.groupby(['arm','fit_seed']).mean(numeric_only=True).reset_index();means.to_csv(DOCS/'fit_means.csv',index=False)
    keys=['fraud_amount_log_w1','fraud_ratio_log_w1','normal_amount_log_w1','class_1_category_amount_tv',
          'class_1_gap_category_amount_history_tv','class_0_category_amount_tv','class_0_merchant_tv','normal_gap_seconds_log_w1']
    summaries=data.groupby('arm')[keys].agg(['min','max','mean'])
    summaries.columns=['_'.join(x) for x in summaries.columns];summaries.reset_index().to_csv(DOCS/'ranges.csv',index=False)
    paired=[]
    for fs in CFG['fit_seeds']:
        cells=means[means.fit_seed.eq(fs)].set_index('arm')
        if not all(x in cells.index for x in ['neither','amount_only','category_only','category_and_amount']):continue
        for key in keys:
            p,a,c,j=[float(cells.loc[x,key]) for x in ['neither','amount_only','category_only','category_and_amount']]
            paired.append(dict(fit_seed=fs,metric=key,neither=p,amount_only=a,category_only=c,joint=j,
                               joint_vs_amount_delta=j-a,interaction_on_error=j-c-a+p))
    pd.DataFrame(paired).to_csv(DOCS/'paired_effects.csv',index=False)
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');ry=pd.to_numeric(real.event_is_fraud);ra=pd.to_numeric(real.amount_or_numeric_value)
    independent=[]
    for name,record in proof.items():
        arm,fs=name.rsplit('_',1);fs=int(fs)
        for gs in CFG['generation_seeds']:
            generated=pd.read_parquet(OUT/f'runs/{name}/generated_validation_{gs}.parquet')
            parent=pd.read_parquet(ROOT/f'artifacts/research_reaudit_20260927/transition_reference/duration_{fs}/generated_validation_{gs}.parquet')
            cols=['entity_id','event_index','event_is_fraud'];same=generated[cols].equals(parent[cols]);assert same
            row=data[(data.arm==arm)&(data.fit_seed==fs)&(data.generation_seed==gs)].iloc[0]
            sy=pd.to_numeric(generated.event_is_fraud);sa=pd.to_numeric(generated.amount_or_numeric_value);errors=[]
            for label,prefix in [(0,'normal'),(1,'fraud')]:
                value=wasserstein_distance(np.log1p(ra[ry.eq(label)]),np.log1p(sa[sy.eq(label)]))
                error=abs(value-row[prefix+'_amount_log_w1']);assert error<1e-12;errors.append(error)
            independent.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,same_label_sequence=same,
                amount_w1_max_error=max(errors),fraud_amount_median=float(sa[sy.eq(1)].median()),
                normal_amount_median=float(sa[sy.eq(0)].median()),rows=len(generated)))
    pd.DataFrame(independent).to_csv(DOCS/'independent_checks.csv',index=False)
    names={'neither':'기존 기준','amount_only':'금액만','category_only':'업종만','category_and_amount':'업종+금액'}
    lines=['# 업종–금액 제거 대조 결과','',f'새 학습 {len(fit_rows)}/2, 새 자유 생성 {len(independent)}/8 완료.',
        '기존4생성+금액만4생성도 동일 고객·seed로 함께 비교했다. 아래는 네 draw 범위.','',
        '| 변경 | 사기 금액 log-W1 | 개인 대비 금액 log-W1 | 정상 금액 log-W1 | 사기 업종–금액 TV |',
        '|---|---:|---:|---:|---:|']
    for arm,name in names.items():
        if arm not in summaries.index:continue
        r=summaries.loc[arm];vals=[f"{r[k+'_min']:.4f}–{r[k+'_max']:.4f}" for k in keys[:4]]
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines+=['','각 W1/TV는 작을수록 좋다. interaction_on_error는 두 fit에서의 기술적 차이이며',
        '비선형 오차 지표의 통계적 유의성 검정이나 금융 인과 추정이 아니다.',
        '금액 가중치는 앞선 balanced_shared와 동일하다. 사기 label/고객 구성은 고정 전환 대조와 같다.','']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines))
    write(DOCS/'completion_evidence.json',dict(complete=not partial,fits=proof,
        new_generated_rows=sum(r['rows'] for r in independent),test_events_read=False,
        report_source_sha256=digest(__file__),created_utc=datetime.now(timezone.utc).isoformat()))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
