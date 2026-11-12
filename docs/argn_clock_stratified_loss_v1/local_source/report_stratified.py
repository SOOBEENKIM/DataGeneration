"""Independent audits plus paired coarse/refined comparisons, frozen before fit."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from run_stratified import ROOT, OUT, DOCS, PRIOR, CFG, ARMS, verify, digest, write
import report_adaptive as report
from report_argn_time_density import markdown_table


def main():
    verify()
    # Reuse the same independent evaluator and conservative preservation screen.
    # The runtime arm names and architectures are unchanged, only their folder differs.
    with patch.object(report,'OUT',OUT),patch.object(report,'DOCS',DOCS), \
         patch.object(report,'CFG',CFG),patch.object(report,'verify',verify), \
         patch.object(report,'__file__',str(Path(__file__).resolve())):
        report.main()
    old_checks=pd.read_csv(PRIOR/'report/independent_checks.csv')
    for r in old_checks.itertuples():
        assert digest(PRIOR/f'runs/{r.arm}_{r.fit_seed}/generated_validation_{r.generation_seed}.parquet')==r.output_sha256
    combined=[]; paired=[]
    keys=['arm','fit_seed','generation_seed']
    for loss,folder in [('coarse',PRIOR),('refined',OUT)]:
        d=pd.read_csv(folder/'report/all_models.csv');d=d[d.arm.isin(ARMS)].copy()
        h=pd.read_csv(folder/'report/horizon_diagnostic.csv')
        tail=h[h.start.eq(1024)&h.end.eq(-1)][keys+['gap_w1']].rename(columns={'gap_w1':'late_normal_gap_w1'})
        extreme=h[h.start.eq(2048)][keys+['gap_w1']].rename(columns={'gap_w1':'tail2048_gap_w1'})
        c=pd.read_csv(folder/'report/customer_metrics.csv')[keys+['equal_customer_normal_gap_w1']]
        d=d.merge(tail,on=keys,validate='one_to_one').merge(extreme,on=keys,validate='one_to_one').merge(c,on=keys,validate='one_to_one')
        d['loss']=loss;combined.append(d)
    both=pd.concat(combined,ignore_index=True);both.to_csv(DOCS/'paired_models.csv',index=False)
    fields=report.audit.KEYS+['late_normal_gap_w1','tail2048_gap_w1','equal_customer_normal_gap_w1']
    pair=combined[0].merge(combined[1],on=keys,suffixes=['_coarse','_refined'],validate='one_to_one')
    for r in pair.to_dict('records'):
        for k in fields:
            paired.append({**{name:r[name] for name in keys},'metric':k,
                'coarse':r[k+'_coarse'],'refined':r[k+'_refined'],
                'delta_refined_minus_coarse':r[k+'_refined']-r[k+'_coarse']})
    p=pd.DataFrame(paired);p.to_csv(DOCS/'paired_draw_differences.csv',index=False)
    pm=p.groupby(['arm','fit_seed','metric'])[['coarse','refined','delta_refined_minus_coarse']].mean()
    pm.to_csv(DOCS/'paired_parent_differences.csv')
    table=both.groupby(['arm','loss'])[fields].mean().round(6).reset_index()
    table.to_csv(DOCS/'PAIRED_SUMMARY.csv',index=False)
    training=[]
    for arm in ARMS:
        f=json.loads((OUT/arm/'FIT_COMPLETE.json').read_text())
        for loss,m in [('coarse',f['old_coefficients_on_same_noise']),('refined',f['selected'])]:
            training.append(dict(arm=arm,loss=loss,**{k:m[k] for k in ['objective','coarse_objective','normal','bin2','bin2_refined','pooling_gap','fine_200','fine_500','fine_1024','fine_2048']}))
    pd.DataFrame(training).to_csv(DOCS/'paired_training.csv',index=False)
    relevant=['arm','loss','normal_gap_seconds_log_w1','late_normal_gap_w1','tail2048_gap_w1',
        'equal_customer_normal_gap_w1','onset_personal_gap_w1','fraud_ratio_log_w1']
    lines=['# 同一 모델에서 손실만 세분화한 paired 결과','',
        'coarse/refined 각각2기존 부모×4draw 평균. W1은 낮을수록 좋음.',
        '같은 팔의 전후만 손실 단독 대조다. 행/고객 질량 및 구조/예산은 그대로다.','',
        markdown_table(table[relevant]),'',
        '원래 clock_prefix 보존 판단은 screen.csv와 MODEL_DECISION.json에 별도로 기록한다.',
        '학습 목적값과 자유 생성 품질을 구분한다. tail2048의 고객 지원이 작으므로 신중히 해석한다.','']
    (DOCS/'PAIRED_RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True);verify()


if __name__=='__main__':main()
