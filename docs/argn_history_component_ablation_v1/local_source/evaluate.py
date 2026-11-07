"""Complete frozen-history factorial comparison; independently read saved rows."""
import importlib.util,json
import numpy as np
import pandas as pd
from study import ROOT,OUT,DOCS,PRIOR,ARMS,NEW_ARMS,SEEDS,PARENTS,DRAWS,verify,preserve,head_path,digest,write,now
spec=importlib.util.spec_from_file_location('previous_evaluation',PRIOR/'local_source/evaluate.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
KEYS=old.KEYS

def main():
    verify();assert (OUT/'EXECUTION_COMPLETE.json').exists()
    real=pd.read_parquet(old.SOURCE/'prepared/validation.parquet')
    prior_table=pd.read_csv(PRIOR/'report/ALL_DRAWS.csv');rows=[];checks=[];supports=[];new_rows=0;reused_rows=0
    for rs in SEEDS:
        for parent in PARENTS:
            for gs in DRAWS:
                base=pd.read_parquet(PRIOR/f'runs/history_only_{rs}_{parent}/generated_validation_{gs}.parquet')
                cols=['entity_id','event_index','event_is_fraud']
                for arm in ARMS:
                    origin=OUT if arm in NEW_ARMS else PRIOR;run=origin/f'runs/{arm}_{rs}_{parent}'
                    assert (run/'COMPLETE.json').exists()
                    rec=json.loads((run/f'GENERATION_{gs}.json').read_text());path=run/f'generated_validation_{gs}.parquet'
                    assert digest(path)==rec['sha256'] and digest(head_path(arm,rs,parent))==rec['head_sha256']
                    raw=pd.read_parquet(path);assert raw[cols].equals(base[cols])
                    if arm in NEW_ARMS:
                        new_rows+=len(raw);writer=pd.read_csv(DOCS/f'evaluation/metrics_{arm}_{rs}_{parent}.csv')
                        saved=writer[writer.generation_seed.eq(gs)].iloc[0].to_dict()
                    else:
                        reused_rows+=len(raw)
                        saved=prior_table[prior_table.arm.eq(arm)&prior_table.time_fit_seed.eq(rs)&prior_table.parent.eq(parent)&prior_table.generation_seed.eq(gs)].iloc[0].to_dict()
                    values,support=old.metrics(real,raw)
                    errors={k:abs(float(v)-float(saved[k])) for k,v in values.items() if k in saved and np.isfinite(v)}
                    assert errors and max(errors.values())<1e-10,errors
                    info=dict(arm=arm,time_fit_seed=rs,parent=parent,generation_seed=gs)
                    row={**saved,**values,**info,'events':len(raw),'new_generation':arm in NEW_ARMS}
                    rows.append(row);supports.extend([{**info,**s} for s in support])
                    checks.append(dict(**info,rows=len(raw),metric_checks=len(errors),max_metric_error=max(errors.values()),labels_lengths_exact=True))
                    print('AUDITED',arm,rs,parent,gs,len(raw),flush=True)
    d=pd.DataFrame(rows);assert len(d)==60
    d.to_csv(DOCS/'ALL_DRAWS.csv',index=False)
    pd.DataFrame(checks).to_csv(DOCS/'INDEPENDENT_CHECKS.csv',index=False)
    pd.DataFrame(supports).to_csv(DOCS/'SUPPORT.csv',index=False)
    d.groupby(['arm','time_fit_seed','parent'])[KEYS].mean().to_csv(DOCS/'PARENT_FIT_MEANS.csv')
    d.groupby(['arm','time_fit_seed'])[KEYS].mean().to_csv(DOCS/'FIT_MEANS.csv')
    means=d.groupby('arm')[KEYS].mean().reindex(ARMS);means.to_csv(DOCS/'MEANS.csv')
    paired={a:d[d.arm.eq(a)].set_index(['time_fit_seed','parent','generation_seed'])[KEYS] for a in ARMS}
    h,i,c,ic=[paired[a] for a in ARMS[:4]]
    contrasts={'I_minus_H':i-h,'IC_minus_C':ic-c,'C_minus_H':c-h,'IC_minus_I':ic-i,
               'IC_minus_H':ic-h,'interaction_IC_minus_I_minus_C_plus_H':ic-i-c+h}
    for a in ARMS[:4]:contrasts[f'{a}_minus_init_only']=paired[a]-paired['init_only']
    effect_rows=[];fit_rows=[];draw_rows=[]
    for name,delta in contrasts.items():
        for key in KEYS:
            s=delta[key];fm=s.groupby(level='time_fit_seed').mean();pm=s.groupby(level=['time_fit_seed','parent']).mean()
            effect_rows.append(dict(contrast=name,metric=key,delta=float(s.mean()),lower_fits=int((fm < -1e-12).sum()),
                lower_parent_fit_means=int((pm < -1e-12).sum()),lower_draws=int((s < -1e-12).sum()),equal_draws=int((s.abs()<=1e-12).sum()),draws=len(s)))
            for rs,value in fm.items():fit_rows.append(dict(contrast=name,metric=key,time_fit_seed=rs,delta=value))
        dd=delta.reset_index();dd.insert(0,'contrast',name);draw_rows.append(dd)
    effects=pd.DataFrame(effect_rows);effects.to_csv(DOCS/'PAIRED_EFFECTS.csv',index=False)
    pd.DataFrame(fit_rows).to_csv(DOCS/'FIT_EFFECTS.csv',index=False)
    pd.concat(draw_rows).to_csv(DOCS/'DRAW_EFFECTS.csv',index=False)
    screens=[]
    for candidate,reference in [('history_init_cal',a) for a in ARMS[:3]]+[(a,'init_only') for a in ARMS[:4]]:
        a,b=means.loc[candidate],means.loc[reference];n,o='normal_gap_seconds_log_w1','onset_personal_gap_w1'
        effect=bool((a[n]<=.9*b[n] and a[o]<=b[o]+.01) or (a[o]<=.9*b[o] and a[n]<=b[n]+.01))
        costs=[name for _,_,name in old.BINS if a[name]>b[name]+.02]
        costs += [name for name in ['fraud_ratio_log_w1','normal_amount_log_w1','class_0_merchant_tv'] if a[name]>b[name]+.01]
        if a.median_unique_merchants_per_customer<.95*b.median_unique_merchants_per_customer:costs.append('merchant_diversity')
        screens.append(dict(candidate=candidate,reference=reference,primary_effect=effect,costs=','.join(costs),
            descriptive_screen=effect and not costs,normal_personal_delta=a.normal_stay_personal_gap_w1-b.normal_stay_personal_gap_w1))
    pd.DataFrame(screens).to_csv(DOCS/'TRADEOFF_SCREEN.csv',index=False)
    prior_count=preserve();verify()
    routing=[]
    for p in (OUT/'routing').glob('*/ROUTING_CHECK.json'):
        r=json.loads(p.read_text());assert r['strict_past_clock_exact'] and r['reordered_compacted_batch'] and r['strict_native_checkpoint'];routing.append(r)
    assert len(routing)==12
    audit=dict(completed_utc=now(),new_fits=0,new_generations=24,new_rows=new_rows,reused_generations=36,reused_rows=reused_rows,
        evaluated_outputs=60,independent_metric_checks=sum(x['metric_checks'] for x in checks),max_metric_error=max(x['max_metric_error'] for x in checks),
        preserved_previous_hashes=prior_count,new_routing_workers=12,labels_lengths_exact=60,test_events_read=False,push=False,model_replacement=False)
    write(DOCS/'FINAL_AUDIT.json',audit)
    write(DOCS/'MODEL_DECISION.json',dict(active='clock_prefix',automatic_promotion=False,preserved=['init_only','history_init_cal'],
        comparisons=screens,new_method_claim=False,whole_generator_comparison_complete=False))
    display=['normal_gap_seconds_log_w1','onset_personal_gap_w1','normal_stay_personal_gap_w1','early_time','time100_199','late1024_time']
    lines=['# 이력 모델 초기 처리/보정 분리 결과','',
        '기존 이력망6개 고정. 새 학습0, 새 전체 생성24. 기존36출력과 합쳐60개 평가. 낮을수록 좋은 오류.',
        'I=초기 처리만, C=보정만. 기존 neither/both와 구성 요소를 제외한 모든 가중치를 공유한다.', '',
        old.markdown_table(means[display].round(6).reset_index()),'',
        old.markdown_table(effects[effects.contrast.isin(list(contrasts)[:6])&effects.metric.isin(display[:3])].round(6)), '',
        '모든 fit/parent/draw 차이는 FIT_EFFECTS.csv와 DRAW_EFFECTS.csv에 보존한다.',
        'W1 차이의 상호작용은 오차 척도상의 기술 통계이며 새로운 구조의 필연성/신규성 증명이 아니다.',
        '이전 개발 자료를 재사용했다. 최종 test/CPAR/최신 전체 생성기 실험은 이번에 하지 않았다.',
        '기존 모델 교체·commit·push 없음.']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
