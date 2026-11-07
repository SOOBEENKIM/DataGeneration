"""Independent saved-output evaluation; fixed primary targets and simple controls."""
import argparse,importlib.util,json
import numpy as np
import pandas as pd
from study import ROOT,OUT,DOCS,PRIOR,ABLATION,ARMS,NEW_ARMS,SEEDS,PARENTS,DRAWS,verify,preserve,head_path,origin,digest,write,now
spec=importlib.util.spec_from_file_location('previous_evaluation',PRIOR/'local_source/evaluate.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
KEYS=old.KEYS
N,O='normal_gap_seconds_log_w1','onset_personal_gap_w1'

def check():
    verify();real=pd.read_parquet(old.SOURCE/'prepared/validation.parquet')
    arm,rs,parent,gs='history_calibration_only',SEEDS[0],PARENTS[0],DRAWS[0]
    values,_=old.metrics(real,pd.read_parquet(ABLATION/f'runs/{arm}_{rs}_{parent}/generated_validation_{gs}.parquet'))
    prior=pd.read_csv(ABLATION/'report/ALL_DRAWS.csv')
    r=prior[prior.arm.eq(arm)&prior.time_fit_seed.eq(rs)&prior.parent.eq(parent)&prior.generation_seed.eq(gs)].iloc[0]
    errors={k:abs(float(v)-float(r[k])) for k,v in values.items() if k in r and np.isfinite(v)}
    assert len(errors)>=len(KEYS) and max(errors.values())<1e-10
    write(OUT/'EVALUATION_READY.json',dict(source_sha256=digest(__file__),registered_utc=now(),
        prior_output_checks=errors,new_output_reads=0,test_events_read=False))
    print('EVALUATION_READY',len(errors),max(errors.values()),flush=True)

def main():
    verify();assert (OUT/'EXECUTION_COMPLETE.json').exists()
    assert digest(__file__)==json.loads((OUT/'EVALUATION_READY.json').read_text())['source_sha256']
    real=pd.read_parquet(old.SOURCE/'prepared/validation.parquet')
    previous={p:pd.read_csv(p/'report/ALL_DRAWS.csv') for p in [PRIOR,ABLATION]}
    rows=[];checks=[];supports=[];new_rows=0;reused_rows=0
    for rs in SEEDS:
        for parent in PARENTS:
            for gs in DRAWS:
                base=pd.read_parquet(PRIOR/f'runs/history_only_{rs}_{parent}/generated_validation_{gs}.parquet')
                cols=['entity_id','event_index','event_is_fraud']
                for arm in ARMS:
                    source=origin(arm);run=source/f'runs/{arm}_{rs}_{parent}';assert (run/'COMPLETE.json').exists()
                    rec=json.loads((run/f'GENERATION_{gs}.json').read_text());path=run/f'generated_validation_{gs}.parquet'
                    assert digest(path)==rec['sha256'] and digest(head_path(arm,rs,parent))==rec['head_sha256']
                    raw=pd.read_parquet(path);assert raw[cols].equals(base[cols])
                    if arm in NEW_ARMS:
                        new_rows+=len(raw);writer=pd.read_csv(DOCS/f'evaluation/metrics_{arm}_{rs}_{parent}.csv')
                        saved=writer[writer.generation_seed.eq(gs)].iloc[0].to_dict()
                    else:
                        reused_rows+=len(raw);table=previous[source]
                        saved=table[table.arm.eq(arm)&table.time_fit_seed.eq(rs)&table.parent.eq(parent)&table.generation_seed.eq(gs)].iloc[0].to_dict()
                    values,support=old.metrics(real,raw)
                    errors={k:abs(float(v)-float(saved[k])) for k,v in values.items() if k in saved and np.isfinite(v)}
                    assert errors and max(errors.values())<1e-10,errors
                    info=dict(arm=arm,time_fit_seed=rs,parent=parent,generation_seed=gs)
                    rows.append({**saved,**values,**info,'events':len(raw),'new_generation':arm in NEW_ARMS})
                    supports.extend([{**info,**s} for s in support])
                    checks.append(dict(**info,rows=len(raw),metric_checks=len(errors),max_metric_error=max(errors.values()),labels_lengths_exact=True))
                    print('AUDITED',arm,rs,parent,gs,len(raw),flush=True)
    d=pd.DataFrame(rows);assert len(d)==72
    d.to_csv(DOCS/'ALL_DRAWS.csv',index=False);pd.DataFrame(checks).to_csv(DOCS/'INDEPENDENT_CHECKS.csv',index=False)
    pd.DataFrame(supports).to_csv(DOCS/'SUPPORT.csv',index=False)
    d.groupby(['arm','time_fit_seed','parent'])[KEYS].mean().to_csv(DOCS/'PARENT_FIT_MEANS.csv')
    fm=d.groupby(['arm','time_fit_seed'])[KEYS].mean();fm.to_csv(DOCS/'FIT_MEANS.csv')
    d.groupby(['arm','parent'])[KEYS].mean().to_csv(DOCS/'PARENT_MEANS.csv')
    means=d.groupby('arm')[KEYS].mean().reindex(ARMS);means.to_csv(DOCS/'MEANS.csv')
    paired={a:d[d.arm.eq(a)].set_index(['time_fit_seed','parent','generation_seed'])[KEYS] for a in ARMS}
    comparisons=[('history_calibration_only',a) for a in NEW_ARMS+['history_only','init_only']]
    comparisons += [(a,'history_only') for a in NEW_ARMS]
    effects=[];fit_effects=[];draw_effects=[];screens=[]
    for candidate,reference in comparisons:
        delta=paired[candidate]-paired[reference];name=f'{candidate}_minus_{reference}'
        for key in KEYS:
            s=delta[key];f=s.groupby(level='time_fit_seed').mean();p=s.groupby(level=['time_fit_seed','parent']).mean()
            effects.append(dict(candidate=candidate,reference=reference,metric=key,delta=float(s.mean()),
                lower_fits=int((f < -1e-12).sum()),lower_parent_fit_means=int((p < -1e-12).sum()),
                lower_draws=int((s < -1e-12).sum()),equal_draws=int((s.abs()<=1e-12).sum()),draws=len(s)))
            for rs,value in f.items():fit_effects.append(dict(candidate=candidate,reference=reference,metric=key,time_fit_seed=rs,delta=value))
        dd=delta.reset_index();dd.insert(0,'contrast',name);draw_effects.append(dd)
        a,b=means.loc[candidate],means.loc[reference]
        n_ok=bool(a[N]<=.9*b[N] and a[O]<=b[O]+.01)
        o_ok=bool(a[O]<=.9*b[O] and a[N]<=b[N]+.01)
        costs=[name for _,_,name in old.BINS if a[name]>b[name]+.02]
        costs += [name for name in ['fraud_ratio_log_w1','normal_amount_log_w1','class_0_merchant_tv'] if a[name]>b[name]+.01]
        if a.median_unique_merchants_per_customer<.95*b.median_unique_merchants_per_customer:costs.append('merchant_diversity')
        normal_repeats=int((fm.loc[candidate,N]<fm.loc[reference,N]-1e-12).sum())
        onset_repeats=int((fm.loc[candidate,O]<fm.loc[reference,O]-1e-12).sum())
        screens.append(dict(candidate=candidate,reference=reference,normal_effect=n_ok,onset_effect=o_ok,
            costs=','.join(costs),descriptive_screen=(n_ok or o_ok) and not costs,
            normal_better_fits=normal_repeats,onset_better_fits=onset_repeats,
            repeating_screen=bool(not costs and ((n_ok and normal_repeats==3) or (o_ok and onset_repeats==3))),
            normal_personal_delta=a.normal_stay_personal_gap_w1-b.normal_stay_personal_gap_w1))
    pd.DataFrame(effects).to_csv(DOCS/'PAIRED_EFFECTS.csv',index=False)
    pd.DataFrame(fit_effects).to_csv(DOCS/'FIT_EFFECTS.csv',index=False)
    pd.concat(draw_effects).to_csv(DOCS/'DRAW_EFFECTS.csv',index=False)
    pd.DataFrame(screens).to_csv(DOCS/'TRADEOFF_SCREEN.csv',index=False)
    fits=[]
    for p in (OUT/'fits').glob('*/CALIBRATION.json'):
        r=json.loads(p.read_text());assert r['frozen_personal_objective_parity']
        fits.append(dict(arm=r['arm'],parent=r['parent'],time_fit_seed=r['time_fit_seed'],evaluations=r['evaluations'],
            optimizer_success=r['optimizer_success'],initial_objective=r['initial']['objective'],
            selected_objective=r['selected']['objective'],frozen_personal_objective=r['frozen_personal_objective'],
            objective_delta_vs_personal=r['selected']['objective']-r['frozen_personal_objective'],
            near_bound_coefficients=sum(abs(x)>=.349 for x in r['selected']['theta'])))
    assert len(fits)==18;pd.DataFrame(fits).to_csv(DOCS/'FIT_AUDIT.csv',index=False)
    previous_count=preserve();verify();routing=[]
    for p in (OUT/'routing').glob('*/ROUTING_CHECK.json'):
        r=json.loads(p.read_text());assert r['strict_past_clock_exact'] and r['reordered_compacted_batch'] and r['strict_native_checkpoint'];routing.append(r)
    assert len(routing)==18
    supported=all(s['repeating_screen'] for s in screens if s['candidate']=='history_calibration_only' and s['reference'] in NEW_ARMS)
    audit=dict(completed_utc=now(),new_calibration_fits=18,new_neural_fits=0,new_native_argn_fits=0,
        new_generations=36,new_rows=new_rows,reused_generations=36,reused_rows=reused_rows,evaluated_outputs=72,
        independent_metric_checks=sum(x['metric_checks'] for x in checks),max_metric_error=max(x['max_metric_error'] for x in checks),
        preserved_previous_hashes=previous_count,new_routing_workers=18,labels_lengths_exact=72,
        all_calibration_optimizer_success=all(f['optimizer_success'] for f in fits),test_events_read=False,push=False,model_replacement=False)
    write(DOCS/'FINAL_AUDIT.json',audit)
    write(DOCS/'MODEL_DECISION.json',dict(active='clock_prefix',automatic_promotion=False,
        current_personal_speed_increment_supported=supported,comparisons=screens,
        new_method_claim=False,whole_generator_comparison_complete=False))
    display=[N,O,'normal_stay_personal_gap_w1','early_time','time100_199','late1024_time']
    lines=['# Frozen calibration specificity results','',
        '18 new coefficient fits;36 new native generations;36 frozen outputs reused. Three existing time-fit seeds, two frozen parents, two draws. Lower errors are better.', '',
        old.markdown_table(means[display].round(6).reset_index()),'',old.markdown_table(pd.DataFrame(screens)), '',
        'All paired fit, parent and draw effects are saved. These are repeatedly reused development data, not untouched test or independent whole-generator refits.',
        'Passing a control screen would not establish novelty or SOTA superiority. Failure stops the present personal-speed necessity claim; it is not proof that no useful research is possible.',
        'No model replacement, commit or push.']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check() if args.check else main()
