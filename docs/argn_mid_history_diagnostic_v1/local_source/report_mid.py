"""Audit and summarize the registered clock/correction diagnosis."""
from datetime import datetime, timezone
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diagnose_mid as d


def saved_outputs():
    real=d.old.frame(d.old.SOURCE/'prepared/validation.parquet');rows=[]
    previous=pd.read_csv(d.PRIOR/'report/horizon_diagnostic.csv')
    verified=0
    for fs in d.CFG['fit_seeds']:
        for gs in d.CFG['generation_seeds']:
            g=d.old.frame(d.PRIOR/f'runs/init_only_{fs}/generated_validation_{gs}.parquet')
            common=pd.concat([real.groupby('entity_id').size(),g.groupby('entity_id').size()],axis=1).min(axis=1)
            a=real[real.event_index.lt(real.entity_id.map(common))]
            b=g[g.event_index.lt(g.entity_id.map(common))]
            for lo,hi in d.BINS:
                x=a[a.event_index.ge(lo)&a.event_index.lt(hi)&a.label.eq(0)&a.raw_clock.gt(0)]
                y=b[b.event_index.ge(lo)&b.event_index.lt(hi)&b.label.eq(0)&b.raw_clock.gt(0)]
                r=dict(fit_seed=fs,generation_seed=gs,start=lo,end=hi,real_events=len(x),generated_events=len(y),
                    **d.old.distances(x.gap,y.gap,x.entity_id,y.entity_id))
                old=previous[previous.arm.eq('init_only')&previous.fit_seed.eq(fs)&previous.generation_seed.eq(gs)&previous.start.eq(lo)&previous.end.eq(hi)]
                if len(old):
                    assert len(old)==1
                    np.testing.assert_allclose(r['gap_w1'],old.gap_w1.iloc[0],atol=1e-12,rtol=0);verified+=1
                rows.append(r)
    pd.DataFrame(rows).to_csv(d.DOCS/'saved_generation_positions.csv',index=False)
    assert verified==40
    return verified


def main():
    d.verify();done=json.loads((d.DOCS/'COMPLETE.json').read_text())
    assert done['recursive_replays']==156 and done['fixed_prefix_resamples']==96
    frame=pd.read_csv(d.DOCS/'all_diagnostics.csv')
    keys=['split','path_source','style','fit_seed','generation_seed','noise_seed','start','end','group']
    assert not frame.duplicated(keys+['mode']).any()
    assert frame.groupby(keys)[['events','customers']].nunique().eq(1).all().all()
    assert np.isfinite(frame[['gap_w1','customer_gap_w1','personal_gap_w1','mean_delta']]).all().all()
    early=frame[frame.start.lt(100)]
    assert early.groupby(keys).gap_w1.nunique().eq(1).all()
    primary=frame[frame.start.eq(100)&frame.end.eq(200)]
    group=['split','path_source','style','mode','group']
    fields=['gap_w1','customer_gap_w1','log_gap_bias','own_log_clock_bias','base_expected_log_gap_bias',
            'mean_delta','q10_delta','q90_delta','saturated_delta_fraction','personal_gap_w1','events','customers']
    primary.groupby(group)[fields].mean().to_csv(d.DOCS/'TARGET_MEANS.csv')
    contrast=[]
    for ref in ['GG_on','RR_on']:
        base=frame[frame['mode'].eq(ref)]
        for mode in d.MODES:
            if mode==ref:continue
            pair=base.merge(frame[frame['mode'].eq(mode)],on=keys,suffixes=('_base','_new'),validate='one_to_one')
            for _,r in pair.iterrows():
                row={k:r[k] for k in keys};row.update(reference=ref,mode=mode,
                    gap_w1_change=r.gap_w1_new-r.gap_w1_base,
                    customer_w1_change=r.customer_gap_w1_new-r.customer_gap_w1_base,
                    log_bias_change=r.log_gap_bias_new-r.log_gap_bias_base,
                    personal_w1_change=r.personal_gap_w1_new-r.personal_gap_w1_base)
                contrast.append(row)
    c=pd.DataFrame(contrast);c.to_csv(d.DOCS/'paired_contrasts.csv',index=False)
    c.groupby(['split','path_source','style','reference','mode','start','end','group']).agg(
        mean_gap_w1_change=('gap_w1_change','mean'),improved=('gap_w1_change',lambda x:int((x<0).sum())),
        comparisons=('gap_w1_change','size'),mean_personal_w1_change=('personal_w1_change','mean')).to_csv(d.DOCS/'contrast_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    selections=[('internal_validation','actual_labels','recursive','Internal / true label paths'),
                ('development','generated_labels','recursive','Generated labels / recursive clocks'),
                ('development','generated_labels','saved_prefix','Saved generated clocks / one-step')]
    for ax,(split,path,style,title) in zip(axes,selections):
        part=primary[primary.split.eq(split)&primary.path_source.eq(path)&primary['style'].eq(style)&primary['group'].eq('NN')]
        mean=part.groupby('mode').gap_w1.mean().reindex(d.MODES)
        ax.bar(range(6),mean,color=['#777777','#288a98','#709959','#4163ad','#c38739','#a15d87'])
        for i,v in enumerate(mean):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(6),d.MODES,rotation=35,ha='right');ax.set_title(title,fontsize=10)
        ax.set_ylabel('Normal-stay gap log-W1 (lower is better)');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Frozen init-only head: local position 100–199 diagnosis (oracle ≠ generation quality)')
    for ext in ['png','pdf']:fig.savefig(d.DOCS/f'clock_correction_diagnosis.{ext}',dpi=180)
    plt.close(fig)
    verified=saved_outputs()
    hashes=json.loads((d.PRIOR/'report/ARTIFACT_HASHES.json').read_text())['sha256']
    for relative,sha in hashes.items():assert d.digest(d.PRIOR/relative)==sha,relative
    jobs=list(d.DOCS.glob('*_COMPLETE.json'));assert len(jobs)==11
    details=[json.loads(p.read_text()) for p in jobs]
    assert sum(x['recursive_replays'] for x in details)==156
    assert sum(x['fixed_prefix_resamples'] for x in details)==96
    assert sum(x['evaluated_time_values'] for x in details)==done['evaluated_time_values']
    assert sum(x['early_prefix_checks'] for x in details)==130
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=d.ROOT,text=True).strip()
    d.verify()
    audit={**done,'completed_utc':datetime.now(timezone.utc).isoformat(),'completed_jobs':len(jobs),
        'metric_rows':len(frame),'matched_supports':True,'all_early_outputs_unchanged_checks':130,
        'prior_artifacts_hash_verified':len(hashes),'saved_full_generation_metrics_reproduced':verified,
        'numpy_torch_and_identity_checks_passed':True,'original_heads_and_generations_preserved':True,
        'test_events_read':False,'committed':False,'pushed':False,'active_model':'clock_prefix',
        'diagnosed_local_candidate':'init_only','new_model_training_started':False,'experiment_running':False}
    d.write(d.DOCS/'FINAL_AUDIT.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
