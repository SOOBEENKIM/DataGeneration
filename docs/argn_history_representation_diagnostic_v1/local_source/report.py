"""Independent saved-sample audit and aggregate tables; never select a model."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from diagnose import OUT,DOC,SEEDS,NOISES,ROOT,HEAD,INITIAL,verify,digest,write

def main():
    verify();frames=[];law=[];nll=[];complete=[];checks=[];negative=0
    for gs in SEEDS:
        run=OUT/f'capture_{gs}'
        c=json.loads((run/'EVALUATION_COMPLETE.json').read_text());complete.append(c)
        m=pd.read_parquet(run/'matched_metadata.parquet');f=pd.read_csv(run/'metrics.csv');frames.append(f)
        law.append(pd.read_csv(run/'law_effects.csv'));nll.append(pd.read_csv(run/'conditional_nll.csv'))
        for anchor in ['generated','actual']:
            for name in ['init_only','mixture_marginal']:
                saved={h:np.load(run/f'samples_{anchor}__{name}__{h}.npz')['gaps']
                    for h in ['generated','actual_aligned','actual_literal']}
                for h,gaps in saved.items():
                    assert gaps.shape==(4,len(m)) and np.isfinite(gaps).all() and gaps.min()>=0
                    if name=='init_only':np.testing.assert_array_equal(gaps,saved['generated']);negative+=gaps.size
                    np.testing.assert_array_equal(gaps[:,m.event_index.le(5)],saved['generated'][:,m.event_index.le(5)])
                    subset=f[(f.anchor==anchor)&(f['head']==name)&(f.history==h)&(f.group=='normal')]
                    for support in ['phase_population','same_phase']:
                        for lo,hi in [(1,-1),(6,50),(100,200),(1024,-1)]:
                            mask=(m.event_index>=lo)&((m.event_index<hi) if hi!=-1 else True)
                            if support=='same_phase':mask &= m.real_phase==m.generated_phase
                            r=mask&(m.real_phase%2==0)
                            generated_phase=m.generated_phase if anchor=='generated' else m.real_phase
                            g=mask&(generated_phase%2==0)
                            if not r.any() or not g.any():continue
                            for i,ns in enumerate(NOISES):
                                row=subset[(subset.support==support)&(subset.start==lo)&(subset.end==hi)&(subset.noise==ns)]
                                assert len(row)==1
                                want=wasserstein_distance(np.log1p(m.real_gap[r]),np.log1p(gaps[i,g]))
                                error=abs(want-float(row.gap_w1.iloc[0]));assert error<2e-12,error
                                assert int(row.actual_events.iloc[0])==int(r.sum()) and int(row.sample_events.iloc[0])==int(g.sum())
                                checks.append(dict(seed=gs,anchor=anchor,head=name,history=h,support=support,start=lo,noise=ns,error=error))
    all_metrics=pd.concat(frames,ignore_index=True);all_metrics.to_csv(DOC/'ALL_METRICS.csv',index=False)
    keys=['anchor','head','history','support','start','end','group']
    means=all_metrics.groupby(keys,sort=False).mean(numeric_only=True).reset_index();means.to_csv(DOC/'MEAN_METRICS.csv',index=False)
    pd.concat(law,ignore_index=True).to_csv(DOC/'LAW_EFFECTS.csv',index=False)
    pd.concat(nll,ignore_index=True).to_csv(DOC/'CONDITIONAL_NLL.csv',index=False)
    # Differences remain paired by native path and sampling noise.
    mix=all_metrics[all_metrics['head']=='mixture_marginal']
    index=['generation_seed','noise','anchor','support','start','end','group']
    p=mix.pivot(index=index,columns='history',values='gap_w1').reset_index()
    base=all_metrics[(all_metrics['head']=='init_only')&(all_metrics.history=='generated')][index+['gap_w1']].rename(columns={'gap_w1':'init_only_same_conditions'})
    p=p.merge(base,on=index,validate='one_to_one')
    p['aligned_rescue']=p.generated-p.actual_aligned
    p['literal_rescue']=p.generated-p.actual_literal
    p['residual_excess']=p.generated-p.init_only_same_conditions
    p['fraction_excess_removed']=np.where(p.residual_excess>.005,p.aligned_rescue/p.residual_excess,np.nan)
    p.to_csv(DOC/'PAIRED_EFFECTS.csv',index=False)
    pd.DataFrame(checks).to_csv(DOC/'INDEPENDENT_CHECKS.csv',index=False)
    counts=[json.loads((OUT/f'capture_{gs}/COMPLETE.json').read_text()) for gs in SEEDS]
    write(DOC/'FINAL_AUDIT.json',dict(native_instrumentation_generations=2,native_rows=sum(c['rows'] for c in counts),
        conditional_resamples=96,conditional_time_values=sum(c['conditional_sample_values'] for c in complete),
        independently_recomputed_metrics=len(checks),max_metric_error=max(c['error'] for c in checks),
        independent_init_only_sample_checks=negative,prior_preserved_hashes=len(json.loads((OUT/'MANIFEST.json').read_text())['sha256']),
        new_fits=0,gpu_jobs=0,device='cpu',native_gpu_results_not_replaced=True,
        test_events_read=False,head_hashes={str(p):digest(p) for p in [HEAD,INITIAL]},jobs=complete))
    write(DOC/'MODEL_DECISION.json',dict(active='clock_prefix',comparison_anchor='init_only',
        model_changes=0,new_fits=0,automatic_promotion=False,commit=False,push=False,
        diagnostic_only=True,no_deployable_oracle_claim=True))
    verify()
    print(means[(means.support=='phase_population')&(means.group=='normal')&(means.start.isin([1,6,100,1024]))][
        ['anchor','head','history','start','end','gap_w1','personal_w1']].to_string(index=False),flush=True)
    print('AUDIT_COMPLETE',len(checks),flush=True)

if __name__=='__main__':main()
