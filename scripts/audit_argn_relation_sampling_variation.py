"""Descriptive real-real customer resampling reference, not a population bound."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from run_argn_state_first import ROOT, SOURCE, digest, write
from evaluate_sparkov_argn_control import extended

OUT=ROOT/'docs/research_reaudit_20260927'


def main():
    seed=20260927;repeats=500
    paths=[SOURCE/'prepared/validation.parquet',SOURCE/'prepared/metric_state.json']
    manifest=dict(seed=seed,repeats=repeats,hashes={str(p):digest(p) for p in paths},
        description='Two independent customer bootstrap samples from observed development customers. '
        'Features calculated before whole-customer resampling. Descriptive finite-sample reference, '
        'not a bound, null significance test or validation-independent model selection.',test_events_read=False)
    assert not (OUT/'relation_sampling_variation_manifest.json').exists()
    write(OUT/'relation_sampling_variation_manifest.json',manifest)
    d=extended(pd.read_parquet(paths[0]),json.loads(paths[1].read_text()))
    ids=pd.Index(d.entity_id.unique());rng=np.random.default_rng(seed)
    wa,wb=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=(2,repeats))
    f=d[d.fraud.eq(1)&d.event_index.gt(0)].copy()
    cols=['gap_bin','category','amount_bin','seen_merchant','amount_vs_history']
    codes,cells=pd.factorize(pd.MultiIndex.from_frame(f[cols]),sort=True)
    counts=np.zeros((len(ids),len(cells)))
    np.add.at(counts,(ids.get_indexer(f.entity_id),codes),1)
    a,b=wa@counts,wb@counts
    tv=.5*np.abs(a/a.sum(1)[:,None]-b/b.sum(1)[:,None]).sum(1)
    f=d[d.fraud.eq(1)&d.event_index.ge(5)&np.isfinite(d.amount_history_ratio_raw)&d.amount_history_ratio_raw.ge(0)]
    values=np.log1p(f.amount_history_ratio_raw.to_numpy());ix=ids.get_indexer(f.entity_id)
    w1=[wasserstein_distance(values,values,u_weights=a[ix],v_weights=b[ix]) for a,b in zip(wa,wb)]
    pd.DataFrame({'repeat':np.arange(repeats),'fraud_relation_tv':tv,'fraud_personal_amount_log_w1':w1}).to_csv(OUT/'relation_sampling_variation_draws.csv',index=False)
    result=[]
    for name,x in [('fraud_relation_tv',tv),('fraud_personal_amount_log_w1',w1)]:
        result.append(dict(metric=name,median=float(np.median(x)),q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975)),repeats=repeats))
    pd.DataFrame(result).to_csv(OUT/'relation_sampling_variation_summary.csv',index=False)
    print(result)


if __name__=='__main__':main()
