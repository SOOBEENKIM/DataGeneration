"""Environment-neutral evaluation of frozen observable-repeat controls."""
import numpy as np
from models.cs_saf_observed_repeat_control import apply_numpy


def conditional_metrics(train,val,controls):
    result=[]
    for kind,control in [('raw',None)]+list(controls.items()):
        p=apply_numpy(val.p.to_numpy(),val.gap_code.to_numpy(),val.group.to_numpy(),control)
        old=val.p.to_numpy().clip(1e-9,1-1e-9);y=val.y.to_numpy()
        obs=val.obs_p.to_numpy() if kind=='raw' else np.where(y==1,p,val.obs_p.to_numpy()*(1-p)/(1-old))
        for group in ('pooled',0,1):
            ti=np.ones(len(train),bool) if group=='pooled' else train.group.to_numpy()==group
            vi=np.ones(len(val),bool) if group=='pooled' else val.group.to_numpy()==group
            mean=float(train.y.to_numpy()[ti].mean())
            edges=np.quantile(train.gap.to_numpy()[ti],[.2,.4,.6,.8])
            bins=np.searchsorted(edges,val.gap.to_numpy()[vi],side='right')
            count=np.bincount(bins,minlength=5);assert (count>0).all()
            expected=np.bincount(bins,weights=p[vi],minlength=5)/count
            observed=np.bincount(bins,weights=y[vi],minlength=5)/count
            result.append(dict(variant=kind,group=str(group),observed_repeat=float(y[vi].mean()),
                predicted_repeat=float(p[vi].mean()),repeat_mean_bias=float(abs(p[vi].mean()-y[vi].mean())),
                repeat_brier=float(np.mean((p[vi]-y[vi])**2)),
                train_mean_brier=float(np.mean((mean-y[vi])**2)),
                mark_nll=float(-np.log(obs[vi].clip(1e-12,1)).mean()),
                uniform_mark_nll=float(np.log(64)),curve_l1=float(np.sum(count*abs(expected-observed))/count.sum())))
    return result
