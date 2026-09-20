"""Nested sampling summaries; five trials never become 25 trained models."""
import numpy as np
from scipy.stats import t
from experiments.cs_saf_replication import seed_statistics


def repeat_statistics(values):
    a=np.asarray(values,dtype=np.float64)
    if a.shape!=(5,5) or not np.isfinite(a).all():
        raise ValueError('five fixed training trials by five fresh tapes required')
    means=a.mean(axis=1);within=a.var(axis=1,ddof=1)
    result=seed_statistics(means)
    j,r=a.shape;variance_terms=within/r
    mc_se=float(np.sqrt(variance_terms.sum())/j)
    if variance_terms.sum()>0:
        df=float(variance_terms.sum()**2/np.sum(variance_terms**2/(r-1)))
        radius=float(t.ppf(.975,df)*mc_se)
    else:df=None;radius=0.
    between_observed=float(means.var(ddof=1));w=float(within.mean())
    between_raw=between_observed-w/r
    return dict(result,tape_values=a.tolist(),within_trial_SD=np.sqrt(within).tolist(),
        trial_tape_ranges=np.stack([a.min(1),a.max(1)],axis=1).tolist(),
        positive_count=int((means>0).sum()),conditional_MC_standard_error=mc_se,
        conditional_MC_degrees_of_freedom=df,
        conditional_MC_95_percent_interval=[result['mean']-radius,result['mean']+radius],
        conditional_MC_interval_excludes_training_and_data_uncertainty=True,
        mean_within_tape_variance=w,observed_between_trial_mean_variance=between_observed,
        expected_MC_variance_in_trial_mean=w/r,
        between_trial_variance_raw=between_raw,between_trial_variance_nonnegative=max(0.,between_raw),
        between_trial_variance_truncated=bool(between_raw<0),
        between_trial_includes_fixed_plan_differences=True)
