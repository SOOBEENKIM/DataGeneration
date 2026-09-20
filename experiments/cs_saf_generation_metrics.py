"""Same 13 native metrics with immutable reference statistics cached per cell."""
from dataclasses import asdict
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from benchmarks.cof_seqgen_saf_metrics import (
    _validate, _stable_mark, _lag1, fit_metric_state)
from generators.cof_seqgen_saf_baselines import validate_raw_generated_events


def normalized_counts(counts):
    total=counts.sum(-1,keepdims=True)
    return np.divide(counts,total,out=np.zeros_like(counts,dtype=float),where=total>0)


def conditional_tv(reference,candidate):
    """Reference-weighted row TV; an absent candidate row has TV=1."""
    weight=reference.sum(-1)
    distance=.5*np.abs(normalized_counts(reference)-normalized_counts(candidate)).sum(-1)
    distance=np.where(candidate.sum(-1)==0,1.,distance)
    return float(np.sum(weight*distance)/weight.sum())


def mutual_information(table):
    p=table/table.sum();expected=np.outer(p.sum(1),p.sum(0));mask=p>0
    return float(np.sum(p[mask]*np.log(p[mask]/expected[mask])))


class CachedGroupMetrics:
    def __init__(self,train,reference):
        self.state=fit_metric_state(train)
        self.reference=self.sufficient_statistics(reference)
        self.reference_entities=reference.entity_id.nunique()

    def sufficient_statistics(self,events):
        f=_validate(events);state=self.state
        raw=f.receiver_or_mark.map(_stable_mark).to_numpy()
        mapping={m:i for i,m in enumerate(state.mark_groups)};n=len(mapping)+1
        marks=np.array([mapping.get(m,n-1) for m in raw],dtype=np.int64)
        gap=f.gap.to_numpy(float);mask=np.isfinite(gap)
        bins=np.searchsorted(state.gap_bin_edges,gap[mask],side='right');b=len(state.gap_bin_edges)+1
        transition=np.bincount((bins*n+np.roll(marks,1)[mask])*n+marks[mask],
                               minlength=b*n*n).reshape(b,n,n)
        repeated=(raw[mask]==np.roll(raw,1)[mask]).astype(int)
        repeat=np.bincount(2*bins+repeated,minlength=b*2).reshape(b,2)
        return dict(gap=gap[mask],amount=f.amount_or_numeric_value.dropna().to_numpy(float),
            lengths=f.groupby('entity_id').size().to_numpy(float),
            marks=np.bincount(marks,minlength=n),transition=transition,repeat=repeat,
            gap_acf=_lag1(f,'gap'),amount_acf=_lag1(f,'amount_or_numeric_value'))

    def score(self,events):
        a=self.reference;b=self.sufficient_statistics(events);s=self.state
        repeat_l1=float(np.sum(a['repeat'].sum(1)/a['repeat'].sum()*np.abs(
            normalized_counts(a['repeat'])[:,1]-normalized_counts(b['repeat'])[:,1])))
        scores=dict(
            gap_conditioned_mark_tv=conditional_tv(a['transition'].sum(1),b['transition'].sum(1)),
            transition_conditioned_mark_tv=conditional_tv(a['transition'],b['transition']),
            short_gap_repeat_curve_l1=repeat_l1,
            gap_repeat_mi_error=abs(mutual_information(a['repeat'])-mutual_information(b['repeat'])),
            gap_lag1_acf_error=abs(a['gap_acf']-b['gap_acf']),
            amount_lag1_acf_error=abs(a['amount_acf']-b['amount_acf']),
            gap_ks=float(ks_2samp(a['gap'],b['gap']).statistic),
            gap_scaled_w1=float(wasserstein_distance(a['gap'],b['gap'])/s.gap_scale),
            gap_zero_rate_error=abs(float(np.mean(a['gap']==0))-float(np.mean(b['gap']==0))),
            mark_sparse_tv=float(.5*np.abs(normalized_counts(a['marks'])-normalized_counts(b['marks'])).sum()),
            amount_ks=float(ks_2samp(a['amount'],b['amount']).statistic),
            amount_scaled_w1=float(wasserstein_distance(a['amount'],b['amount'])/s.amount_scale),
            length_scaled_w1=float(wasserstein_distance(a['lengths'],b['lengths'])/s.length_scale))
        if not np.isfinite(list(scores.values())).all():raise ValueError('nonfinite native score')
        return scores


class CachedGenerationMetrics:
    def __init__(self,dataset,plan):
        self.plan=plan;self.groups={};self.generated_ids={}
        train_ids=set(dataset.entity_ids_for_split('train'))
        val_ids=set(dataset.entity_ids_for_split('validation'))
        self.allowed_marks=set(dataset.events[dataset.events.entity_id.isin(train_ids)].receiver_or_mark)
        for name in ('pooled','context_0','context_1'):
            if name=='pooled':real_ids=set(dataset.static_context.entity_id);gen_ids=set(plan.entity_ids)
            else:
                label=int(name[-1])
                real_ids=set(dataset.static_context.loc[dataset.static_context.entity_label==label,'entity_id'])
                gen_ids=set(plan.static_context.loc[plan.static_context.entity_label==label,'entity_id'])
            train=dataset.events[dataset.events.entity_id.isin(real_ids&train_ids)]
            reference=dataset.events[dataset.events.entity_id.isin(real_ids&val_ids)]
            self.groups[name]=CachedGroupMetrics(train,reference);self.generated_ids[name]=gen_ids

    def score(self,generated):
        validate_raw_generated_events(generated,self.plan)
        if not set(generated.receiver_or_mark)<=self.allowed_marks:raise ValueError('unknown generated mark')
        return {name:dict(generated_entities=len(self.generated_ids[name]),
            validation_entities=int(group.reference_entities),train_metric_state=asdict(group.state),
            metrics=group.score(generated[generated.entity_id.isin(self.generated_ids[name])]))
            for name,group in self.groups.items()}
