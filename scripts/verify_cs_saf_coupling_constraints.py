"""Two exact limitations of a frozen-marginal coupling candidate, not model fits."""
import json
from pathlib import Path
import numpy as np


def sinkhorn(a,b,score):
    kernel=np.outer(a,b)*np.exp(score)
    value=kernel.copy()
    for _ in range(10000):
        value*= (a/value.sum(1))[:,None]
        value*= (b/value.sum(0))[None,:]
        if max(np.max(abs(value.sum(1)-a)),np.max(abs(value.sum(0)-b)))<1e-12:
            return value
    raise RuntimeError('scaling did not converge')


def main():
    # Arbitrary illustrative probabilities, not estimates of an actual fitted
    # model's marginal after integrating over its predicted gap distribution.
    a=np.array([.5,.5]); b=np.array([.94,.06]); target_repeat=.31
    cases=[]
    for strength in [-5.,0.,5.]:
        joint=sinkhorn(a,b,strength*np.array([[1.,-1.],[-1.,1.]]))
        repeat_given_gap=joint[:,1]/a
        integrated=float(a@repeat_given_gap)
        assert abs(integrated-b[1])<1e-12
        cases.append(dict(strength=strength,conditional_repeat=repeat_given_gap.tolist(),
                          integrated_repeat=integrated,level_error=abs(target_repeat-integrated)))
    old=np.array([[.25,.25],[.25,.25]])
    new=np.array([[.45,.05],[.05,.45]])
    np.testing.assert_allclose(old.sum(0),new.sum(0))
    np.testing.assert_allclose(old.sum(1),new.sum(1))
    next_gap=np.array([[.9,.1],[.1,.9]])
    p0=float((old*next_gap).sum());p1=float((new*next_gap).sum())
    assert abs(p0-.5)<1e-12 and abs(p1-.82)<1e-12
    report=dict(role='exact_toy_counterexamples_not_neural_success_or_novel_theorems',
        fixed_wrong_repeat_marginal_cannot_be_corrected_by_coupling=cases,
        local_marginals_do_not_preserve_future_marginals={'before':p0,'after':p1},
        decision='do_not_freeze_uncalibrated_history_conditioned_marginals_or_claim_global_preservation')
    out=Path(__file__).resolve().parents[1]/'docs/cs_saf/external_audit_v1/coupling_feasibility_limits.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
