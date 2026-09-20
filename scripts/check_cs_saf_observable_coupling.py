"""CPU numerical feasibility only; not training or a claim of model quality."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.cs_saf_observable_coupling import observable_gap_repeat_joint
from scripts.run_cs_saf_external_audit_v1 import write


def main():
    torch.set_num_threads(1);torch.manual_seed(20260920)
    values=[torch.randn(2,4,dtype=torch.float64,requires_grad=True),
            torch.randn(2,dtype=torch.float64,requires_grad=True),
            torch.randn(2,4,dtype=torch.float64,requires_grad=True)]
    assert torch.autograd.gradcheck(observable_gap_repeat_joint,tuple(values),eps=1e-6,atol=1e-5,rtol=1e-4)
    joint=observable_gap_repeat_joint(*values)
    a=values[0].softmax(-1);rho=values[1].sigmoid()
    row=float((joint.sum(-1)-a).abs().max().detach())
    column=float((joint[...,1].sum(-1)-rho).abs().max().detach())
    zero=observable_gap_repeat_joint(values[0],values[1],torch.zeros_like(values[2]))
    independent=a[...,None]*torch.stack((1-rho,rho),-1)[...,None,:]
    null=float((zero-independent).abs().max().detach())
    shifted=observable_gap_repeat_joint(values[0],values[1],values[2]+9)
    shift=float((shifted-joint).abs().max().detach())
    assert max(row,column,null,shift)<1e-12
    real_size=[torch.randn(8,31,requires_grad=True),torch.randn(8,requires_grad=True),torch.randn(8,31,requires_grad=True)]
    j32=observable_gap_repeat_joint(*real_size)
    real_error=float((j32[...,1].sum(-1)-real_size[1].sigmoid()).abs().max().detach())
    (-j32.clamp_min(1e-12).log().mean()).backward()
    assert real_error<1e-6 and all(torch.isfinite(v.grad).all() for v in real_size)
    report=dict(first_order_gradient_check=True,row_marginal_max_error=row,
        repeat_marginal_max_error=column,zero_score_independence_max_error=null,
        common_score_shift_max_error=shift,neural_training_performed=False,
        float32_31_gap_repeat_error=real_error,float32_gradient_finite=True,
        final_architecture_adopted=False,novel_algorithm_claim=False,
        scope='known conditional coupling component; not a complete trained generator')
    write(ROOT/'docs/cs_saf/external_audit_v1/observable_coupling_cpu_check.json',report)
    print(report)


if __name__=='__main__':main()
