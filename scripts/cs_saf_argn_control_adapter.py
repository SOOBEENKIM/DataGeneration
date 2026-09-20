"""Scoped instrumentation of pinned official sampling; no on-disk library edits.

The official generation loop supplies aligned previous marks/context immediately
before each step. Only mark sampling probabilities change. The sampled mark is
then used by the unchanged numeric predictor and recurrent history update.
"""
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import numpy as np
import torch

from models.cs_saf_observed_repeat_control import apply_torch, redistribute


def ordered_gap_code(code, stats):
    """Native numeric special MAX is code 2, not the second-smallest gap."""
    codes=stats['columns']['gap']['codes'];maximum=stats['columns']['gap']['cardinalities']['bin']
    result=np.asarray(code,dtype=np.int64).copy()
    result[np.asarray(code)==codes['<<MAX>>']]=maximum
    result[np.asarray(code)==codes['<<MIN>>']]=-1
    return result


class SamplingControl:
    def __init__(self, control, stats):
        self.control=control;self.stats=stats;self.column=None;self.steps=0;self.corrected=0
        self.previous=None;self.group=None;self.gap=None;self.max_probability_sum_error=0.

    def before_step(self, out_df, step, keys, context, primary_key):
        self.steps+=1;self.gap=None
        self.previous=None if step==0 else out_df['tgt:t1/c1__cat'].to_numpy(np.int64)
        self.group=context.set_index(primary_key).loc[keys.to_numpy(),'entity_label'].astype(int).to_numpy()
        assert set(np.unique(self.group)) <= {0,1}
        assert self.previous is None or len(self.previous)==len(self.group)

    def sample(self, native, probs, temperature=None, top_p=None, fixed_probs=None):
        if self.column=='tgt:t1/c1__cat' and self.previous is not None and self.control is not None:
            assert temperature in (None,1,1.) and top_p in (None,1,1.)
            assert self.gap is not None and len(self.gap)==len(self.previous)==len(probs)
            # Same rare-token suppression as the official sampler and replay.
            p=probs.clone();p[:,0]=0;p=p/p.sum(-1,keepdim=True)
            previous=torch.as_tensor(self.previous,device=p.device,dtype=torch.long)
            group=torch.as_tensor(self.group,device=p.device,dtype=torch.long)
            gap=torch.as_tensor(ordered_gap_code(self.gap,self.stats),device=p.device,dtype=torch.long)
            original=p.gather(1,previous[:,None])[:,0]
            new=apply_torch(original,gap,group,self.control)
            probs=redistribute(p,previous,new)
            error=float((probs.sum(-1)-1).abs().max())
            self.max_probability_sum_error=max(self.max_probability_sum_error,error)
            assert error<1e-5 and torch.isfinite(probs).all() and (probs>=0).all()
            self.corrected+=len(probs)
        sampled=native(probs=probs,temperature=temperature,top_p=top_p,fixed_probs=fixed_probs)
        if self.column=='tgt:t0/c0__bin':
            self.gap=sampled.detach().cpu().numpy().reshape(-1)
        return sampled


@contextmanager
def generation_adapter(control, stats):
    generation=importlib.import_module('mostlyai.engine._tabular.generation')
    argn=importlib.import_module('mostlyai.engine._tabular.argn')
    source=inspect.getsource(generation.generate)
    needle='                    out_dct, history, history_state = model('
    assert source.count(needle)==1, 'pinned official generation source changed'
    insertion=('                    _stage1_control.before_step(out_df, seq_step, step_ctx_keys, '
               'ctx_batch, ctx_primary_key)\n')
    state=SamplingControl(control,stats)
    namespace=dict(generation.__dict__,_stage1_control=state)
    exec(compile(source.replace(needle,insertion+needle),'<registered-argn-sampling-adapter>','exec'),namespace)
    native_sample=argn._sample;native_predict=argn.Predictors.forward
    def predictor(module,x,sub_col):
        state.column=sub_col
        return native_predict(module,x,sub_col)
    def sampler(probs,temperature=None,top_p=None,fixed_probs=None):
        return state.sample(native_sample,probs,temperature,top_p,fixed_probs)
    try:
        argn.Predictors.forward=predictor;argn._sample=sampler
        yield namespace['generate'],state,hashlib.sha256(source.encode()).hexdigest()
    finally:
        argn.Predictors.forward=native_predict;argn._sample=native_sample
