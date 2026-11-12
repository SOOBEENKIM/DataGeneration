"""One train-only conditional log-mixture for each previous/current label pair."""
from contextlib import contextmanager
import numpy as np
import torch
from benchmarks.argn_onset_output import onset_output_generation


def sample_values(parameters,labels,rng):
    values=np.empty(len(labels));clipped=0
    for phase in range(4):
        ix=np.flatnonzero(labels==phase)
        if not len(ix):continue
        p=parameters['phases'][str(phase)];component=rng.choice(p['k'],len(ix),p=p['weights'])
        z=rng.normal(np.asarray(p['means'])[component],np.sqrt(np.asarray(p['variances'])[component]))
        raw=np.expm1(z);lo,hi=parameters['limits'];clipped+=int(((raw<lo)|(raw>hi)).sum());values[ix]=np.clip(raw,lo,hi)
    return values,clipped


def encode_values(values,stats,keys,cardinalities):
    units=np.floor(np.asarray(values)/10.**stats['min_decimal']+1e-8)*10.**stats['min_decimal'];tokens={}
    for key in keys:
        exponent=int(key.rsplit('__E',1)[1]);name=f'E{exponent}'
        value=(np.floor(units/10.**exponent+1e-8).astype(np.int64)%10)-stats['min_digits'][name]
        assert (value>=0).all() and (value<cardinalities[key]).all(),(key,value.min(),value.max())
        tokens[key]=value
    return tokens


@contextmanager
def phase_gap_generation(*args,phase_parameters,**kwargs):
    import mostlyai.engine._tabular.generation as generation
    audit=[]
    with onset_output_generation(*args,**kwargs):
        parent=generation.SequentialModel
        class PhaseModel(parent):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                s=args[0]['columns']['gap'];prefix=f"{s['argn_processor']}:{s['argn_table']}/{s['argn_column']}__"
                self.phase_keys=[k for k in self.tgt_cardinalities if k.startswith(prefix)]
                self.phase_stats=s;self.phase_rng=np.random.default_rng(torch.initial_seed());self.phase_clipped=0
                self.phase_audit=dict(clipped_values=0,sampled_values=0);audit.append(self.phase_audit)
                self.predictors.register_forward_hook(self.replace_phase)

            def replace_phase(self,module,a,out):
                key=a[1]
                if key not in self.phase_keys:return out
                if key==self.phase_keys[0]:
                    f=self._state_features;y=self._amount_labels
                    self.phase_mask=f[...,0].gt(.5)&f[...,2].le(.5)
                    pair=(2*f[...,1].gt(.5).long()+y).reshape(-1).cpu().numpy()
                    valid=self.phase_mask.reshape(-1).cpu().numpy();values=np.repeat(phase_parameters['limits'][0],len(pair))
                    sampled,n=sample_values(phase_parameters,pair[valid],self.phase_rng);values[valid]=sampled;self.phase_clipped+=n
                    self.phase_audit['clipped_values']+=n;self.phase_audit['sampled_values']+=int(valid.sum())
                    self.phase_tokens=encode_values(values,self.phase_stats,self.phase_keys,self.tgt_cardinalities)
                token=torch.as_tensor(self.phase_tokens[key],device=self.device).reshape(out.shape[:-1])
                logits=torch.full_like(out,-torch.inf);logits.scatter_(-1,token.unsqueeze(-1),0.)
                return torch.where(self.phase_mask.unsqueeze(-1),logits,out)
        generation.SequentialModel=PhaseModel
        try:yield audit
        finally:generation.SequentialModel=parent
