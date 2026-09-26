"""Pinned PAR recurrent-cache and conversion acceleration, with native semantics."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import numpy as np
import torch
from experiments.cs_saf_cpar_loss import PINNED_PAR_SOURCE_SHA256

def cached_sequence(par,context,min_length,max_length):
    model=par._model;current=torch.zeros(1,1,par._data_dims,device=par.device)
    current[0,0,par._data_map['<TOKEN>']['indices']['<START>']]=1.
    hidden=None;parts=[];log_likelihood=0.
    for step in range(max_length):
        inp=torch.cat([current,context.unsqueeze(0)],dim=2) if model.context_size else current
        state,hidden=model.rnn(model.down(inp),hidden)
        next_x,ll=par._sample_state(model.up(state));log_likelihood+=ll
        # Native torch.cat copies next_x BEFORE its later premature-END rewrite.
        # Preserve that convention: the stored next input remains the sampled END.
        current=next_x.clone();parts.append(current)
        if next_x[0,0,par._data_map['<TOKEN>']['indices']['<END>']]>0.:
            if min_length<=step+1<=max_length:break
            next_x[0,0,par._data_map['<TOKEN>']['indices']['<BODY>']]=1.
            next_x[0,0,par._data_map['<TOKEN>']['indices']['<END>']]=0.
    return torch.cat(parts,dim=0),log_likelihood

def vectorized_decode(par,x):
    assert x.shape[1]==1
    a=x[:,0].detach().cpu().numpy().astype(np.float64)
    result=[None]*(len(par._data_map)-1)
    for key,props in par._data_map.items():
        if key=='<TOKEN>':continue
        kind=props['type']
        if kind in ('continuous','datetime','count'):
            value,_,missing=props['indices']
            decoded=a[:,value]*(props['range'] if kind=='count' else props['std'])+(props['min'] if kind=='count' else props['mu'])
            values=decoded.astype(np.int64).tolist() if kind=='count' else decoded.tolist()
            if props['nulls']:values=[None if m>0 else v for v,m in zip(values,a[:,missing])]
        elif kind in ('categorical','ordinal'):
            values=list(props['indices']);indices=list(props['indices'].values())
            values=[values[int(i)] for i in a[:,indices].argmax(1)]
        else:raise ValueError(kind)
        result[key]=values
    return result

@contextmanager
def equivalent_sampling():
    from deepecho.models.par import PARModel
    assert hashlib.sha256(Path(inspect.getfile(PARModel)).read_bytes()).hexdigest()==PINNED_PAR_SOURCE_SHA256
    original_sample=PARModel._sample_sequence;original_decode=PARModel._tensor_to_data
    PARModel._sample_sequence=cached_sequence;PARModel._tensor_to_data=vectorized_decode
    try:yield
    finally:PARModel._sample_sequence=original_sample;PARModel._tensor_to_data=original_decode
