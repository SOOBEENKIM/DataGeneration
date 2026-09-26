"""Equivalent full-sequence PAR training without differentiable packing roundtrips.

Valid recurrent outputs and gradients are unchanged mathematically. Mask padded
outputs exactly as pad_packed_sequence does, including native loss conventions.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import textwrap
import torch
import torch.nn.functional as F
from experiments.cs_saf_cpar_loss import PINNED_PAR_SOURCE_SHA256

def dense_forward(model,x,context,lengths):
    if model.context_size:
        x=torch.cat([x,context.unsqueeze(0).expand(x.shape[0],-1,-1)],dim=2)
    y=model.down(x);y,_=model.rnn(y);y=model.up(y)
    mask=torch.arange(y.shape[0],device=y.device)[:,None]<torch.as_tensor(lengths,device=y.device)[None,:]
    return y*mask.unsqueeze(-1)

def sliced_native_loss(par,x_full,y,lengths,global_customers):
    """Same native loss, retaining GLOBAL padding alignment for numeric targets."""
    steps=y.shape[0];global_steps=x_full.shape[0]
    lengths=torch.as_tensor(lengths,device=y.device).clamp(max=global_steps)
    times=torch.arange(steps,device=y.device)[:,None];mask=times<lengths[None,:]
    source=(global_steps-lengths[None,:]+times).clamp(max=global_steps-1)
    total=y.new_zeros(())
    for props in par._data_map.values():
        if props['type'] in ('continuous','timestamp'):
            mu,sigma,missing=props['indices']
            target=x_full[:,:,mu].gather(0,source)
            value=torch.distributions.Normal(y[:,:,mu],F.softplus(y[:,:,sigma])).log_prob(target)
            log_missing=F.logsigmoid(y[:,:,missing]);truth=x_full[:steps,:,missing]
            terms=value+truth*log_missing+(1-truth)*torch.log(1-torch.exp(log_missing))
        elif props['type'] in ('categorical','ordinal'):
            ix=list(props['indices'].values())
            target=x_full[:steps,:,ix].argmax(-1,keepdim=True)
            terms=F.log_softmax(y[:,:,ix],dim=2).gather(2,target).squeeze(2)
        else:raise ValueError('Sliced execution only verified for categorical and continuous targets')
        total=total+terms.masked_select(mask).sum()
    return -total/(global_customers*len(par._data_map)*global_customers)

def full_batch_gradient(par,x,context,lengths,chunk_size=64):
    """Accumulate the original full-batch gradient; optimizer still steps once."""
    n=x.shape[1];total=x.new_zeros(())
    # Sorting reduces padding compute; recurrent state is reset for each customer.
    for indices in torch.argsort(lengths,descending=True).split(chunk_size):
        local_lengths=lengths[indices];local_max=int(local_lengths.max())
        local_x=x[:,indices];local_c=context[indices] if par._ctx_dims else context
        y=dense_forward(par._model,local_x[:local_max],local_c,local_lengths)
        loss=sliced_native_loss(par,local_x[1:],y[:min(local_max,x.shape[0]-1)],local_lengths,n)
        loss.backward();total+=loss.detach()
    return total

@contextmanager
def equivalent_microbatch_training(progress_callback=None):
    import deepecho.models.par as module
    cls=module.PARModel
    assert hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()==PINNED_PAR_SOURCE_SHA256
    original=cls.fit_sequences;source=textwrap.dedent(inspect.getsource(original))
    old='''Y = self._model(X, C)
        Y_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(Y)

        optimizer.zero_grad()
        loss = self._compute_loss(X_padded[1:, :, :], Y_padded[:-1, :, :], seq_len)
        loss.backward()'''
    assert source.count(old)==1
    source=source.replace(old,'''optimizer.zero_grad()
        loss = full_batch_gradient(self, X_padded, C, seq_len)
        print(f'CPAR completed gradient epoch={epoch+1} loss={float(loss):.9g}', flush=True)''')
    source=source.replace('optimizer.step()', 'optimizer.step()\n        if progress_callback is not None: progress_callback(self, optimizer, epoch+1)')
    namespace=dict(module.__dict__);namespace['full_batch_gradient']=full_batch_gradient;namespace['progress_callback']=progress_callback
    exec(compile(source,__file__,'exec'),namespace);cls.fit_sequences=namespace['fit_sequences']
    try:yield
    finally:cls.fit_sequences=original

@contextmanager
def equivalent_dense_training():
    import deepecho.models.par as module
    cls=module.PARModel
    assert hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()==PINNED_PAR_SOURCE_SHA256
    original=cls.fit_sequences;source=textwrap.dedent(inspect.getsource(original))
    old='Y = self._model(X, C)\n        Y_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(Y)'
    assert source.count(old)==1
    source=source.replace(old,'Y_padded = dense_forward(self._model, X_padded, C, seq_len)')
    namespace=dict(module.__dict__);namespace['dense_forward']=dense_forward
    exec(compile(source,__file__,'exec'),namespace);cls.fit_sequences=namespace['fit_sequences']
    try:yield
    finally:cls.fit_sequences=original
