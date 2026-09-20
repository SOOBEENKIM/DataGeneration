"""External U/G ports with a common, explicitly changed context adapter.

U: latent copy + fresh-mark mixture. G: observable repeat + nonrepeat marks.
The single rank-32 route replaces the controlled two rank-16 banks. No new
regularizer, calibration, constraint, rollout objective, or nonrepeat gap route.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from models.cof_seqgen_saf import CoFSeqGenSAF, SAFModelConfig


class ExternalUG(CoFSeqGenSAF):
    def __init__(self, mode, support, *, context_window=32, rank=32, **kwargs):
        if mode not in ('U','G'):
            raise ValueError('external port supports U and G only')
        super().__init__(SAFModelConfig(candidate_id='SAF-U1', context_window=context_window, **kwargs),support)
        if self.config.auxiliary_numeric_dim:
            raise ValueError('v1 external view excludes auxiliary numeric fields explicitly')
        del self.current_auxiliary_categorical
        self.mode,self.rank=mode,rank
        self.calibration=None
        del self.copy_base_logit_head,self.copy_gap_gate_head,self.copy_gap_delta_head
        sdim=self.config.static_dim+sum(e.embedding_dim for e in self.encoder.static_categorical)
        self.external_static=nn.Linear(sdim,8) if sdim else None
        dim=self.config.hidden_dim+8
        self.copy_base=nn.Linear(dim,1)
        self.route_context=nn.Linear(dim,rank)
        self.route_gap=nn.Linear(self.config.gap_embedding_dim,rank,bias=False)
        self.interaction=nn.Parameter(torch.zeros(rank))
        self.new_mark_head=nn.Linear(dim,self.config.receiver_vocab_size)

    def context(self,hidden,static,static_categorical):
        if self.external_static is None:
            s=hidden.new_zeros(len(hidden),8)
        else:
            parts=[static] if self.config.static_dim else []
            parts.extend(e(v) for e,v in zip(self.encoder.static_categorical,static_categorical))
            s=self.external_static(torch.cat(parts,-1))
        return torch.cat([hidden,s],-1)

    def mark_log_probabilities(self,context,gap,previous,has_previous):
        logits=self.new_mark_head(context).clone()
        # UNK/MISSING retain probability in both scoring and sampling. Never
        # mask an unseen validation target; PAD alone is not an event.
        logits[:,0]=-torch.inf
        logfresh=logits.log_softmax(-1)
        u=torch.tanh(self.route_context(context))
        v=torch.tanh(self.route_gap(self.gap_route(self._support_code(gap))))
        r=self.copy_base(context).squeeze(-1)+(u*v*self.interaction).sum(-1)/math.sqrt(self.rank)
        yes,no=F.logsigmoid(r),F.logsigmoid(-r)
        same=F.one_hot(previous,self.config.receiver_vocab_size).bool()
        if self.mode=='U':
            fresh=no[:,None]+logfresh
            logp=torch.where(same,torch.logaddexp(fresh,yes[:,None]),fresh)
        else:
            other=logits.masked_fill(same,-torch.inf).log_softmax(-1)
            logp=torch.where(same,yes[:,None],no[:,None]+other)
        logp=torch.where(has_previous[:,None],logp,logfresh)
        if self.calibration is not None:
            edges=torch.as_tensor(self.calibration['edges'],device=gap.device,dtype=gap.dtype)
            code=torch.bucketize(torch.nan_to_num(gap),edges)+1
            code=torch.where(gap.eq(0),torch.zeros_like(code),code)
            beta=torch.as_tensor(self.calibration['beta'],device=gap.device,dtype=gap.dtype)[code]
            oldyes=logp.gather(1,previous[:,None]).squeeze(1)
            oldno=logp.masked_fill(same,-torch.inf).logsumexp(-1)
            shifted=oldyes-oldno+beta
            calibrated=torch.where(same,F.logsigmoid(shifted)[:,None],
                logp-oldno[:,None]+F.logsigmoid(-shifted)[:,None])
            logp=torch.where(has_previous[:,None],calibrated,logp)
        return logp

    def target_outputs(self,**batch):
        pos=batch['target_position']
        inp={k:v for k,v in batch.items() if k!='target_position'}
        hidden_all=self.encoder(**inp)
        row=torch.arange(len(pos),device=pos.device)
        h=hidden_all[row,pos]
        g=batch['gap'][row,pos]
        m=batch['receiver'][row,pos]
        val=batch['numeric_value'][row,pos]
        prev=batch['receiver'][row,(pos-1).clamp_min(0)]
        has=pos>0
        prev=torch.where(has,prev,torch.ones_like(prev))
        c=self.context(h,batch['static'],batch['static_categorical'])
        logmark=self.mark_log_probabilities(c,g,prev,has)
        mu,sd=self._value_parameters(h,g,m)
        core=self._event_core_features(h,g,m,val)
        auxiliary=tuple(x[row,pos] for x in batch['auxiliary_categorical'])
        auxlogits=[]
        for head in self.auxiliary_categorical_heads:
            logits=head(core)
            logits[:,0]=-torch.inf
            auxlogits.append(logits.log_softmax(-1))
        return dict(hidden=h,gap=g,mark=m,numeric=val,previous=prev,has_previous=has,
                    logmark=logmark,location=mu,scale=sd,auxiliary=auxiliary,auxlogits=tuple(auxlogits))

    def terms(self,**batch):
        o=self.target_outputs(**batch)
        gapmask=torch.isfinite(o['gap'])
        gc=(self._support_code(o['gap'])-3).clamp_min(0)
        gap=self.gap_decoder.nll(o['hidden'],gc)
        mark=-o['logmark'].gather(1,o['mark'][:,None]).squeeze(1)
        numeric=o['scale'].log()+.5*math.log(2*math.pi)+.5*((o['numeric']-o['location'])/o['scale']).square()
        result={'gap':(gap[gapmask].sum(),gapmask.sum()),'mark':(mark.sum(),mark.new_tensor(len(mark))),
                'amount':(numeric.sum(),numeric.new_tensor(len(numeric)))}
        for i,(logp,target) in enumerate(zip(o['auxlogits'],o['auxiliary'])):
            nll=-logp.gather(1,target[:,None]).squeeze(1)
            result[f'aux_{i}']=(nll.sum(),nll.new_tensor(len(nll)))
        return result,o

    def compute_loss(self,**batch):
        terms,_=self.terms(**batch)
        values={k:s/n.clamp_min(1) for k,(s,n) in terms.items()}
        return {'loss':sum(values.values()),**values}

    @torch.no_grad()
    def sample_fixed_lengths(self,lengths,*,static=None,static_categorical=(),device=None):
        if not lengths or min(lengths)<1:
            raise ValueError('positive lengths required')
        dev=device or next(self.parameters()).device
        lens=torch.as_tensor(lengths,device=dev)
        b,t=len(lengths),max(lengths)
        valid=torch.arange(t,device=dev)[None,:]<lens[:,None]
        g=torch.full((b,t),float('nan'),device=dev)
        m=torch.zeros((b,t),dtype=torch.long,device=dev)
        val=torch.zeros((b,t),device=dev)
        aux=tuple(torch.zeros_like(m) for _ in self.config.auxiliary_categorical_vocab_sizes)
        auxnum=torch.zeros(b,t,0,device=dev)
        static=static.to(dev) if static is not None else torch.zeros(b,0,device=dev)
        static_categorical=tuple(v.to(dev) for v in static_categorical)
        for step in range(t):
            active=valid[:,step]
            rows=active.nonzero().squeeze(1)
            start=max(0,step-self.config.context_window+1)
            mg=m[rows,start:step+1].clone(); mg[:,-1]=1
            ag=tuple(v[rows,start:step+1].clone() for v in aux)
            for v in ag:v[:,-1]=1
            h=self.encoder(g[rows,start:step+1],mg,val[rows,start:step+1],valid[rows,start:step+1],
                           static[rows],tuple(v[rows] for v in static_categorical),ag,
                           auxnum[rows,start:step+1])[:,-1]
            if step:
                g[rows,step]=self.support.decode_tensor(self.gap_decoder.sample(h))
            current=g[rows,step]
            prev=m[rows,step-1] if step else torch.ones(len(rows),dtype=torch.long,device=dev)
            c=self.context(h,static[rows],tuple(v[rows] for v in static_categorical))
            logp=self.mark_log_probabilities(c,current,prev,torch.full_like(prev,step>0,dtype=torch.bool))
            sampled=torch.distributions.Categorical(logits=logp).sample()
            m[rows,step]=sampled
            mu,sd=self._value_parameters(h,current,sampled)
            numeric=mu+sd*torch.randn_like(mu)
            val[rows,step]=numeric
            core=self._event_core_features(h,current,sampled,numeric)
            for head,output in zip(self.auxiliary_categorical_heads,aux):
                logits=head(core);logits[:,0]=-torch.inf
                output[rows,step]=torch.distributions.Categorical(logits=logits).sample()
        return dict(gap=g,receiver=m,numeric_value=val,auxiliary_categorical=aux,
                    auxiliary_numeric=auxnum,valid_mask=valid,lengths=lens)

    def architecture_contract(self):
        return dict(version='cs-saf-external-port-v1',mode=self.mode,
                    parameters=sum(p.numel() for p in self.parameters()),rank=self.rank,
                    history_window_including_target=self.config.context_window,
                    history='strict_past_recomputed_window',context_banks=1,
                    auxiliary_factorization='gap -> mark -> amount -> auxiliary category',
                    reserved_outputs='PAD masked; UNK/MISSING scored and sampled',
                    base_likelihood_only=True,old_controlled_implementation_modified=False)
