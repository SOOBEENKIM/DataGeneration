"""Observed root-transition + emission + amount-resampling controls.

Root is operation for Berka, category for Sparkov. No copied trajectories,
oracle formulas, generated-output correction, static fraud labels or test data.
"""
import numpy as np
import pandas as pd
from benchmarks.cs_saf_external import gap_bin


class ObservedTransition:
    def __init__(self,name,edges,shrink=20.,use_history=True):
        self.name,self.edges,self.shrink,self.use_history=name,edges,shrink,use_history
        self.root='category' if name=='sparkov' else 'receiver_or_mark'
        self.other='receiver_or_mark' if name=='sparkov' else 'transaction_type'

    def encode(self,frame):
        x=frame.sort_values(['entity_id','event_index'],kind='stable').copy()
        z=x[self.root].fillna('<MISSING>').astype(str).map(self.rootmap).fillna(0).to_numpy(int)
        o=x[self.other].fillna('<MISSING>').astype(str).map(self.othermap).fillna(0).to_numpy(int)
        p=pd.Series(z).groupby(x.entity_id.to_numpy()).shift().fillna(0).to_numpy(int)
        g=gap_bin(x.gap,self.edges).clip(0)
        return x,z,o,p,g

    def fit(self,frame):
        self.roots=['<UNK>']+sorted(frame[self.root].fillna('<MISSING>').astype(str).unique())
        self.others=['<UNK>']+sorted(frame[self.other].fillna('<MISSING>').astype(str).unique())
        self.rootmap={v:i for i,v in enumerate(self.roots)};self.othermap={v:i for i,v in enumerate(self.others)}
        x,z,o,p,g=self.encode(frame)
        k,q,b=len(self.roots),len(self.others),len(self.edges)+2
        mask=x.event_index.gt(0).to_numpy()
        c0=np.bincount(z,minlength=k);self.p0=(c0+.5)/(len(z)+.5*k)
        cfirst=np.bincount(z[~mask],minlength=k);self.first=(cfirst+20*self.p0)/(cfirst.sum()+20)
        c1=np.bincount(p[mask]*k+z[mask],minlength=k*k).reshape(k,k)
        p1=(c1+self.shrink*self.p0)/(c1.sum(-1,keepdims=True)+self.shrink)
        c2=np.bincount((p[mask]*b+g[mask])*k+z[mask],minlength=k*b*k).reshape(k,b,k)
        self.p2=(c2+self.shrink*p1[:,None,:])/(c2.sum(-1,keepdims=True)+self.shrink)
        cm=np.bincount(z*q+o,minlength=k*q).reshape(k,q)
        marginal=(cm.sum(0)+.5)/(len(z)+.5*q)
        self.emission=(cm+20*marginal)/(cm.sum(-1,keepdims=True)+20)
        self.gaps=x.loc[x.event_index.gt(0),'gap'].to_numpy()
        self.amounts=x.amount_or_numeric_value.to_numpy()
        self.pools={key:np.asarray(ix) for key,ix in pd.DataFrame({'z':z,'o':o}).groupby(['z','o']).indices.items()}
        self.rootpools={key:np.asarray(ix) for key,ix in pd.DataFrame({'z':z}).groupby('z').indices.items()}
        return self

    def nll(self,frame):
        x,z,o,p,g=self.encode(frame);valid=x.event_index.gt(0).to_numpy()
        probs=self.p2[p,g,z] if self.use_history else self.p0[z]
        return float((-np.log(probs[valid])-np.log(self.emission[z[valid],o[valid]])).mean())

    def sample(self,plan,seed):
        rng=np.random.default_rng(seed);records=[]
        for eid,length in zip(plan.entity_id,plan.length):
            previous=0;timestamp=0.
            for t in range(int(length)):
                gap=float(rng.choice(self.gaps)) if t else np.nan
                if t:timestamp+=gap
                g=int(gap_bin([gap],self.edges)[0]) if t else 0
                probability=self.first if not t else (self.p2[previous,g] if self.use_history else self.p0)
                z=int(rng.choice(len(self.roots),p=probability));o=int(rng.choice(len(self.others),p=self.emission[z]))
                pool=self.pools.get((z,o),self.rootpools.get(z))
                amount=float(self.amounts[int(rng.choice(pool))] if pool is not None else rng.choice(self.amounts))
                records.append(dict(entity_id=eid,event_index=t,timestamp=timestamp,gap=gap,
                    **{self.root:self.roots[z],self.other:self.others[o]},amount_or_numeric_value=amount))
                previous=z
        return pd.DataFrame(records)
