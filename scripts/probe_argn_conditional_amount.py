"""Sample only amount from real histories/current fields; no training."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from audit_argn_fit_generalization import inputs,folder,ROOT,OUT,DOCS
from run_argn_state_first import seed,digest,write
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities,get_ctx_sequence_length
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.generation import _fix_rare_token_probs,_translate_fixed_probs
from benchmarks.argn_label_first_control import label_first_model_class
from benchmarks.argn_state_adapter import FullHistoryCollator
from benchmarks.argn_frozen_probe import teacher_with_history

A=OUT/'amount_probe';D=DOCS/'amount_probe'


def prepare():
    assert not (A/'MANIFEST.json').exists()
    A.mkdir(exist_ok=True);D.mkdir(exist_ok=True)
    paths=[ROOT/'scripts/probe_argn_conditional_amount.py',DOCS/'AMOUNT_PROBE_PROTOCOL.md']
    paths += [Workspace(folder('B_event_label_first',fs)/'workspace').model_tabular_weights_path for fs in [20260930,20261001]]
    write(A/'MANIFEST.json',dict(hashes={str(p):digest(p) for p in paths},selection_seed=20260927,
        sampling_seeds=[20261101,20261102,20261103,20261104],test_events_read=False))


@torch.no_grad()
def worker(fs,device):
    start=time.monotonic();seed(fs)
    manifest=json.loads((A/'MANIFEST.json').read_text())
    for p,h in manifest['hashes'].items():assert digest(p)==h
    shared,codec,frames,metas=inputs();dev=torch.device(device)
    ws=Workspace(folder('B_event_label_first',fs)/'workspace');ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read()
    model=label_first_model_class(ts)(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
        tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
        ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),model_size=ws.model_tabular_configs.read()['model_units'],
        column_order=None,device=dev)
    load_model_weights(model=model,path=ws.model_tabular_weights_path,device=dev);model.to(dev).eval()
    collator=FullHistoryCollator(True,None,dev);key=codec.prefixes['event_is_fraud']+'__cat'
    amount_prefix=codec.prefixes['amount_or_numeric_value']+'__'
    masks=_translate_fixed_probs(_fix_rare_token_probs(ts),ts);results=[];selections={};equivalence_checks=0
    first_amount=next(k for k in model.tgt_cardinalities if k.startswith(amount_prefix))
    for split in ['optimization','development']:
        records=frames[split];meta=metas[split].copy();rng=np.random.default_rng(20260927)
        nf=int(meta.label.sum());selected=np.r_[meta.index[meta.label.eq(1)],rng.choice(meta.index[meta.label.eq(0)],nf,replace=False)]
        meta['selected']=meta.index.isin(selected);selections[split]=meta.loc[meta.selected,['record','event_index','label']].to_dict('list')
        rows=[]
        for lo in range(0,len(records),2):
            batch=collator(records[lo:lo+2]);logits,hist,ctx=teacher_with_history(model,batch)
            loss=sum(-v.log_softmax(-1).gather(-1,batch[k]).squeeze(-1) for k,v in logits.items() if k.startswith(amount_prefix))
            for b in range(len(records[lo:lo+2])):
                m=meta[meta.record.eq(lo+b)&meta.selected];ix=m.event_index.to_numpy(dtype=int)
                if not len(ix):continue
                n=len(ix);hh=hist[b,ix].unsqueeze(1)
                cx=[[v[b:b+1].expand(n,*v.shape[1:]) for v in group] for group in ctx]
                fixed={k:v[b,ix] for k,v in batch.items() if k in model.tgt_cardinalities and not k.startswith(amount_prefix)}
                real=codec.numeric({k:v[b,ix].cpu().numpy() for k,v in batch.items() if k.startswith(amount_prefix)},'amount_or_numeric_value')
                lstm=model.history_compressor.get();recurrent=tuple(torch.zeros(lstm.num_layers,n,lstm.hidden_size,device=dev) for _ in range(2))
                # S is disabled in this parent: cached history supplies all actual past information.
                memory=torch.zeros(1,n,9,dtype=torch.float64,device=dev)
                for case in ['observed_label','flipped_label']:
                    values=dict(fixed)
                    if case=='flipped_label':values[key]=torch.where(fixed[key].eq(codec.codes['1']),codec.codes['0'],codec.codes['1'])
                    for gs in manifest['sampling_seeds']:
                        seed(gs+(lo+b)*10)
                        captured={}
                        def capture(module,args,out):
                            if args[1]==first_amount:captured['first']=out
                        hook=model.predictors.register_forward_hook(capture)
                        try:
                            output,_,_=model(None,mode='gen',batch_size=n,history=hh,history_state=(*recurrent,memory),
                                context=cx,fixed_values=values,fixed_probs=masks)
                        finally:hook.remove()
                        if case=='observed_label' and gs==manifest['sampling_seeds'][0]:
                            torch.testing.assert_close(captured['first'][:,0],logits[first_amount][b,ix],rtol=1e-4,atol=2e-5)
                            equivalence_checks+=n
                        generated=codec.numeric({k:v.cpu().numpy() for k,v in output.items()},'amount_or_numeric_value')
                        row=m[['record','event_index','label']].copy();row['case']=case;row['sampling_seed']=gs
                        row['real_amount']=real;row['generated_amount']=generated;row['teacher_digit_nll']=loss[b,ix].cpu().numpy();rows.append(row)
        data=pd.concat(rows,ignore_index=True);data.to_parquet(A/f'amount_{fs}_{split}.parquet',index=False)
        for (case,label),g in data.groupby(['case','label']):
            actual=g.real_amount.to_numpy();synthetic=g.generated_amount.to_numpy()
            results.append(dict(fit_seed=fs,split=split,case=case,label=int(label),positions=len(g)//4,customers=g.record.nunique(),
                real_median=float(np.median(actual)),generated_median=float(np.median(synthetic)),
                generated_p90=float(np.quantile(synthetic,.9)),real_p90=float(np.quantile(actual,.9)),
                log_w1=float(wasserstein_distance(np.log1p(actual),np.log1p(synthetic))),
                teacher_digit_nll=float(g.teacher_digit_nll.mean()) if case=='observed_label' else None))
        pd.DataFrame(results).to_csv(D/f'metrics_{fs}.csv',index=False)
        print('AMOUNT_PROBE',fs,split,'seconds',round(time.monotonic()-start,2),flush=True)
    write(A/f'SELECTIONS_{fs}.json',selections)
    for p,h in manifest['hashes'].items():assert digest(p)==h
    write(D/f'COMPLETE_{fs}.json',dict(seconds=time.monotonic()-start,selection_sha256=digest(A/f'SELECTIONS_{fs}.json'),
        rows={s:digest(A/f'amount_{fs}_{s}.parquet') for s in ['optimization','development']},
        first_digit_teacher_generation_equivalence_checks=equivalence_checks,weights_unchanged=True,test_events_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=[20260930,20261001]);p.add_argument('--device',default='cpu')
    a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
