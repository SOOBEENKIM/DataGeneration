"""Separate generated current-field effects from generated-history effects."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch

from run_argn_amount_learning import (ROOT, OUT as STUDY, DOCS as STUDY_DOCS, CFG,
    inputs, folder, seed, digest, write, AmountHeads, amount_generation, Workspace,
    get_cardinalities, get_ctx_sequence_length, load_model_weights,
    _translate_fixed_probs, _fix_rare_token_probs)

OUT=STUDY/'condition_probe';DOCS=STUDY_DOCS/'condition_probe'
CASES={
    'all_current_true':['category','gap','receiver_or_mark'],
    'all_current_sampled':[],
    'category_true':['category'],
    'gap_true':['gap'],
    'merchant_true':['receiver_or_mark'],
    'category_merchant_true':['category','receiver_or_mark'],
}


def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),DOCS/'PROTOCOL.md',ROOT/'benchmarks/argn_amount_control.py',
           STUDY/'MANIFEST.json']
    for fs in CFG['fit_seeds']:
        paths.extend([STUDY/f'runs/balanced_shared_{fs}/amount_head.pt',
                      STUDY/f'worker_{fs}/development_cache.pt',
                      Workspace(folder('B_event_label_first',fs)/'workspace').model_tabular_weights_path])
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=CASES,arm='balanced_shared',
        generation_seeds=CFG['sampling_seeds'],batch_size=768,split='development',
        hashes={str(p):digest(p) for p in paths},test_events_read=False)
    write(OUT/'MANIFEST.json',record);write(DOCS/'MANIFEST.json',record)


@torch.no_grad()
def worker(fs,device):
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for p,h in manifest['hashes'].items():assert digest(p)==h
    started=time.monotonic();seed(fs);device=torch.device(device)
    shared,codec,frames,metas=inputs();records=frames['development'];meta=metas['development']
    cache=torch.load(STUDY/f'worker_{fs}/development_cache.pt',map_location='cpu',weights_only=True)
    ws=Workspace(folder('B_event_label_first',fs)/'workspace');ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read()
    params=json.loads((ROOT/'artifacts/research_reaudit_20260927/simple_label_parameters.json').read_text())
    import mostlyai.engine._tabular.generation as generation
    with amount_generation(ts,params,STUDY/f'runs/balanced_shared_{fs}/amount_head.pt'):
        model=generation.SequentialModel(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
            tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
            ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),model_size=ws.model_tabular_configs.read()['model_units'],
            column_order=None,device=device)
        load_model_weights(model=model,path=ws.model_tabular_weights_path,device=device);model.to(device).eval()
    assert not model.context_compressor.ctxseq_cardinalities
    context_dim=model.context_compressor.dim_output;history_dim=model.history_dim
    amount_keys=model.amount_heads.keys
    encoded={k:torch.as_tensor(np.concatenate([np.asarray(r[k]).reshape(-1) for r in records]),device=device).reshape(-1,1)
             for k in model.tgt_cardinalities}
    rng=np.random.default_rng(CFG['selection_seed']);fraud=np.flatnonzero(meta.label.to_numpy()==1)
    selected=np.sort(np.r_[fraud,rng.choice(np.flatnonzero(meta.label.to_numpy()==0),len(fraud),replace=False)])
    base=cache['base'][selected].to(device);truth=cache['targets'][selected].numpy()
    real=codec.numeric({k:truth[:,j] for j,k in enumerate(amount_keys)},'amount_or_numeric_value')
    masks=_translate_fixed_probs(_fix_rare_token_probs(ts),ts);rows=[];checks=0
    label_key=codec.prefixes['event_is_fraud']+'__cat'
    structural=[k for k in encoded if k.startswith('tgt:/')]
    for case,fields in CASES.items():
        fixed_keys=structural+[label_key]
        for field in fields:fixed_keys.extend(k for k in encoded if k.startswith(codec.prefixes[field]+'__'))
        for gs in CFG['sampling_seeds']:
            seed(gs)
            for lo in range(0,len(selected),manifest['batch_size']):
                chosen=selected[lo:lo+manifest['batch_size']];n=len(chosen)
                b=base[lo:lo+n];hh=b[:,context_dim:context_dim+history_dim].unsqueeze(1)
                context=([b[:,:context_dim]],[],[])
                lstm=model.history_compressor.get()
                zero=tuple(torch.zeros(lstm.num_layers,n,lstm.hidden_size,device=device) for _ in range(2))
                memory=torch.zeros(1,n,9,dtype=torch.float64,device=device)
                captured={}
                def capture(module,args,out):
                    if args[1]==amount_keys[0]:captured['first']=out[:,0]
                handle=model.predictors.register_forward_hook(capture)
                try:
                    output,_,_=model(None,mode='gen',batch_size=n,history=hh,
                        history_state=(*zero,memory),context=context,
                        fixed_values={k:encoded[k][chosen] for k in fixed_keys},fixed_probs=masks)
                finally:handle.remove()
                if case=='all_current_true' and gs==CFG['sampling_seeds'][0]:
                    labels=encoded[label_key][chosen,0].eq(codec.codes['1']).long()
                    expected=model.amount_heads(b,amount_keys[0],labels)
                    torch.testing.assert_close(captured['first'],expected,rtol=1e-4,atol=2e-5);checks+=n
                generated=codec.numeric({k:v.cpu().numpy() for k,v in output.items()},'amount_or_numeric_value')
                part=meta.iloc[chosen][['record','event_index','label']].copy()
                part['case']=case;part['sampling_seed']=gs;part['real_amount']=real[lo:lo+n];part['generated_amount']=generated
                for field in ['category','receiver_or_mark']:
                    key=codec.prefixes[field]+'__cat'
                    part[field+'_matches_observed']=output[key][:,0].eq(encoded[key][chosen,0]).cpu().numpy()
                rows.append(part)
        print('CONDITION_PROBE',fs,case,round(time.monotonic()-started,1),flush=True)
    data=pd.concat(rows,ignore_index=True);path=OUT/f'rows_{fs}.parquet';data.to_parquet(path,index=False)
    metrics=[]
    for (case,label),g in data.groupby(['case','label']):
        metrics.append(dict(fit_seed=fs,case=case,label=int(label),positions=len(g)//4,
            real_median=float(g.real_amount.median()),generated_median=float(g.generated_amount.median()),
            log_w1=float(wasserstein_distance(np.log1p(g.real_amount),np.log1p(g.generated_amount))),
            category_exact_match=float(g.category_matches_observed.mean()),
            merchant_exact_match=float(g.receiver_or_mark_matches_observed.mean())))
    pd.DataFrame(metrics).to_csv(DOCS/f'metrics_{fs}.csv',index=False)
    for p,h in manifest['hashes'].items():assert digest(p)==h
    write(DOCS/f'COMPLETE_{fs}.json',dict(seconds=time.monotonic()-started,pid=os.getpid(),
        first_digit_equivalence_checks=checks,rows_sha256=digest(path),test_events_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--seed',type=int,choices=CFG['fit_seeds'])
    p.add_argument('--device',default='cpu');a=p.parse_args();prepare() if a.mode=='prepare' else worker(a.seed,a.device)
