"""D1: frozen teacher vs true-prefix one-step field-generation probes."""
import argparse
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from run_argn_state_first import ROOT, OUT, SOURCE, digest, write, seed, check_manifest
from diagnose_argn_followup_data import DEST, DOCS, CORE, ARMS, SEEDS, folder
from benchmarks.argn_past_state import PastState, STATE_COLUMN
from benchmarks.argn_state_adapter import FullHistoryCollator, model_class
from benchmarks.argn_frozen_probe import teacher_with_history, one_step, label_probability
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length, SLEN_SIDX_SDEC_COLUMN
from mostlyai.engine._tabular.encoding import encode_df, flatten_frame, _enrich_slen_sidx_sdec
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.generation import _fix_rare_token_probs, _translate_fixed_probs

MC_DRAWS = 16
CASES = ['sample_all', 'fix_amount_or_numeric_value', 'fix_gap', 'fix_receiver_or_mark', 'fix_category']


def prepare():
    check_manifest(); DEST.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    assert not (DEST/'D1_INPUTS.json').exists()
    seed(20260927)
    ws = Workspace(OUT/'prepared/workspace'); ts,cs = ws.tgt_stats.read(),ws.ctx_stats.read()
    codec = PastState(ts)
    real = pd.read_parquet(SOURCE/'prepared/validation.parquet').rename(columns={'entity_id':'customer_id'})
    real = real.sort_values(['customer_id','event_index']).reset_index(drop=True)
    context = pd.read_parquet(SOURCE/'prepared/validation_context.parquet')
    real.gap = real.gap.fillna(0)
    encoded, _, key = encode_df(real[['customer_id',*CORE]],ts,tgt_context_key='customer_id',n_jobs=1)
    encoded = flatten_frame(_enrich_slen_sidx_sdec(encoded,key,ts['seq_len']['max']),key)
    static, pk, _ = encode_df(context,cs,ctx_primary_key='customer_id',n_jobs=1)
    frame = encoded.merge(static,left_on=key,right_on=pk,validate='one_to_one')
    ids = frame[key].tolist()
    records = frame.drop(columns=[key,pk]).to_dict('records')
    rng = np.random.default_rng(20260927)
    metadata, memories = [], []
    label_key = codec.prefixes['event_is_fraud']+'__cat'
    for ri,(cid,record) in enumerate(zip(ids,records)):
        g = real[real.customer_id == cid]
        y = pd.to_numeric(g.event_is_fraud).to_numpy(dtype=int)
        assert len(y) == len(record[label_key])
        assert np.array_equal(np.asarray(record[label_key]) == codec.codes['1'],y.astype(bool))
        prev = np.r_[-1,y[:-1]]
        selected = (y == 1) | (prev == 1)
        selected[0] = True
        eligible = np.flatnonzero(~selected)
        if len(eligible): selected[rng.choice(eligible,min(4,len(eligible)),replace=False)] = True
        cohort = 'normal_only' if not y.any() else ('all_fraud' if y.all() else 'mixed')
        memory = codec.empty(1)
        features, selected_memories = [], {}
        for t in range(len(y)):
            features.append(codec.features(memory)[0])
            if selected[t]: selected_memories[t] = memory[0].copy()
            transition = 'first' if t == 0 else ('onset' if prev[t] == 0 and y[t] == 1 else
                          'termination' if prev[t] == 1 and y[t] == 0 else
                          'continuation' if prev[t] == 1 else 'normal_stay')
            metadata.append(dict(record=ri,customer_id=int(cid),event_index=t,label=int(y[t]),
                                 previous_label=int(prev[t]),prior_run_age=int(memory[0,2]),
                                 cohort=cohort,transition=transition,selected=bool(selected[t]),
                                 amount=float(g.iloc[t].amount_or_numeric_value)))
            event = {k:np.asarray([v[t]]) for k,v in record.items() if k.startswith('tgt:')}
            memory = codec.advance(memory,event)
        record[STATE_COLUMN] = np.stack(features)
        memories.append(selected_memories)
        if (ri+1)%25 == 0: print('PREPARE',ri+1,len(records),flush=True)
    meta = pd.DataFrame(metadata)
    meta.to_parquet(DEST/'positions.parquet',index=False)
    meta[meta.selected].to_csv(DOCS/'selected_positions.csv',index=False)
    with (DEST/'encoded_development.pkl').open('wb') as f:
        pickle.dump(dict(records=records,memories=memories),f,protocol=5)
    weights = {}; stats = {}
    for arm in ARMS:
        for fs in SEEDS:
            w = Workspace(folder(arm,fs)/'workspace')
            weights[str(w.model_tabular_weights_path)] = digest(w.model_tabular_weights_path)
            assert w.tgt_stats.read() == ts and w.ctx_stats.read() == cs
    code_paths = [ROOT/'scripts/diagnose_argn_followup_models.py',ROOT/'benchmarks/argn_frozen_probe.py',
                  ROOT/'docs/argn_state_first_v1/FOLLOWUP_DIAGNOSTIC_PROTOCOL.md']
    write(DEST/'D1_INPUTS.json',dict(weights=weights,positions_sha256=digest(DEST/'positions.parquet'),
          encoded_sha256=digest(DEST/'encoded_development.pkl'),
          source_hashes={str(p):digest(p) for p in code_paths},
          real_events=len(meta),customers=len(records),selected_prefixes=int(meta.selected.sum()),
          max_real_length=int(meta.groupby('customer_id').size().max()),
          model_stats_max_length=ts['seq_len']['max'],
          mc_draws=MC_DRAWS,cases=CASES,selection_seed=20260927,test_events_read=False,
          future_information='actual total lengths only for teacher and one-step diagnostic'))
    print('D1_PREPARED',len(meta),int(meta.selected.sum()),flush=True)


@torch.no_grad()
def probe(arm,fs,device):
    start = time.monotonic(); seed(20260927 + fs)
    manifest = json.loads((DEST/'D1_INPUTS.json').read_text())
    for path,h in manifest['source_hashes'].items(): assert digest(path) == h
    assert digest(DEST/'positions.parquet') == manifest['positions_sha256']
    assert digest(DEST/'encoded_development.pkl') == manifest['encoded_sha256']
    run = DEST/f'{arm}_{fs}'; run.mkdir(exist_ok=False)
    write(run/'START.json',dict(arm=arm,fit_seed=fs,pid=os.getpid(),device=device,
          cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),mc_draws=MC_DRAWS))
    ws = Workspace(folder(arm,fs)/'workspace'); ts,cs = ws.tgt_stats.read(),ws.ctx_stats.read()
    before = digest(ws.model_tabular_weights_path)
    assert before == manifest['weights'][str(ws.model_tabular_weights_path)]
    codec = PastState(ts); label_key = codec.prefixes['event_is_fraud']+'__cat'
    masks = _translate_fixed_probs(_fix_rare_token_probs(ts),ts)
    dev = torch.device(device)
    model = model_class(ts,arm == 'B_S')(
        tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
        tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
        ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),
        model_size=ws.model_tabular_configs.read()['model_units'],column_order=None,device=dev)
    load_model_weights(model=model,path=ws.model_tabular_weights_path,device=dev)
    # Native constructors leave some recurrent modules on CPU until model.to().
    model.to(dev).eval()
    assert all(p.device.type == dev.type for p in model.parameters())
    with (DEST/'encoded_development.pkl').open('rb') as f: data = pickle.load(f)
    records,memories = data['records'],data['memories']
    meta = pd.read_parquet(DEST/'positions.parquet')
    collator = FullHistoryCollator(True,None,dev)
    teacher_rows, caches = [], []
    for lo in range(0,len(records),2):
        batch = collator(records[lo:lo+2])
        logits,history,context = teacher_with_history(model,batch)
        probs = label_probability(logits[label_key],label_key,codec.codes['1'],masks)
        raw_p = logits[label_key].softmax(-1)[...,codec.codes['1']]
        raw_nll = -logits[label_key].log_softmax(-1).gather(-1,batch[label_key]).squeeze(-1)
        amount_nll = sum(-v.log_softmax(-1).gather(-1,batch[k]).squeeze(-1)
                         for k,v in logits.items() if k.startswith(codec.prefixes['amount_or_numeric_value']+'__'))
        for b in range(len(records[lo:lo+2])):
            ri = lo+b
            m = meta[meta.record == ri].copy(); n = len(m)
            m['fraud_probability'] = probs[b,:n].cpu().numpy()
            m['raw_fraud_probability'] = raw_p[b,:n].cpu().numpy()
            m['label_nll'] = raw_nll[b,:n].cpu().numpy()
            m['amount_digit_nll'] = amount_nll[b,:n].cpu().numpy()
            teacher_rows.append(m)
            indices = m.loc[m.selected,'event_index'].to_numpy(dtype=int)
            caches.append(dict(meta=m[m.selected].copy(),history=history[b,indices].unsqueeze(1).cpu(),
                memory=torch.from_numpy(np.stack([memories[ri][int(t)] for t in indices])),
                context=[[v[b:b+1].expand(len(indices),*v.shape[1:]).cpu() for v in group] for group in context],
                fixed={k:v[b,indices].cpu() for k,v in batch.items() if k.startswith('tgt:')}))
        if (lo//2)%15 == 0: print('TEACHER',arm,fs,lo+2,len(records),flush=True)
    teacher = pd.concat(teacher_rows,ignore_index=True)
    teacher.to_parquet(run/'teacher.parquet',index=False)
    selected = pd.concat([c['meta'] for c in caches],ignore_index=True)
    histories = torch.cat([c['history'] for c in caches])
    memory = torch.cat([c['memory'] for c in caches])
    contexts = [[torch.cat([c['context'][i][j] for c in caches]) for j in range(len(caches[0]['context'][i]))]
                for i in range(len(caches[0]['context']))]
    fixed = {k:torch.cat([c['fixed'][k] for c in caches]) for k in caches[0]['fixed']}
    # Check real checkpoint equivalence, in addition to the native unit test.
    n = min(128,len(selected))
    check,_ = one_step(model,histories[:n].to(dev),memory[:n].to(dev),
                       [[v[:n].to(dev) for v in g] for g in contexts],
                       {k:v[:n].to(dev) for k,v in fixed.items() if k != label_key},label_key,masks)
    observed = label_probability(check,label_key,codec.codes['1'],masks).cpu().numpy()
    np.testing.assert_allclose(observed,selected.fraud_probability.to_numpy()[:n],rtol=1e-4,atol=2e-6)
    rows=[]
    for ci,case in enumerate(CASES):
        seed(20260927 + fs + ci*1000)
        p_draws = np.empty((len(selected),MC_DRAWS),dtype=np.float32)
        for lo in range(0,len(selected),64):
            hi=min(lo+64,len(selected)); count=hi-lo
            idx=torch.arange(lo,hi).repeat_interleave(MC_DRAWS)
            usekeys=[k for k in fixed if k.startswith(SLEN_SIDX_SDEC_COLUMN+'__')]
            if case != 'sample_all':
                field=case.removeprefix('fix_')
                usekeys += [k for k in fixed if k.startswith(codec.prefixes[field]+'__')]
            result,_=one_step(model,histories[idx].to(dev),memory[idx].to(dev),
                              [[v[idx].to(dev) for v in g] for g in contexts],
                              {k:fixed[k][idx].to(dev) for k in usekeys},label_key,masks)
            p_draws[lo:hi]=label_probability(result,label_key,codec.codes['1'],masks).cpu().numpy().reshape(count,MC_DRAWS)
        result=selected.copy(); result['case']=case
        result['fraud_probability']=p_draws.mean(axis=1)
        result['mc_se']=p_draws.std(axis=1,ddof=1)/np.sqrt(MC_DRAWS)
        result['mc_draws']=MC_DRAWS
        rows.append(result)
        result.to_parquet(run/f'{case}.parquet',index=False)
        print('ONE_STEP',arm,fs,case,'seconds',round(time.monotonic()-start,1),flush=True)
    pd.concat(rows,ignore_index=True).to_parquet(run/'one_step.parquet',index=False)
    assert digest(ws.model_tabular_weights_path)==before
    write(run/'COMPLETE.json',dict(arm=arm,fit_seed=fs,seconds=time.monotonic()-start,
          checkpoint_sha256=before,weights_unchanged=True,real_events=len(teacher),
          selected_prefixes=len(selected),test_events_read=False,mc_draws=MC_DRAWS,
          equivalence_check_passed=True,source_hashes=manifest['source_hashes']))
    print('D1_COMPLETE',arm,fs,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','probe','worker'])
    parser.add_argument('--arm',choices=ARMS); parser.add_argument('--fit-seed',type=int,choices=SEEDS)
    parser.add_argument('--device',default='cpu'); args=parser.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='probe': probe(args.arm,args.fit_seed,args.device)
    else:
        for arm in ARMS: probe(arm,args.fit_seed,args.device)


if __name__=='__main__': main()
