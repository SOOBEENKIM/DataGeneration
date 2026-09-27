"""Fresh order-only fits, free generation, and unchanged development diagnostics."""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import time

from run_argn_state_first import ROOT, OUT as PARENT, SOURCE, CFG, seed, digest, write, check_manifest
import numpy as np
import pandas as pd
import torch
from importlib.metadata import version
from mostlyai.engine import generate
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length, SLEN_SIDX_SDEC_COLUMN
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.generation import _fix_rare_token_probs, _translate_fixed_probs
from benchmarks.argn_label_first_control import label_first_adapter, label_first_model_class
from benchmarks.argn_state_adapter import FullHistoryCollator
from benchmarks.argn_past_state import PastState
from benchmarks.argn_frozen_probe import teacher_with_history, one_step, label_probability

OUT = ROOT / 'artifacts/argn_label_first_control_v1'
DOCS = ROOT / 'docs/argn_label_first_control_v1'
CONFIG = ROOT / 'configs/argn_label_first_control_v1.json'
CONTROL = json.loads(CONFIG.read_text())
PROBES = ROOT / 'artifacts/argn_followup_diagnostics_v1'


def source_hashes():
    paths = [CONFIG, Path(__file__), ROOT/'scripts/dispatch_argn_label_first_control.py',
             ROOT/'scripts/report_argn_label_first_control.py',
             ROOT/'benchmarks/argn_label_first_control.py',
             ROOT/'benchmarks/argn_event_weight_control.py',
             ROOT/'benchmarks/argn_frozen_probe.py', ROOT/'benchmarks/argn_state_evaluation.py',
             ROOT/'benchmarks/argn_fraud_audit.py', ROOT/'scripts/evaluate_sparkov_argn_control.py',
             ROOT/'scripts/summarize_argn_followup.py', DOCS/'PROTOCOL.md',
             ROOT/'tests/test_argn_label_first_control.py', ROOT/'tests/test_argn_label_first_dispatch.py',
             DOCS/'LAUNCH_AMENDMENT.md']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def verify_inputs():
    prep = check_manifest()
    assert torch.__version__ == CFG['torch_version']
    assert version('mostlyai-engine') == CFG['engine_version']
    manifest = json.loads((PROBES/'D1_INPUTS.json').read_text())
    assert digest(PROBES/'positions.parquet') == manifest['positions_sha256']
    assert digest(PROBES/'encoded_development.pkl') == manifest['encoded_sha256']
    return prep, manifest


def prepare():
    assert not (OUT/'MANIFEST.json').exists(), 'never overwrite a registered study'
    prep, probes = verify_inputs()
    d0 = json.loads((PROBES/'D0_COMPLETE.json').read_text())
    for path, sha in {**probes['weights'], **d0['generation_hashes']}.items():
        assert digest(path) == sha
    write(OUT/'MANIFEST.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        config=CONTROL, inherited_config=CFG, source_hashes=source_hashes(),
        preparation_sha256=digest(PARENT/'prepared/MANIFEST.json'),
        probe_manifest_sha256=digest(PROBES/'D1_INPUTS.json'),
        paired_weights=probes['weights'], paired_generations=d0['generation_hashes'],
        test_events_read=False))
    print('ORDER_CONTROL_PREPARED', flush=True)


def check_registered():
    prep, probes = verify_inputs()
    manifest = json.loads((OUT/'MANIFEST.json').read_text())
    assert manifest['source_hashes'] == source_hashes(), 'registered source drift'
    assert manifest['config'] == CONTROL and manifest['inherited_config'] == CFG
    assert manifest['preparation_sha256'] == digest(PARENT/'prepared/MANIFEST.json')
    assert manifest['probe_manifest_sha256'] == digest(PROBES/'D1_INPUTS.json')
    return prep


def evaluate(folder, arm, fs):
    from evaluate_sparkov_argn_control import extended, numeric_metrics, additional_relations
    from benchmarks.argn_fraud_audit import summaries
    from benchmarks.argn_state_evaluation import episode_features, scalar_summary, log_w1, age_amount_table, hazard_table
    state = json.loads((SOURCE/'prepared/metric_state.json').read_text())
    raw_real = pd.read_parquet(SOURCE/'prepared/validation.parquet')
    real, real_runs = episode_features(extended(raw_real, state))
    results, numbers, ages, hazards = [], [], [], []
    for gs in CONTROL['generation_seeds']:
        raw = pd.read_parquet(folder/f'generated_validation_{gs}.parquet')
        assert set(raw.entity_id) == set(real.entity_id)
        d, runs = episode_features(extended(raw, state))
        name = f'{arm}_{fs}_{gs}'
        row = dict(run=name, arm=arm, fit_seed=fs, generation_seed=gs,
                   **summaries(raw_real, raw, state), **scalar_summary(d, runs))
        row.update(additional_relations(real, d))
        row['first_event_fraud_rate'] = float(d.loc[d.event_index.eq(0),'fraud'].eq(1).mean())
        for field, metric in [('length','run_length_log_w1'),('span_seconds','run_span_log_w1')]:
            row[metric] = log_w1(real_runs[field], runs[field])
            rr, ss = real_runs[real_runs.completed_known_start], runs[runs.completed_known_start]
            row['completed_'+metric] = log_w1(rr[field], ss[field])
        bands, values = age_amount_table(real,d,name)
        row.update(values); ages.extend(bands); hazards.extend(hazard_table(d,name))
        num = numeric_metrics(real,d,name)
        for n in num: row[f"{n['label']}_{n['field']}_log_w1"] = n['wasserstein_log1p']
        numbers.extend(num)
        row['customer_length_log_w1'] = log_w1(real.groupby('entity_id').size(),d.groupby('entity_id').size())
        results.append(row)
    dest = DOCS/'evaluation'; dest.mkdir(parents=True,exist_ok=True)
    for name,rows in [('metrics',results),('numeric',numbers),('age_amounts',ages),('hazards',hazards)]:
        pd.DataFrame(rows).to_csv(dest/f'{name}_{arm}_{fs}.csv',index=False)


@torch.no_grad()
def probe(folder, arm, fs, device):
    from summarize_argn_followup import summarize
    ws = Workspace(folder/'workspace'); ts,cs = ws.tgt_stats.read(),ws.ctx_stats.read()
    codec = PastState(ts); key = codec.prefixes['event_is_fraud']+'__cat'
    masks = _translate_fixed_probs(_fix_rare_token_probs(ts),ts)
    dev = torch.device(device)
    model = label_first_model_class(ts)(
        tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
        tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
        ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),
        model_size=ws.model_tabular_configs.read()['model_units'],column_order=None,device=dev)
    load_model_weights(model=model,path=ws.model_tabular_weights_path,device=dev)
    model.to(dev).eval()
    with (PROBES/'encoded_development.pkl').open('rb') as handle: data = pickle.load(handle)
    meta = pd.read_parquet(PROBES/'positions.parquet'); records = data['records']
    collator = FullHistoryCollator(True,None,dev); rows=[]
    equivalence_checked = 0
    for lo in range(0,len(records),2):
        batch = collator(records[lo:lo+2])
        logits,histories,context = teacher_with_history(model,batch)
        probs = label_probability(logits[key],key,codec.codes['1'],masks)
        label_nll = -logits[key].log_softmax(-1).gather(-1,batch[key]).squeeze(-1)
        amount_nll = sum(-v.log_softmax(-1).gather(-1,batch[k]).squeeze(-1)
            for k,v in logits.items() if k.startswith(codec.prefixes['amount_or_numeric_value']+'__'))
        for b in range(len(records[lo:lo+2])):
            ri = lo+b; m = meta[meta.record.eq(ri)].copy(); n = len(m)
            m['fraud_probability'] = probs[b,:n].cpu().numpy()
            m['label_nll'] = label_nll[b,:n].cpu().numpy()
            m['amount_digit_nll'] = amount_nll[b,:n].cpu().numpy(); rows.append(m)
            indices = m.loc[m.selected,'event_index'].to_numpy(dtype=int)
            histories_i = histories[b,indices].unsqueeze(1)
            memory = torch.from_numpy(np.stack([data['memories'][ri][int(t)] for t in indices])).to(dev)
            ctx = [[v[b:b+1].expand(len(indices),*v.shape[1:]) for v in group] for group in context]
            structural = {k:v[b,indices] for k,v in batch.items() if k.startswith(SLEN_SIDX_SDEC_COLUMN+'__')}
            observed,_ = one_step(model,histories_i,memory,ctx,structural,key,masks)
            np.testing.assert_allclose(label_probability(observed,key,codec.codes['1'],masks).cpu().numpy(),
                m.loc[m.selected,'fraud_probability'].to_numpy(),rtol=1e-4,atol=2e-6)
            equivalence_checked += len(indices)
    teacher = pd.concat(rows,ignore_index=True)
    teacher.to_parquet(folder/'teacher.parquet',index=False)
    summary = summarize(teacher,arm,fs,'teacher','all_development')
    summary += summarize(teacher[teacher.selected],arm,fs,'teacher','matched_prefixes')
    pd.DataFrame(summary).to_csv(DOCS/'evaluation'/f'teacher_{arm}_{fs}.csv',index=False)
    write(folder/'PROBE.json',dict(real_events=len(teacher),prefix_equivalence_checks=equivalence_checked,
        true_length_diagnostic=True,current_fields_cannot_affect_label=True,test_events_read=False))


def run(arm, fs, device):
    prep = check_registered()
    assert arm in CONTROL['arms'] and fs in CONTROL['fit_seeds']
    folder = OUT/'runs'/f'{arm}_{fs}'; folder.mkdir(parents=True,exist_ok=False)
    status = OUT/f'queue_{arm}_{fs}.json'
    hashes = source_hashes(); split = prep['encoded_splits']['part.000000-trn.parquet']
    mean_length = split['events']/split['customers']
    def stage(value, **extra):
        write(status,dict(stage=value,arm=arm,fit_seed=fs,pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    try:
        stage('preparing_workspace')
        shutil.copytree(PARENT/'prepared/workspace',folder/'workspace')
        ws = Workspace(folder/'workspace'); assert not ws.model_tabular_weights_path.exists()
        write(folder/'START.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
            config=CONTROL,inherited_config=CFG,arm=arm,fit_seed=fs,source_hashes=hashes,
            mean_training_length=mean_length,device=device,pid=os.getpid(),fresh_initialization=True,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        seed(fs); started=time.monotonic(); stage('training')
        with label_first_adapter(ws.tgt_stats.read(),CONTROL['arms'][arm],mean_length,folder) as train:
            train(model=CFG['model'],max_training_time=CFG['max_training_minutes'],max_epochs=CFG['max_epochs'],
                batch_size=CFG['batch_size'],gradient_accumulation_steps=CFG['gradient_accumulation_steps'],
                max_sequence_window=ws.tgt_stats.read()['seq_len']['max'],enable_flexible_generation=False,
                device=device,workspace_dir=folder/'workspace')
        weights_sha=digest(ws.model_tabular_weights_path)
        weights=torch.load(ws.model_tabular_weights_path,map_location='cpu',weights_only=True)
        write(folder/'FIT.json',dict(seconds=time.monotonic()-started,weights_sha256=weights_sha,
            total_parameters=sum(x.numel() for x in weights.values()),source_hashes=hashes))
        del weights
        assert hashes==source_hashes(); print('ORDER_FIT_COMPLETE',arm,fs,flush=True)
        context=pd.read_parquet(PARENT/'prepared/validation_context.parquet')
        with label_first_adapter(ws.tgt_stats.read(),CONTROL['arms'][arm],mean_length):
            for gs in CONTROL['generation_seeds']:
                stage('generating',generation_seed=gs); seed(gs); started=time.monotonic()
                generate(ctx_data=context,device=device,workspace_dir=folder/'workspace')
                raw=pd.read_parquet(folder/'workspace/SyntheticData')
                raw.to_parquet(folder/f'native_validation_{gs}.parquet',index=False)
                frame=raw.rename(columns={'customer_id':'entity_id'}).copy()
                frame['event_index']=frame.groupby('entity_id',sort=False).cumcount()
                frame.loc[frame.event_index.eq(0),'gap']=np.nan
                dest=folder/f'generated_validation_{gs}.parquet'; frame.to_parquet(dest,index=False)
                assert digest(ws.model_tabular_weights_path)==weights_sha
                write(folder/f'generation_validation_{gs}.json',dict(seconds=time.monotonic()-started,
                    events=len(frame),customers=frame.entity_id.nunique(),sha256=digest(dest),
                    weights_sha256=weights_sha,native_generated_length=True,labels_generated_jointly=True,
                    no_prevalence_or_run_length_repair=True))
                print('ORDER_GENERATION_COMPLETE',arm,fs,gs,flush=True)
        stage('evaluating'); evaluate(folder,arm,fs)
        stage('teacher_diagnostic'); seed(20260927+fs); probe(folder,arm,fs,device)
        assert digest(ws.model_tabular_weights_path)==weights_sha
        check_registered(); stage('complete',generated_datasets=2,test_events_read=False)
        write(folder/'COMPLETE.json',dict(arm=arm,fit_seed=fs,source_hashes=hashes,
            checkpoint_sha256=weights_sha,generated_datasets=2,test_events_read=False))
        print('ORDER_CONTROL_COMPLETE',arm,fs,flush=True)
    except Exception as error:
        stage('failed',error=repr(error)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--arm',choices=list(CONTROL['arms'])); p.add_argument('--seed',type=int)
    p.add_argument('--device',default='cuda:0'); args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s',force=True)
    if args.action=='prepare': prepare()
    else: run(args.arm,args.seed,args.device)
