"""One preregistered external output-control fit and its complete evaluation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizer, SAFTensorizerState
from data.cs_saf_external import TargetWindows
from models.cs_saf_external_controls import ExternalControls
from scripts.run_cs_saf_external_port import inputs, prediction, calibration_fit, seed, digest, write
from benchmarks.cs_saf_external import evaluate

OUT = ROOT / 'artifacts/cs_saf/external_controls_v1'
OLD = ROOT / 'artifacts/cs_saf/external_port_v1'
CONFIG = ROOT / 'configs/cs_saf_external_controls_v1.json'


def amount_state(fit, tensor_state):
    a = fit.amount_or_numeric_value.to_numpy(float)
    assert np.isfinite(a).all() and (a >= 0).all()
    positive = np.log(a[a > 0])
    mean, scale = float(positive.mean()), max(float(positive.std()), 1e-8)
    codec = tensor_state.event_numeric_codecs[0][1]
    return dict(codec_mean=codec.mean, codec_scale=codec.scale, log_mean=mean, log_scale=scale,
                zero_rate=float((a == 0).mean()), fit_events=len(a),
                initial_locations=((np.quantile(positive, [.2, .5, .8]) - mean) / scale).tolist())


def build_model(model_id, state, ast, initial):
    mode, change = model_id.split('_')
    model = ExternalControls(mode, state.gap_support,
                             amount_kind='legacy' if change == 'action' else 'hurdle_lognormal3',
                             full_gap=change in ('action', 'both'), amount_state=ast,
                             **state.model_config_kwargs())
    shared = model.initialize_shared(initial)
    return model, shared


def train(name, model_id, folder, cfg):
    inp, parents, ids, frames, plan, metric_state = inputs(name)
    data = torch.load(inp / 'prepared.pt', map_location='cpu')
    state = SAFTensorizerState.from_dict(data['state'])
    tf = SAFTensorizer(state)
    ast = amount_state(frames['fit'], state)
    write(folder / 'amount_state.json', ast)
    training = TargetWindows(data['sequences']['fit'], device='cuda')
    check = TargetWindows(data['sequences']['check'], device='cuda')
    reference_mode = 'G' if model_id.startswith('D') else model_id[0]
    initial_path = OLD / 'runs' / name / reference_mode / 'initial.pt'
    initial = torch.load(initial_path, map_location='cpu')
    seed(cfg['fit_seed'])
    model, shared = build_model(model_id, state, ast, initial)
    for k in shared:
        assert torch.equal(model.state_dict()[k].cpu(), initial[k]), k
    model = model.cuda()
    torch.save(model.state_dict(), folder / 'initial.pt')
    write(folder / 'architecture.json', dict(**model.architecture_contract(), shared_initial_keys=shared,
                                           source_initial_sha256=digest(initial_path)))
    options = cfg['ug']
    opt = torch.optim.Adam(model.parameters(), lr=options['learning_rate'], weight_decay=0.)
    best, best_epoch, history, updates = float('inf'), 0, [], 0
    stopped = 'epoch_budget'
    started = time.monotonic()
    for epoch in range(1, options['max_epochs'] + 1):
        model.train()
        order = np.random.default_rng(cfg['fit_seed'] + epoch).permutation(len(training))
        sums, counts = {}, {}
        epoch_start = time.monotonic()
        for start in range(0, len(order), options['batch_size']):
            terms, _ = model.terms(**training.batch(order[start:start + options['batch_size']]))
            loss = sum(v / c.clamp_min(1) for v, c in terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'])
            opt.step(); updates += 1
            for k, (v, c) in terms.items():
                sums[k] = sums.get(k, 0.) + float(v.detach())
                counts[k] = counts.get(k, 0) + int(c)
        check_score, _ = prediction(model, check, options['batch_size'], state)
        if not np.isfinite(check_score['loss']):
            raise FloatingPointError('nonfinite check loss')
        improved = check_score['loss'] < best
        if improved:
            best, best_epoch = check_score['loss'], epoch
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, check=check_score), folder / 'best.pt')
        record = dict(epoch=epoch, fit_loss=sum(sums[k] / counts[k] for k in sums),
                      check=check_score, selected=improved, seconds=time.monotonic() - epoch_start,
                      optimizer_updates=updates)
        history.append(record); write(folder / 'history.json', history)
        print(name, model_id, 'EPOCH', epoch, 'fit', record['fit_loss'], 'check', check_score['loss'],
              'best', best_epoch, 'seconds', record['seconds'], flush=True)
        if epoch - best_epoch >= options['patience']:
            stopped = 'check_patience'; break
        if time.monotonic() - started >= options['max_fit_seconds']:
            stopped = 'time_budget'; break
    fit_seconds = time.monotonic() - started
    torch.save(dict(state_dict=model.state_dict(), epoch=epoch), folder / 'last.pt')
    model.load_state_dict(torch.load(folder / 'best.pt')['state_dict'])
    _, caldata = prediction(model, training, options['batch_size'], state, collect=True)
    correction = calibration_fit(caldata, metric_state['gap_edges'], cfg['calibration'])
    write(folder / 'calibration.json', correction)
    validation = TargetWindows(data['sequences']['validation'], device='cuda')
    positions = plan.fit_position.to_numpy(int)
    results = {}
    for variant in cfg['variants']:
        model.calibration = None if variant == 'raw' else correction
        pred, _ = prediction(model, validation, options['batch_size'], state)
        generations = []
        for gs in cfg['generation_seeds']:
            seed(gs); begun = time.monotonic()
            sample = model.sample_fixed_lengths(plan.length.tolist(), static=training.static[positions],
                static_categorical=tuple(v[positions] for v in training.static_cat))
            output = tf.decode_generated(entity_ids=plan.entity_id.tolist(), lengths=plan.length.tolist(),
                **{k: sample[k] for k in ('gap', 'receiver', 'numeric_value', 'auxiliary_categorical', 'auxiliary_numeric')})
            # Exact raw draws; the float32 codec is only the history representation.
            if model.amount_kind != 'legacy':
                output['amount_or_numeric_value'] = np.concatenate([
                    sample['raw_amount'][i, :n].cpu().numpy() for i, n in enumerate(plan.length)])
            assert np.isfinite(output.amount_or_numeric_value).all()
            assert len(output) == int(plan.length.sum())
            path = folder / f'generated_{variant}_{gs}.parquet'
            output.to_parquet(path, index=False)
            metrics = evaluate(frames['validation'], output, metric_state)
            generations.append(dict(seed=gs, seconds=time.monotonic() - begun, metrics=metrics, sha256=digest(path)))
            print(name, model_id, variant, 'GENERATED', gs, len(output), flush=True)
        results[variant] = dict(prediction=pred, generations=generations)
    return dict(dataset=name, model=model_id, selected_epoch=best_epoch, last_epoch=epoch,
                stop_reason=stopped, optimizer_updates=updates, fit_seconds=fit_seconds,
                architecture=model.architecture_contract(), initial_sha256=digest(folder / 'initial.pt'),
                best_sha256=digest(folder / 'best.pt'), results=results)


def main():
    p = argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('dataset')
    a = p.parse_args(); cfg = json.loads(CONFIG.read_text())
    assert a.model in cfg['new_models'] and a.dataset in cfg['datasets']
    folder = OUT / 'runs' / a.dataset / a.model
    assert not folder.exists(), 'scientific runs are never overwritten'
    assert torch.cuda.is_available(), 'admitted workstation GPU required'
    scientific = ['configs/cs_saf_external_controls_v1.json', 'models/cs_saf_external_controls.py',
                  'scripts/run_cs_saf_external_controls.py', 'models/cs_saf_external.py',
                  'models/cof_seqgen_saf.py', 'data/cs_saf_external.py', 'data/cof_seqgen_saf_tensorizer.py',
                  'benchmarks/cs_saf_external.py', 'scripts/run_cs_saf_external_port.py']
    subprocess.check_call(['git', 'ls-files', '--error-unmatch', *scientific], cwd=ROOT, stdout=subprocess.DEVNULL)
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *scientific], cwd=ROOT), 'commit scientific source before running'
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    write(folder / 'START.json', dict(config_sha256=digest(CONFIG), model=a.model, dataset=a.dataset,
          source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          scientific_source_sha256={f: digest(ROOT / f) for f in scientific},
          input_sha256=digest(OLD / 'input' / a.dataset / 'preflight.json'), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
          fit_seed=cfg['fit_seed'], versions={k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'pandas')},
          test_outcomes_accessed=False))
    begun = time.monotonic()
    try:
        result = train(a.dataset, a.model, folder, cfg)
        result.update(total_seconds=time.monotonic() - begun, config_sha256=digest(CONFIG),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      files={f.name: digest(f) for f in folder.iterdir() if f.is_file()})
        write(folder / 'DONE.json', result)
        print('DONE', a.dataset, a.model, flush=True)
    except Exception as exc:
        write(folder / 'FAILED.json', dict(error_type=type(exc).__name__, message=str(exc), elapsed_seconds=time.monotonic() - begun))
        raise


if __name__ == '__main__':
    main()
