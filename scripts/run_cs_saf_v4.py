"""Run the two preregistered E-forward / residual-penalty fits on a free GPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from experiments.cs_saf_v4 import (ROOT, CONTRACT_SHA, CANDIDATE, CANDIDATES,
    load_contract, train, make_model, save_accuracy, contrasts, accuracy_screens,
    frozen_source)
from experiments.cs_saf_pilot import audit_model, decide_pilot, batch, state_digest
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json


def cpu_gate(cache_root, output):
    commit = frozen_source()
    _, cfg = load_contract()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for candidate in CANDIDATES:
        runs, audits, reload_checks = [], [], []
        for repeat in (1, 2):
            folder = output/candidate/f"run_{repeat}"
            model, payload, report = train(cache_root/"pi_0.05_kappa_1.pt", folder,
                                           candidate, torch.device("cpu"), cpu=True)
            copy = make_model(payload, candidate, torch.device("cpu")).eval()
            copy.load_state_dict(torch.load(folder/"checkpoint_best.pt", map_location="cpu")["model_state"])
            x = batch(payload["validation"], torch.arange(len(payload["validation"]["lengths"])), torch.device("cpu"))
            with torch.no_grad():
                a, b = model.loss_terms(**x), copy.loss_terms(**x)
            reload_checks.append(state_digest(copy) == report["best_state_sha256"]
                                 and all(torch.equal(a[k], b[k]) for k in a))
            audits.append(audit_model(copy, payload, dict(cfg, generation_entities=32, generation_batch_size=32),
                                      torch.device("cpu"), sample_output=folder/"generated_sample.pt"))
            runs.append(report)
        first, second = runs
        drop = (first["history"][0]["train_objective"]-min(x["train_objective"] for x in first["history"]))/abs(first["history"][0]["train_objective"])
        checks = {"training_history_identical": first["history"] == second["history"],
                  "best_model_identical": first["best_state_sha256"] == second["best_state_sha256"],
                  "loss_decrease": drop >= cfg["cpu_gate"]["minimum_relative_train_loss_decrease"],
                  "checkpoint_reload_identical": all(reload_checks),
                  "generation_support": all(a["gap_support_violations"] == 0 for a in audits),
                  "generation_valid": all(a["invalid_reserved_marks"] == 0 and a["finite_generated_values"] for a in audits),
                  "zero_gap_invariant": all(a["zero_gap_control_max_range"] <= 1e-8 for a in audits)}
        results[candidate] = {"checks": checks, "decision": "PASS" if all(checks.values()) else "FAIL",
                              "relative_loss_decrease": drop, "best_state_sha256": first["best_state_sha256"],
                              "initial_state_sha256": first["initial_state_sha256"], "audits": audits}
    paired = len({r["initial_state_sha256"] for r in results.values()}) == 1
    result = {"decision": "PASS" if paired and all(r["decision"] == "PASS" for r in results.values()) else "FAIL",
              "source_commit": commit, "config_sha256": CONTRACT_SHA, "candidates": results,
              "matched_initialization": paired, "test_accessed": False}
    write_json(output/"COMPLETE.json", result)
    print(json.dumps(result), flush=True)
    return result


def validate_cpu(path):
    cpu = json.loads(path.read_text())
    if cpu["source_commit"] != frozen_source() or cpu["config_sha256"] != CONTRACT_SHA or cpu["decision"] != "PASS":
        raise RuntimeError("same-source passing CPU gate required")



def job(cache,output,candidate,device,cpu_path):
    validate_cpu(cpu_path)
    if output.exists():raise FileExistsError('prior output is immutable')
    try:
        model,payload,report=train(cache,output,candidate,device)
        _,cfg=load_contract();kappa=int(cache.stem[-1])
        audit=audit_model(model,payload,cfg,device,sample_output=output/'generated_sample.pt')
        write_json(output/'intervention_audit.json',audit)
        diagnostic=save_accuracy(model,payload,output,kappa,device)
        for label in ('0','1'):
            for metric in ('copy','repeat'):
                detailed=diagnostic['checkpoints']['best']['splits']['validation']['groups'][label]['metrics'][metric+'_range']['mean']
                if abs(detailed-audit['responses'][label][f'mean_{metric}_range'])>1e-6:
                    raise RuntimeError('independent response aggregations disagree')
        if state_digest(model)!=report['best_state_sha256']:raise RuntimeError('analysis changed best state')
        write_json(output/'COMPLETE.json',{'status':'COMPLETE','source_commit':report['source_commit'],
            'config_sha256':CONTRACT_SHA,'test_accessed':False,
            'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    except Exception as exc:
        if output.is_dir() and not (output/'COMPLETE.json').exists():
            write_json(output/'FAILED.json',{'error':type(exc).__name__,'message':str(exc)})
        raise


def saved_controls(kappa, cache):
    contract, _ = load_contract()
    parent = json.loads((ROOT/contract['parent_result']).read_text())
    runtime = ROOT/contract['execution']['saved_control_root']
    if sha256(runtime/'COMPLETE.json') != parent['runtime_terminal_sha256']:
        raise RuntimeError('saved v3 terminal changed')
    result = parent['result']
    folders, evidence = {}, {}
    for candidate in contract['saved_comparators']:
        key = f'pi_0.05_kappa_{kappa}/{candidate}'
        folder = runtime/key
        terminal = result['worker_terminals'][key]
        if json.loads((folder/'COMPLETE.json').read_text()) != terminal:
            raise RuntimeError('saved terminal differs from committed evidence')
        for name, expected in terminal['artifact_sha256'].items():
            if sha256(folder/name) != expected:
                raise RuntimeError('saved comparator artifact changed: '+str(folder/name))
        diag = result['conditional_accuracy'][key]
        if json.loads((folder/'conditional_accuracy.json').read_text()) != diag:
            raise RuntimeError('saved conditional audit mismatch')
        for item in diag['checkpoints'].values():
            if sha256(Path(item['checkpoint_path'])) != item['checkpoint_sha256']:
                raise RuntimeError('saved comparator checkpoint changed')
        if candidate != 'CS2-U1' and sha256(cache) != result['training'][key]['cache_sha256']:
            raise RuntimeError('comparator data differs')
        folders[candidate] = folder
        evidence[key] = {'terminal': terminal, 'diagnostic': diag}
    return result, folders, evidence


def matched_controls(kappa, report, audit, diagnostic, parent):
    for candidate in ('CS3-E1', 'CS3-C1', 'CS3-R1'):
        key = f'pi_0.05_kappa_{kappa}/{candidate}'
        old = parent['training'][key]
        for field in ('initial_state_sha256', 'cache_sha256', 'seed', 'options',
                      'parameters', 'train_entities', 'validation_entities',
                      'transition_counts_by_code', 'initialization_contract'):
            if report[field] != old[field]:
                raise RuntimeError('new/saved comparator mismatch: '+field)
        for new_epoch, old_epoch in zip(report['history'], old['history']):
            if new_epoch['entity_order_prefix_sha256'] != old_epoch['entity_order_prefix_sha256']:
                raise RuntimeError('entity orders differ')
        if audit['sampling_plan_sha256'] != parent['audits'][key]['sampling_plan_sha256']:
            raise RuntimeError('sampling plan differs')
    for candidate in ('CS3-E1', 'CS3-C1', 'CS3-R1', 'CS2-U1'):
        old = parent['conditional_accuracy'][f'pi_0.05_kappa_{kappa}/{candidate}']
        for name, item in diagnostic['checkpoints'].items():
            if item['reference_probabilities'] != old['checkpoints'][name]['reference_probabilities']:
                raise RuntimeError('evaluation measures differ')


def run(cache_root, cpu_path, output, gpu):
    validate_cpu(cpu_path)
    source = frozen_source()
    contract, cfg = load_contract()
    # Validate both cells before launching any new fit.
    controls = {k: saved_controls(k, cache_root/f'pi_0.05_kappa_{k}.pt') for k in (0, 1)}
    output.mkdir(parents=True, exist_ok=False)
    training, audits, diagnostics, terminals, paired, reuse = {}, {}, {}, {}, {}, {}
    for kappa in (0, 1):
        parent, folders, evidence = controls[kappa]
        reuse.update(evidence)
        cell = output/f'pi_0.05_kappa_{kappa}'
        cell.mkdir()
        folder = cell/CANDIDATE
        command = [sys.executable, '-m', 'scripts.run_cs_saf_v4', 'job',
            '--cache', str(cache_root/f'pi_0.05_kappa_{kappa}.pt'), '--output', str(folder),
            '--candidate', CANDIDATE, '--device', f'cuda:{gpu}', '--cpu-gate', str(cpu_path)]
        with (cell/f'{CANDIDATE}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(output/'FAILED.json', {'source_commit': source, 'kappa': kappa,
                'candidate': CANDIDATE, 'exit_code': result.returncode})
            raise RuntimeError('technical failure; preserve outputs')
        key = f'pi_0.05_kappa_{kappa}/{CANDIDATE}'
        terminal = json.loads((folder/'COMPLETE.json').read_text())
        if terminal['source_commit'] != source or terminal['config_sha256'] != CONTRACT_SHA:
            raise RuntimeError('source mismatch')
        for name, expected in terminal['artifact_sha256'].items():
            if sha256(folder/name) != expected:
                raise RuntimeError('new artifact changed')
        terminals[key] = terminal
        diagnostics[key] = json.loads((folder/'conditional_accuracy.json').read_text())
        training[key] = json.loads((folder/'training_report.json').read_text())
        audits[key] = json.loads((folder/'intervention_audit.json').read_text())
        matched_controls(kappa, training[key], audits[key], diagnostics[key], parent)
        # Recheck read-only controls after the job, then calculate paired contrasts.
        saved_controls(kappa, cache_root/f'pi_0.05_kappa_{kappa}.pt')
        folders[CANDIDATE] = folder
        paired[str(kappa)] = contrasts(folders)
        print(f'Completed {key}; saved comparator matching PASS', flush=True)
    gate = decide_pilot({k: audits[f'pi_0.05_kappa_{k}/{CANDIDATE}'] for k in (0, 1)}, cfg['pilot_gate'])
    screens = accuracy_screens(paired)
    eligible = gate['decision'] == 'PASS' and all(s['PASS'] for s in screens.values())
    result = {'status': 'COMPLETE', 'source_commit': source, 'config_sha256': CONTRACT_SHA,
        'scientific_fits': len(training), 'training': training, 'audits': audits,
        'conditional_accuracy': diagnostics, 'worker_terminals': terminals,
        'paired_contrasts': paired, 'reused_controls': reuse, 'response_gate': gate,
        'accuracy_screen': screens, 'next_registered_pilot_eligible': eligible,
        'new_method_success': False, 'test_accessed': False, 'later_stage_started': False,
        'saved_comparator_new_fits': 0, 'matched_saved_controls': True,
        'cpu_gate_sha256': sha256(cpu_path), 'prepared_index_sha256': sha256(cache_root/'COMPLETE.json')}
    write_json(output/'COMPLETE.json', result)
    print(json.dumps({'status': 'COMPLETE', 'response_gate': gate, 'accuracy_screen': screens,
                     'next_registered_pilot_eligible': eligible}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='phase', required=True)
    a = sub.add_parser('cpu')
    a.add_argument('--cache-root', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('job')
    a.add_argument('--cache', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--candidate', choices=CANDIDATES, required=True)
    a.add_argument('--device', required=True)
    a.add_argument('--cpu-gate', type=Path, required=True)
    a = sub.add_parser('run')
    a.add_argument('--cache-root', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--cpu-gate', type=Path, required=True)
    a.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    if args.phase == 'cpu':
        if cpu_gate(args.cache_root, args.output)['decision'] != 'PASS':
            raise SystemExit(2)
    elif args.phase == 'job':
        job(args.cache, args.output, args.candidate, torch.device(args.device), args.cpu_gate)
    else:
        run(args.cache_root, args.cpu_gate, args.output, args.gpu)


if __name__ == '__main__':
    main()
