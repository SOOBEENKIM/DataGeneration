"""Bounded continuation of copied official workspaces; original runs are immutable."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cs_saf_external_audit_v1 import digest, write

CONFIG = ROOT / 'configs/benchmark_v2/cs_saf_baseline_adequacy_v1.json'
OUT = ROOT / 'artifacts/cs_saf/baseline_adequacy_v1'
OLD = ROOT / 'artifacts/cs_saf/external_audit_v1'


def run(kappa, seed, device, smoke=False):
    from mostlyai.engine import set_random_state
    from mostlyai.engine._workspace import Workspace
    tr = importlib.import_module('mostlyai.engine._tabular.training')
    c = json.loads(CONFIG.read_text())
    assert kappa in c['kappas'] and seed in c['argn_seeds']
    source = OLD / ('cpu_smoke' if smoke else f'kappa_{kappa}/seed_{seed}')
    out = OUT / ('continuation_cpu_smoke' if smoke else f'continuations/kappa_{kappa}/seed_{seed}')
    out.mkdir(parents=True, exist_ok=False)
    old_ws = Workspace(source / 'workspace')
    original = {str(p.relative_to(source / 'workspace')): digest(p)
                for p in [old_ws.model_tabular_weights_path, old_ws.model_optimizer_path,
                          old_ws.model_lr_scheduler_path, old_ws.model_progress_messages_path]}
    shutil.copytree(source / 'workspace', out / 'workspace')
    ws = Workspace(out / 'workspace')
    progress = pd.read_csv(ws.model_progress_messages_path)
    best_index = progress[progress.is_checkpoint == 1].index[-1]
    selected = progress.loc[best_index]
    progress.loc[:best_index].to_csv(ws.model_progress_messages_path, index=False)
    torch.set_num_threads(c['cpu_threads'])
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'])
    set_random_state(seed + 10000)
    old_stopper, old_checkpoint, old_loss = tr.EarlyStopper, tr.TabularModelCheckpoint, tr._calculate_sample_losses
    latest = {}

    class RegisteredPatience(old_stopper):
        def __init__(self, val_loss_patience):
            super().__init__(c['early_stopping_patience_during_continuation'])

    class IncludeExistingBest(old_checkpoint):
        def __init__(self, workspace):
            super().__init__(workspace, initial_best_val_loss=float(selected.val_loss))
            self.save_count = 1  # existing checkpoint is already a valid saved candidate

    def keep_final_model(model, data):
        latest['model'] = model
        return old_loss(model, data)

    write(out / 'start.json', dict(config_sha256=digest(CONFIG), original_hashes=original,
        original_path=str(source.relative_to(ROOT)), selected_epoch=float(selected.epoch),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), smoke=smoke,
        change_model_or_loss=False, optimizer_restored=True, progress_aligned_to_saved_best=True))
    started = time.monotonic()
    try:
        tr.EarlyStopper = RegisteredPatience
        tr.TabularModelCheckpoint = IncludeExistingBest
        tr._calculate_sample_losses = keep_final_model
        tr.train(workspace_dir=out / 'workspace', model_state_strategy='resume',
                 max_epochs=float(selected.epoch) + (.01 if smoke else c['additional_epochs']),
                 max_training_time=float(selected.total_time) / 60 + (2 if smoke else c['additional_minutes']),
                 batch_size=32 if smoke else c['batch_size'], max_sequence_window=32,
                 enable_flexible_generation=True, device=device)
        torch.save(latest['model'].state_dict(), out / 'checkpoint_last.pt')
        for name, expected in original.items():
            assert digest(source / 'workspace' / name) == expected
        done_progress = pd.read_csv(ws.model_progress_messages_path)
        write(out / 'DONE.json', dict(config_sha256=digest(CONFIG), seconds=time.monotonic()-started,
            start_epoch=float(selected.epoch), last_epoch=float(done_progress.epoch.max()),
            best_epoch=float(done_progress[done_progress.is_checkpoint == 1].iloc[-1].epoch),
            weights_sha256=digest(ws.model_tabular_weights_path), last_weights_sha256=digest(out / 'checkpoint_last.pt'),
            originals_unchanged=True, smoke=smoke, test_accessed=False,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.startswith('cuda') else None))
        print('CONTINUATION_DONE', kappa, seed, flush=True)
    except Exception as exc:
        write(out / 'FAILED.json', dict(error=type(exc).__name__, message=str(exc)))
        raise
    finally:
        tr.EarlyStopper, tr.TabularModelCheckpoint, tr._calculate_sample_losses = old_stopper, old_checkpoint, old_loss


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--kappa',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(a.kappa,a.seed,a.device,a.smoke)
