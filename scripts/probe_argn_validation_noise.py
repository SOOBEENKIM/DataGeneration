"""Factorial read-only check of official checkpoint-selection noise."""
from run_argn_fraud_audit import OUT, DOCS, config, digest, write, pd, np, torch
from mostlyai.engine import set_random_state
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._tabular.probability import _initialize_model
from mostlyai.engine._tabular.training import BatchCollator, _calculate_val_loss
from torch.utils.data import DataLoader


def main():
    cfg = config(); rows = []; hashes = {}
    for seed in cfg['fit_seeds']:
        ws = Workspace(OUT/f'seed_{seed}/workspace')
        before = digest(ws.model_tabular_weights_path)
        model, *_ = _initialize_model(workspace=ws, device='cpu')
        data = pd.concat([pd.read_parquet(p) for p in ws.encoded_data_val.fetch_all()],ignore_index=True)
        assert len(data) == 138
        prefix = data.copy()
        for c in prefix:
            if c.startswith('tgt:'): prefix[c] = prefix[c].map(lambda x:x[:101])
        for window in ['random','prefix']:
            collator = BatchCollator(True, 100 if window == 'random' else None, torch.device('cpu'))
            loader = DataLoader((data if window == 'random' else prefix).to_dict('records'),
                                batch_size=32,shuffle=False,collate_fn=collator)
            for order in ['random','fixed']:
                model.column_order = None if order == 'random' else list(model.tgt_columns)
                for replicate in range(6):
                    set_random_state(20261010+replicate)
                    loss = _calculate_val_loss(model, loader)
                    rows.append(dict(seed=seed,window=window,column_order=order,replicate=replicate,check_loss=loss))
        assert digest(ws.model_tabular_weights_path) == before
        hashes[str(seed)] = before
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['seed','window','column_order']).check_loss.agg(['mean','std','min','max']).reset_index()
    fixed = summary[summary.window.eq('prefix') & summary.column_order.eq('fixed')]
    assert fixed['std'].max() < 1e-6
    frame.to_csv(DOCS/'validation_noise.csv',index=False)
    summary.to_csv(DOCS/'validation_noise_summary.csv',index=False)
    write(DOCS/'validation_noise.json',dict(checkpoint_hashes=hashes,checkpoints_unchanged=True,
        protocol_sha256=digest(DOCS/'VALIDATION_PROBE_PROTOCOL.md'),script_sha256=digest(__file__),
        repeated_evaluations=len(rows),training_updates=0,test_outcomes_accessed=False))
    print(summary.to_string(index=False))


if __name__ == '__main__': main()
