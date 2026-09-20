"""Independent report arithmetic from saved features and raw generated events."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cs_saf_structure_v1 import DOC, OLD


def curve(frame, edges):
    # Use within-entity shift and explicit per-bin averages, not cached histograms.
    frame = frame.sort_values(['entity_id', 'event_index'])
    previous = frame.groupby('entity_id').receiver_or_mark.shift()
    keep = frame.event_index > 0
    y = (frame.receiver_or_mark[keep] == previous[keep]).to_numpy(float)
    bins = np.searchsorted(edges, frame.gap[keep].to_numpy(float), side='right')
    counts = np.array([(bins == b).sum() for b in range(5)])
    assert (counts > 0).all()
    rates = np.array([y[bins == b].mean() for b in range(5)])
    return counts / counts.sum(), rates


def main():
    summary = json.loads((DOC/'execution_summary.json').read_text())
    assert summary['status'] == 'COMPLETE'
    conditional = pd.read_csv(DOC/'conditional_metrics.csv', dtype={'group': str})
    generated = pd.read_csv(DOC/'generation_metrics.csv', dtype={'group': str})
    references = {}
    for k in (0, 1):
        inp = OLD/f'kappa_{k}/input'
        tr = pd.read_parquet(inp/'train_events.parquet')
        tr['event_index'] = tr.groupby('entity_id').cumcount()
        tr.loc[tr.event_index == 0, 'gap'] = np.nan
        va = pd.read_parquet(inp/'validation_canonical.parquet')
        tc = pd.read_parquet(inp/'train_context.parquet')
        vc = pd.read_parquet(inp/'validation_context.parquet')
        plan = pd.read_parquet(inp/'plan.parquet')
        for g in ('pooled', '0', '1'):
            def ids(c):
                return set(c.entity_id if g == 'pooled' else c[c.entity_label.astype(str) == g].entity_id)
            edges = np.unique(np.quantile(tr[tr.entity_id.isin(ids(tc))].gap.dropna(), np.linspace(0, 1, 6)[1:-1]))
            weights, rates = curve(va[va.entity_id.isin(ids(vc))], edges)
            references[k, g] = edges, weights, rates, ids(plan)
    ce, ge = [], []
    for parent in summary['inventory']:
        k, t, name = parent['kappa'], parent['trial'], parent['name']
        folder = ROOT/parent['path']; ev = folder/'evaluation'
        train = json.loads((folder/'TRAIN_DONE.json').read_text())
        assert train['matmul_allow_tf32'] is False and train['cudnn_allow_tf32'] is False
        assert train['source']['commit'] == '6afb05b4ec06b255d0cd9b74a4f9df8276ecbd9e'
        fit = json.loads((ev/'fit.json').read_text())['controls']
        vf = pd.read_parquet(ev/'validation_features.parquet')
        base = vf.p.to_numpy(float); y = vf.y.to_numpy(float)
        for variant in ('raw', 'gap', 'direct'):
            p = base.copy()
            if variant != 'raw':
                control = fit[variant]
                for g in (0, 1):
                    ix = vf.group.to_numpy() == g
                    bins = np.searchsorted(control['edges'][g], vf.gap_code.to_numpy()[ix], side='right')
                    coefficients = np.asarray(control['parameters'][g])[bins]
                    p[ix] = coefficients if variant == 'direct' else expit(logit(base[ix].clip(1e-9, 1-1e-9))+coefficients)
            obs = vf.obs_p.to_numpy(float) if variant == 'raw' else np.where(y == 1, p, vf.obs_p.to_numpy(float)*(1-p)/(1-base.clip(1e-9, 1-1e-9)))
            for g in ('pooled', '0', '1'):
                ix = np.ones(len(vf), bool) if g == 'pooled' else vf.group.to_numpy() == int(g)
                row = conditional[(conditional.kappa == k)&(conditional.trial == t)&(conditional.name == name)&
                                  (conditional.variant == variant)&(conditional.group == g)].iloc[0]
                ce.extend([abs(np.mean((p[ix]-y[ix])**2)-row.repeat_brier),
                           abs(-np.log(obs[ix].clip(1e-12, 1)).mean()-row.mark_nll)])
        done = json.loads((folder/'EVAL_DONE.json').read_text())
        for run in done['generations']:
            frame = pd.read_parquet(ev/run['file'])
            for g in ('pooled', '0', '1'):
                edges, w, real, ids = references[k, g]
                _, fake = curve(frame[frame.entity_id.isin(ids)], edges)
                l1 = np.dot(w, np.abs(fake-real))
                shape = np.dot(w, np.abs((fake-np.dot(w, fake))-(real-np.dot(w, real))))
                row = generated[(generated.kappa == k)&(generated.trial == t)&(generated.name == name)&
                                (generated.variant == run['variant'])&(generated.tape == run['tape'])&(generated.group == g)].iloc[0]
                ge.extend([abs(l1-row.short_gap_repeat_curve_l1), abs(shape-row.null_centered_shape_l1)])
    assert len(ce) == 324 and len(ge) == 972
    assert max(ce) < 2e-6 and max(ge) < 1e-12
    report = dict(status='PASS', conditional_values=len(ce), independent_conditional_max_error=max(ce),
                  generation_values=len(ge), independent_generation_max_error=max(ge),
                  precision_unchanged=True, training_implementation_unchanged=True,
                  scope='saved outputs only; no fitting, generation, selection or threshold change')
    (DOC/'independent_arithmetic.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
