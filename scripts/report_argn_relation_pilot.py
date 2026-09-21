"""Frozen-screen aggregation and independent histogram/ECDF arithmetic checks."""
import json
from pathlib import Path
import sys
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_argn_relation_pilot import OUT, CONFIG, OLD, inputs, digest, original_manifest


def independent_features(frame, state):
    d = frame.sort_values(['entity_id', 'event_index'], kind='stable')
    has = d.event_index.to_numpy() > 0
    maps = {name: {s:i+1 for i,s in enumerate(state[key])}
            for name,key in [('category','roots'), ('receiver_or_mark','marks')]}
    category = d.category.fillna('<MISSING>').astype(str).map(maps['category']).fillna(0).to_numpy(int)
    merchant = d.receiver_or_mark.fillna('<MISSING>').astype(str).map(maps['receiver_or_mark']).fillna(0).to_numpy(int)
    previous = np.roll(category,1); previous[~has] = -1
    g = d.gap.to_numpy(float)
    gb = np.searchsorted(np.asarray(state['gap_edges']), g, side='left') + 1
    gb[g == 0] = 0; gb[g < 0] = -2; gb[~np.isfinite(g)] = -1
    a = d.amount_or_numeric_value.to_numpy(float)
    good = np.isfinite(a) & (a >= 0)
    ab = np.full(len(a), -1)
    ab[good] = np.searchsorted(np.asarray(state['amount_edges']), np.log1p(a[good]), side='left')
    return dict(gap=gb, root=category, merchant=merchant, previous=previous, amount=ab,
                has=has, raw_amount=a, lengths=d.groupby('entity_id').size().to_numpy())


def histogram_tv(a, b):
    aa, ac = np.unique(a, axis=0, return_counts=True)
    bb, bc = np.unique(b, axis=0, return_counts=True)
    hist = {tuple(np.atleast_1d(k)): float(v)/len(a) for k,v in zip(aa,ac)}
    for k,v in zip(bb,bc):
        key = tuple(np.atleast_1d(k)); hist[key] = hist.get(key,0.) - float(v)/len(b)
    return float(sum(abs(v) for v in hist.values())/2)


def ecdf_ks(a,b):
    a,b = np.sort(a),np.sort(b)
    support = np.unique(np.concatenate([a,b]))
    return float(np.max(np.abs(np.searchsorted(a,support,side='right')/len(a)
                               -np.searchsorted(b,support,side='right')/len(b))))


def independent_metrics(real, syn, state):
    r,s = independent_features(real,state),independent_features(syn,state)
    result = {}
    definitions = {
        'gap_tv':(['gap'],True), 'root_tv':(['root'],False), 'mark_tv':(['merchant'],False),
        'root_amount_joint_tv':(['root','amount'],False),
        'merchant_amount_joint_tv':(['merchant','amount'],False),
        'merchant_category_joint_tv':(['merchant','root'],False),
        'gap_root_transition_joint_tv':(['gap','previous','root'],True)}
    for metric,(keys,transitions) in definitions.items():
        a,b = np.column_stack([r[k] for k in keys]),np.column_stack([s[k] for k in keys])
        if transitions:
            a,b = a[r['has']],b[s['has']]
        result[metric] = histogram_tv(a,b)
    result['amount_ks'] = ecdf_ks(r['raw_amount'],s['raw_amount'])
    result['length_ks'] = ecdf_ks(r['lengths'],s['lengths'])
    result['invalid_amount_rate'] = float(np.mean(~np.isfinite(s['raw_amount']) | (s['raw_amount'] < 0)))
    rawgap = syn.loc[syn.event_index.gt(0),'gap'].to_numpy(float)
    result['invalid_gap_rate'] = float(np.mean(~np.isfinite(rawgap) | (rawgap < 0)))
    result['unknown_mark_rate'] = float(np.mean(s['merchant']==0))
    result['unknown_root_rate'] = float(np.mean(s['root']==0))
    return result


def main():
    cfg = json.loads(CONFIG.read_text())
    docs = ROOT / 'docs/argn_relation_pilot_v1'
    runs = {a:json.loads((OUT/a/'DONE.json').read_text()) for a in cfg['arms']}
    assert len({r['initial']['base_tensor_sha256'] for r in runs.values()}) == 1
    assert runs['G']['initial']['extra_sha256'] == runs['R']['initial']['extra_sha256']
    assert runs['G']['initial']['extra_parameters'] == runs['R']['initial']['extra_parameters'] == 2143
    frames,_,_,state = inputs()
    verification, rows, predrows = [],[],[]
    originals = original_manifest()
    for arm,run in runs.items():
        for field in ('category', 'receiver_or_mark', 'amount_or_numeric_value'):
            assert run['prediction']['counts'][field] == len(frames['validation'])
        assert run['prediction']['counts']['gap'] == len(frames['validation']) - frames['validation'].entity_id.nunique()
        assert originals == json.loads((OUT/arm/'original_manifest.json').read_text())
        assert run['config_sha256'] == digest(CONFIG)
        assert digest(OUT/arm/'workspace/ModelStore/model-data/model-weights.pt') == run['weights_sha256']
        for generation in run['generations']:
            seed = generation['seed']; path = OUT/arm/f'generated_{seed}.parquet'
            assert digest(path) == run['files'][path.name]
            sample = pd.read_parquet(path)
            independent = independent_metrics(frames['validation'], sample, state)
            diffs = {k:abs(v-generation['metrics'][k]) for k,v in independent.items()}
            assert max(diffs.values()) < 1e-12, (arm,seed,diffs)
            verification.append(dict(arm=arm,seed=seed,verified_metrics=len(diffs),
                max_difference=max(diffs.values()),generated_events=len(sample),sha256=digest(path)))
            rows.append(dict(arm=arm,generation_seed=seed,**generation['metrics']))
        predrows.append(dict(arm=arm,**run['prediction']['nll']))
    frame = pd.DataFrame(rows); means = frame.groupby('arm').mean(numeric_only=True)
    predictions = pd.DataFrame(predrows).set_index('arm')
    primary = cfg['primary_metric']
    contrasts = {}
    for control in ('A','G'):
        reference = float(means.loc[control,primary]); candidate = float(means.loc['R',primary])
        paired = frame.pivot(index='generation_seed',columns='arm',values=primary)
        wins = int((paired.R < paired[control]).sum())
        contrasts[control] = dict(relative_improvement=(reference-candidate)/reference,
            seed_wins=wins, passed=(reference-candidate)/reference >= cfg['required_mean_relative_improvement_against_A_and_G']
                and wins == cfg['required_generation_seed_wins_against_A_and_G'])
    guards = {k:dict(cost=float(means.loc['R',k]-means.loc['A',k]),
                    passed=bool(means.loc['R',k]-means.loc['A',k] <= cfg['generation_guard_absolute_cost']))
              for k in cfg['generation_guards']}
    prediction_guards = {k:dict(relative_cost=float(predictions.loc['R',k]/predictions.loc['A',k]-1),
        passed=bool(predictions.loc['R',k]/predictions.loc['A',k]-1 <= cfg['prediction_max_relative_cost_vs_A'])) for k in cfg['prediction_guards']}
    invalids = ['invalid_amount_rate','invalid_gap_rate','unknown_mark_rate','unknown_root_rate']
    valid = bool((frame.loc[frame.arm.eq('R'),invalids].to_numpy() == 0).all())
    passed = all(x['passed'] for x in contrasts.values()) and all(x['passed'] for x in guards.values()) \
        and all(x['passed'] for x in prediction_guards.values()) and valid
    summary = dict(screen='PASS' if passed else 'FAIL', primary_contrasts=contrasts,
        generation_guards=guards,prediction_guards=prediction_guards,validity_passed=valid,
        means=json.loads(means.to_json(orient='index')),prediction=predictions.to_dict(orient='index'),
        training={a:{k:r[k] for k in ('selected_epoch','last_epoch','optimizer_updates','best_check_loss','fit_seconds')}
                  for a,r in runs.items()}, originals_unchanged=True,
        equal_backbone_initialization=True,equal_extra_initialization=True,extra_parameters=2143,
        train_seeds=1,generation_seeds=2,datasets=1,independent_confirmation=False,
        config_sha256=digest(CONFIG),run_commits={a:json.loads((OUT/a/'START.json').read_text())['source_commit'] for a in runs})
    frame.to_csv(docs/'generation_metrics.csv',index=False)
    predictions.to_csv(docs/'prediction_metrics.csv')
    (docs/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (docs/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    for arm in runs:
        shutil_path = docs/f'{arm}_training.csv'
        shutil_path.write_bytes((OUT/arm/'history.csv').read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(11,3.2))
    labels = ['ARGN','ARGN + MLP','ARGN + interaction']
    metrics = [(primary,'Gap-category transition TV'),('root_amount_joint_tv','Category-amount TV')]
    for ax,(metric,title) in zip(axes,metrics):
        for i,arm in enumerate(cfg['arms']):
            values = frame.loc[frame.arm.eq(arm),metric].to_numpy()
            ax.bar(i,values.mean(),color=['#64748b','#0d9488','#2563eb'][i],alpha=.8)
            ax.scatter([i-.06,i+.06],values,color='black',s=15,zorder=3)
        ax.set_xticks(range(3),labels,rotation=15,ha='right');ax.set_title(title+' (lower better)')
        ax.set_ylim(bottom=0)
    axes[2].bar(range(3),[predictions.loc[a,'category'] for a in cfg['arms']],color=['#64748b','#0d9488','#2563eb'])
    axes[2].set_xticks(range(3),labels,rotation=15,ha='right')
    axes[2].set_title('Observed-history category NLL')
    fig.suptitle('Sparkov feasibility pilot | 1 fit seed, 2 generation seeds | '+summary['screen'])
    fig.tight_layout();fig.savefig(docs/'results.png',dpi=180);fig.savefig(docs/'results.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
