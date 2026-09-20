"""Export all completed pilot outcomes and static figures; no model selection."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/cs_saf/external_port_v1'
DOC = ROOT/'docs/cs_saf/external_port_v1'
ORDER = ['U', 'U+gap', 'G', 'G+gap', 'ARGN', 'CPAR-tail', 'Marginal', 'Transition']
COLORS = ['#5487b4', '#81a8cb', '#dc8b36', '#e6b26a', '#8571ac', '#be6a75', '#8b969d', '#429789']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def label(model, variant):
    if model in ('U', 'G'):
        return model+('+gap' if variant == 'gap' else '')
    if model == 'empirical':
        return variant.title()
    return 'CPAR-tail' if model == 'CPAR_tail' else model


def main():
    assert read(ART/'independent_verification.json')['passed']
    assert read(ART/'checkpoint_audit.json')['passed']
    native_metadata = {r['dataset']: r for r in read(ART/'native_metadata.json')}
    rows, predictions, trainings, results, traces = [], [], [], [], {}
    for dataset in ('berka', 'sparkov'):
        for model in ('U', 'G', 'ARGN', 'CPAR_tail', 'empirical'):
            folder = ART/'runs'/dataset/model
            done = read(folder/'DONE.json')
            start = read(folder/'START.json')
            results.append(dict(start=start, done=done))
            if model != 'empirical':
                tr = dict(dataset=dataset, model=model, fit_seed=start['fit_seed'],
                          fit_seconds=done['fit_seconds'], total_seconds=done['total_seconds'],
                          peak_reserved_mib=done['peak_reserved_bytes']/1024**2,
                          source_commit=start['source_commit'])
                if model in ('U', 'G'):
                    history = read(folder/'history.json')
                    traces[dataset+'_'+model] = history
                    tr.update(selected_epoch=done['selected_epoch'], last_epoch=done['last_epoch'],
                              optimizer_updates=done['optimizer_updates'], parameters=done['architecture']['parameters'],
                              stop_reason=done['stop_reason'])
                elif model == 'ARGN':
                    assert native_metadata[dataset]['load_verified']
                    assert native_metadata[dataset]['checkpoint_sha256'] == done['weights_sha256']
                    history = pd.read_csv(folder/'workspace/ModelStore/model-data/progress-messages.csv')
                    traces[dataset+'_'+model] = json.loads(history.to_json(orient='records'))
                    selected = history.loc[history.is_checkpoint.eq(1)].iloc[-1]
                    tr.update(selected_epoch=int(selected.epoch), last_epoch=int(history.iloc[-1].epoch),
                              optimizer_updates=int(history.iloc[-1].steps), parameters=native_metadata[dataset]['parameters'],
                              stop_reason='native_check_early_stop')
                else:
                    history = pd.read_csv(folder/'history.csv')
                    traces[dataset+'_'+model] = history.to_dict('records')
                    tr.update(selected_epoch=done['epochs'], last_epoch=done['epochs'], optimizer_updates=done['epochs'],
                              parameters=done['parameters'], stop_reason='registered_final_epoch')
                trainings.append(tr)
            for variant, result in done['results'].items():
                name = label(model, variant)
                if 'prediction' in result:
                    pred = result['prediction']
                    predictions.append(dict(dataset=dataset, model=name,
                        **{k: v for k, v in pred.items() if k != 'components'},
                        **{k+'_nll': v for k, v in pred['components'].items()}))
                for generation in result['generations']:
                    rows.append(dict(dataset=dataset, model=name, generation_seed=generation['seed'],
                                     generation_seconds=generation['seconds'], **generation['metrics']))
    frame = pd.DataFrame(rows)
    assert len(frame) == 32
    frame.to_csv(DOC/'generation_metrics.csv', index=False)
    numeric = frame.columns.drop(['dataset', 'model', 'generation_seed'])
    aggregate = frame.groupby(['dataset', 'model'])[numeric].agg(['mean', 'min', 'max'])
    aggregate.columns = [a+'_'+b for a, b in aggregate.columns]
    aggregate.reset_index().to_csv(DOC/'generation_summary.csv', index=False)
    pd.DataFrame(predictions).to_csv(DOC/'prediction_metrics.csv', index=False)
    pd.DataFrame(trainings).to_csv(DOC/'training_budgets.csv', index=False)
    # Descriptive lower bound, not a repaired model or replacement primary score:
    # changing only invalid-amount records moves at most that fraction of mass.
    support_bound = frame.loc[frame.model.isin(['U', 'U+gap', 'G', 'G+gap']),
        ['dataset', 'model', 'generation_seed', 'root_amount_joint_tv', 'invalid_amount_rate']].copy()
    support_bound['tv_lower_bound_after_changing_only_invalid_amount_records'] = (
        support_bound.root_amount_joint_tv-support_bound.invalid_amount_rate).clip(lower=0)
    support_bound.to_csv(DOC/'amount_support_bound.csv', index=False)
    write(DOC/'results.json', results)
    write(DOC/'training_traces.json', traces)
    preflight, support = {}, {}
    for name in ('berka', 'sparkov'):
        pre = read(ART/'input'/name/'preflight.json')
        preflight[name] = {k: v for k, v in pre.items() if k != 'tensorizer'}
        preflight[name]['transforms'] = dict(
            amount_codec=pre['tensorizer']['event_numeric_codecs'],
            gap_support=pre['tensorizer']['gap_support'],
            cardinalities={k: len(v['learned_keys']) for k, v in pre['tensorizer']['auxiliary_categorical_codecs'].items()})
        amount = pd.read_parquet(ART/'input'/name/'events.parquet', columns=['amount_or_numeric_value']).iloc[:, 0].to_numpy()
        support[name] = dict(development_events=len(amount), negative_amounts=int((amount<0).sum()),
            zero_amounts=int((amount==0).sum()), nonfinite_amounts=int((~np.isfinite(amount)).sum()),
            minimum=float(amount.min()), maximum=float(amount.max()))
    write(DOC/'input_verification.json', preflight)
    assert support == read(ART/'observed_amount_support.json')
    write(DOC/'observed_amount_support.json', support)
    for file in ('independent_verification.json', 'checkpoint_audit.json', 'gpu_profile.json', 'native_metadata.json'):
        shutil.copyfile(ART/file, DOC/file)
    exclusions = {}
    for name in ('berka', 'sparkov'):
        p = ART/'runs'/name/'CPAR'
        exclusions[name] = {file: read(p/file) for file in ('START.json', 'ABORTED_PROTOCOL.json', 'FAILED.json') if (p/file).exists()}
    write(DOC/'excluded_attempts.json', exclusions)
    write(DOC/'calibrations.json', {name+'_'+m: read(ART/'runs'/name/m/'calibration.json')
                                  for name in ('berka', 'sparkov') for m in ('U', 'G')})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for i, dataset in enumerate(('berka', 'sparkov')):
        relation = 'daily_gap_mark_joint_tv' if dataset == 'berka' else 'gap_root_transition_joint_tv'
        for j, metric in enumerate((relation, 'root_amount_joint_tv')):
            ax = axes[i, j]
            data = frame.loc[frame.dataset.eq(dataset)].groupby('model')[metric].agg(['mean', 'min', 'max']).reindex(ORDER)
            y = np.arange(len(ORDER))
            ax.barh(y, data['mean'], color=COLORS, alpha=.9)
            ax.errorbar(data['mean'], y, xerr=np.array([data['mean']-data['min'], data['max']-data['mean']]),
                        fmt='none', ecolor='#17222b', capsize=3, linewidth=1)
            for row, (_, values) in enumerate(data.iterrows()):
                ax.text(values['max']+.009, row, f"{values['mean']:.3f}", va='center', fontsize=9)
            ax.set_yticks(y, ORDER)
            ax.invert_yaxis()
            ax.set_xlim(0, data['max'].max()*1.2+.015)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            name = ('Day-gap x operation' if dataset == 'berka' else 'Gap x previous/current category') if j == 0 else ('Operation x amount' if dataset == 'berka' else 'Category x amount')
            ax.set_title(dataset.title()+': '+name, fontsize=11)
            ax.set_xlabel('Empirical joint-distribution TV (lower is better)')
    fig.suptitle('External pilot: one training seed, two generation seeds\nBars: mean; whiskers: generation range (not confidence intervals)', fontsize=13)
    fig.savefig(DOC/'primary_results.png', dpi=170)
    fig.savefig(DOC/'primary_results.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for i, dataset in enumerate(('berka', 'sparkov')):
        ax = axes[i, 0]
        for mode, color in [('U', COLORS[0]), ('G', COLORS[2])]:
            h = traces[dataset+'_'+mode]
            ax.plot([r['epoch'] for r in h], [r['check']['loss'] for r in h], label=mode+' check', color=color)
            ax.plot([r['epoch'] for r in h], [r['fit_loss'] for r in h], '--', label=mode+' fit', color=color, alpha=.65)
        ax.set_title(dataset.title()+': U/G component NLL')
        ax.legend(fontsize=8)
        h = pd.DataFrame(traces[dataset+'_ARGN'])
        axes[i, 1].plot(h.epoch, h.val_loss, label='check', color=COLORS[4])
        axes[i, 1].plot(h.epoch, h.trn_loss, '--', label='fit', color=COLORS[4], alpha=.6)
        axes[i, 1].set_title(dataset.title()+': ARGN native loss')
        axes[i, 1].legend(fontsize=8)
        h = pd.DataFrame(traces[dataset+'_CPAR_tail'])
        axes[i, 2].plot(h.Epoch+1, h.Loss, color=COLORS[5])
        axes[i, 2].set_title(dataset.title()+': CPAR native fit loss')
        for ax in axes[i]:
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.2)
    fig.suptitle('Different loss definitions and update counts: compare curves only within each model\nCPAR was still improving at the fixed 64-epoch cap; convergence is not established', fontsize=12)
    fig.savefig(DOC/'training_curves.png', dpi=170)
    plt.close(fig)
    table = ['| 모델 | Berka 날짜 간격–종류 | Berka 종류–금액 | Sparkov 간격–범주 전이 | Sparkov 범주–금액 |',
             '|---|---:|---:|---:|---:|']
    for model in ORDER:
        values = []
        for dataset, metric in [('berka', 'daily_gap_mark_joint_tv'), ('berka', 'root_amount_joint_tv'),
                                ('sparkov', 'gap_root_transition_joint_tv'), ('sparkov', 'root_amount_joint_tv')]:
            v = frame.loc[frame.dataset.eq(dataset) & frame.model.eq(model), metric].mean()
            values.append(f'{v:.4f}')
        table.append('| '+model+' | '+' | '.join(values)+' |')
    (DOC/'primary_table.md').write_text('\n'.join(table)+'\n')
    readme = (DOC/'README.md').read_text()
    before, rest = readme.split('<!-- primary:start -->', 1)
    _, after = rest.split('<!-- primary:end -->', 1)
    (DOC/'README.md').write_text(before+'<!-- primary:start -->\n'+'\n'.join(table)+'\n<!-- primary:end -->'+after)
    print('\n'.join(table))
    print('Exported', len(frame), 'generated datasets and', len(predictions), 'prediction rows.')


if __name__ == '__main__':
    main()
