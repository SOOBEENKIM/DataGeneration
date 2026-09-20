"""Aggregate all registered runs, with an independent repeat-curve calculation."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_cs_saf_external_audit_v1 import OUT, CONTRACT, write, digest
from benchmarks.cof_seqgen_saf_metrics import fit_metric_state


def curve(frame, edges, prediction=None):
    f=frame.sort_values(['entity_id','event_index'],kind='stable')
    mask=f.event_index.to_numpy()>0
    bins=np.searchsorted(edges,f.gap.to_numpy(float)[mask],side='right')
    values=(f.receiver_or_mark.eq(f.groupby('entity_id').receiver_or_mark.shift()).to_numpy(float)[mask]
            if prediction is None else f[prediction].to_numpy(float)[mask])
    n=np.bincount(bins,minlength=len(edges)+1)
    assert (n>0).all()
    sums=np.bincount(bins,weights=values,minlength=len(edges)+1)
    return sums/n,n


def main(allow_partial=False):
    c=json.loads(CONTRACT.read_text()); report=ROOT/'docs/cs_saf/external_audit_v1'
    metrics=pd.read_csv(report/'metrics.csv'); records=[]; binrows=[]; fits=[]
    verification=[]
    for k in c['kappas']:
        inp=OUT/f'kappa_{k}/input'
        tr=pd.read_parquet(inp/'train_events.parquet')
        tr['event_index']=tr.groupby('entity_id').cumcount(); tr.loc[tr.event_index==0,'gap']=np.nan
        tp=pd.read_parquet(inp/'train_context.parquet'); vp=pd.read_parquet(inp/'validation_context.parquet')
        val=pd.read_parquet(inp/'validation_canonical.parquet'); plan=pd.read_parquet(inp/'plan.parquet')
        for seed in c['training_seeds']:
            f=OUT/f'kappa_{k}/seed_{seed}'
            if allow_partial and not (f/'replay_checks.json').exists():
                continue
            done=json.loads((f/'DONE.json').read_text()); check=json.loads((f/'replay_checks.json').read_text())
            prog=pd.read_csv(f/'workspace/ModelStore/model-data/progress-messages.csv')
            best=prog[prog.is_checkpoint==1].iloc[-1]
            fits.append(dict(kappa=k,seed=seed,parameters=check['parameters'],last_epoch=float(prog.epoch.max()),
                selected_epoch=float(best.epoch),fit_seconds=done['fit_seconds'],weights_sha256=done['weights_sha256'],
                final_logged_loss=float(prog.iloc[-1].val_loss),selected_logged_loss=float(best.val_loss),
                stopped_at_epoch_cap=bool(prog.epoch.max()>=c['max_epochs']),
                replay_checks=check))
            real_pred=pd.read_parquet(f/'replay_validation.parquet')
            for group in ['pooled','context_0','context_1']:
                def ids(p):
                    return set(p.entity_id if group=='pooled' else p.loc[p.entity_label.astype(str)==group[-1],'entity_id'])
                ti,vi,gi=ids(tp),ids(vp),ids(plan)
                edges=fit_metric_state(tr[tr.entity_id.isin(ti)]).gap_bin_edges
                reference,n=curve(val[val.entity_id.isin(vi)],edges); weights=n/n.sum()
                teacher,_=curve(real_pred[real_pred.entity_id.isin(vi)],edges,'predicted_repeat')
                for tape in c['generation_seeds']:
                    generated=pd.read_parquet(f/f'replay_generated_{tape}.parquet')
                    generated=generated[generated.entity_id.isin(gi)]
                    predicted,_=curve(generated,edges,'predicted_repeat')
                    empirical,_=curve(generated,edges)
                    row=dict(kappa=k,seed=seed,tape=tape,group=group,
                        observed_history_curve_l1=float(weights@np.abs(teacher-reference)),
                        generated_history_reencoded_curve_l1=float(weights@np.abs(predicted-reference)),
                        native_generated_curve_l1=float(weights@np.abs(empirical-reference)),
                        reference_repeat_level=float(weights@reference),
                        observed_history_repeat_level=float(weights@teacher),
                        generated_repeat_level_reference_weighted=float(weights@empirical),
                        native_level_error=float(abs(weights@(empirical-reference))),
                        native_centered_shape_l1=float(weights@np.abs((empirical-weights@empirical)-(reference-weights@reference))))
                    records.append(row)
                    registered=metrics[(metrics.kappa==k)&(metrics.seed==seed)&(metrics.tape==tape)&(metrics.group==group)&(metrics.kind=='generated')].iloc[0].short_gap_repeat_curve_l1
                    verification.append(abs(registered-row['native_generated_curve_l1']))
                    for b in range(len(weights)):
                        binrows.append(dict(kappa=k,seed=seed,tape=tape,group=group,bin=b,weight=weights[b],
                            observed_repeat=reference[b],observed_history_prediction=teacher[b],
                            generated_history_reencoded_prediction=predicted[b],generated_repeat=empirical[b],
                            observed_history_bias=teacher[b]-reference[b],
                            history_and_condition_composition_shift=predicted[b]-teacher[b],
                            outcome_minus_reencoded_prediction=empirical[b]-predicted[b]))
    assert max(verification)<1e-12
    pd.DataFrame(records).to_csv(report/'prediction_diagnostics.csv',index=False)
    pd.DataFrame(binrows).to_csv(report/'repeat_curves.csv',index=False)
    write(report/'execution_summary.json',dict(contract_sha256=digest(CONTRACT),fits=fits,
         status='COMPLETE' if len(fits)==len(c['kappas'])*len(c['training_seeds']) else 'PARTIAL_GPU_WAIT',
         registered_fits=len(c['kappas'])*len(c['training_seeds']),
         completed_fits=len(fits),new_generated_datasets=len(fits)*5,new_generated_sequences=len(fits)*5*2048,
         independent_repeat_l1_max_error=max(verification),test_accessed=False,
         independent_new_DGP=False,version='official-mostlyai-engine-2.4.0',
         causal_exposure_bias_isolated=False))
    numeric=[x for x in metrics.columns if x not in ['kappa','seed','tape','kind','group']]
    summary=metrics.groupby(['kappa','kind','group'])[numeric].mean().reset_index()
    summary.to_csv(report/'metrics_mean.csv',index=False)
    print(summary[['kappa','kind','group','short_gap_repeat_curve_l1','gap_repeat_mi_error','gap_ks','mark_sparse_tv']].to_string(index=False))
    print(pd.DataFrame(records).groupby(['kappa','group'])[['observed_history_curve_l1','generated_history_reencoded_curve_l1','native_generated_curve_l1']].mean().to_string())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,(k,g,title) in zip(axes,[(1,'context_1','Relationship present / 10% group'),(1,'context_0','Relationship absent / 90% group'),(0,'context_1','Null data / 10% group')]):
        rr=summary[(summary.kappa==k)&(summary.group==g)]
        replay=pd.DataFrame(records); replay=replay[(replay.kappa==k)&(replay.group==g)]
        values=[float(rr[rr.kind=='roundtrip'].short_gap_repeat_curve_l1.iloc[0]),replay.observed_history_curve_l1.mean(),replay.generated_history_reencoded_curve_l1.mean(),replay.native_generated_curve_l1.mean()]
        ax.bar(range(4),values,color=['#90a4ae','#3498db','#9075bb','#e67e22'])
        ax.set_xticks(range(4),['Encode/\ndecode','Observed\nhistory','Generated\nhistory*','Generated\noutcomes'])
        ax.set_title(title);ax.set_ylabel('Repeat-curve L1 (lower is better)')
    fig.text(.01,.015,f'{len(fits)}/4 fits completed. * Re-encoded histories with realized length controls; not an isolated causal decomposition.',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(report/'diagnostic_overview.png',dpi=180);fig.savefig(report/'diagnostic_overview.pdf');plt.close(fig)
    curves = pd.DataFrame(binrows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (k, title) in zip(axes, [(1, 'Relationship present / 10% group'), (0, 'Null data / 10% group')]):
        selected = curves[(curves.kappa == k) & (curves.group == 'context_1')]
        ref = selected.groupby('bin').observed_repeat.mean()
        ax.plot(ref.index + 1, ref.values, 'o-', color='#263238', label='Observed validation')
        for i, (seed, rows) in enumerate(selected.groupby('seed')):
            mean = rows.groupby('bin')[['observed_history_prediction', 'generated_repeat']].mean()
            color = ['#2471a3', '#d35400'][i]
            ax.plot(mean.index + 1, mean.observed_history_prediction, '--', color=color,
                    label=f'Real-history prediction / seed {i+1}')
            ax.plot(mean.index + 1, mean.generated_repeat, 'o-', color=color,
                    label=f'Generated repetition / seed {i+1}')
        ax.set_title(title); ax.set_xlabel('Train-defined gap bin (short to long)')
        ax.set_xticks(range(1, 6)); ax.set_ylim(0, 0.8)
    axes[0].set_ylabel('Observable repetition probability')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, 1))
    fig.savefig(report/'repeat_probability_curves.png', dpi=180)
    fig.savefig(report/'repeat_probability_curves.pdf'); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true')
    main(p.parse_args().allow_partial)
