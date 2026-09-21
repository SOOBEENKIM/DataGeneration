"""Descriptive aggregation of the registered frozen-D diagnostic."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs-saf-amount-diagnostic-matplotlib')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.diagnose_cs_saf_external_amount import CONFIG, OUT, DOC, OLD, CONTROLS, survival, decompose, digest, write


def main():
    cfg = json.loads(CONFIG.read_text())
    summary, strata, components, effects, sampling = [], [], [], [], []
    verification = dict(passed=True, input_hashes_verified=True, new_fits=0, new_rollouts=0,
                        test_outcomes_accessed=False, max_probability_reduction_error=0.,
                        max_decomposition_error=0., arrays=0, independent_raw_rate_checks=0,
                        max_independent_raw_rate_error=0.)
    for name in cfg['datasets']:
        audit = json.loads((OUT/name/'DONE.json').read_text())
        cuts = np.array(list(audit['thresholds'].values()))
        cases = {}
        for label in audit['cases']:
            path = OUT/name/f'{label}.npz'
            assert digest(path) == audit['array_sha256'][path.name]
            z = dict(np.load(path)); cases[label] = z
            p, comp = survival(z, cuts)
            err = float(abs(p-z['probability']).max())
            assert err < 1e-15
            verification['max_probability_reduction_error'] = max(err,verification['max_probability_reduction_error'])
            verification['arrays'] += 1
            obs = z['amount'][:,None] > cuts
            if label == 'validation':
                seed, case = 0, 'validation'
            else:
                seed, case = label.split('_'); seed = int(seed)
            for j, cut in enumerate(cuts):
                row = dict(dataset=name,seed=seed,case=case,threshold=cfg['thresholds'][j],amount_threshold=cut,
                           events=len(p),expected_exceedance=p[:,j].mean(),
                           observed_query_amount_exceedance=obs[:,j].mean(),
                           difference=p[:,j].mean()-obs[:,j].mean())
                # Brier is meaningful only when actual query history/amount pair is intact.
                row['brier'] = float(((p[:,j]-obs[:,j])**2).mean()) if case in ('validation','RR','GG') else None
                summary.append(row)
                for c in range(3):
                    components.append(dict(dataset=name,seed=seed,case=case,threshold=cfg['thresholds'][j],
                        component=c,mean_weight=z['weight'][:,c].mean(),mean_log_sigma=z['sigma'][:,c].mean(),
                        q99_log_sigma=np.quantile(z['sigma'][:,c],.99),
                        tail_probability_contribution=comp[:,c,j].mean()))
            groupings = dict(position=np.digitize(z['position'],cfg['position_edges']),mark=z['mark'])
            for kind, codes in groupings.items():
                for code in np.unique(codes):
                    mask = codes == code
                    for j in range(len(cuts)):
                        strata.append(dict(dataset=name,seed=seed,case=case,group=kind,code=int(code),events=int(mask.sum()),
                            threshold=cfg['thresholds'][j],expected_exceedance=p[mask,j].mean(),
                            observed_query_amount_exceedance=obs[mask,j].mean()))
        for seed in cfg['saved_generation_seeds']:
            p = {c:cases[f'{seed}_{c}']['probability'] for c in cfg['cases']}
            first = cases[f'{seed}_RR']['position'] == 0
            assert np.max(abs(p['GR'][first]-p['RR'][first])) < 1e-12
            assert np.max(abs(p['GG'][first]-p['RG'][first])) < 1e-12
            decomposition = decompose(p['RR'],p['GR'],p['RG'],p['GG'])
            sub = decompose(p['RG'],p['AG'],p['OG'],p['GG'])
            error = max(float(abs(decomposition['history']+decomposition['current']-decomposition['total']).max()),
                        float(abs(sub['history']+sub['current']-sub['total']).max()))
            assert error < 1e-15
            verification['max_decomposition_error'] = max(error,verification['max_decomposition_error'])
            for j in range(len(cuts)):
                effects.append(dict(dataset=name,seed=seed,threshold=cfg['thresholds'][j],
                    generated_vs_real_total=decomposition['total'][:,j].mean(),
                    history_symmetric=decomposition['history'][:,j].mean(),
                    current_symmetric=decomposition['current'][:,j].mean(),
                    history_current_interaction=decomposition['interaction'][:,j].mean(),
                    history_at_real_current=(p['GR']-p['RR'])[:,j].mean(),
                    history_at_generated_current=(p['GG']-p['RG'])[:,j].mean(),
                    amount_history_at_generated_current=sub['history'][:,j].mean(),
                    other_history_at_generated_current=sub['current'][:,j].mean(),
                    amount_other_interaction=sub['interaction'][:,j].mean()))
                gg = cases[f'{seed}_GG']; observed = gg['amount'] > cuts[j]
                sampling.append(dict(dataset=name,seed=seed,threshold=cfg['thresholds'][j],events=len(observed),
                    predicted_count=p['GG'][:,j].sum(),observed_count=int(observed.sum()),
                    count_difference=observed.sum()-p['GG'][:,j].sum()))
        # RR is seed-independent; repeated evaluation must be identical.
        s1,s2 = cfg['saved_generation_seeds']
        np.testing.assert_array_equal(cases[f'{s1}_RR']['probability'],cases[f'{s2}_RR']['probability'])
    frames = dict(tail_summary=pd.DataFrame(summary),stratified_tail=pd.DataFrame(strata),
                  mixture_components=pd.DataFrame(components),paired_effects=pd.DataFrame(effects),
                  sampling_check=pd.DataFrame(sampling))
    # Independently count original parquet amounts without the sequence adapter.
    for name in cfg['datasets']:
        events = pd.read_parquet(OLD/'input'/name/'events.parquet', columns=['entity_id','amount_or_numeric_value'])
        roles = pd.read_parquet(OLD/'input'/name/'roles.parquet')
        validation_ids = roles.loc[roles.role.eq('validation'),'entity_id']
        amount = events.loc[events.entity_id.isin(validation_ids),'amount_or_numeric_value'].to_numpy()
        plan = pd.read_parquet(OLD/'input'/name/'plan.parquet')
        multiplicity = plan.source_entity_id.value_counts()
        weighted = events.entity_id.map(multiplicity).fillna(0).to_numpy()
        audit = json.loads((OUT/name/'DONE.json').read_text())
        for rel, sha in audit['files'].items():
            assert digest(ROOT/rel) == sha
        for threshold,cut in audit['thresholds'].items():
            checks = [(0,'validation',float((amount>cut).mean()))]
            matched = float(np.sum(weighted*(events.amount_or_numeric_value.to_numpy()>cut))/weighted.sum())
            for seed in cfg['saved_generation_seeds']:
                generated=pd.read_parquet(CONTROLS/'runs'/name/cfg['model']/f'generated_raw_{seed}.parquet',columns=['amount_or_numeric_value'])
                checks.extend([(seed,'RR',matched),(seed,'GG',float((generated.amount_or_numeric_value>cut).mean()))])
            for seed,case,expected in checks:
                t=frames['tail_summary']; actual=t.loc[(t.dataset==name)&(t.seed==seed)&(t.case==case)&(t.threshold==threshold),'observed_query_amount_exceedance'].item()
                err=abs(actual-expected);assert err<1e-15
                verification['independent_raw_rate_checks']+=1
                verification['max_independent_raw_rate_error']=max(err,verification['max_independent_raw_rate_error'])
    for label, frame in frames.items():
        frame.to_csv(DOC/f'{label}.csv',index=False)
    write(DOC/'verification.json',verification)
    plt.rcParams.update({'font.size':10})
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    table = frames['tail_summary']; case_order=['RR','GR','RG','GG','AG','OG']
    for ax,name in zip(axes,cfg['datasets']):
        t=table[(table.dataset==name)&(table.threshold=='fit_q999')]
        x=np.arange(6)
        for i,seed in enumerate(cfg['saved_generation_seeds']):
            sub=t[t.seed==seed].set_index('case').loc[case_order]
            ax.plot(x+(i-.5)*.07,100*sub.expected_exceedance,marker='o',lw=1,label=f'Saved draw {i+1}')
        observed=t[t.case=='RR'].observed_query_amount_exceedance.iloc[0]*100
        ax.axhline(observed,color='black',ls='--',lw=1,label='Matched real exceedance')
        ax.set_xticks(x,case_order);ax.set_ylabel('Predicted amount above fit q99.9 (%)')
        ax.set_title(name.capitalize());ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylim(bottom=0)
    fig.suptitle('Frozen D: crossed histories and current gap/mark\nRR/GR/RG/GG: history/current; AG: generated past amount only; OG: generated other past fields')
    fig.savefig(DOC/'paired_tail.png',dpi=180);fig.savefig(DOC/'paired_tail.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,name in zip(axes,cfg['datasets']):
        t=table[(table.dataset==name)&(table.case=='validation')].set_index('threshold').loc[cfg['thresholds']]
        x=np.arange(4)
        ax.bar(x-.18,100*t.observed_query_amount_exceedance,.36,label='Observed validation')
        ax.bar(x+.18,100*t.expected_exceedance,.36,label='Predicted on real histories')
        ax.set_xticks(x,['Fit q99','Fit q99.9','Fit max','10 x fit max'])
        ax.set_ylabel('Exceedance rate (%)');ax.set_title(name.capitalize());ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Amount tail mismatch before generated-history feedback')
    fig.savefig(DOC/'real_history_tail.png',dpi=180);fig.savefig(DOC/'real_history_tail.pdf');plt.close(fig)
    print(table[table.threshold=='fit_q999'].to_string(index=False))
    print(frames['paired_effects'][frames['paired_effects'].threshold=='fit_q999'].to_string(index=False))
    print('VERIFICATION',json.dumps(verification))


if __name__=='__main__':main()
