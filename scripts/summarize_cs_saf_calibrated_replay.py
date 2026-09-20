"""Nested five-by-five summaries of every registered calibrated replay contrast."""
import csv,json
import numpy as np
from experiments.cs_saf_calibrated_replay import (ROOT,OUTPUT,CONFIG_SHA,contract,folder_for,verify,sha256,write_json,GROUPS,NAMES)
from experiments.cs_saf_generation_repeat_stats import repeat_statistics

COMPONENTS=('predictor_F','source_H','empirical_remainder_S','expected_difference','native_difference',
            'predictor_on_U','predictor_on_E','source_under_U','source_under_E','interaction')


def export_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),rows=len(rows))


def main():
    c=contract();terminal=json.loads((OUTPUT/'GRID_COMPLETE.json').read_text())
    verification=json.loads((OUTPUT/'verification.json').read_text())
    if verification['status']!='PASS' or verification['source_group_checks']!=1200:raise ValueError('complete independent verification required')
    cells={};manifests={};scalar=[];bins=[]
    for pi in c['prevalences']:
        for k in c['kappas']:
            for trial in c['trials']:
                folder=folder_for(pi,k,trial);m=verify(folder)
                if m['source_commit']!=terminal['source_commit']:raise ValueError('mixed sources')
                key=f'{pi:.2f}/{k}/{trial}';cells[key]=json.loads((folder/'replay.json').read_text());manifests[key]=sha256(folder/'COMPLETE.json')
                for path,record in cells[key]['paths'].items():
                    source,repeat=path.split('_repeat_');repeat=int(repeat)
                    for group,block in record['scores'].items():
                        base=dict(prevalence=pi,kappa=k,trial=trial,repeat=repeat,source=source,group=group)
                        scalar.append(dict(base,empirical_L1=block['empirical_L1'],
                            Ucal_curve_L1=block['expected_curve_L1']['Ucal'],Ecal_curve_L1=block['expected_curve_L1']['Ecal'],
                            **{m:block[m] for m in ('mark_TV_transition_mean','mark_TV_entity_mean','signed_repeat_difference','absolute_repeat_difference')},
                            **{f'{stage}_{m}':v for stage,values in block['stages'].items() for m,v in values.items()}))
                        for b in range(len(block['counts'])):
                            bins.append(dict(base,bin=b,count=block['counts'][b],reference_repeat=block['reference_repeat'][b],
                                reference_weight=block['reference_weights'][b],empirical_repeat=block['empirical_repeat'][b],
                                Ucal_repeat=block['predicted_repeat']['Ucal'][b],Ecal_repeat=block['predicted_repeat']['Ecal'][b],
                                matched_mark_TV=block['bin_mark_TV'][b],signed_repeat_difference=block['bin_signed_repeat_difference'][b]))
    summary={};matrices={};matched={};screens={}
    for pi in c['prevalences']:
        pk=f'{pi:.2f}';summary[pk]={};matrices[pk]={};matched[pk]={}
        for k in c['kappas']:
            kk=str(k);summary[pk][kk]={};matrices[pk][kk]={};matched[pk][kk]={}
            for group in GROUPS:
                summary[pk][kk][group]={name:repeat_statistics([[cells[f'{pk}/{k}/{j}']['decomposition'][str(r)][group][name]
                    for r in c['repeats']] for j in c['trials']]) for name in COMPONENTS}
                matrices[pk][kk][group]={};matched[pk][kk][group]={}
                for source in NAMES:
                    for target in NAMES:
                        matrices[pk][kk][group][target+'_on_'+source]=repeat_statistics([[cells[f'{pk}/{k}/{j}']['paths'][f'{source}_repeat_{r}']['scores'][group]['expected_curve_L1'][target]
                            for r in c['repeats']] for j in c['trials']])
                    matched[pk][kk][group][source]={m:repeat_statistics([[cells[f'{pk}/{k}/{j}']['paths'][f'{source}_repeat_{r}']['scores'][group][m]
                        for r in c['repeats']] for j in c['trials']]) for m in ('mark_TV_transition_mean','mark_TV_entity_mean','signed_repeat_difference','absolute_repeat_difference')}
    for name in ('predictor_F','source_H','empirical_remainder_S','native_difference'):
        screens[name]={}
        for direction in ('positive','negative'):
            qualified=[]
            for pk in summary:
                x=summary[pk]['1']['context_1'][name];sign=1 if direction=='positive' else -1
                count=x['positive_count'] if sign==1 else x['negative_count']
                if sign*x['mean']>=c['materiality'] and count>=c['minimum_directional_trials']:qualified.append(pk)
            screens[name][direction]=dict(qualifying_prevalences=qualified,consistently_material=len(qualified)>=c['minimum_prevalences'])
    # Recover the prior native contrast exactly; expected-score C is a different quantity.
    previous=json.loads((ROOT/'docs/cs_saf/generation_repeats_v1_result.json').read_text())
    max_native_summary=0.
    for pk in summary:
        a=summary[pk]['1']['context_1']['native_difference']['tape_values']
        b=previous['contrasts'][pk]['Ecal_minus_Ucal']['active_short_gap_repeat_curve_l1']['tape_values']
        max_native_summary=max(max_native_summary,float(np.max(abs(np.array(a)-b))))
    if max_native_summary>1e-12:raise ValueError('prior nested native difference mismatch')
    exports=dict(scalars=export_csv(ROOT/'docs/cs_saf/calibrated_replay_v1_scalars.csv',scalar),
        bins=export_csv(ROOT/'docs/cs_saf/calibrated_replay_v1_bins.csv',bins))
    result=dict(source_commit=terminal['source_commit'],config_sha256=CONFIG_SHA,source_paths=400,
        predictor_path_evaluations=800,new_fits=0,new_sequences=0,summary=summary,matrices=matrices,matched=matched,
        diagnostic_screens=screens,verification=verification,max_native_summary_error=max_native_summary,exports=exports,
        cell_manifests=manifests,gates={g:json.loads((OUTPUT/g/'gate.json').read_text()) for g in ('cpu_gate','gpu_gate')},
        max_sequential_error=max(v['max_sequential_error'] for v in cells.values()),
        max_formula_error=max(v['max_formula_error'] for v in cells.values()),
        raw_arrays_path=str(OUTPUT),test_accessed=False,independent_data_confirmation=False,
        unique_causal_attribution=False,prior_failure_decisions_unchanged=True)
    write_json(OUTPUT/'summary.json',result);write_json(ROOT/'docs/cs_saf/calibrated_replay_v1_result.json',result)
    print(json.dumps(dict(screens=screens,active={pk:{m:summary[pk]['1']['context_1'][m]['mean'] for m in COMPONENTS} for pk in summary},verification=verification),indent=2))

if __name__=='__main__':main()
