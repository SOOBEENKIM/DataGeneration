"""Registered nested comparisons of five tapes within five frozen trials."""
import json
from pathlib import Path
import numpy as np
from experiments.cs_saf_generation_repeats import ROOT,OUTPUT,CONFIG_SHA,contract,folder_for,verify,sha256,write_json
from experiments.cs_saf_generation_repeat_stats import repeat_statistics

PAIRS=(('Ecal','Ucal'),('Ucal','U'),('Ecal','E'),('E','U'),('Ecal','U'))
PRIMARY='active_short_gap_repeat_curve_l1'


def main():
    c=contract();terminal=json.loads((OUTPUT/'GRID_COMPLETE.json').read_text())
    expected={(p,k,t) for p in c['prevalences'] for k in c['kappas'] for t in c['trials']}
    if {tuple(v) for v in terminal['completed']}!=expected or terminal['new_generated_datasets']!=800:
        raise ValueError('all registered cells required')
    if terminal['config_sha256']!=CONFIG_SHA:raise ValueError('completed contract differs')
    source=terminal['source_commit'];cells={};manifests={};inputs={};sample_seconds=[]
    for pi,k,trial in sorted(expected):
        key=f'{pi:.2f}/{k}/{trial}';cell=folder_for(pi,k,trial);manifest=verify(cell)
        if manifest['source_commit']!=source or manifest['generated_datasets']!=20:raise ValueError('cell source or size differs')
        manifests[key]=sha256(cell/'COMPLETE.json');inputs[key]=json.loads((cell/'inputs.json').read_text())
        for name,item in inputs[key].items():
            if sha256(Path(item['checkpoint_path']))!=item['checkpoint_sha256']:
                raise ValueError('input checkpoint changed')
            if sha256(Path(item['reference_folder'])/'COMPLETE.json')!=item['reference_manifest_sha256']:
                raise ValueError('historical manifest changed')
        records={};plans=set()
        for name in c['models']:
            records[name]=[]
            for repeat in c['repeats']:
                folder=cell/name/f'repeat_{repeat}';m=verify(folder)
                if m['source_commit']!=source:raise ValueError('sample source differs')
                v=json.loads((folder/'comparison.json').read_text())
                if (v['prevalence'],v['kappa'],v['trial'],v['model'],v['repeat'])!=(pi,k,trial,name,repeat):
                    raise ValueError('generation identity differs')
                if v['sampling_seed']!=c['generation_seeds'][str(trial)][repeat]:raise ValueError('sampling seed changed')
                plans.add(v['sampling_plan_sha256']);records[name].append(v['metrics']);sample_seconds.append(v['sample_seconds'])
        if len(plans)!=1:raise ValueError('unpaired plans')
        cells[key]=records
    previous=json.loads((ROOT/'docs/cs_saf/calibration_u_control_v1_result.json').read_text())
    vectors={};summaries={};contrasts={};historical={};screens={}
    metrics=list(cells['0.05/1/0']['U'][0]['context_1']['metrics'])
    for pi in c['prevalences']:
        pk=f'{pi:.2f}';vectors[pk]={};summaries[pk]={};contrasts[pk]={};historical[pk]={}
        for name in c['models']:
            vec={}
            for metric in metrics:
                vec['active_'+metric]=np.array([[cells[f'{pk}/1/{j}'][name][r]['context_1']['metrics'][metric]
                    for r in c['repeats']] for j in c['trials']])
                vec['three_null_generation_'+metric]=np.array([[np.mean([
                    cells[f'{pk}/{k}/{j}'][name][r][group]['metrics'][metric]
                    for k,group in ((0,'context_0'),(0,'context_1'),(1,'context_0'))])
                    for r in c['repeats']] for j in c['trials']])
            vectors[pk][name]=vec;summaries[pk][name]={m:repeat_statistics(a) for m,a in vec.items()}
        for a,b in PAIRS:
            pair=a+'_minus_'+b
            contrasts[pk][pair]={m:repeat_statistics(vectors[pk][a][m]-vectors[pk][b][m]) for m in vectors[pk][a]}
            old=previous['contrasts'][pk][pair][PRIMARY];new=contrasts[pk][pair][PRIMARY]
            historical[pk][pair]=dict(original_single_tape=old,
                fresh_five_tape_trial_means=new['values'],fresh_minus_original_trial_differences=(np.array(new['values'])-old['values']).tolist(),
                original_within_fresh_trial_range=[lo<=x<=hi for x,(lo,hi) in zip(old['values'],new['trial_tape_ranges'])],
                old_tape_included_in_primary=False)
        contrasts[pk]['factorial_interaction']={m:repeat_statistics(
            vectors[pk]['Ecal'][m]-vectors[pk]['E'][m]-vectors[pk]['Ucal'][m]+vectors[pk]['U'][m])
            for m in vectors[pk]['U']}
    for a,b in PAIRS:
        pair=a+'_minus_'+b;benefit=[];cost=[];joint=[]
        for pk in contrasts:
            d=contrasts[pk][pair][PRIMARY]
            if d['mean']<0 and d['negative_count']>=4:
                benefit.append(pk)
                if (contrasts[pk][pair]['active_gap_repeat_mi_error']['mean']<=0 and
                    all(previous['contrasts'][pk][pair][m]['mean']<=0 for m in ('three_null_copy_range','three_null_repeat_range'))):
                    joint.append(pk)
            if d['mean']>0 and d['positive_count']>=4:cost.append(pk)
        screens[pair]=dict(benefit_qualifying_prevalences=benefit,cost_qualifying_prevalences=cost,
            joint_qualifying_prevalences=joint,consistent_primary_benefit=len(benefit)>=3,
            consistent_primary_cost=len(cost)>=3,joint_improvement_lead=len(joint)>=3,
            exploratory_not_confirmatory=True)
    native=json.loads((OUTPUT/'native_verification.json').read_text())
    if native['status']!='PASS' or native['metrics_checked']!=4800 or native['max_absolute_error']>1e-12:
        raise ValueError('independent raw-array verification missing')
    gates={name:json.loads((OUTPUT/name/'gate.json').read_text()) for name in ('cpu_gate','gpu_gate','metric_gate')}
    if any(g['decision']!='PASS' or g['source_commit']!=source for g in gates.values()):raise ValueError('same-source gates missing')
    result=dict(source_commit=source,config_sha256=CONFIG_SHA,new_generated_datasets=800,new_sequences=800*2048,
        frozen_model_states=160,new_neural_fits=0,new_calibration_fits=0,
        repeats_within_trial=5,training_trials=5,original_tape_excluded=True,
        summaries=summaries,contrasts=contrasts,historical_comparison=historical,diagnostic_screens=screens,
        all_cell_metrics=cells,input_checkpoints=inputs,cell_manifest_sha256=manifests,
        fixed_model_endpoint_contrasts={pk:{pair:{m:v for m,v in cols.items()
            if m.startswith('active_conditional_') or m.startswith('three_null_')}
            for pair,cols in contrasts0.items()} for pk,contrasts0 in previous['contrasts'].items()},
        fixed_endpoint_source_sha256=sha256(ROOT/'docs/cs_saf/calibration_u_control_v1_result.json'),
        fixed_endpoint_repeats_not_counted_as_new_observations=True,
        native_verification={k:v for k,v in native.items() if k!='records'},
        native_verification_sha256=sha256(OUTPUT/'native_verification.json'),
        gates={k:{a:b for a,b in v.items() if a!='records'} for k,v in gates.items()},
        mean_sample_seconds=float(np.mean(sample_seconds)),test_accessed=False,
        independent_data_confirmation=False,old_ER_failure_unchanged=True,
        between_trial_variation_includes_fixed_generation_plan=True)
    write_json(OUTPUT/'summary.json',result)
    compact=dict(result,full_evidence_path=str(OUTPUT/'summary.json'),full_evidence_sha256=sha256(OUTPUT/'summary.json'))
    write_json(ROOT/'docs/cs_saf/generation_repeats_v1_result.json',compact)
    print(json.dumps(dict(screens=screens,native=result['native_verification']),indent=2))


if __name__=='__main__':main()
