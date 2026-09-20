"""Frozen pair contrasts; report primary gains and every conditional/null cost."""
import csv,json
import numpy as np
from experiments.cs_saf_gap_calibration import ROOT,OUTPUT,CONFIG_SHA,contract,folder_for,verify,repeats,sha256,write_json
from experiments.cs_saf_generation_repeat_stats import repeat_statistics
from experiments.cs_saf_replication import seed_statistics
NAMES=('Ucal','Ugap','Ecal','Egap')
PAIRS=(('Egap','Ugap'),('Ugap','Ucal'),('Egap','Ecal'),('Ecal','Ucal'))
NULLS=((0,'0'),(0,'1'),(1,'0'))


def main():
    c=contract();end=json.loads((OUTPUT/'GRID_COMPLETE.json').read_text());verification=json.loads((OUTPUT/'verification.json').read_text())
    if verification['status']!='PASS' or verification['native_values_checked']!=2400:raise ValueError('independent complete verification required')
    records={};fits=[];manifests={};rows=[]
    for pi in c['prevalences']:
        pk=f'{pi:.2f}'
        for k in c['kappas']:
            for t in c['trials']:
                for parent,new in c['variants'].items():
                    folder=folder_for(pi,k,t,new);m=verify(folder)
                    if m['source_commit']!=end['source_commit']:raise ValueError('mixed sources')
                    fit=json.loads((folder/'fit.json').read_text());fits.append(fit);manifests[f'{pk}/{k}/{t}/{new}']=sha256(folder/'COMPLETE.json')
                    for name,dirname,prefix in [(new,folder,''),(parent,repeats.folder_for(pi,k,t)/parent,'control_')]:
                        conditional=json.loads((folder/(prefix+'conditional_accuracy.json')).read_text())['groups']
                        gen=[]
                        for r in c['repeats']:
                            if name==parent:repeats.verify(dirname/f'repeat_{r}')
                            report=json.loads((dirname/f'repeat_{r}/comparison.json').read_text());gen.append(report['metrics'])
                            for g,block in report['metrics'].items():rows.append(dict(prevalence=pi,kappa=k,trial=t,repeat=r,model=name,group=g,**block['metrics']))
                        records[(pk,k,t,name)]=dict(conditional=conditional,generation=gen)
    metrics=list(rows[0])[6:];vectors={};summaries={};contrasts={};worst_cell_means={}
    for pi in c['prevalences']:
        pk=f'{pi:.2f}';vectors[pk]={};summaries[pk]={};contrasts[pk]={};worst_cell_means[pk]={}
        for name in NAMES:
            vals={};worst_cell_means[pk][name]={}
            for metric in metrics:
                vals['active_'+metric]=np.array([[records[(pk,1,t,name)]['generation'][r]['context_1']['metrics'][metric] for r in c['repeats']] for t in c['trials']])
                vals['three_null_generation_'+metric]=np.array([[np.mean([records[(pk,k,t,name)]['generation'][r]['context_'+label]['metrics'][metric] for k,label in NULLS]) for r in c['repeats']] for t in c['trials']])
            cm=records[(pk,1,0,name)]['conditional']['1']['metrics']
            for metric in cm:
                vals['active_conditional_'+metric]=np.array([records[(pk,1,t,name)]['conditional']['1']['metrics'][metric]['mean'] for t in c['trials']])
                for k,label in NULLS:
                    vals[f'null_k{k}_c{label}_'+metric]=np.array([records[(pk,k,t,name)]['conditional'][label]['metrics'][metric]['mean'] for t in c['trials']])
                ns=np.stack([vals[f'null_k{k}_c{label}_'+metric] for k,label in NULLS])
                vals['three_null_'+metric]=ns.mean(0)
                if metric in ('copy_range','repeat_range'):
                    # Distinguish E_trial[max_cell(range)] from max_cell[E_trial(range)].
                    vals['mean_trial_worst_null_'+metric]=ns.max(0)
                    cell_means={f'k{k}_c{g}':float(ns[i].mean()) for i,(k,g) in enumerate(NULLS)}
                    worst_cell_means[pk][name][metric]=dict(cell_means=cell_means,
                        worst_cell=max(cell_means,key=cell_means.get),mean=max(cell_means.values()))
            vectors[pk][name]=vals
            summaries[pk][name]={m:(repeat_statistics(a) if a.ndim==2 else seed_statistics(a)) for m,a in vals.items()}
        for a,b in PAIRS:
            contrasts[pk][a+'_minus_'+b]={m:(repeat_statistics(v-vectors[pk][b][m]) if v.ndim==2 else seed_statistics(v-vectors[pk][b][m])) for m,v in vectors[pk][a].items()}
        v=vectors[pk];interaction={m:v['Egap'][m]-v['Ecal'][m]-v['Ugap'][m]+v['Ucal'][m] for m in v['Egap']}
        contrasts[pk]['factorial_interaction']={m:(repeat_statistics(a) if a.ndim==2 else seed_statistics(a)) for m,a in interaction.items()}
    screens={}
    joint_metrics=['active_gap_repeat_mi_error','active_conditional_grid_mark_TV','active_conditional_factual_mark_TV']+[f'null_k{k}_c{g}_{m}_range' for k,g in NULLS for m in ('copy','repeat')]
    for a,b in PAIRS:
        pair=a+'_minus_'+b;good=[];cost=[];joint=[];details={}
        for pk,values in contrasts.items():
            d=values[pair];p=d['active_short_gap_repeat_curve_l1']
            if p['mean']<0 and p['negative_count']>=4:good.append(pk)
            if p['mean']>0 and p['positive_count']>=4:cost.append(pk)
            failed=[m for m in joint_metrics if d[m]['mean']>0];details[pk]=failed
            if pk in good and not failed:joint.append(pk)
        screens[pair]=dict(primary_prevalences=good,cost_prevalences=cost,joint_prevalences=joint,
            primary_consistent_improvement=len(good)>=3,primary_consistent_cost=len(cost)>=3,joint_lead=len(joint)>=3,
            positive_cost_endpoints=details,exploratory_not_confirmatory=True)
    csvpath=ROOT/'docs/cs_saf/gap_calibration_v1_all_metrics.csv'
    with csvpath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result=dict(source_commit=end['source_commit'],config_sha256=CONFIG_SHA,new_calibration_fits=80,new_base_fits=0,new_generated_datasets=400,
        existing_generation_datasets=400,summaries=summaries,contrasts=contrasts,diagnostic_screens=screens,fits=fits,
        worst_null_cell_mean_ranges=worst_cell_means,
        shared_conditional_weight_verification=json.loads((OUTPUT/'shared_conditional_weights_check.json').read_text()),
        shared_conditional_weight_verification_sha256=sha256(OUTPUT/'shared_conditional_weights_check.json'),
        environment=json.loads((OUTPUT/'environment.json').read_text()),
        worst_null_aggregation_note='max of three cell means is separate from mean of per-trial maxima; joint screen checks each cell',
        technical_attempt_01=dict(unique_interrupted_fits=2,technical_reruns=2,
            amendment='docs/cs_saf/gap_calibration_v1_execution_amendment_01.md',
            rerun_identity=[json.loads((folder_for(.05,0,0,n)/'technical_rerun_identity.json').read_text()) for n in ('Ugap','Egap')]),
        bound_hits=sum(len(g['bound_hits']) for fit in fits for g in fit['groups'].values()),
        verification={k:v for k,v in verification.items() if k!='records'},verification_sha256=sha256(OUTPUT/'verification.json'),
        gates={g:json.loads((OUTPUT/g/'gate.json').read_text()) for g in ('cpu_gate','gpu_gate')},manifests=manifests,
        all_metrics_csv=dict(path=str(csvpath.relative_to(ROOT)),sha256=sha256(csvpath),rows=len(rows),values=len(rows)*len(metrics)),
        joint_screen_endpoints=joint_metrics,test_accessed=False,independent_confirmation=False,old_failures_unchanged=True)
    write_json(OUTPUT/'summary.json',result);write_json(ROOT/'docs/cs_saf/gap_calibration_v1_result.json',result)
    print(json.dumps(dict(screens=screens,bound_hits=result['bound_hits'],verification=result['verification']),indent=2))

if __name__=='__main__':main()
