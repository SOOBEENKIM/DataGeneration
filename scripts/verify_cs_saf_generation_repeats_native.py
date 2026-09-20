"""Independently recompute new L1 and MI from saved raw generation arrays."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.cs_saf_generation_repeats import ROOT,OUTPUT,contract,folder_for,verify,sha256,write_json
from data.cof_seqgen_saf_tensorizer import load_canonical_dataset
from scripts.verify_cs_saf_calibration_u_native import frame_table,sample_table,scores


def main():
    c=contract();records=[];largest=0.;torch.set_num_threads(1)
    if not (OUTPUT/'GRID_COMPLETE.json').exists():raise ValueError('complete grid required')
    for pi in c['prevalences']:
        for k in c['kappas']:
            dataset=load_canonical_dataset(ROOT/f'data/cs_saf/prevalence_v1/pi_{pi:.2f}_kappa_{k}',allowed_splits=('train','validation'))
            val=set(dataset.entity_ids_for_split('validation'));refs={};states={}
            for trial in c['trials']:
                cell=folder_for(pi,k,trial);verify(cell)
                plan=torch.load(cell/'plan.pt',map_location='cpu')
                inputs=json.loads((cell/'inputs.json').read_text())
                for name in c['models']:
                    for repeat in c['repeats']:
                        folder=cell/name/f'repeat_{repeat}'
                        report=json.loads((folder/'comparison.json').read_text())
                        saved=torch.load(folder/'generated_sample.pt',map_location='cpu');sample=saved['sample']
                        if saved['sampling_seed']!=c['generation_seeds'][str(trial)][repeat] or saved['sampling_seed']!=report['sampling_seed']:
                            raise ValueError('saved sampling seed differs')
                        if saved['model_state_sha256']!=inputs[name]['state_sha256'] or saved['plan_seed']!=plan['plan_seed']:
                            raise ValueError('saved fixed model or plan identity differs')
                        np.testing.assert_array_equal(sample['plan_train_indices'].numpy(),plan['positions'])
                        torch.testing.assert_close(sample['static_codes'],plan['contexts'],rtol=0,atol=0)
                        torch.testing.assert_close(sample['lengths'],plan['lengths'],rtol=0,atol=0)
                        torch.testing.assert_close(sample['valid_mask'],torch.arange(32)[None,:]<plan['lengths'][:,None],rtol=0,atol=0)
                        if report['sampling_plan_sha256']!=hashlib.sha256(plan['positions'].tobytes()).hexdigest():
                            raise ValueError('saved plan checksum differs')
                        for group,label in (('pooled',None),('context_0',0),('context_1',1)):
                            block=report['metrics'][group];edges=np.array(block['train_metric_state']['gap_bin_edges'])
                            if group not in refs:
                                ids=val if label is None else val&set(dataset.static_context.loc[dataset.static_context.entity_label==label,'entity_id'])
                                refs[group]=frame_table(dataset.events[dataset.events.entity_id.isin(ids)],edges);states[group]=edges
                            np.testing.assert_array_equal(states[group],edges)
                            generated=sample_table(sample,label,edges);actual=scores(refs[group],generated)
                            expected=[block['metrics']['short_gap_repeat_curve_l1'],block['metrics']['gap_repeat_mi_error']]
                            error=float(np.max(np.abs(np.array(actual)-expected)));largest=max(largest,error)
                            if error>1e-12:raise ValueError(f'independent native score mismatch {pi} {k} {trial} {name} {repeat} {group}')
                            records.append(dict(pi=pi,kappa=k,trial=trial,model=name,repeat=repeat,group=group,
                                repeat_L1=actual[0],MI_error=actual[1],max_error=error,
                                reference_counts=refs[group].tolist(),generated_counts=generated.tolist()))
            print(f'independent raw-array verification PASS pi={pi} kappa={k}',flush=True)
    result=dict(status='PASS',groups_checked=len(records),metrics_checked=2*len(records),
        max_absolute_error=largest,verifier_sha256=sha256(Path(__file__)),saved_sample_identity_checks=len(records)//3,records=records)
    write_json(OUTPUT/'native_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
