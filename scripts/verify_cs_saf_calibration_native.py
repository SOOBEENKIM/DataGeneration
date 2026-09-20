"""Independent NumPy contingency-table checks of saved native generation endpoints."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.cs_saf_calibration import OUTPUT,contract,parent
from data.cof_seqgen_saf_tensorizer import load_canonical_dataset
from scripts.materialize_cs_saf_prevalence import write_json


def frame_table(frame,edges):
    f=frame.sort_values(['entity_id','event_index'])
    ids=f.entity_id.to_numpy();marks=f.receiver_or_mark.to_numpy();gap=f.gap.to_numpy(float)
    mask=np.isfinite(gap)&(ids==np.roll(ids,1));mask[0]=False
    equality=(marks==np.roll(marks,1))[mask].astype(int)
    bins=np.searchsorted(edges,gap[mask],side='right')
    return np.bincount(2*bins+equality,minlength=2*(len(edges)+1)).reshape(-1,2)


def sample_table(sample,label,edges):
    mask=sample['valid_mask'].numpy().copy();mask[:,0]=False
    if label is not None:mask&=(sample['static_codes'].numpy()[:,None]==label+3)
    marks=sample['receiver'].numpy();repeat=(marks==np.roll(marks,1,axis=1))[mask].astype(int)
    gap=sample['gap'].numpy()[mask].astype(float)
    bins=np.searchsorted(edges,gap,side='right')
    return np.bincount(2*bins+repeat,minlength=2*(len(edges)+1)).reshape(-1,2)


def mi(table):
    p=table/table.sum();q=np.outer(p.sum(1),p.sum(0));nonzero=p>0
    return float((p[nonzero]*np.log(p[nonzero]/q[nonzero])).sum())


def scores(ref,syn):
    rc=ref.sum(1);sc=syn.sum(1)
    r=np.divide(ref[:,1],rc,out=np.zeros_like(rc,dtype=float),where=rc>0)
    s=np.divide(syn[:,1],sc,out=np.zeros_like(sc,dtype=float),where=sc>0)
    return float(np.dot(rc/rc.sum(),np.abs(r-s))),abs(mi(ref)-mi(syn))


def main():
    c=contract();records=[];largest=0.;torch.set_num_threads(1)
    if not (OUTPUT/'GRID_COMPLETE.json').exists():raise RuntimeError('complete grid required')
    for pi in c['prevalences']:
        for k in c['kappas']:
            dataset=load_canonical_dataset(ROOT/f'data/cs_saf/prevalence_v1/pi_{pi:.2f}_kappa_{k}',allowed_splits=('train','validation'))
            val=set(dataset.entity_ids_for_split('validation'));refs={};states={}
            for t in c['trials']:
                for name in ('U','E','Ecal','L003','L003cal'):
                    if name.endswith('cal'):
                        folder=OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_{t}/{name}'
                        comparison=folder/'comparison.json'
                    else:
                        folder=parent.folder_for(pi,k,t,name)
                        comparison=parent.OUTPUT/f'generation/pi_{pi:.2f}/trial_{t}/kappa_{k}/{name}/comparison.json'
                    report=json.loads(comparison.read_text());sample=torch.load(folder/'generated_sample.pt',map_location='cpu')['sample']
                    for group,label in (('pooled',None),('context_0',0),('context_1',1)):
                        block=report['metrics'][group];edges=np.array(block['train_metric_state']['gap_bin_edges'])
                        if group not in refs:
                            ids=val if label is None else val&set(dataset.static_context.loc[dataset.static_context.entity_label==label,'entity_id'])
                            refs[group]=frame_table(dataset.events[dataset.events.entity_id.isin(ids)],edges);states[group]=edges
                        np.testing.assert_array_equal(states[group],edges)
                        generated=sample_table(sample,label,edges)
                        actual=scores(refs[group],generated)
                        expected=[block['metrics']['short_gap_repeat_curve_l1'],block['metrics']['gap_repeat_mi_error']]
                        error=float(np.max(np.abs(np.array(actual)-expected)));largest=max(largest,error)
                        if error>1e-12:raise ValueError(f'native arithmetic differs: {pi} {k} {t} {name} {group} {error}')
                        records.append(dict(pi=pi,kappa=k,trial=t,model=name,group=group,
                            independent_repeat_L1=actual[0],independent_MI_error=actual[1],max_error=error,
                            reference_counts=refs[group].tolist(),generated_counts=generated.tolist()))
            print(f'native arithmetic verified pi={pi} kappa={k}',flush=True)
    result=dict(status='PASS',groups_checked=len(records),metrics_checked=2*len(records),
        max_absolute_error=largest,method='independent_numpy_counts_from_saved_marks_and_gaps',
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records)
    write_json(OUTPUT/'native_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
