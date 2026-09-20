"""Independent scalar reductions and index/censoring checks on saved diagnostic outputs."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from experiments.cs_saf_history_diagnostic_v1 import DOC,OUT,ROOT,write,digest,parent


def main():
    done=json.loads((OUT/'DONE.json').read_text());audit=json.loads((DOC/'audit.json').read_text())
    for path,h in done['files'].items():assert digest(ROOT/path)==h
    for path,h in audit['source_inventory'].items():assert digest(ROOT/path)==h
    pred=pd.read_csv(DOC/'prediction_by_tape.csv');gen=pd.read_csv(DOC/'generation_by_tape.csv')
    maximum=0.;checks=0;states=0
    bounds={'all':{'all':(0,999)},'history':{'1-4':(1,5),'5-8':(5,9),'9-16':(9,17),'17-31':(17,32)},
        'run':{'1':(1,2),'2-3':(2,4),'4-7':(4,8),'8+':(8,33)}}
    for (k,t,src,tape,name),rows in pred.groupby(['kappa','trial','source','tape','predictor']):
        f=pd.read_parquet(OUT/f'events_k{k}_t{t}_{src}_{tape}_{name}.parquet')
        assert (f['run']<=f.history).all()
        for _,row in rows.iterrows():
            sub=f[f.group==row.group]
            if row.dimension!='all':
                lo,hi=bounds[row.dimension][row.bin];sub=sub[(sub[row.dimension]>=lo)&(sub[row.dimension]<hi)]
            assert len(sub)==row.transitions and sub.entity.nunique()==row.entities
            if not len(sub):continue
            p=sub.p.to_numpy();q=sub.q.to_numpy();y=sub.y.to_numpy()
            values=dict(bias=np.mean(p-q),mae=np.mean(abs(p-q)),squared_probability_error=np.mean((p-q)**2),
                brier=np.mean((p-y)**2),mark_nll=sub.mark_nll.mean(),sampled_residual=np.mean(y-p),
                continuation=np.mean(y),oracle_continuation=np.mean(q),mean_prediction=np.mean(p))
            for key,value in values.items():maximum=max(maximum,abs(value-row[key]));checks+=1
        if src!='real' and name==src:
            raw=pd.read_parquet(parent.folder(int(k),int(t),src)/f'evaluation/generated_A_{int(tape)}.parquet')
            # Independent scalar strict-prefix/run loop from decoded category strings.
            pieces=[]
            for entity,part in raw.groupby('entity_id',sort=False):
                part=part.sort_values('event_index');marks=part.receiver_or_mark.tolist();run=1
                for step in range(1,len(part)):
                    pieces.append((step,run,int(marks[step]==marks[step-1])))
                    run=run+1 if marks[step]==marks[step-1] else 1
            expected=np.asarray(pieces)
            np.testing.assert_array_equal(expected,f[['event_index','run','y']].to_numpy());states+=len(f)
    assert maximum<1e-12
    # Descriptive signed identity, explicitly not a causal attribution.
    rows=[]
    for k in (0,1):
        for group in (0,1):
            for dim in ('history','run'):
                for b in bounds[dim]:
                    ref=gen[(gen.kappa==k)&(gen.group==group)&(gen.dimension==dim)&(gen.bin==b)&(gen.source=='JOINT_BIN')]
                    for name in ('U','G'):
                        own=pred[(pred.kappa==k)&(pred.group==group)&(pred.dimension==dim)&(pred.bin==b)&(pred.source==name)&(pred.predictor==name)]
                        total=own.continuation.mean()-ref.continuation.mean()
                        prediction=own.bias.mean()
                        history=own.oracle_continuation.mean()-ref.oracle_continuation.mean()
                        sampling=own.sampled_residual.mean()+(ref.oracle_continuation-ref.continuation).mean()
                        assert abs(total-prediction-history-sampling)<1e-12
                        rows.append(dict(kappa=k,group=group,model=name,dimension=dim,bin=b,total=total,
                            same_history_prediction=prediction,history_composition=history,sampling_residual=sampling,
                            adequate=bool(own.adequate.all() and ref.adequate.all())))
    pd.DataFrame(rows).to_csv(DOC/'signed_descriptive_identity.csv',index=False)
    # Primary published scientific decisions remain byte-for-byte unchanged in Git parent.
    import subprocess
    previous='e19897b3adf1851ddfc59c11232ad898138f3f0f'
    for p in ('docs/cs_saf/rollout_calibration_v1/decision.json','docs/cs_saf/structure_v1/decision.json'):
        assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show',f'{previous}:{p}'],cwd=ROOT)
    write(DOC/'verification.json',dict(status='PASS',independent_scalar_checks=checks,
        scalar_max_error=maximum,independent_state_and_outcome_rows=states,
        input_hashes_unchanged=True,previous_failure_decisions_unchanged=True,
        diagnostic_runtime_seconds=done['end_time']-done['start_time'],
        fits=0,new_generations=0,gpu_used=False))
    print('VERIFIED',checks,'metrics;',states,'strict-prefix states; max error',maximum)


if __name__=='__main__':main()
