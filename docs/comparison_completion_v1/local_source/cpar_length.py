"""Frozen CPAR with matched generated counts, explicitly a supplemental control."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sdv.sequential import PARSynthesizer
from run_sparkov_argn_control import CORE,STATIC,seed,native_frame,write,digest
from experiments.sparkov_cpar_sampling import equivalent_sampling
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/comparison_completion_v1'
def main():
    fs=int(sys.argv[1]);assert fs in [20261021,20261022,20261023];assert torch.cuda.is_available();torch.set_num_threads(2)
    original=ROOT/'artifacts/cpar_adequacy_v1' if fs==20261021 else OUT/f'cpar/{fs}'
    assert (original/'COMPLETE.json').exists()
    selected=json.loads((original/'SELECTION.json').read_text())['chosen_epochs'];directory=original/f'budget_{selected}'
    path=directory/'model.pkl';before=digest(path);model=PARSynthesizer.load(path)
    model._model.device=torch.device('cuda');model._model._model.to('cuda');model._model._model.eval()
    # The pickle's constructor still describes the 256-epoch trajectory; verify actual selected tensors.
    state=torch.load(original/f'run/epoch_{selected}.pt',map_location='cpu',weights_only=True)
    for k,v in model._model._model.state_dict().items():torch.testing.assert_close(v.cpu(),state[k],rtol=0,atol=0)
    out=OUT/f'cpar_length/{fs}';out.mkdir(parents=True,exist_ok=False)
    ctx=pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/validation_context.parquet')
    supported=json.loads((directory/'development_SUPPORT.json').read_text())['supported']
    stub=pd.DataFrame(index=ctx.index,columns=['customer_id',*CORE,*STATIC])
    for c in ctx:stub[c]=ctx[c].to_numpy()
    proc=model._data_processor.transform(stub)[['customer_id',*model.context_columns]]
    proc=proc[proc.customer_id.isin(supported)];assert len(proc)==len(supported)
    write(out/'NATIVE_LENGTH_BOUNDS.json',dict(min_length=model._model._min_length,max_length=model._model._max_length,fixed=model._model._fixed_length,selected_epochs=selected))
    for gs in [20261031,20261032]:
        count_path=ROOT/f'artifacts/argn_whole_validation_v1/runs/history_calibration_only_{fs}/generated_validation_{gs}.parquet'
        count_frame=pd.read_parquet(count_path,columns=['entity_id']);counts=count_frame.groupby('entity_id').size()
        seed(gs);model._data_processor.reset_sampling();parts=[];start=time.monotonic()
        model._model_kwargs['verbose']=False
        with equivalent_sampling():
            for i,(_,row) in enumerate(proc.iterrows()):
                length=int(counts.loc[row.customer_id]);raw=model._sample(row.to_frame().T,sequence_length=length)
                assert len(raw)==length,(len(raw),length);parts.append(raw)
                if i%20==0:print('COUNT_CONTROL',fs,gs,i,flush=True)
        raw=pd.concat(parts,ignore_index=True);d=native_frame(raw);target=out/f'generated_development_{gs}.parquet';d.to_parquet(target,index=False)
        assert d.groupby('entity_id').size().sort_index().equals(counts.loc[supported].sort_index())
        write(out/f'GENERATION_{gs}.json',dict(rows=len(d),sha256=digest(target),seconds=time.monotonic()-start,counts_sha256=digest(count_path),true_future_lengths=False,borrowed_labels=False))
    assert digest(path)==before
    write(out/'COMPLETE.json',dict(fit_seed=fs,generations=2,model_sha256=before,variant='CPAR+shared-synthetic-count',test_reads=0))
if __name__=='__main__':main()
