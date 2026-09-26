"""Separate crop/permutation variation in native validation at fixed weights."""
import argparse
from run_sparkov_argn_control import OUT,DOCS,pd,np,torch,seed,write,digest
from probe_sparkov_argn_control import load
from mostlyai.engine._tabular.training import BatchCollator,_calculate_val_loss
from torch.utils.data import DataLoader

@torch.no_grad()
def probe(name):
    folder=OUT/'runs'/name;seed(20261100)
    model,_,_,_,_,ws=load(folder);original_order=model.column_order
    frame=pd.read_parquet(OUT/'prepared/child/OriginalData/encoded-data/part.000000-val.parquet')
    before=digest(ws.model_tabular_weights_path)
    loader=DataLoader(frame.to_dict('records'),batch_size=4,shuffle=False,collate_fn=BatchCollator(True,100,torch.device('cpu')))
    rows=[]
    for mode in ['native_random_both','fixed_crop','fixed_order','fixed_both']:
        for repeat in range(20):
            seed(20261100+repeat)
            if mode in ['fixed_crop','fixed_both']:np.random.seed(20261100)
            model.column_order=original_order if mode in ['fixed_order','fixed_both'] else None
            loss=_calculate_val_loss(model,loader)
            rows.append(dict(run=name,mode=mode,repeat=repeat,loss=loss))
    result=pd.DataFrame(rows)
    assert result.loc[result['mode'].eq('fixed_both'),'loss'].std()<1e-6
    assert digest(ws.model_tabular_weights_path)==before
    result.to_csv(DOCS/f'validation_noise_{name}.csv',index=False)
    write(DOCS/f'validation_noise_{name}.json',dict(weights_unchanged=True,weights_sha256=before,customers=len(frame),batch_size=4,window=100,repeats=20,script_sha256=digest(__file__),protocol_sha256=digest(DOCS/'VALIDATION_NOISE_PROTOCOL.md')))
    print(result.groupby('mode').loss.agg(['mean','std','min','max']).to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');probe(p.parse_args().name)
