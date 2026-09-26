"""Generate the preregistered common-support cohort from the completed CPAR fit."""
import json
import time
from run_sparkov_argn_control import OUT,DOCS,CFG,CORE,STATIC,pd,torch,seed,write,digest,native_frame
from sdv.sequential import PARSynthesizer
from experiments.sparkov_cpar_sampling import equivalent_sampling

def sample_supported(model=None):
    folder=OUT/'runs/cpar_20260928'
    fit=json.loads((folder/'FIT.json').read_text());assert digest(folder/'model.pkl')==fit['model_sha256']
    if model is None:model=PARSynthesizer.load(folder/'model.pkl')
    parent=pd.read_parquet(OUT/'prepared/validation_context.parquet')
    support=pd.read_parquet(OUT/'prepared/supported_validation_context.parquet')
    stub=pd.DataFrame(index=parent.index,columns=['customer_id',*CORE,*STATIC])
    for c in parent:stub[c]=parent[c].to_numpy()
    processed=model._data_processor.transform(stub)[['customer_id',*model.context_columns]]
    supported=[];failures={}
    for row in processed.itertuples(index=False,name=None):
        try:model._model._context_to_tensor(list(row[1:]));supported.append(row[0])
        except KeyError as error:failures[str(error)]=failures.get(str(error),0)+1
    assert set(supported)==set(support.customer_id)
    write(DOCS/'cpar_context_encoder_check.json',dict(requested_customers=len(parent),supported_customers=len(supported),native_errors=failures,observed_support_matches_registered_cohort=True,protocol_sha256=digest(DOCS/'CPAR_CONTEXT_SUPPORT_PROTOCOL.md'),model_sha256=fit['model_sha256']))
    processed=processed[processed.customer_id.isin(supported)]
    for gs in CFG['generation_seeds']:
        dest=folder/f'generated_supported_validation_{gs}.parquet'
        if dest.exists():continue
        seed(gs);torch.set_num_threads(1);model._data_processor.reset_sampling();start=time.monotonic()
        with equivalent_sampling():raw=model._sample(processed,sequence_length=None)
        raw.to_parquet(folder/f'native_supported_validation_{gs}.parquet',index=False)
        d=native_frame(raw);d.to_parquet(dest,index=False)
        assert digest(folder/'model.pkl')==fit['model_sha256']
        write(folder/f'generation_supported_validation_{gs}.json',dict(seconds=time.monotonic()-start,events=len(d),customers=d.entity_id.nunique(),future_lengths_supplied=False,execution='verified recurrent cache and vectorized exact conversion',script_sha256=digest(__file__),context_support_sha256=digest(OUT/'prepared/supported_validation_context.parquet')))
        print('CPAR GENERATED',gs,len(d),flush=True)

if __name__=='__main__':sample_supported()
