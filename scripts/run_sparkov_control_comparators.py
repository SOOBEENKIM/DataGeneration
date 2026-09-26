"""CPAR and descriptive resampling controls; no validation outcomes used to fit."""
import argparse
import importlib.metadata
import json
import time
from run_sparkov_argn_control import OUT,DOCS,CFG,CONFIG,CORE,STATIC,LABEL,MERCHANT,AMOUNT,pd,np,torch,seed,write,digest,native_frame

def empirical():
    d=pd.read_parquet(OUT/'prepared/train.parquet').sort_values(['entity_id','event_index'])
    ctx=pd.read_parquet(OUT/'prepared/validation_context.parquet')
    values=d[CORE].to_numpy();first=np.flatnonzero(d.event_index.eq(0));later=np.flatnonzero(d.event_index.gt(0))
    previous=d.groupby('entity_id')[['category',LABEL]].shift()
    pools={key:np.asarray(index) for key,index in previous.groupby(['category',LABEL]).groups.items()}
    # group indices use d's RangeIndex, inherited from parquet; assert explicitly.
    assert np.array_equal(d.index,np.arange(len(d)))
    lengths=d.groupby('entity_id').size().to_numpy()
    for arm in ['row_resampling','transition_resampling']:
        folder=OUT/'runs'/arm;folder.mkdir(parents=True,exist_ok=False)
        for gs in CFG['generation_seeds']:
            rng=np.random.default_rng(gs);pieces=[]
            for customer in ctx.customer_id:
                length=int(rng.choice(lengths));indices=np.empty(length,dtype=int);indices[0]=rng.choice(first)
                if arm=='row_resampling':indices[1:]=rng.choice(later,length-1)
                else:
                    for i in range(1,length):
                        prev=values[indices[i-1]];pool=pools.get((prev[3],prev[4]),later)
                        indices[i]=rng.choice(pool)
                chunk=pd.DataFrame(values[indices],columns=CORE);chunk.insert(0,'customer_id',customer);pieces.append(chunk)
            raw=pd.concat(pieces,ignore_index=True)
            for c in ['gap',AMOUNT]:raw[c]=pd.to_numeric(raw[c]).astype(float)
            native_frame(raw).to_parquet(folder/f'generated_validation_{gs}.parquet',index=False)
        write(folder/'FIT.json',dict(training_events=len(d),control='training row resampling; no privacy guarantee',context_used=False,native_length=False,length_source='empirical training customer length',condition='previous category and fraud' if arm.startswith('transition') else 'first vs subsequent position'))

def cpar():
    from sdv.metadata import SingleTableMetadata
    from sdv.sequential import PARSynthesizer
    from deepecho.models.par import PARModel
    from experiments.cs_saf_cpar_loss import equivalent_par_loss
    from experiments.sparkov_cpar_dense import equivalent_microbatch_training
    assert importlib.metadata.version('sdv')=='1.38.0'
    assert importlib.metadata.version('deepecho')=='0.8.1'
    folder=OUT/'runs/cpar_20260928';folder.mkdir(parents=True,exist_ok=False)
    d=pd.read_parquet(OUT/'prepared/train.parquet').rename(columns={'entity_id':'customer_id'})[['customer_id',*CORE]]
    d['gap']=d.gap.fillna(0)
    parent=pd.read_parquet(OUT/'prepared/train_context.parquet')
    data=d.merge(parent,on='customer_id',validate='many_to_one')
    for c in [MERCHANT,'category',LABEL,'cardholder_gender','cardholder_state']:data[c]=data[c].astype(str)
    metadata=SingleTableMetadata();metadata.detect_from_dataframe(data)
    metadata.update_column('customer_id',sdtype='id');metadata.set_sequence_key('customer_id')
    for c in data.columns.drop('customer_id'):
        metadata.update_column(c,sdtype='categorical' if c in [MERCHANT,'category',LABEL,'cardholder_gender','cardholder_state'] else 'numerical')
    model=PARSynthesizer(metadata,context_columns=STATIC,epochs=CFG['cpar_epochs'],segment_size=None,sample_size=1,cuda=False,verbose=True)
    seed(20260928);start=time.monotonic()
    write(folder/'START.json',dict(config_sha256=digest(CONFIG),script_sha256=digest(__file__),versions={p:importlib.metadata.version(p) for p in ['sdv','deepecho','torch','numpy','pandas']},training_events=len(data),training_customers=data.customer_id.nunique(),segment_size=None,epochs=CFG['cpar_epochs'],device='cpu',source_of_context='validation static attributes only'))
    def progress(par,optimizer,epoch):
        if epoch%8==0 or epoch==CFG['cpar_epochs']:
            torch.save(dict(model=par._model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch),folder/'latest_training_state.pt')
            par.loss_values.to_csv(folder/'history_running.csv',index=False)
        write(folder/'PROGRESS.json',dict(epoch=epoch,elapsed_seconds=time.monotonic()-start,loss=float(par.loss_values.iloc[-1].Loss)))
    with equivalent_par_loss(),equivalent_microbatch_training(progress):
        original=PARModel._compute_loss
        def logged_loss(self,*args):
            loss=original(self,*args)
            print(f'CPAR epoch={len(self.loss_values)+1} loss={float(loss):.9g} elapsed={time.monotonic()-start:.1f}',flush=True)
            return loss
        PARModel._compute_loss=logged_loss
        try:model.fit(data)
        finally:PARModel._compute_loss=original
    model.save(folder/'model.pkl');model.get_loss_values().to_csv(folder/'history.csv',index=False)
    write(folder/'FIT.json',dict(seconds=time.monotonic()-start,model_sha256=digest(folder/'model.pkl'),epochs=CFG['cpar_epochs']))
    from sample_sparkov_cpar_control import sample_supported
    sample_supported(model)

def sample_cpar(model,columns,folder):
    parent=pd.read_parquet(OUT/'prepared/validation_context.parquet')
    stub=pd.DataFrame(index=parent.index,columns=columns)
    for c in parent:stub[c]=parent[c].to_numpy()
    processed=model._data_processor.transform(stub)[['customer_id',*STATIC]]
    for gs in CFG['generation_seeds']:
        seed(gs);torch.set_num_threads(1);model._data_processor.reset_sampling();start=time.monotonic()
        raw=model._sample(processed,sequence_length=None)
        raw.to_parquet(folder/f'native_validation_{gs}.parquet',index=False)
        d=native_frame(raw);d.to_parquet(folder/f'generated_validation_{gs}.parquet',index=False)
        write(folder/f'generation_validation_{gs}.json',dict(seconds=time.monotonic()-start,events=len(d),customers=d.entity_id.nunique(),future_lengths_supplied=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['empirical','cpar']);a=p.parse_args()
    (empirical if a.model=='empirical' else cpar)()
