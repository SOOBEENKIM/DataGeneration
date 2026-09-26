"""Frozen-weight diagnostic using real prefixes, plus isolated information removal."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from run_sparkov_argn_control import OUT,DOCS,CORE,STATIC,LABEL,MERCHANT,AMOUNT,pd,np,torch,seed,write,digest
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities,get_sequence_length_stats,get_ctx_sequence_length,get_argn_name,SLEN_SUB_COLUMN_PREFIX
from mostlyai.engine._tabular.argn import SequentialModel
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.generation import _resolve_gen_column_order
from mostlyai.engine._tabular.encoding import encode_df,flatten_frame,_enrich_slen_sidx_sdec
from mostlyai.engine._tabular.training import BatchCollator
import torch.nn.functional as F

@contextmanager
def intervene(model,mode,merchant):
    handle=None
    if mode=='shuffle_current_merchant':
        def hook(module,args,result):
            result=dict(result);result[merchant]=result[merchant].roll(1,dims=0);return result
        handle=model.column_embedders.register_forward_hook(hook)
    elif mode=='zero_history_after_first':
        def hook(module,args,result):
            history,state=result;history=history.clone();history[:,1:]=0;return history,state
        handle=model.history_compressor.register_forward_hook(hook)
    try:yield
    finally:
        if handle is not None:handle.remove()

def load(folder):
    ws=Workspace(folder/'workspace');ts=ws.tgt_stats.read();cs=ws.ctx_stats.read();cfg=ws.model_tabular_configs.read()
    cardinalities=get_cardinalities(ts);lengths=get_sequence_length_stats(ts)
    # Generation uses this order; fix the same order for all diagnostic interventions.
    order=_resolve_gen_column_order(column_stats=ts['columns'],cardinalities=cardinalities,rebalancing=None,imputation=None,sample_seed=None,fairness=None)
    model=SequentialModel(tgt_cardinalities=cardinalities,tgt_seq_len_median=lengths['median'],tgt_seq_len_max=lengths['max'],ctx_cardinalities=get_cardinalities(cs),ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),model_size=cfg['model_units'],column_order=order,device=torch.device('cpu'))
    load_model_weights(model=model,path=ws.model_tabular_weights_path,device=torch.device('cpu'));model.eval()
    raw=pd.read_parquet(OUT/'prepared/validation.parquet').rename(columns={'entity_id':'customer_id'});raw.gap=raw.gap.fillna(0)
    encoded,_,key=encode_df(raw[['customer_id',*CORE]],ts,tgt_context_key='customer_id',n_jobs=1)
    encoded=flatten_frame(_enrich_slen_sidx_sdec(encoded,key,lengths['max']),key)
    for col in encoded:
        if col.startswith('tgt:'):encoded[col]=encoded[col].map(lambda x:x[:512])
    ctx=pd.read_parquet(OUT/'prepared/validation_context.parquet');context,pk,_=encode_df(ctx,cs,ctx_primary_key='customer_id',n_jobs=1)
    frame=encoded.merge(context,left_on=key,right_on=pk,validate='one_to_one').drop(columns=[key,pk])
    columns={c:get_argn_name(st['argn_processor'],st['argn_table'],st['argn_column']) for c,st in ts['columns'].items()}
    names={c:next(k for k in cardinalities if k.startswith(prefix+'__')) for c,prefix in columns.items()}
    assert all(sum(k.startswith(prefix+'__') for k in cardinalities)==1 for prefix in columns.values())
    return model,frame,columns,names,ts,ws

@torch.no_grad()
def probe(name):
    folder=OUT/'runs'/name;assert (folder/'FIT.json').exists();seed(20260928)
    model,frame,columns,names,ts,ws=load(folder);before=digest(ws.model_tabular_weights_path)
    collate=BatchCollator(True,None,torch.device('cpu'));totals=defaultdict(lambda:[0.,0,0,0.,0.])
    for start in range(0,len(frame),3):
        batch=collate(frame.iloc[start:start+3].to_dict('records'))
        mask=torch.zeros_like(batch[names['category']].squeeze(-1),dtype=torch.bool)
        for key in batch:
            if key.startswith(SLEN_SUB_COLUMN_PREFIX):mask |= batch[key].squeeze(-1).ne(0)
        position=torch.arange(mask.shape[1])[None,:].expand_as(mask)
        y=batch[names[LABEL]].squeeze(-1)
        groups={'all':mask,'first':mask&position.eq(0),'positions_1_99':mask&position.gt(0)&position.lt(100),'positions_100_511':mask&position.ge(100),'fraud':mask&y.eq(ts['columns'][LABEL]['codes']['1']),'normal':mask&y.eq(ts['columns'][LABEL]['codes']['0'])}
        previous=y.roll(1,dims=1)
        groups['fraud_onset']=groups['fraud']&position.gt(0)&previous.eq(ts['columns'][LABEL]['codes']['0'])
        groups['fraud_continuation']=groups['fraud']&position.gt(0)&previous.eq(ts['columns'][LABEL]['codes']['1'])
        for mode in ['teacher','shuffle_current_merchant','zero_history_after_first']:
            with intervene(model,mode,columns[MERCHANT]):output,_=model(batch,mode='trn')
            for field,key in names.items():
                target=batch[key].squeeze(-1);logits=output[key]
                loss=F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').reshape_as(target)
                correct=logits.argmax(-1).eq(target)
                fraud_prob=logits.softmax(-1)[...,ts['columns'][LABEL]['codes']['1']] if field==LABEL else torch.zeros_like(loss)
                true_probability=logits.softmax(-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
                for group,use in groups.items():
                    if field=='gap':use=use&position.gt(0)
                    a=totals[mode,field,group];a[0]+=float(loss[use].double().sum());a[1]+=int(use.sum());a[2]+=int(correct[use].sum());a[3]+=float(fraud_prob[use].double().sum());a[4]+=float(true_probability[use].double().sum())
    for field in ['gap',MERCHANT]:np.testing.assert_allclose(totals['teacher',field,'all'],totals['shuffle_current_merchant',field,'all'])
    for field in names:np.testing.assert_allclose(totals['teacher',field,'first'],totals['zero_history_after_first',field,'first'])
    assert digest(ws.model_tabular_weights_path)==before
    rows=[dict(run=name,intervention=k[0],field=k[1],group=k[2],events=v[1],nll=v[0]/v[1] if v[1] else None,accuracy=v[2]/v[1] if v[1] else None,mean_fraud_probability=v[3]/v[1] if v[1] and k[1]==LABEL else None,mean_true_probability=v[4]/v[1] if v[1] else None) for k,v in totals.items()]
    pd.DataFrame(rows).to_csv(DOCS/f'probe_{name}.csv',index=False)
    write(DOCS/f'probe_{name}.json',dict(weights_sha256=before,weights_unchanged=True,customers=len(frame),prefix_limit=512,uses_true_sequence_length_positional_encoding=True,interpretation='conditional teacher diagnostic; not free-running quality or architectural proof',script_sha256=digest(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');probe(p.parse_args().name)
