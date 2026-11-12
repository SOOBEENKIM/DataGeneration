"""Real checkpoint integration: online episode memory and shrinking/reordered batches."""
import json
import numpy as np
import pandas as pd
import torch
from run_argn_gap_episode import (OUT,DOCS,OLD,CFG,foundation,inputs,folder,Workspace,get_cardinalities,
    get_ctx_sequence_length,load_model_weights,_translate_fixed_probs,_fix_rare_token_probs,write,digest,verify)
from benchmarks.argn_episode_control import episode_generation,episode_probabilities
from benchmarks.argn_state_adapter import FullHistoryCollator
from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX,SIDX_SUB_COLUMN_PREFIX,SDEC_SUB_COLUMN_PREFIX,encode_slen_sidx_sdec


@torch.no_grad()
def main():
    verify();_,codec,records,_=inputs();fs=CFG['fit_seeds'][0];device=torch.device('cpu')
    ws=Workspace(folder('B_event_label_first',fs)/'workspace');ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read()
    params=json.loads((OUT/'episode_parameters.json').read_text());old=json.loads(OLD.read_text())
    import mostlyai.engine._tabular.generation as generation
    with episode_generation(ts,old,foundation(fs)/'amount_head.pt',params,'joint'):
        model=generation.SequentialModel(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
            tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],
            ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),model_size=ws.model_tabular_configs.read()['model_units'],
            column_order=None,device=device)
        load_model_weights(model=model,path=ws.model_tabular_weights_path,device=device);model.eval()
    batch=FullHistoryCollator(True,None,device)(records['optimization'][:3]);context=model.context_compressor(batch)
    assert len(context[1])==0
    masks=_translate_fixed_probs(_fix_rare_token_probs(ts),ts);key=codec.prefixes['event_is_fraud']+'__cat'
    histories={i:[] for i in range(3)};ids=np.arange(3);length={0:30,1:300,2:10};history=None;state=None
    checks=0;maximum_error=0
    captured={}
    def hook(module,args,output):
        if args[1]==key:captured['p']=output.softmax(-1)[:,0,codec.codes['1']].numpy()
    handle=model.predictors.register_forward_hook(hook)
    for step in range(12):
        if step==6:
            ix=torch.tensor([2,0]);ids=ids[[2,0]]
            history=history[ix];state=tuple(s[:,ix] for s in state)
            context=([c[ix] for c in context[0]],[],[])
        fixed={}
        for prefix,values in [(SLEN_SUB_COLUMN_PREFIX,[length[i] for i in ids]),(SIDX_SUB_COLUMN_PREFIX,[step]*len(ids)),
                              (SDEC_SUB_COLUMN_PREFIX,[min(9,10*step//length[i]) for i in ids])]:
            df=encode_slen_sidx_sdec(pd.Series(values),ts['seq_len']['max'],prefix)
            fixed.update({k:torch.tensor(df[k].to_numpy()).reshape(-1,1) for k in df})
        rows=[]
        for i in ids:
            labels=histories[i];prev=labels[-1] if labels else -1;age=0
            for v in reversed(labels):
                if v!=prev:break
                age+=1
            if step==0:
                from benchmarks.argn_episode_control import first_probability
                expected=first_probability(np.array([length[i]]),params['first_tree'])[0]
            else:
                expected=np.asarray(params['table'])[prev+1,min(age,21),int(any(labels)),labels[0]]
            rows.append(expected)
        outputs,history,state=model(None,mode='gen',batch_size=len(ids),context=context,history=history,
            history_state=state,fixed_values=fixed,fixed_probs=masks)
        np.testing.assert_allclose(captured['p'],rows,rtol=2e-6,atol=1e-7)
        maximum_error=max(maximum_error,float(np.max(np.abs(captured['p']-rows))));checks+=len(ids)
        labels=outputs[key].reshape(-1).eq(codec.codes['1']).long().tolist()
        for i,y in zip(ids,labels):histories[i].append(y)
        for j,i in enumerate(ids):
            assert state[3][0,j,0].item()==int(any(histories[i]))
            assert state[3][0,j,1].item()==histories[i][0]
    handle.remove()
    write(DOCS/'RUNTIME_INTEGRATION_CHECK.json',dict(passed=True,probability_checks=checks,max_probability_error=maximum_error,
        shrink_and_reorder=True,real_checkpoint=True,synthetic_fixed_lengths_for_unit_verification=True,
        full_transaction_generations=0,source_sha256=digest(__file__)))
    print('RUNTIME_INTEGRATION_PASSED',checks,maximum_error)


if __name__=='__main__':main()
