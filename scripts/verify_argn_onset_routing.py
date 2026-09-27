"""Real-checkpoint routing equivalence through first/onset/continuation and batch compaction."""
import json
import torch
from run_argn_onset_output import ROOT,OUT,DOCS,CFG,verify,inputs,write,digest
import run_argn_joint_preservation as study
from run_argn_amount_learning import load_parent,get_cardinalities,get_ctx_sequence_length,load_model_weights
from run_argn_amount_learning import _translate_fixed_probs,_fix_rare_token_probs
from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX
from benchmarks.argn_onset_output import onset_output_generation
import mostlyai.engine._tabular.generation as generation


@torch.no_grad()
def main():
    verify();_,codec,records,metas=inputs();checks=[]
    for fs in CFG['fit_seeds']:
        ws,native=load_parent(fs,torch.device('cpu'));c=study.context_cache(native,records['development'],metas['development'])['x'][:4]
        ts,cs=ws.tgt_stats.read(),ws.ctx_stats.read();labelkey=codec.prefixes['event_is_fraud']+'__cat'
        old=json.loads(study.OLD.read_text());ep=json.loads((study.PRIOR/'episode_parameters.json').read_text());hazard=json.loads((study.OUT/'hazard_parameters.json').read_text())
        with onset_output_generation(ts,old,study.foundation(fs)/'amount_head.pt',ep,hazard,study.OUT/f'worker_{fs}/unconditional/count_head.pt',True,
                onset_payload=OUT/f'worker_{fs}/onset_fit/onset_head.pt'):
            model=generation.SequentialModel(tgt_cardinalities=get_cardinalities(ts),ctx_cardinalities=get_cardinalities(cs),
                tgt_seq_len_median=ts['seq_len']['median'],tgt_seq_len_max=ts['seq_len']['max'],ctxseq_len_median=get_ctx_sequence_length(cs,key='median'),
                model_size=ws.model_tabular_configs.read()['model_units'],column_order=None,device=torch.device('cpu'))
            assert not any(k.startswith(('onset_head.','joint_count.')) for k in model.state_dict())
            load_model_weights(model=model,path=ws.model_tabular_weights_path,device=torch.device('cpu'));model.eval()
            captured={};logits={}
            def before(module,args):
                if args[1] in model.onset_head.keys:captured[args[1]]=torch.cat(args[0],-1)
            def after(module,args,output):
                if args[1] in model.onset_head.keys:logits[args[1]]=output
            h1=model.regressors.register_forward_pre_hook(before);h2=model.predictors.register_forward_hook(after)
            history=state=None;structural={};positions=0
            for step,labels in enumerate([torch.tensor([[0],[0],[1],[1]]),torch.tensor([[1],[0],[1],[0]]),torch.tensor([[1],[1]])]):
                if step==2:
                    selected=torch.tensor([3,0]);c=c[selected];history=history[selected];state=tuple(v[:,selected] for v in state)
                    structural={k:v[selected] for k,v in structural.items()}
                fixed={**structural,labelkey:torch.where(labels.eq(1),codec.codes['1'],codec.codes['0'])}
                generated,history,state=model(None,mode='gen',batch_size=len(c),context=([c],[],[]),history=history,history_state=state,
                    fixed_values=fixed,fixed_probs=_translate_fixed_probs(_fix_rare_token_probs(ts),ts))
                expected=torch.tensor([[False],[False],[False],[False]]) if step==0 else torch.tensor([[True],[False],[False],[False]]) if step==1 else torch.tensor([[True],[False]])
                for key,x in captured.items():
                    original=model.amount_heads(x,key,labels);adapted=model.onset_head(x,key,labels)
                    torch.testing.assert_close(logits[key],torch.where(expected.unsqueeze(-1),adapted,original),rtol=0,atol=0)
                    positions+=len(c)
                structural={k:v for k,v in generated.items() if k.startswith(SLEN_SUB_COLUMN_PREFIX)}
            h1.remove();h2.remove();checks.append(dict(fit_seed=fs,token_positions=positions,exact_routing=True,shrinking_reordered_batch=True))
    write(DOCS/'NATIVE_ROUTING_CHECK.json',dict(checks=checks,source_sha256=digest(__file__),test_events_read=False));print(checks)


if __name__=='__main__':main()
