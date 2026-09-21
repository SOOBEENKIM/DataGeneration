"""Read-only checkpoint and independent validation-score audit after all fits."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logsumexp
import torch
from data.cs_saf_external import TargetWindows
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from scripts.run_cs_saf_external_controls import OUT,OLD,CONFIG,build_model,digest,write


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    cfg=json.loads(CONFIG.read_text())
    result=dict(passed=True,new_fits=0,new_generations=0,test_outcomes_accessed=False,
                forward_reused_score_reductions_independent=True,rows=[],verifier_sha256=digest(__file__))
    for name in cfg['datasets']:
        data=torch.load(OLD/'input'/name/'prepared.pt',map_location='cpu')
        state=SAFTensorizerState.from_dict(data['state']);codec=state.event_numeric_codecs[0][1]
        windows=TargetWindows(data['sequences']['validation'],device='cuda')
        fit_events=sum(s.length for s in data['sequences']['fit'])
        for mid in cfg['new_models']:
            folder=OUT/'runs'/name/mid
            done=json.loads((folder/'DONE.json').read_text());start=json.loads((folder/'START.json').read_text())
            for f,h in start['scientific_source_sha256'].items():assert digest(ROOT/f)==h
            best=torch.load(folder/'best.pt',map_location='cpu');history=json.loads((folder/'history.json').read_text())
            assert best['epoch']==done['selected_epoch']==min(history,key=lambda x:x['check']['loss'])['epoch']
            assert done['optimizer_updates']==math.ceil(fit_events/cfg['ug']['batch_size'])*len(history)
            reference=torch.load(OLD/'runs'/name/('G' if mid[0]=='D' else mid[0])/'initial.pt',map_location='cpu')
            actual_initial=torch.load(folder/'initial.pt',map_location='cpu')
            architecture=json.loads((folder/'architecture.json').read_text())
            for k in architecture['shared_initial_keys']:assert torch.equal(actual_initial[k],reference[k]),k
            ast=json.loads((folder/'amount_state.json').read_text())
            model,_=build_model(mid,state,ast,reference);model.load_state_dict(best['state_dict']);model=model.cuda().eval()
            original_hash=digest(folder/'best.pt')
            for variant in cfg['variants']:
                model.calibration=None if variant=='raw' else json.loads((folder/'calibration.json').read_text())
                sums=dict(gap=0.,mark=0.,amount=0.,aux_0=0.,brier=0.,mae=0.);n=nt=0
                for first in range(0,len(windows),512):
                    o=model.target_outputs(**windows.batch(np.arange(first,min(first+512,len(windows)))))
                    row=np.arange(len(o['mark']));has=o['has_previous'].cpu().numpy()
                    gap=o['gap'].cpu().numpy()
                    # Match the frozen input representation, then reduce independently.
                    bounds=np.asarray(state.gap_support.upper_bounds,dtype=gap.dtype)
                    gc=np.searchsorted(bounds,np.nan_to_num(gap),side='left')
                    gl=model.gap_decoder.logits(o['hidden']).double().log_softmax(-1).cpu().numpy()
                    sums['gap']+=float(-gl[row[has],gc[has]].sum())
                    lp=o['logmark'].double().cpu().numpy();mark=o['mark'].cpu().numpy();prev=o['previous'].cpu().numpy()
                    sums['mark']+=float(-lp[row,mark].sum())
                    sums['brier']+=float(((np.exp(lp[row,prev])[has]-(mark==prev)[has])**2).sum())
                    value=o['numeric'].double().cpu().numpy();mu=o['location'].double().cpu().numpy()
                    sums['mae']+=float(abs(value-mu).sum())*codec.scale
                    if model.amount_kind=='legacy':
                        sd=o['scale'].double().cpu().numpy()
                        nll=np.log(sd)+.5*np.log(2*np.pi)+.5*((value-mu)/sd)**2
                    else:
                        p=model.amount_parameters(o['hidden'],o['gap'],o['mark'])
                        weights,loc,scale,zero=[x.double().cpu().numpy() for x in p]
                        is_zero=value==np.float32(-ast['codec_mean']/ast['codec_scale'])
                        u=value*ast['codec_scale']+ast['codec_mean'];u[is_zero]=1.
                        loga=u+np.log(-np.expm1(-u));x=(loga-ast['log_mean'])/ast['log_scale']
                        component=-.5*((x[:,None]-loc)/scale)**2-np.log(scale)-.5*np.log(2*np.pi)
                        nll=-logsumexp(weights+component,axis=1)+np.log(ast['log_scale'])-np.log(ast['codec_scale'])-u+loga
                        if ast['zero_rate']>0:
                            nll+=np.logaddexp(0,zero);nll[is_zero]=np.logaddexp(0,-zero[is_zero])
                    sums['amount']+=float(nll.sum())
                    al=o['auxlogits'][0].double().cpu().numpy();at=o['auxiliary'][0].cpu().numpy()
                    sums['aux_0']+=float(-al[row,at].sum())
                    n+=len(row);nt+=int(has.sum())
                components={k:sums[k]/(nt if k=='gap' else n) for k in ('gap','mark','amount','aux_0')}
                computed=dict(loss=sum(components.values()),repeat_brier=sums['brier']/nt,amount_log_mae=sums['mae']/n,**components)
                recorded=done['results'][variant]['prediction']
                expected={**{k:recorded[k] for k in ('loss','repeat_brier','amount_log_mae')},**recorded['components']}
                errors={k:abs(v-expected[k]) for k,v in computed.items()}
                assert max(errors.values())<2e-5,(name,mid,variant,errors)
                result['rows'].append(dict(dataset=name,model=mid,variant=variant,events=n,transitions=nt,
                    selected_epoch_verified=True,shared_initial_tensors_verified=True,optimizer_updates_verified=True,
                    max_absolute_error=max(errors.values()),errors=errors,checkpoint_sha256=original_hash))
            assert digest(folder/'best.pt')==original_hash
            del model
            print(name,mid,'checkpoint audit PASS',flush=True)
    path=OUT/'checkpoint_audit.json';assert not path.exists()
    write(path,result)


if __name__=='__main__':main()
