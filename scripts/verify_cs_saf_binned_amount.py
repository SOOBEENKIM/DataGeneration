"""Independent metric/expectation reductions and saved-checkpoint integrity."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from data.cs_saf_external import TargetWindows
from models.cs_saf_binned_amount import BinnedAmountD
from scripts.verify_cs_saf_external_port import transforms,distributions,distance,ks,sha as digest

OUT=ROOT/'artifacts/cs_saf/external_binned_amount_v1'
DOC=ROOT/'docs/cs_saf/external_binned_amount_v1'
CONFIG=ROOT/'configs/cs_saf_external_binned_amount_v1.json'
OLD=ROOT/'artifacts/cs_saf/external_port_v1'
PARENT=ROOT/'artifacts/cs_saf/external_controls_v1'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    cfg=json.loads(CONFIG.read_text())
    assert 'benchmarks.cs_saf_external' not in sys.modules
    result=dict(passed=True,metrics=0,tail_scalars=0,max_metric_error=0.,max_prediction_error=0.,
                max_tail_error=0.,new_fits=0,new_generations=0,test_outcomes_accessed=False,
                evaluator_imported=False,forward_reused_reductions_independent=True,
                verifier_sha256=digest(__file__),rows=[])
    for name in cfg['datasets']:
        folder=OUT/'runs'/name;done=json.loads((folder/'DONE.json').read_text())
        start=json.loads((folder/'START.json').read_text())
        for f,sha in start['scientific_source_sha256'].items():assert digest(ROOT/f)==sha
        for f,sha in done['files'].items():assert digest(folder/f)==sha
        inp=OLD/'input'/name;pre=json.loads((inp/'preflight.json').read_text())
        for f,sha in pre['files'].items():assert digest(inp/f)==sha
        events=pd.read_parquet(inp/'events.parquet');roles=pd.read_parquet(inp/'roles.parquet');plan=pd.read_parquet(inp/'plan.parquet')
        assert set(roles.role)=={'fit','check','validation'}
        frames={r:events[events.entity_id.isin(roles.loc[roles.role.eq(r),'entity_id'])] for r in ('fit','validation')}
        pool=dict(np.load(folder/'pool.npz'))
        np.testing.assert_array_equal(np.sort(pool['values']),np.sort(frames['fit'].amount_or_numeric_value.to_numpy(float)))
        assert (pool['counts']>0).all()
        state_metric=transforms(frames['fit'],name)
        reference,info_ref=distributions(frames['validation'],state_metric,name)
        cuts=np.r_[np.quantile(frames['fit'].amount_or_numeric_value,[.99,.999]),
                   frames['fit'].amount_or_numeric_value.max(),10*frames['fit'].amount_or_numeric_value.max()]
        np.testing.assert_array_equal(cuts,done['results']['raw']['tail_thresholds'])
        root='receiver_or_mark' if name=='berka' else 'category'
        for record in done['results']['raw']['generations']:
            gs=record['seed'];frame=pd.read_parquet(folder/f'generated_raw_{gs}.parquet')
            assert digest(folder/f'generated_raw_{gs}.parquet')==record['sha256']
            actual,info=distributions(frame,state_metric,name)
            computed={k:distance(reference[k],v) for k,v in actual.items()}
            computed.update(amount_ks=ks(info_ref['amounts'],info.pop('amounts')),
                            length_ks=ks(info_ref['lengths'],info.pop('lengths')),**info)
            for k,v in computed.items():
                expected=record['metrics'][k]
                if v is None:assert expected is None
                else:
                    err=abs(v-expected);assert err<1e-11,(k,err)
                    result['max_metric_error']=max(result['max_metric_error'],err)
                result['metrics']+=1
            np.testing.assert_array_equal(frame.groupby('entity_id').size().reindex(plan.entity_id),plan.length)
            np.testing.assert_array_equal(frame.event_index,frame.groupby('entity_id',sort=False).cumcount())
            rebuilt=frame.gap.fillna(0).groupby(frame.entity_id,sort=False).cumsum()
            np.testing.assert_allclose(frame.timestamp,rebuilt,rtol=1e-7,atol=1e-5)
            support=np.unique(pool['values']);amounts=frame.amount_or_numeric_value.to_numpy(float)
            ix=np.searchsorted(support,amounts);assert (ix<len(support)).all()
            np.testing.assert_array_equal(support[ix],amounts)
            for j,cut in enumerate(cuts):
                counts=[];rates=[]
                for source in (frames['validation'],frame):
                    roots=source[root].fillna('<MISSING>').astype(str).to_numpy()
                    above=source.amount_or_numeric_value.to_numpy()>cut
                    labels,n=np.unique(roots[above],return_counts=True)
                    counts.append(dict(zip(labels,n/len(source))));rates.append(float(above.mean()))
                labels=set(counts[0])|set(counts[1])
                l1=sum(abs(counts[0].get(k,0.)-counts[1].get(k,0.)) for k in labels)
                for key,value in zip(('real_rate','generated_rate','root_tail_l1'),(*rates,l1)):
                    error=abs(value-record['tails'][j][key]);assert error<1e-12
                    result['tail_scalars']+=1;result['max_tail_error']=max(result['max_tail_error'],error)
        data=torch.load(inp/'prepared.pt',map_location='cpu');state=SAFTensorizerState.from_dict(data['state'])
        ast=json.loads((folder/'amount_state.json').read_text())
        model=BinnedAmountD(state.gap_support,ast,pool,**state.model_config_kwargs())
        best=torch.load(folder/'best.pt',map_location='cpu');model.load_state_dict(best['state_dict']);model.to(a.device).eval()
        original=torch.load(PARENT/'runs'/name/cfg['reference']/'initial.pt',map_location='cpu')
        initial=torch.load(folder/'initial.pt',map_location='cpu')
        keys=done['architecture']['shared_initial_keys']
        for k in keys:assert torch.equal(initial[k],original[k])
        history=json.loads((folder/'history.json').read_text())
        assert best['epoch']==done['selected_epoch']==min(history,key=lambda r:r['check']['loss'])['epoch']
        assert done['optimizer_updates']==len(history)*math.ceil(len(frames['fit'])/cfg['training']['batch_size'])
        w=TargetWindows(data['sequences']['validation'],device=a.device)
        sums=dict(gap=0.,mark=0.,amount=0.,aux_0=0.,brier=0.,mae=0.);tail=np.zeros(4);n=nt=0
        # Construct empirical means/survival directly from every bin's raw pool.
        means=[];matrix=[]
        for lo,count in zip(pool['starts'],pool['counts']):
            values=pool['values'][lo:lo+count]
            means.append(np.log1p(values).sum()/count)
            matrix.append([(values>cut).sum()/count for cut in cuts])
        means=np.array(means);matrix=np.array(matrix)
        for first in range(0,len(w),512):
            o=model.target_outputs(**w.batch(np.arange(first,min(first+512,len(w)))))
            row=np.arange(len(o['mark']));has=o['has_previous'].cpu().numpy()
            g=o['gap'].cpu().numpy();bounds=np.asarray(state.gap_support.upper_bounds,dtype=g.dtype)
            gc=np.searchsorted(bounds,np.nan_to_num(g),side='left')
            gap_logp=model.gap_decoder.logits(o['hidden']).double().log_softmax(-1).cpu().numpy()
            sums['gap']-=gap_logp[row[has],gc[has]].sum()
            lp=o['logmark'].double().cpu().numpy();mark=o['mark'].cpu().numpy();prev=o['previous'].cpu().numpy()
            sums['mark']-=lp[row,mark].sum()
            sums['brier']+=((np.exp(lp[row,prev])[has]-(mark==prev)[has])**2).sum()
            al=o['auxlogits'][0].double().cpu().numpy();at=o['auxiliary'][0].cpu().numpy()
            sums['aux_0']-=al[row,at].sum()
            z=o['numeric'].cpu().numpy()
            code=np.searchsorted(pool['edges'],z,side='left')+int(pool['zero_bin'])
            if pool['zero_bin']:code[z==np.float32(-ast['codec_mean']/ast['codec_scale'])]=0
            # The logits are reused; normalization and all reductions are independent.
            features=model._event_core_features(o['hidden'],o['gap'],o['mark'],torch.zeros_like(o['gap']))[:,:-1]
            logits=model.value_head(features).double().cpu().numpy()
            logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
            sums['amount']-=np.log(prob[row,code]).sum()
            point=prob@means;actual=z.astype(float)*ast['codec_scale']+ast['codec_mean']
            sums['mae']+=abs(point-actual).sum();tail+=(prob@matrix).sum(0)
            n+=len(row);nt+=int(has.sum())
        comp={k:float(sums[k]/(nt if k=='gap' else n)) for k in ('gap','mark','amount','aux_0')}
        recomputed=dict(loss=sum(comp.values()),amount_log_mae=float(sums['mae']/n),repeat_brier=float(sums['brier']/nt),**comp)
        p=done['results']['raw']['prediction'];expected={**p,**p['components']}
        errors={k:abs(v-expected[k]) for k,v in recomputed.items()}
        assert max(errors.values())<2e-6,(name,errors)
        result['max_prediction_error']=max(result['max_prediction_error'],max(errors.values()))
        terr=float(abs(tail/n-done['results']['raw']['teacher_tail']).max());assert terr<2e-8
        result['max_tail_error']=max(result['max_tail_error'],terr)
        result['rows'].append(dict(dataset=name,generations=2,prediction_events=n,
            selected_epoch=best['epoch'],initial_tensors_verified=True,optimizer_updates_verified=True,
            exact_fit_pool_and_generated_support_verified=True,prediction_errors=errors,teacher_tail_error=terr))
        print(name,'ALL CHECKS PASS',flush=True)
    for f in ('models/cs_saf_external_controls.py','models/cs_saf_external.py','models/cof_seqgen_saf.py',
              'data/cof_seqgen_saf_tensorizer.py','data/cs_saf_external.py','benchmarks/cs_saf_external.py'):
        old=subprocess.check_output(['git','show',cfg['parent_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(old).hexdigest()==digest(ROOT/f)
    write(DOC/'verification.json',result);write(OUT/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
