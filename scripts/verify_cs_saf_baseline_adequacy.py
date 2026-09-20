"""Independent scalar-root checks of fitted controls and immutable inputs/results."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continue_cs_saf_argn_adequacy import CONFIG,OUT,OLD
from scripts.run_cs_saf_external_audit_v1 import digest,write
from models.cs_saf_observed_repeat_control import assign_bins


def main():
    c=json.loads(CONFIG.read_text());report=ROOT/'docs/cs_saf/baseline_adequacy_v1'
    summary=json.loads((report/'execution_summary.json').read_text());assert summary['status']=='COMPLETE'
    records=[];original_probability_errors=[]
    for parent in summary['parents']:
        folder=ROOT/parent['path'];fit=json.loads((folder/'fit.json').read_text())
        assert digest(folder/'fit.json')==parent['fit_sha256']
        tr=pd.read_parquet(folder/'train_features.parquet');vf=pd.read_parquet(folder/'validation_features.parquet')
        assert digest(folder/'train_features.parquet')==parent['train_feature_sha256']
        assert digest(folder/'validation_features.parquet')==parent['validation_feature_sha256']
        assert not set(tr.entity_id)&set(vf.entity_id)
        max_delta_error=0.;max_direct_error=0.
        for kind,control in fit['controls'].items():
            edges=[np.quantile(tr.loc[tr.group==g,'gap_code'],c['coarse_quantiles']).tolist() for g in (0,1)]
            np.testing.assert_array_equal(edges,control['edges'])
            bins=assign_bins(tr.gap_code.to_numpy(),tr.group.to_numpy(),edges)
            for g in (0,1):
                for b in range(1 if kind=='level' else 5):
                    mask=(tr.group.to_numpy()==g)&(np.ones(len(tr),bool) if kind=='level' else bins==b)
                    y=tr.y.to_numpy(float)[mask];p=tr.p.to_numpy(float)[mask]
                    saved=control['parameters'][g][b]
                    if kind=='direct':
                        expected=(y.sum()+c['direct_beta_pseudocount'])/(len(y)+2*c['direct_beta_pseudocount'])
                        max_direct_error=max(max_direct_error,abs(saved-expected))
                    else:
                        z=logit(np.clip(p,1e-9,1-1e-9))
                        # Independent monotone 1-D optimum; no L-BFGS or its supplied gradient.
                        def derivative(delta):return float(np.mean(expit(z+delta)-y)+c['ridge']*delta)
                        bound=c['offset_bound']
                        expected=(-bound if derivative(-bound)>=0 else bound if derivative(bound)<=0
                                  else brentq(derivative,-bound,bound,xtol=1e-12))
                        max_delta_error=max(max_delta_error,abs(saved-expected))
        assert max_delta_error<1e-4 and max_direct_error<1e-12
        for name,sha in parent['generated_sha256'].items():assert digest(folder/name)==sha
        k,index=parent['kappa'],parent['index']
        if parent['family']=='argn_original':
            old=OLD/f'kappa_{k}/seed_{index}'
            assert digest(old/'workspace/ModelStore/model-data/model-weights.pt')==fit['weights_sha256']
            prior=pd.read_parquet(old/'replay_validation.parquet')
            prior=prior[prior.event_index>0].sort_values(['entity_id','event_index'])
            current=vf.sort_values(['entity_id','event_index'])
            np.testing.assert_array_equal(prior.entity_id,current.entity_id)
            np.testing.assert_array_equal(prior.event_index,current.event_index)
            original_probability_errors.append(float(np.max(abs(prior.predicted_repeat.to_numpy()-current.p.to_numpy()))))
        elif parent['family']=='argn_continued':
            continued=OUT/f'continuations/kappa_{k}/seed_{index}'
            start=json.loads((continued/'start.json').read_text())
            done=json.loads((continued/'DONE.json').read_text())
            for name,sha in start['original_hashes'].items():
                assert digest(ROOT/start['original_path']/'workspace'/name)==sha
            assert digest(continued/'workspace/ModelStore/model-data/model-weights.pt')==fit['weights_sha256']
            last=json.loads((folder/'last_conditional.json').read_text())
            assert digest(continued/'checkpoint_last.pt')==done['last_weights_sha256']==last['checkpoint_sha256']
            assert last['invariance_error']==0
            assert done['last_epoch']-done['start_epoch']<=c['additional_epochs']
            assert done['seconds']<=c['additional_minutes']*60+60
        else:
            assert digest(fit['source']['checkpoint_path'])==fit['source']['checkpoint_sha256']
        records.append(dict(family=parent['family'],kappa=k,index=index,
             independent_offset_optimum_max_error=max_delta_error,direct_probability_max_error=max_direct_error))
    assert max(original_probability_errors)<1e-6
    implementation_commit='a03739e1d8564709f79ed5984b1ebee4131eed01'
    files=['models/cs_saf_observed_repeat_control.py','scripts/continue_cs_saf_argn_adequacy.py',
           'scripts/cs_saf_argn_control_adapter.py','scripts/run_cs_saf_argn_repeat_controls.py',
           'scripts/cs_saf_repeat_control_common.py','scripts/run_cs_saf_u_repeat_controls.py']
    for file in files:
        assert (ROOT/file).read_bytes()==subprocess.check_output(['git','show',f'{implementation_commit}:{file}'],cwd=ROOT)
    write(report/'verification.json',dict(status='PASS',parents=records,
        old_argn_replay_max_probability_difference=max(original_probability_errors),
        implementation_commit=implementation_commit,implementation_unchanged_during_execution=True,
        originals_unchanged=True,all_generated_hashes_match=True,
        calibration_optimizer_verified_with_independent_scalar_roots=True))
    print('PASS: all twelve parents, independent fitted-control optima, native replay, source/weight/data hashes')


if __name__=='__main__':main()
