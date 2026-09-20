"""Check Ucal/Ecal use identical observed transitions and deployed sampling plans."""
import json
import numpy as np
import torch
from experiments.cs_saf_calibration_u import ROOT,OUTPUT,contract,verify,prior,parent,sha256,write_json


def main():
    c=contract();records=[]
    for pi in c['prevalences']:
        for k in c['kappas']:
            for t in c['trials']:
                u=OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_{t}/Ucal'
                e=prior.OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_{t}/Ecal'
                verify(u);prior.verify(e)
                uf=json.loads((u/'fit.json').read_text());ef=json.loads((e/'fit.json').read_text())
                for field in ('train_entity_ids_sha256','train_entities','transitions','cache_sha256'):
                    if uf[field]!=ef[field]:raise ValueError('unequal train input: '+field)
                with np.load(u/'train_features.npz') as ua,np.load(e/'train_features.npz') as ea:
                    for field in ('equality','code','entity_index','event_index'):
                        np.testing.assert_array_equal(ua[field],ea[field])
                up=torch.load(u/'generated_sample.pt',map_location='cpu')['sample']
                ep=torch.load(e/'generated_sample.pt',map_location='cpu')['sample']
                for field in ('static_codes','valid_mask','lengths','plan_train_indices'):
                    torch.testing.assert_close(up[field],ep[field],rtol=0,atol=0)
                for field in ('matmul_allow_tf32','cudnn_allow_tf32','torch'):
                    if json.loads((u/'execution.json').read_text())[field]!=json.loads((e/'execution.json').read_text())[field]:
                        raise ValueError('unequal numerical execution: '+field)
                paired=json.loads((u/'paired_reference.json').read_text())
                if paired['manifest_sha256']!=sha256(e/'COMPLETE.json'):
                    raise ValueError('Ecal reference manifest changed')
                records.append(dict(pi=pi,kappa=k,trial=t,train_transitions=uf['transitions'],
                    U_manifest_sha256=sha256(u/'COMPLETE.json'),E_manifest_sha256=sha256(e/'COMPLETE.json')))
    evidence=dict(status='PASS',paired_cells=len(records),observed_train_targets_identical=True,
        training_transition_order_identical=True,generated_contexts_and_lengths_identical=True,
        scientific_precision_identical=True,reference_manifests_unchanged=True,records=records)
    write_json(OUTPUT/'pairing_verification.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k!='records'}))


if __name__=='__main__':main()
