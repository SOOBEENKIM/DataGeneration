import json
import torch
from experiments.cs_saf_gap_calibration import repeats,ROOT,OUTPUT,contract,sha256,write_json
from experiments.cs_saf_route_decomposition import reference_measure
from pathlib import Path
torch.set_num_threads(1);c=contract();records=[]
for pi in c['prevalences']:
    for k in c['kappas']:
        payload=repeats.parent.load_cache(repeats.parent.CACHE/f'pi_{pi:.2f}_kappa_{k}.pt')
        u,_,_=repeats.load_fixed(payload,pi,k,0,'Ucal','cpu');reference,_=reference_measure(u,payload['train'])
        for trial in c['trials']:
            e,provenance,_=repeats.load_fixed(payload,pi,k,trial,'Ecal','cpu')
            torch.testing.assert_close(e.reference_probabilities,reference,atol=0,rtol=0)
            assert e.support==u.support
            records.append(dict(pi=pi,kappa=k,trial=trial,reference_exactly_equal=True,support_exactly_equal=True))
        print(pi,k,'shared conditional weights PASS',flush=True)
result=dict(status='PASS',records=records,models_checked=40,script_sha256=sha256(Path(__file__)))
write_json(OUTPUT/'shared_conditional_weights_check.json',result)
print('all 40 matched U/E conditional reference distributions identical')
