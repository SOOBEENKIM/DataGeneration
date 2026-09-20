"""Post-run, read-only checkpoint/provenance and independent score reductions.

Reuses the frozen model/window forward pass, not the experiment's loss/reducer.
The Gaussian support calculation is descriptive; it does not select a model.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.cs_saf_external import TargetWindows
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from models.cs_saf_external import ExternalUG

OUT = ROOT / 'artifacts/cs_saf/external_port_v1'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    result = dict(passed=True, phase='post_run_read_only', new_fits=0,
                  new_generations=0, test_outcomes_accessed=False,
                  reuses_frozen_forward_not_loss_reducer=True,
                  verifier_sha256=sha(Path(__file__)), rows=[])
    for name in ('berka', 'sparkov'):
        data = torch.load(OUT/'input'/name/'prepared.pt', map_location='cpu')
        state = SAFTensorizerState.from_dict(data['state'])
        codec = state.event_numeric_codecs[0][1]
        windows = TargetWindows(data['sequences']['validation'], device='cuda')
        initials = [torch.load(OUT/'runs'/name/m/'initial.pt', map_location='cpu') for m in ('U', 'G')]
        assert initials[0].keys() == initials[1].keys()
        assert all(torch.equal(initials[0][k], initials[1][k]) for k in initials[0])
        for mode in ('U', 'G'):
            folder = OUT/'runs'/name/mode
            done = json.loads((folder/'DONE.json').read_text())
            checkpoint = torch.load(folder/'best.pt', map_location='cpu')
            history = json.loads((folder/'history.json').read_text())
            assert checkpoint['epoch'] == done['selected_epoch']
            assert min(history, key=lambda r: r['check']['loss'])['epoch'] == done['selected_epoch']
            fit_events = sum(s.length for s in data['sequences']['fit'])
            assert done['optimizer_updates'] == math.ceil(fit_events/512)*len(history)
            original_hash = sha(folder/'best.pt')
            model = ExternalUG(mode, state.gap_support, **state.model_config_kwargs()).cuda().eval()
            model.load_state_dict(checkpoint['state_dict'])
            for variant in ('raw', 'gap'):
                model.calibration = None if variant == 'raw' else json.loads((folder/'calibration.json').read_text())
                sums = dict(gap=0., mark=0., amount=0., aux_0=0., brier=0., mae=0., negative_mass=0.)
                n = nt = 0
                for start in range(0, len(windows), 512):
                    batch = windows.batch(np.arange(start, min(start+512, len(windows))))
                    o = model.target_outputs(**batch)
                    row = torch.arange(len(o['mark']), device='cuda')
                    has = o['has_previous']
                    gc = torch.bucketize(torch.nan_to_num(o['gap']), torch.tensor(state.gap_support.upper_bounds[:-1], device='cuda'))
                    gl = model.gap_decoder.logits(o['hidden']).double().log_softmax(-1)
                    sums['gap'] += float(-gl[row, gc][has].sum())
                    logmark = o['logmark'].double()
                    sums['mark'] += float(-logmark[row, o['mark']].sum())
                    mu, sd, target = (o[k].double() for k in ('location', 'scale', 'numeric'))
                    sums['amount'] += float((sd.log()+.5*math.log(2*math.pi)+.5*((target-mu)/sd).square()).sum())
                    sums['aux_0'] += float(-o['auxlogits'][0].double()[row, o['auxiliary'][0]].sum())
                    repeat = logmark[row, o['previous']].exp()
                    sums['brier'] += float((repeat[has]-o['mark'].eq(o['previous'])[has].double()).square().sum())
                    sums['mae'] += float(abs(target-mu).sum())*codec.scale
                    # Inverse signed-log is negative iff standardized value < -mean/scale.
                    z = (-codec.mean/codec.scale-mu)/sd
                    sums['negative_mass'] += float((.5*(1+torch.erf(z/math.sqrt(2)))).sum())
                    n += len(row)
                    nt += int(has.sum())
                components = {k: sums[k]/(nt if k == 'gap' else n) for k in ('gap', 'mark', 'amount', 'aux_0')}
                computed = dict(loss=sum(components.values()), repeat_brier=sums['brier']/nt,
                                amount_log_mae=sums['mae']/n, **components)
                recorded = done['results'][variant]['prediction']
                expected = {**{k: recorded[k] for k in ('loss', 'repeat_brier', 'amount_log_mae')}, **recorded['components']}
                errors = {k: abs(v-expected[k]) for k, v in computed.items()}
                assert max(errors.values()) < 2e-6, (name, mode, variant, errors)
                result['rows'].append(dict(dataset=name, model=mode, variant=variant,
                    validation_events=n, validation_transitions=nt, identical_ug_initial_tensors=True,
                    checkpoint_and_update_count_verified=True, max_score_absolute_error=max(errors.values()),
                    real_history_expected_negative_amount_probability=sums['negative_mass']/n,
                    checkpoint_sha256=original_hash))
            assert sha(folder/'best.pt') == original_hash
            print(name, mode, 'checkpoint audit PASS', flush=True)
    destination = OUT/'checkpoint_audit.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
