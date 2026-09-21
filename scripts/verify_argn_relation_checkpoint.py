"""Read saved extension tensors; no model fit, sampling, or old-file writes."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.argn_relation_pilot import RelationPath


def main():
    cfg = json.loads((ROOT/'configs/argn_relation_pilot_v1.json').read_text())
    root = ROOT/'artifacts/argn_relation_pilot_v1'
    out = {}
    torch.set_num_threads(1)
    for arm in ('G','R'):
        state = torch.load(root/arm/'workspace/ModelStore/model-data/model-weights.pt',
                           map_location='cpu',weights_only=True)
        tensors = {k.removeprefix('relation_path.'):v for k,v in state.items() if k.startswith('relation_path.')}
        assert tensors and all(torch.isfinite(v).all() for v in tensors.values())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg['extra_seed'])
            initial = RelationPath(arm,15,103,cfg['rank'])
        changes = {k:float((v-initial.state_dict()[k]).norm()) for k,v in tensors.items()}
        assert all(v>0 for v in changes.values())
        out[arm] = dict(selected_extra_parameter_changes_l2=changes,all_finite=True,
                        all_extra_parameter_blocks_updated=True)
    (ROOT/'docs/argn_relation_pilot_v1/checkpoint_verification.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__ == '__main__':
    main()
