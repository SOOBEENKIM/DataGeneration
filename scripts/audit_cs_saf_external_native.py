"""Read-only ARGN capacity/provenance export in the pinned official runtime."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from mostlyai.engine._common import (get_cardinalities, get_columns_from_cardinalities,
    get_ctx_sequence_length, get_sequence_length_stats, SLEN_SUB_COLUMN_PREFIX,
    RIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX)
from mostlyai.engine._tabular.common import create_and_load_model, load_model_artifacts
from mostlyai.engine._workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/cs_saf/external_port_v1'


def main():
    torch.set_num_threads(1)
    rows = []
    for name in ('berka', 'sparkov'):
        ws = Workspace(ART/'runs'/name/'ARGN/workspace')
        config, target, context, sequential = load_model_artifacts(ws)
        units = config['model_units']
        flags = [any(prefix in k for k in units) for prefix in
                 (SLEN_SUB_COLUMN_PREFIX, RIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX)]
        cardinalities = get_cardinalities(target, *flags)
        lengths = get_sequence_length_stats(target)
        before = hashlib.sha256(ws.model_tabular_weights_path.read_bytes()).hexdigest()
        model = create_and_load_model(ws, sequential, cardinalities, get_cardinalities(context),
            units, get_ctx_sequence_length(context, key='median'),
            get_columns_from_cardinalities(cardinalities), torch.device('cpu'),
            seq_len_median=lengths['median'], seq_len_max=lengths['max'])
        assert before == hashlib.sha256(ws.model_tabular_weights_path.read_bytes()).hexdigest()
        rows.append(dict(dataset=name, model='ARGN', parameters=sum(p.numel() for p in model.parameters()),
            checkpoint_sha256=before, sequence_length_stats=lengths,
            version=importlib.metadata.version('mostlyai-engine'), model_config=config,
            load_verified=True, new_training_updates=0, new_generations=0))
        print(name, rows[-1]['parameters'], flush=True)
    path = ART/'native_metadata.json'
    assert not path.exists()
    path.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
