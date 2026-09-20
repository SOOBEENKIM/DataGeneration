"""Explicit external view and target-complete finite-history windows.

No changes to historical canonical data or controlled U/G contracts.
"""
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.cof_seqgen_saf_contract import CanonicalEntitySequenceDataset, CanonicalSchema, CORE_EVENT_COLUMNS, hash_entity_split_assignment
from data.cof_seqgen_saf_tensorizer import CategoryCodec, NumericCodec, SAFTensorizer, SAFTensorizerState
from models.cof_seqgen_saf import fit_train_only_gap_support

VIEWS = {
    'berka': dict(categorical=('account_frequency','account_district_id','city','region'),
                  numeric=('account_open_day',), auxiliary=('transaction_type',),
                  excluded=('balance','k_symbol'), unit='unix_day'),
    'sparkov': dict(categorical=('cardholder_gender','cardholder_state'),
                    numeric=('birth_year','city_population'), auxiliary=('category',),
                    excluded=('entity_any_fraud','event_is_fraud','merchant_latitude','merchant_longitude'),
                    unit='unix_second_from_trans_date_trans_time'),
}


def load_view(path, name):
    spec = VIEWS[name]
    path = Path(path)
    splits = pd.read_parquet(path/'entity_splits.parquet')
    dev = splits.loc[splits.split.isin(['train','validation'])].copy()
    filters = [('entity_id','in',dev.entity_id.tolist())]
    static_names = spec['categorical']+spec['numeric']
    # Excluded columns are not even materialized by this view.
    events = pd.read_parquet(path/'events.parquet', columns=[*CORE_EVENT_COLUMNS,*spec['auxiliary']], filters=filters)
    static = pd.read_parquet(path/'static_context.parquet', columns=['entity_id',*static_names], filters=filters)
    schema = CanonicalSchema(name, 'absolute', spec['unit'], static_names,
                             auxiliary_categorical_columns=spec['auxiliary'])
    return CanonicalEntitySequenceDataset(schema, static, events, dev, required_splits=('train','validation'),
                                          source_split_assignment_sha256=hash_entity_split_assignment(splits))


def fit_check_ids(dataset, seed=20260921):
    name = dataset.schema.dataset_id
    ids = dataset.entity_ids_for_split('train')
    ranked = sorted(ids, key=lambda x: hashlib.sha256(f'{name}:{x}:{seed}'.encode()).digest())
    cutoff = int(len(ranked)*.8)
    return tuple(ranked[:cutoff]), tuple(ranked[cutoff:])


def fit_tensorizer(dataset, fit_ids, max_gap_states=31):
    if not set(fit_ids) <= set(dataset.entity_ids_for_split('train')) or not fit_ids:
        raise ValueError('transforms may use fitting identities only')
    spec = VIEWS[dataset.schema.dataset_id]
    events = dataset.events.loc[dataset.events.entity_id.isin(fit_ids)]
    static = dataset.static_context.loc[dataset.static_context.entity_id.isin(fit_ids)]
    support = fit_train_only_gap_support(events.gap.dropna(), max_positive_states=max_gap_states)
    # External high-cardinality gaps: zero must be an exact atom, not a bin of
    # small positive values. Keep the old controlled-study helper unchanged.
    if support.zero_is_explicit and support.n_states > 1:
        support = replace(support, upper_bounds=(0.,*support.upper_bounds[1:]))
    state = SAFTensorizerState(
        dataset_id=dataset.schema.dataset_id, schema_sha256=dataset.schema.schema_sha256,
        split_assignment_sha256=dataset.split_assignment_sha256,
        receiver_codec=CategoryCodec.fit(events.receiver_or_mark),
        auxiliary_categorical_codecs=tuple((c,CategoryCodec.fit(events[c])) for c in spec['auxiliary']),
        event_numeric_codecs=(('amount_or_numeric_value',NumericCodec.fit(events.amount_or_numeric_value,transform='signed_log1p_zscore')),),
        static_categorical_codecs=tuple((c,CategoryCodec.fit(static[c])) for c in spec['categorical']),
        static_numeric_codecs=tuple((c,NumericCodec.fit(static[c],transform='zscore')) for c in spec['numeric']),
        gap_support=support)
    return SAFTensorizer(state)


class TargetWindows:
    """Every event is one target; each target sees at most window-1 past events.

    Store only original arrays. Overlapping windows are gathered on demand and
    right-padded; neither tail truncation nor duplicate target weighting occurs.
    """
    def __init__(self, sequences, window=32, device='cpu'):
        if not sequences or window < 2:
            raise ValueError('nonempty sequences and window >=2 required')
        self.sequences, self.window = sequences, window
        self.entity_ids = [s.entity_id for s in sequences]
        self.lengths = np.array([s.length for s in sequences],dtype=np.int64)
        self.starts = torch.as_tensor(np.repeat(np.cumsum(np.r_[0,self.lengths[:-1]]),self.lengths),device=device)
        self.entities = torch.as_tensor(np.repeat(np.arange(len(sequences)),self.lengths),device=device)
        self.flat = {key:torch.as_tensor(np.concatenate([getattr(s,key) for s in sequences]),device=device)
                     for key in ('gap','receiver','numeric_value','auxiliary_numeric')}
        self.aux = tuple(torch.as_tensor(np.concatenate([s.auxiliary_categorical[i] for s in sequences]),device=device)
                         for i in range(len(sequences[0].auxiliary_categorical)))
        self.static = torch.as_tensor(np.stack([s.static_numeric for s in sequences]),device=device)
        self.static_cat = tuple(torch.as_tensor([s.static_categorical[i] for s in sequences],device=device)
                                for i in range(len(sequences[0].static_categorical)))

    def __len__(self):
        return len(self.starts)

    def batch(self, targets, device=None):
        t = torch.as_tensor(targets,device=self.starts.device,dtype=torch.long)
        if t.ndim!=1 or not len(t) or (t<0).any() or (t>=len(self)).any():
            raise ValueError('invalid target indexes')
        begin = torch.maximum(self.starts[t], t-self.window+1)
        length = t-begin+1
        ix = begin[:,None]+torch.arange(self.window,device=t.device)[None,:]
        valid = ix <= t[:,None]
        ix = torch.minimum(ix,t[:,None])
        output = {}
        for key,value in self.flat.items():
            z = value[ix].clone()
            z[~valid] = float('nan') if key=='gap' else 0
            output[key] = z
        output['auxiliary_categorical'] = tuple(v[ix].masked_fill(~valid,0) for v in self.aux)
        eid = self.entities[t]
        output.update(valid_mask=valid, target_position=length-1,
                      static=self.static[eid], static_categorical=tuple(v[eid] for v in self.static_cat))
        if device is not None:
            output={k:tuple(vv.to(device) for vv in v) if isinstance(v,tuple) else v.to(device) for k,v in output.items()}
        return output
