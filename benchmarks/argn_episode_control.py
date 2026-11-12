"""Known statistical episode controls, composed with frozen ARGN output heads.

Rates are estimated on exposed optimization events, not class-balanced labels.
Length is an ARGN-sampled planned length, never a supplied development length.
"""
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier

from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX
from benchmarks.argn_amount_control import amount_generation
from benchmarks.argn_transition_reference import reference_table
from benchmarks.argn_past_state import PastState


def episode_metadata(meta):
    m = meta.copy()
    m['ever_fraud'] = m.groupby('record').label.cumsum().sub(m.label).gt(0).astype(int)
    m['initial_fraud'] = m.groupby('record').label.transform('first')
    # Initial state is unavailable BEFORE the first label itself.
    m.loc[m.event_index.eq(0), 'initial_fraud'] = 0
    m['planned_length'] = m.groupby('record').label.transform('size')
    return m


def fit_episode(meta, old_parameters):
    m = episode_metadata(meta)
    old = reference_table(old_parameters, 'duration')
    table = np.broadcast_to(old[:, :, None, None], (3, 22, 2, 2)).copy()
    counts = []
    # Normal onset depends on whether fraud has ever occurred; continuation
    # distinguishes left-boundary episodes from episodes starting in the window.
    for (prev, age, ever, initial), g in m.assign(age=m.prior_age.clip(upper=21)).groupby(
            ['previous_label', 'age', 'ever_fraud', 'initial_fraud']):
        if prev < 0:
            continue
        p = (g.label.sum() + .5) / (len(g) + 1)
        table[int(prev)+1, int(age), int(ever), int(initial)] = p
        counts.append(dict(previous=int(prev), age=int(age), ever=int(ever), initial=int(initial),
                           events=len(g), frauds=int(g.label.sum()), probability=p))
    first = m[m.event_index.eq(0)]
    x = np.log1p(first.planned_length.to_numpy()).reshape(-1, 1)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10,
                                  criterion='log_loss', random_state=20260927).fit(x, first.label)
    leaf = tree.apply(x)
    probs = np.full(tree.tree_.node_count, old_parameters['previous_label']['-1'])
    for node in np.unique(leaf):
        y = first.label.to_numpy()[leaf == node]
        probs[node] = (y.sum() + .5) / (len(y) + 1)
    return dict(table=table.tolist(), exposure_counts=counts,
                first_tree=dict(left=tree.tree_.children_left.tolist(), right=tree.tree_.children_right.tolist(),
                                threshold=tree.tree_.threshold.tolist(), probability=probs.tolist()),
                fit_split='optimization', smoothing=.5, initial_tree_max_depth=2,
                initial_tree_min_leaf=10, no_forced_episode_limit=True)


def first_probability(length, tree):
    x = np.log1p(np.asarray(length))
    node = np.zeros(x.shape, dtype=int)
    left, right = np.asarray(tree['left']), np.asarray(tree['right'])
    threshold = np.asarray(tree['threshold'])
    while np.any(left[node] >= 0):
        active = left[node] >= 0
        node[active] = np.where(x[active] <= threshold[node[active]], left[node[active]], right[node[active]])
    return np.asarray(tree['probability'])[node]


def episode_probabilities(meta, parameters, joint):
    m = episode_metadata(meta)
    p = np.asarray(parameters['table'])[m.previous_label.to_numpy()+1,
        m.prior_age.clip(upper=21).to_numpy(), m.ever_fraud.to_numpy(), m.initial_fraud.to_numpy()]
    first = m.event_index.eq(0).to_numpy()
    if joint:
        p[first] = first_probability(m.loc[first, 'planned_length'].to_numpy(), parameters['first_tree'])
    return p


def decode_length(tokens, minimum):
    if SLEN_SUB_COLUMN_PREFIX+'cat' in tokens:
        length = tokens[SLEN_SUB_COLUMN_PREFIX+'cat']
    else:
        length = sum(v * 10**int(k.rsplit('E', 1)[1]) for k, v in tokens.items())
    return length.clamp(min=minimum)


@contextmanager
def episode_generation(stats, old_parameters, head_path, parameters, mode):
    assert mode in {'duration', 'episode', 'joint'}
    import mostlyai.engine._tabular.generation as generation
    with amount_generation(stats, old_parameters, head_path):
        if mode == 'duration':
            yield
            return
        parent = generation.SequentialModel
        codec = PastState(stats)
        label_key = codec.prefixes['event_is_fraud']+'__cat'

        class EpisodeModel(parent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.episode_table = torch.tensor(parameters['table'], device=self.device)
                for key in self.tgt_cardinalities:
                    if key.startswith(SLEN_SUB_COLUMN_PREFIX):
                        self.embedders.get(key).register_forward_pre_hook(self.length_hook(key))
                self.predictors.register_forward_hook(self.episode_label)

            def length_hook(self, key):
                def capture(module, args):
                    self.length_tokens[key] = args[0].reshape(-1)
                return capture

            def episode_label(self, module, args, output):
                if args[1] != label_key:
                    return output
                f = self._state_features[:, 0]
                present = f[:, 0] > .5
                prev = f[:, 1] > .5
                age = torch.expm1(f[:, 3].double()*np.log1p(stats['seq_len']['max'])).round().long().clamp(0, 21)
                row = torch.where(present, prev.long()+1, torch.zeros_like(age))
                p = self.episode_table[row, age, self.episode_memory[:, 0].long(), self.episode_memory[:, 1].long()]
                if mode == 'joint' and (~present).any():
                    length = decode_length(self.length_tokens, stats['seq_len']['min'])
                    initial = first_probability(length.detach().cpu().numpy(), parameters['first_tree'])
                    p = torch.where(present, p, torch.as_tensor(initial, device=self.device))
                logits = torch.full_like(output, -1e9)
                logits[:, 0, codec.codes['1']] = p.log().to(output.dtype)
                logits[:, 0, codec.codes['0']] = torch.log1p(-p).to(output.dtype)
                return logits

            def forward(self, x, mode, **kwargs):
                assert mode == 'gen', 'true-prefix rates are evaluated separately with explicit planned-length scope'
                state = kwargs.get('history_state')
                if state is None:
                    memory = torch.zeros((kwargs['batch_size'], 2), dtype=torch.long, device=self.device)
                    first = torch.ones(kwargs['batch_size'], dtype=torch.bool, device=self.device)
                else:
                    assert len(state) == 4
                    memory = state[3][0]
                    first = state[2][0, :, 0].eq(0)
                    kwargs['history_state'] = state[:3]
                self.episode_memory = memory
                self.length_tokens = {}
                outputs, history, recurrent = super().forward(x, mode, **kwargs)
                label = outputs[label_key].reshape(-1).eq(codec.codes['1']).long()
                memory = torch.stack([torch.maximum(memory[:, 0], label), torch.where(first, label, memory[:, 1])], -1)
                return outputs, history, (*recurrent, memory.unsqueeze(0))

        generation.SequentialModel = EpisodeModel
        try:
            yield
        finally:
            generation.SequentialModel = parent
