"""Scoped extensions of official mostlyai-engine 2.4.0; no library files edited.

The native sequential forward remains intact except for an additive category
logit path. All native loss, encodings, recurrent updates and stopping stay native.
G and R receive exactly the same previous-category/current-gap information and
have exactly the same trainable parameter count. This is a feasibility control,
not a claim that bilinear interactions or residual MLPs are new methods.
"""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import textwrap

import torch
from torch import nn
import torch.nn.functional as F

GAP = 'tgt:t0/c0__bin'
CATEGORY = 'tgt:t3/c3__cat'


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RelationPath(nn.Module):
    def __init__(self, arm, n_category, n_gap, rank=16):
        super().__init__()
        if arm not in ('G', 'R'):
            raise ValueError(arm)
        self.arm = arm
        self.previous = nn.Embedding(n_category, rank)
        self.gap = nn.Embedding(n_gap, rank)
        self.output = nn.Linear(rank, n_category)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, previous, gap, valid):
        u, v = self.previous(previous), self.gap(gap)
        features = torch.tanh(u + v) if self.arm == 'G' else torch.tanh(u) * torch.tanh(v)
        return self.output(features) * valid[..., None]


@contextmanager
def official_extension(arm, rank=16, extra_seed=20260922, on_init=None):
    """Restore all official classes after the experiment (also on exceptions)."""
    if importlib.metadata.version('mostlyai-engine') != '2.4.0':
        raise RuntimeError('This adapter is registered for mostlyai-engine 2.4.0 only')
    if arm not in ('A', 'G', 'R'):
        raise ValueError(arm)
    argn = importlib.import_module('mostlyai.engine._tabular.argn')
    training = importlib.import_module('mostlyai.engine._tabular.training')
    generation = importlib.import_module('mostlyai.engine._tabular.generation')
    native = argn.SequentialModel
    forward_source = textwrap.dedent(inspect.getsource(native.forward))
    needle = 'xs = self.predictors(xs, sub_col)'
    if forward_source.count(needle) != 2:
        raise RuntimeError('Native forward changed: expected training and generation predictors')
    forward_source = forward_source.replace(
        needle, needle + '\n' + ' ' * 12 +
        'xs = self._relation_adjust(xs, sub_col, mode, x, outputs)', 1)
    # The generation predictor is nested one level deeper than training.
    pos = forward_source.rfind(needle)
    forward_source = (forward_source[:pos] + forward_source[pos:].replace(
        needle, needle + '\n' + ' ' * 16 +
        'xs = self._relation_adjust(xs, sub_col, mode, x, outputs)', 1))
    namespace = dict(argn.__dict__)
    exec(compile(forward_source, '<argn-relation-forward>', 'exec'), namespace)

    class ExtendedSequentialModel(native):
        forward = namespace['forward']

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if GAP not in self.tgt_cardinalities or CATEGORY not in self.tgt_cardinalities:
                raise RuntimeError('Expected unchanged Sparkov external-pilot schema')
            self.relation_previous = None
            self.relation_calls = 0
            self.relation_sampled_events = 0
            self.relation_trace = []
            base_hash = tensor_hash(self.state_dict())
            if arm != 'A':
                # Preserve the native RNG stream, including all training randomness.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(extra_seed)
                    self.relation_path = RelationPath(
                        arm, self.tgt_cardinalities[CATEGORY], self.tgt_cardinalities[GAP], rank)
                self.relation_path.to(self.device)
            if on_init:
                on_init(self, base_hash)

        def _relation_adjust(self, logits, sub_col, mode, x, outputs):
            if sub_col != CATEGORY or arm == 'A':
                return logits
            if mode == 'trn':
                current = x[CATEGORY].squeeze(-1).long()
                previous = F.pad(current[:, :-1], (1, 0), value=0)
                gap = x[GAP].squeeze(-1).long()
                valid = previous.ne(0)
            else:
                if GAP not in outputs:
                    raise RuntimeError('Generation must put current gap before category')
                gap = outputs[GAP].long()
                if self.relation_previous is None:
                    previous = torch.zeros_like(gap)
                else:
                    previous = self.relation_previous.to(device=gap.device).reshape_as(gap)
                valid = previous.ne(0)
                self.relation_calls += 1
                self.relation_sampled_events += int(valid.sum())
                if len(self.relation_trace) < 4:
                    self.relation_trace.append(dict(previous=previous.flatten()[:8].cpu().tolist(),
                                                   gap=gap.flatten()[:8].cpu().tolist()))
            if tuple(logits.shape[:-1]) != tuple(previous.shape):
                raise RuntimeError(f'Unaligned relation path {logits.shape}, {previous.shape}')
            return logits + self.relation_path(previous, gap, valid)

        def set_previous_generated(self, out_df, step):
            # Native generation has already filtered out sequences that ended.
            self.relation_previous = None if step == 0 else torch.as_tensor(
                out_df[CATEGORY].to_numpy().copy(), dtype=torch.long)

    source = inspect.getsource(generation.generate)
    needle = '                    out_dct, history, history_state = model('
    if source.count(needle) != 1:
        raise RuntimeError('Native generator changed')
    modified = source.replace(needle,
        '                    model.set_previous_generated(out_df, seq_step)\n' + needle)
    gen_namespace = dict(generation.__dict__)
    exec(compile(modified, '<argn-relation-generation>', 'exec'), gen_namespace)
    old_classes = [(module, module.SequentialModel) for module in (argn, training, generation)]
    try:
        for module, _ in old_classes:
            module.SequentialModel = ExtendedSequentialModel
        # isinstance inside the compiled official generator must see the extension.
        gen_namespace['SequentialModel'] = ExtendedSequentialModel
        yield dict(generate=gen_namespace['generate'], model_class=ExtendedSequentialModel,
                   forward_source_sha256=hashlib.sha256(inspect.getsource(native.forward).encode()).hexdigest(),
                   generation_source_sha256=hashlib.sha256(source.encode()).hexdigest())
    finally:
        for module, original in old_classes:
            module.SequentialModel = original
