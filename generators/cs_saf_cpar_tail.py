"""Explicit input-segmentation adapter; official PAR network/loss unchanged."""
from contextlib import contextmanager
import importlib.metadata


@contextmanager
def retain_cpar_tails():
    import deepecho.sequences as module
    if importlib.metadata.version('deepecho')!='0.8.1':
        raise RuntimeError('tail adapter is checked for DeepEcho 0.8.1 only')
    original=module.segment_by_size
    coverage={'input_events':0,'segmented_events':0,'sequences':0,'segments':0,'short_segments':0}
    def split_all(sequence,size):
        if not isinstance(size,int) or size<1:raise ValueError('positive integer segment size required')
        pieces=[sequence.iloc[s:s+size].reset_index(drop=True) for s in range(0,len(sequence),size)]
        coverage['input_events']+=len(sequence)
        coverage['segmented_events']+=sum(map(len,pieces))
        coverage['sequences']+=1;coverage['segments']+=len(pieces)
        coverage['short_segments']+=sum(len(p)<size for p in pieces)
        return pieces
    module.segment_by_size=split_all
    try:yield coverage
    finally:module.segment_by_size=original
