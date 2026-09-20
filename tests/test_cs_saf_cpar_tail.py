import numpy as np
import pandas as pd
import torch
from generators.cs_saf_cpar_tail import retain_cpar_tails


def test_tail_adapter_preserves_all_values_and_restores_native_function():
    import deepecho.sequences as m
    original=m.segment_by_size
    frame=pd.DataFrame({'gap':np.r_[np.nan,np.arange(1,65)],'mark':np.arange(65)})
    assert sum(map(len,original(frame,32)))==64
    with retain_cpar_tails() as coverage:
        chunks=m.segment_by_size(frame,32)
        assert list(map(len,chunks))==[32,32,1]
        pd.testing.assert_frame_equal(pd.concat(chunks,ignore_index=True),frame)
        assert coverage['input_events']==coverage['segmented_events']==65
        assert len(m.segment_by_size(frame.iloc[:3],32)[0])==3
    assert m.segment_by_size is original


def test_native_cpar_end_to_end_tail_and_seeded_sampling(tmp_path):
    from sdv.metadata import SingleTableMetadata
    from sdv.sequential import PARSynthesizer
    from experiments.cs_saf_cpar_loss import equivalent_par_loss
    rows=[]
    for eid,n in zip(range(4),[3,5,7,9]):
        for t in range(n):rows.append({'entity_id':str(eid),'context':str(eid%2),'gap':float(t%3),'mark':str(t%2),'amount':float(t+1)})
    data=pd.DataFrame(rows);meta=SingleTableMetadata();meta.detect_from_dataframe(data)
    meta.update_column('entity_id',sdtype='id');meta.set_sequence_key('entity_id')
    for c in ('context','mark'):meta.update_column(c,sdtype='categorical')
    model=PARSynthesizer(meta,context_columns=['context'],epochs=1,segment_size=4,sample_size=1,cuda=False)
    with retain_cpar_tails() as coverage,equivalent_par_loss():model.fit(data)
    assert coverage['input_events']==coverage['segmented_events']==len(data)
    stub=data.groupby('entity_id',sort=False).first().reset_index().iloc[:2].copy()
    transformed=model._data_processor.transform(stub)[['entity_id','context']]
    def sample():
        import random
        random.seed(42);np.random.seed(42);torch.manual_seed(42);model._data_processor.reset_sampling()
        return model._sample(transformed,sequence_length=9)
    first,second=sample(),sample()
    pd.testing.assert_frame_equal(first,second)
    assert first.groupby('entity_id').size().tolist()==[9,9]
