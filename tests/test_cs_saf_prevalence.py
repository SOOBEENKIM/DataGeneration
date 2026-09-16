import numpy as np
import pandas as pd
import pytest

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, _split
from data.cs_saf_prevalence import production_label_uniforms, prevalence_view


@pytest.fixture
def source():
    cfg = BenchmarkConfig(scenario="joint_semimarkov_v2b", kappa=1, min_length=4, max_length=5)
    values, latent = _split(cfg, 100, 42, 0)
    ids = [f"controlled-{i:06d}" for i in range(100)]
    static = pd.DataFrame({"entity_id": ids, "entity_label": values["y_entity"]})
    rows, states = [], []
    for i, entity in enumerate(ids):
        for t in range(values["lengths"][i]):
            identity = {"entity_id": entity, "event_id": i*5+t, "event_index": t}
            rows.append({**identity, "gap": np.nan if t == 0 else float(latent["raw_gap"][i, t]),
                         "receiver_or_mark": f"receiver-{values['x_cat'][i,t,0]}", "amount": float(values["x_num"][i,t,0])})
            states.append({**identity, "gap_state": latent["gap_state"][i,t]})
    return static, pd.DataFrame(rows), pd.DataFrame(states)


def test_production_uniforms_reproduce_labels_and_do_not_depend_on_requested_subset(source):
    static, _, _ = source
    draws = production_label_uniforms(static.entity_id)
    assert np.array_equal(draws < .05, static.entity_label)
    selected = [4, 17, 43, 81]
    np.testing.assert_array_equal(production_label_uniforms(static.entity_id.iloc[selected]), draws[selected])


def test_base_view_is_exact_and_null_marks_never_change(source):
    static, events, oracle = source
    s, e, _ = prevalence_view(static, events, oracle, prevalence=.05, kappa=1)
    pd.testing.assert_frame_equal(s, static)
    pd.testing.assert_frame_equal(e, events)
    for pi in (.1, .25, .5):
        s, e, _ = prevalence_view(static, events, oracle, prevalence=pi, kappa=0)
        pd.testing.assert_frame_equal(e, events)
        assert s.entity_label.sum() >= static.entity_label.sum()


def test_paired_promotions_change_only_active_marks_and_are_reproducible(source):
    static, events, oracle = source
    s1, e1, _ = prevalence_view(static, events, oracle, prevalence=.1, kappa=1)
    s2, e2, _ = prevalence_view(static, events, oracle, prevalence=.5, kappa=1)
    for column in events.columns:
        if column != "receiver_or_mark":
            pd.testing.assert_series_equal(e1[column], events[column])
            pd.testing.assert_series_equal(e2[column], events[column])
    active_already = set(s1.loc[s1.entity_label == 1, "entity_id"])
    pd.testing.assert_series_equal(e1.loc[e1.entity_id.isin(active_already), "receiver_or_mark"],
                                  e2.loc[e2.entity_id.isin(active_already), "receiver_or_mark"])
    _, repeat, _ = prevalence_view(static, events, oracle, prevalence=.5, kappa=1)
    pd.testing.assert_frame_equal(e2, repeat)


def test_materializer_rejects_wrong_latent_identity_or_label_stream(source):
    static, events, oracle = source
    corrupt = oracle.copy(); corrupt.loc[0, "event_id"] = -1
    with pytest.raises(ValueError, match="identities"):
        prevalence_view(static, events, corrupt, prevalence=.1, kappa=1)
    bad = static.copy(); bad.loc[0, "entity_label"] = 1-bad.loc[0, "entity_label"]
    with pytest.raises(ValueError, match="label stream"):
        prevalence_view(bad, events, oracle, prevalence=.1, kappa=1)
