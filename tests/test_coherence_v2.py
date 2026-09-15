import numpy as np

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from eval.behavior_summaries_v2 import compute_behavior_summaries_v2, fit_short_gap_threshold
from eval.coherence_v2 import coherence_report, fit_coherence_reference


def test_joint_alignment_population_covariance_and_padding():
    bundle = generate_benchmark(BenchmarkConfig(n_train=100, n_test=20), 42)
    tau = np.asarray(bundle.metadata["tau"])
    cutoff = fit_short_gap_threshold(bundle.train, tau=tau)
    summaries = compute_behavior_summaries_v2(
        bundle.test, tau=tau, short_gap_threshold=cutoff, window_width=7
    )
    i, length = 0, int(bundle.test.lengths[0])
    short = (tau[bundle.test.dt_bin[i, 1:length]] <= cutoff).astype(float)
    cats = bundle.test.x_cat[i, :length, 0]
    repeat = (cats[1:] == cats[:-1]).astype(float)
    expected = np.mean((short-short.mean())*(repeat-repeat.mean()))
    assert np.isclose(summaries["joint_alignment"][i], expected)


def test_short_gap_cutoff_uses_pooled_train_without_label_argument():
    bundle = generate_benchmark(BenchmarkConfig(n_train=100, n_test=20), 42)
    cutoff = fit_short_gap_threshold(bundle.train, tau=np.asarray(bundle.metadata["tau"]))
    assert np.isfinite(cutoff)


def test_coherence_2_4_8_bins_and_jeffreys_rates():
    bundle = generate_benchmark(BenchmarkConfig(n_train=500, n_test=200), 42)
    tau = np.asarray(bundle.metadata["tau"])
    cutoff = fit_short_gap_threshold(bundle.train, tau=tau)
    summaries = compute_behavior_summaries_v2(bundle.test, tau=tau, short_gap_threshold=cutoff, window_width=7)
    for bins in (2, 4, 8):
        reference = fit_coherence_reference(summaries, bundle.test.y_entity, n_bins=bins, min_bin_count=5)
        report = coherence_report(summaries, bundle.test.y_entity, summaries, bundle.test.y_entity, reference)
        assert all(value["macro_gap"] == 0 for value in report.values())
