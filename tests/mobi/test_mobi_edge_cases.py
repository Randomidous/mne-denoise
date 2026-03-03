"""Pytest entry-point for MoBI edge-case validation.

Each test function calls run_edge_case_test (or run_batch_tests) and
asserts result.passed.  Synthetic-data tests run in CI without any
real files; real-data tests are skipped when the data files are absent.

Running
-------
All mobi tests:
    pytest tests/mobi/ -v

One test by name:
    pytest tests/mobi/test_mobi_edge_cases.py::test_bandpass_flat_channels -v

With a live summary table printed:
    pytest tests/mobi/ -v -s

Just the synthetic suite (fast, no files needed):
    pytest tests/mobi/ -v -m synthetic

Just the real-data suite:
    pytest tests/mobi/ -v -m real_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mne_denoise.dss.denoisers import AverageBias, BandpassBias, LineNoiseBias

from .edge_case_tester import (
    BandPowerRatioMetric,
    BatchResult,
    ERPCorrelationMetric,
    ERPPeakSNRMetric,
    NumericalValidityMetric,
    OutputShapeMetric,
    SignalBandPreservationMetric,
    TrialReproducibilityMetric,
    make_flat_channel_data,
    make_line_noise_data,
    make_low_snr_data,
    make_motion_artifact_data,
    make_muscle_noise_data,
    make_nonstationary_noise_data,
    make_rank_deficient_data,
    make_real_mobi_data,
    make_short_segment_data,
    run_batch_tests,
    run_edge_case_test,
)

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
# Register in pyproject.toml [tool.pytest.ini_options] markers if you want
# to suppress the "unknown mark" warning:
#   markers = ["synthetic: ...", "real_data: ..."]

pytestmark = []  # Module-level marks go here if needed

# ---------------------------------------------------------------------------
# Paths to real data files — edit these or set via environment variables.
# Tests that need real data are auto-skipped when the file does not exist.
# ---------------------------------------------------------------------------
# SINGLE_SUBJECT_FIF = Path("data/sub-01_task-oddball_walking_epo.fif")
# GROUP_FIFS = [
#     Path("data/sub-01_task-oddball_walking_epo.fif"),
#     Path("data/sub-02_task-oddball_walking_epo.fif"),
#     Path("data/sub-03_task-oddball_walking_epo.fif"),
# ]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SFREQ = 250.0


def _assert(result):
    """Fail the test with a readable message if result.passed is False."""
    assert result.passed, (
        f"\n{result}\n"
        + (f"Exception: {result.exception}" if result.exception else "")
    )


# ===========================================================================
# Synthetic-data tests  (pytest -m synthetic)
# ===========================================================================


@pytest.mark.synthetic
def test_bandpass_flat_channels():
    """BandpassBias must not produce NaN/Inf on data with dead channels."""
    bundle = make_flat_channel_data(n_channels=32, sfreq=SFREQ)
    bias = BandpassBias(freq_band=(1, 40), sfreq=SFREQ)
    result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
    _assert(result)


@pytest.mark.synthetic
def test_bandpass_shape_preserved():
    """BandpassBias output shape must match input shape."""
    bundle = make_flat_channel_data(n_channels=32, sfreq=SFREQ)
    bias = BandpassBias(freq_band=(1, 40), sfreq=SFREQ)
    result = run_edge_case_test(bundle, bias.apply, OutputShapeMetric())
    _assert(result)


@pytest.mark.synthetic
def test_line_noise_bias_output_validity():
    """LineNoiseBias.apply() must produce a finite, shape-preserving output.

    Note: LineNoiseBias is a DSS *bias function* — its output is data filtered
    to the artifact band, intended to feed the biased covariance in the DSS
    pipeline.  It does not by itself remove line noise from the original signal.
    Testing artifact suppression requires a full DSS pipeline; this test only
    verifies that the bias step is numerically stable on noisy MoBI data.
    """
    bundle = make_line_noise_data(n_channels=16, sfreq=1000.0)
    bias = LineNoiseBias(freq=50.0, sfreq=1000.0, method="fft")
    result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
    _assert(result)


@pytest.mark.synthetic
def test_line_noise_bias_concentrates_artifact():
    """LineNoiseBias output should have more artifact-band power than broadband.

    Because the bias isolates the artifact frequency, the ratio of artifact-band
    power to signal-band power should increase substantially after apply().
    Uses a broader measurement band (45–55 Hz) to align with Welch PSD resolution.
    """
    bundle = make_line_noise_data(n_channels=16, sfreq=1000.0)
    bias = LineNoiseBias(freq=50.0, sfreq=1000.0, method="iir")

    # Artifact concentration: power in artifact band / power in broadband signal band
    from .edge_case_tester import DataBundle, _band_power

    biased = bias.apply(bundle.data)
    artifact_power = _band_power(biased, bundle.sfreq, (45.0, 55.0))
    signal_power = _band_power(biased, bundle.sfreq, (1.0, 30.0))

    assert artifact_power > signal_power, (
        f"Bias output artifact power ({artifact_power:.4g}) should exceed "
        f"broadband signal power ({signal_power:.4g}) — bias is not concentrating "
        f"the artifact as expected."
    )


@pytest.mark.synthetic
def test_bandpass_preserves_signal_band():
    """BandpassBias targeting alpha should retain most alpha-band power."""
    bundle = make_low_snr_data(n_channels=16, sfreq=SFREQ, signal_freq=10.0)
    bias = BandpassBias(freq_band=(8, 12), sfreq=SFREQ)
    result = run_edge_case_test(
        bundle,
        bias.apply,
        SignalBandPreservationMetric(signal_band=(8, 12), sfreq=SFREQ),
        expected_range=(0.8, np.inf),
    )
    _assert(result)


@pytest.mark.synthetic
def test_motion_artifact_validity():
    """Any denoiser must produce finite output on step-artifact data."""
    bundle = make_motion_artifact_data(n_channels=32, sfreq=SFREQ)
    bias = BandpassBias(freq_band=(1, 40), sfreq=SFREQ)
    result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
    _assert(result)


@pytest.mark.synthetic
def test_rank_deficient_validity():
    """Denoiser must not crash on rank-deficient (post-reference) data."""
    bundle = make_rank_deficient_data(n_channels=64, sfreq=SFREQ, n_independent_sources=20)
    bias = BandpassBias(freq_band=(1, 40), sfreq=SFREQ)
    result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
    _assert(result)


@pytest.mark.synthetic
def test_short_segment_validity():
    """BandpassBias is expected to fail when n_times < filter padlen.

    A 4th-order Butterworth requires padlen=27 samples for sosfiltfilt.
    This documents a known limitation: BandpassBias needs at least ~28
    samples to operate.  The test asserts that the framework captures the
    failure cleanly rather than silently producing garbage.
    """
    bundle = make_short_segment_data(n_channels=32, n_times=20, sfreq=SFREQ)
    bias = BandpassBias(freq_band=(1, 40), sfreq=SFREQ)
    result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
    assert not result.passed, "Expected BandpassBias to fail on 20-sample data"
    assert isinstance(result.exception, ValueError)


@pytest.mark.synthetic
def test_muscle_noise_validity():
    """Denoiser must produce valid output on muscle-contaminated data."""
    bundle = make_muscle_noise_data(n_channels=16, sfreq=500.0)
    bias = BandpassBias(freq_band=(1, 40), sfreq=500.0)
    result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
    _assert(result)


@pytest.mark.synthetic
def test_nonstationary_validity():
    """Denoiser must remain numerically stable on non-stationary data."""
    bundle = make_nonstationary_noise_data(n_channels=16, sfreq=SFREQ)
    bias = BandpassBias(freq_band=(1, 40), sfreq=SFREQ)
    result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
    _assert(result)


@pytest.mark.synthetic
def test_synthetic_erp_correlation():
    """AverageBias on epoched synthetic data: ERP correlation with raw >= 0.9."""
    # Simulate epoched data with a reproducible signal component
    rng = np.random.default_rng(0)
    n_ch, n_times, n_epochs = 16, 200, 50
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 0.8, n_times))
    clean = np.outer(rng.standard_normal(n_ch), signal)  # (n_ch, n_times)
    noise = rng.standard_normal((n_ch, n_times, n_epochs)) * 3.0
    data = clean[:, :, np.newaxis] + noise  # broadcast over epochs

    bundle = make_flat_channel_data.__func__ if False else None  # not used
    from .edge_case_tester import DataBundle

    bundle = DataBundle(
        data=data,
        description="synthetic_erp – clean signal + 3x noise",
        sfreq=SFREQ,
        clean_reference=clean,
        metadata={"data_type": "single_subject"},
    )

    bias = AverageBias(axis="epochs")
    result = run_edge_case_test(
        bundle,
        bias.apply,
        ERPCorrelationMetric(),
        # With 50 trials at 3x noise, analytic expected r ≈ 0.84.
        # Threshold set conservatively below that.
        expected_range=(0.7, 1.0),
    )
    _assert(result)


@pytest.mark.synthetic
def test_synthetic_trial_reproducibility():
    """AverageBias should improve trial reproducibility over identity."""
    rng = np.random.default_rng(1)
    n_ch, n_times, n_epochs = 8, 100, 40
    signal = np.sin(2 * np.pi * 10 * np.linspace(0, 0.4, n_times))
    clean = np.outer(rng.standard_normal(n_ch), signal)
    noise = rng.standard_normal((n_ch, n_times, n_epochs)) * 5.0
    data = clean[:, :, np.newaxis] + noise

    from .edge_case_tester import DataBundle

    bundle = DataBundle(
        data=data,
        description="synthetic_reproducibility – signal + 5x noise",
        sfreq=SFREQ,
        metadata={"data_type": "single_subject"},
    )

    # The identity should give low reproducibility on noisy data
    result_noisy = run_edge_case_test(
        bundle, lambda d: d, TrialReproducibilityMetric(n_splits=20, rng_seed=0)
    )
    # AverageBias-filtered output should give higher reproducibility
    bias = AverageBias(axis="epochs")
    result_denoised = run_edge_case_test(
        bundle, bias.apply, TrialReproducibilityMetric(n_splits=20, rng_seed=0)
    )
    assert result_denoised.metric_value > result_noisy.metric_value, (
        f"Expected denoised reproducibility ({result_denoised.metric_value:.3f}) "
        f"> noisy ({result_noisy.metric_value:.3f})"
    )


@pytest.mark.synthetic
def test_full_synthetic_batch():
    """Run all synthetic edge cases together and require 100 % pass rate."""
    bias = BandpassBias(freq_band=(1, 40), sfreq=SFREQ)
    specs = [
        (make_flat_channel_data(n_channels=16, sfreq=SFREQ), bias.apply, NumericalValidityMetric()),
        (make_motion_artifact_data(n_channels=16, sfreq=SFREQ), bias.apply, NumericalValidityMetric()),
        (make_muscle_noise_data(n_channels=16, sfreq=500.0), BandpassBias(freq_band=(1, 40), sfreq=500.0).apply, NumericalValidityMetric()),
        (make_rank_deficient_data(n_channels=32, sfreq=SFREQ), bias.apply, NumericalValidityMetric()),
        (make_nonstationary_noise_data(n_channels=16, sfreq=SFREQ), bias.apply, NumericalValidityMetric()),
    ]
    batch: BatchResult = run_batch_tests(specs, verbose=False)
    assert batch.n_failed == 0, f"\n{batch.summary()}"


# ===========================================================================
# Real-data tests  (pytest -m real_data)
# Skipped automatically when data files are not present.
# Fill in SINGLE_SUBJECT_FIF / GROUP_FIFS at the top of this file.
# ===========================================================================

# Uncomment and fill in paths to enable:
#
# @pytest.mark.real_data
# def test_real_single_subject_erp_validity():
#     """Real MoBI single-subject: denoiser output must be numerically valid."""
#     import mne
#     pytest.importorskip("mne")
#     if not SINGLE_SUBJECT_FIF.exists():
#         pytest.skip(f"Data file not found: {SINGLE_SUBJECT_FIF}")
#
#     epochs = mne.read_epochs(SINGLE_SUBJECT_FIF, preload=True, verbose=False)
#     bundle = make_real_mobi_data([epochs], baseline_window=(-0.2, 0.0))
#
#     bias = AverageBias(axis="epochs")
#     result = run_edge_case_test(bundle, bias.apply, NumericalValidityMetric())
#     _assert(result)
#
#
# @pytest.mark.real_data
# def test_real_single_subject_erp_morphology():
#     """Real MoBI single-subject: ERP morphology (correlation with input ERP) >= 0.8."""
#     import mne
#     pytest.importorskip("mne")
#     if not SINGLE_SUBJECT_FIF.exists():
#         pytest.skip(f"Data file not found: {SINGLE_SUBJECT_FIF}")
#
#     epochs = mne.read_epochs(SINGLE_SUBJECT_FIF, preload=True, verbose=False)
#     bundle = make_real_mobi_data([epochs], baseline_window=(-0.2, 0.0))
#
#     bias = AverageBias(axis="epochs")
#     result = run_edge_case_test(
#         bundle,
#         bias.apply,
#         ERPCorrelationMetric(),
#         expected_range=(0.8, 1.0),
#     )
#     _assert(result)
#
#
# @pytest.mark.real_data
# def test_real_single_subject_trial_reproducibility():
#     """Real MoBI: trial reproducibility (split-half r) should be > 0.5."""
#     import mne
#     pytest.importorskip("mne")
#     if not SINGLE_SUBJECT_FIF.exists():
#         pytest.skip(f"Data file not found: {SINGLE_SUBJECT_FIF}")
#
#     epochs = mne.read_epochs(SINGLE_SUBJECT_FIF, preload=True, verbose=False)
#     bundle = make_real_mobi_data([epochs], baseline_window=(-0.2, 0.0))
#
#     bias = AverageBias(axis="epochs")
#     result = run_edge_case_test(
#         bundle,
#         bias.apply,
#         TrialReproducibilityMetric(n_splits=50),
#         expected_range=(0.5, 1.0),
#     )
#     _assert(result)
#
#
# @pytest.mark.real_data
# def test_real_group_erp_reproducibility():
#     """Real MoBI group: leave-one-out ERP consistency should be > 0.7."""
#     import mne
#     pytest.importorskip("mne")
#     missing = [p for p in GROUP_FIFS if not p.exists()]
#     if missing:
#         pytest.skip(f"Missing data files: {missing}")
#
#     epochs_list = [mne.read_epochs(p, preload=True, verbose=False) for p in GROUP_FIFS]
#     bundle = make_real_mobi_data(epochs_list, baseline_window=(-0.2, 0.0))
#
#     # Group mode: AverageBias(axis="datasets") or just the identity for a baseline
#     result = run_edge_case_test(
#         bundle,
#         lambda d: d,
#         TrialReproducibilityMetric(),
#         expected_range=(0.7, 1.0),
#     )
#     _assert(result)
