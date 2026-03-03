"""Edge-case validation framework for DSS denoisers on MoBI data.

MoBI (Mobile Brain/Body Imaging) recordings are characterized by heavy
motion artifacts, muscle contamination, line noise, and rank-deficient
sensor arrays after re-referencing.  This module provides a thin
framework for running a denoiser against such edge cases and evaluating
the result with a pluggable metric.

Usage example
-------------
>>> from tests.mobi.edge_case_tester import (
...     run_edge_case_test,
...     make_flat_channel_data,
...     BandPowerRatioMetric,
... )
>>> from mne_denoise.dss.denoisers import BandpassBias
>>>
>>> bundle = make_flat_channel_data(n_channels=32, sfreq=250)
>>> bias = BandpassBias(freq_band=(8, 12), sfreq=250)
>>> result = run_edge_case_test(bundle, method=bias.apply, metric=NumericalValidityMetric())
>>> assert result.passed
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# 1.  Data container
# ---------------------------------------------------------------------------


@dataclass
class DataBundle:
    """Data and metadata for a single edge-case scenario.

    Parameters
    ----------
    data : ndarray
        The raw input data.  Shape conventions follow the DSS library:
        - ``(n_channels, n_times)``            - continuous data
        - ``(n_channels, n_times, n_epochs)``  - epoched data
        - ``(n_datasets, n_channels, n_times)``- group/JDSS data
    description : str
        Human-readable label identifying the edge case (e.g.
        ``"flat_channel - 4 dead channels out of 64"``).
    sfreq : float
        Sampling frequency in Hz.  Used by metrics that need frequency
        information.
    clean_reference : ndarray, optional
        Ground-truth clean signal with the same shape as ``data``.
        Required by metrics such as :class:`SNRMetric`.
    metadata : dict
        Free-form key-value store for anything else (e.g. artifact
        onset indices, channel labels, expected SNR bounds).
    """

    data: np.ndarray
    description: str
    sfreq: float = 1000.0
    clean_reference: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2.  Result container
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Outcome of a single edge-case test.

    Attributes
    ----------
    passed : bool
        ``True`` when the denoiser ran without exceptions *and* the
        metric value fell within ``expected_range`` (if provided).
    metric_name : str
        Name of the metric that was evaluated.
    metric_value : float or None
        Scalar outcome of the metric, or ``None`` if an exception
        prevented evaluation.
    description : str
        Copied from :attr:`DataBundle.description` for traceability.
    elapsed_s : float
        Wall-clock time (seconds) spent inside the denoiser call.
    exception : Exception or None
        The exception raised by the denoiser or metric, if any.
    output : ndarray or None
        The array returned by the denoiser (for further inspection).
    """

    passed: bool
    metric_name: str
    metric_value: float | None
    description: str
    elapsed_s: float = 0.0
    exception: Exception | None = None
    output: np.ndarray | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        val = f"{self.metric_value:.4g}" if self.metric_value is not None else "N/A"
        return (
            f"[{status}] {self.description!r}  "
            f"{self.metric_name}={val}  ({self.elapsed_s:.3f}s)"
        )


# ---------------------------------------------------------------------------
# 3.  Metric protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Metric(Protocol):
    """Interface that every metric must satisfy.

    Implementations should be stateless callables.  The ``name``
    attribute is used in :class:`TestResult` and printed summaries.
    """

    name: str

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        """Compute a scalar score from the denoiser output.

        Parameters
        ----------
        bundle : DataBundle
            The original input bundle (gives access to ``data``,
            ``clean_reference``, ``sfreq``, and ``metadata``).
        output : ndarray
            The array returned by the denoiser.

        Returns
        -------
        score : float
            A scalar value.  Higher is *not* necessarily better; the
            direction is metric-specific.
        """
        ...


# ---------------------------------------------------------------------------
# 4.  Core orchestration function
# ---------------------------------------------------------------------------


def run_edge_case_test(
    bundle: DataBundle,
    method: Callable[[np.ndarray], np.ndarray],
    metric: Metric,
    *,
    expected_range: tuple[float, float] | None = None,
) -> TestResult:
    """Run *method* on *bundle* and evaluate with *metric*.

    This is the single entry-point for all edge-case tests.

    Parameters
    ----------
    bundle : DataBundle
        Input data and associated metadata.
    method : callable
        A function ``(data: ndarray) -> ndarray``.  Wrap your denoiser
        here; for example::

            bias = BandpassBias(freq_band=(8, 12), sfreq=250)
            run_edge_case_test(bundle, method=bias.apply, ...)

            # Or a full DSS pipeline:
            def dss_method(data):
                dss = DSS(bias=bias, n_components=3, return_type="raw")
                dss.fit(data)
                return dss.transform(data)

    metric : Metric
        An object satisfying the :class:`Metric` protocol.
    expected_range : (low, high), optional
        If provided, ``result.passed`` is ``True`` only when the metric
        value falls in ``[low, high]`` (inclusive).  If omitted, the
        test passes as long as no exception is raised and the output is
        numerically valid (no NaN / Inf).

    Returns
    -------
    result : TestResult
    """
    output: np.ndarray | None = None
    metric_value: float | None = None
    exc: Exception | None = None
    passed = False

    t0 = time.perf_counter()
    try:
        output = method(bundle.data)
    except Exception as e:  # noqa: BLE001
        exc = e
        elapsed = time.perf_counter() - t0
        return TestResult(
            passed=False,
            metric_name=metric.name,
            metric_value=None,
            description=bundle.description,
            elapsed_s=elapsed,
            exception=exc,
            output=None,
        )
    elapsed = time.perf_counter() - t0

    # --- Numeric sanity check (always performed) ---------------------------
    if not _is_numerically_valid(output):
        exc = ValueError("Denoiser output contains NaN or Inf values.")
        return TestResult(
            passed=False,
            metric_name=metric.name,
            metric_value=None,
            description=bundle.description,
            elapsed_s=elapsed,
            exception=exc,
            output=output,
        )

    # --- Metric evaluation -------------------------------------------------
    try:
        metric_value = float(metric(bundle, output))
    except Exception as e:  # noqa: BLE001
        exc = e
        return TestResult(
            passed=False,
            metric_name=metric.name,
            metric_value=None,
            description=bundle.description,
            elapsed_s=elapsed,
            exception=exc,
            output=output,
        )

    # --- Pass / fail decision ----------------------------------------------
    if expected_range is not None:
        low, high = expected_range
        passed = low <= metric_value <= high
    else:
        passed = True  # No range constraint → pass if we got here

    return TestResult(
        passed=passed,
        metric_name=metric.name,
        metric_value=metric_value,
        description=bundle.description,
        elapsed_s=elapsed,
        exception=None,
        output=output,
    )


# ---------------------------------------------------------------------------
# 5.  Built-in metrics
# ---------------------------------------------------------------------------


class NumericalValidityMetric:
    """Check that the output is finite and shape-preserving.

    Returns ``1.0`` on success, ``0.0`` otherwise.
    """

    name = "numerical_validity"

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        if not _is_numerically_valid(output):
            return 0.0
        if output.shape != bundle.data.shape:
            return 0.0
        return 1.0


class SNRMetric:
    """SNR improvement in dB relative to a clean reference.

    Requires ``bundle.clean_reference`` to be set.

    The score is ``SNR_out - SNR_in``.  Positive means the denoiser
    improved the SNR.
    """

    name = "snr_improvement_dB"

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        if bundle.clean_reference is None:
            raise ValueError(
                "SNRMetric requires bundle.clean_reference to be set."
            )
        ref = bundle.clean_reference
        snr_in = _snr_db(bundle.data, ref)
        snr_out = _snr_db(output, ref)
        return snr_out - snr_in


class BandPowerRatioMetric:
    """Ratio of power in *artifact_band* before vs. after denoising.

    A value > 1 means power in the artifact band was reduced.

    Parameters
    ----------
    artifact_band : (low_hz, high_hz)
        Frequency band of the artifact to suppress.
    sfreq : float, optional
        Override the sampling frequency from the bundle.
    """

    name = "artifact_band_power_ratio"

    def __init__(
        self,
        artifact_band: tuple[float, float],
        sfreq: float | None = None,
    ) -> None:
        self.artifact_band = artifact_band
        self._sfreq_override = sfreq

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        sfreq = self._sfreq_override if self._sfreq_override is not None else bundle.sfreq
        power_before = _band_power(bundle.data, sfreq, self.artifact_band)
        power_after = _band_power(output, sfreq, self.artifact_band)
        if power_after < 1e-30:
            return np.inf
        return float(power_before / power_after)


class SignalBandPreservationMetric:
    """Fraction of signal-band power retained after denoising.

    A value close to 1 means the denoiser left the signal band intact.

    Parameters
    ----------
    signal_band : (low_hz, high_hz)
        Frequency band of the signal of interest.
    sfreq : float, optional
        Override the sampling frequency from the bundle.
    """

    name = "signal_band_preservation"

    def __init__(
        self,
        signal_band: tuple[float, float],
        sfreq: float | None = None,
    ) -> None:
        self.signal_band = signal_band
        self._sfreq_override = sfreq

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        sfreq = self._sfreq_override if self._sfreq_override is not None else bundle.sfreq
        power_before = _band_power(bundle.data, sfreq, self.signal_band)
        if power_before < 1e-30:
            return 1.0  # Nothing to preserve
        power_after = _band_power(output, sfreq, self.signal_band)
        return float(power_after / power_before)


class OutputShapeMetric:
    """Verify that the output shape matches the input shape.

    Returns ``1.0`` when shapes match, ``0.0`` otherwise.
    """

    name = "output_shape_preserved"

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        return 1.0 if output.shape == bundle.data.shape else 0.0


class ERPCorrelationMetric:
    """Pearson correlation between the ERP extracted from the output and a reference.

    The ERP is obtained by averaging across trials (single-subject) or subjects
    (group).  If ``bundle.clean_reference`` is set it is used as the reference;
    otherwise the ERP of the raw ``bundle.data`` is used (measuring morphology
    preservation rather than absolute recovery).

    A score of 1 means perfect morphology match; 0 means orthogonal.

    The data type (single-subject vs. group) is read from
    ``bundle.metadata['data_type']`` (set automatically by
    :func:`make_real_mobi_data`).  Defaults to ``'single_subject'``.
    """

    name = "erp_correlation"

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        data_type = bundle.metadata.get("data_type", "single_subject")
        erp_out = _compute_erp(output, data_type)

        if bundle.clean_reference is not None:
            ref = bundle.clean_reference
            erp_ref = _compute_erp(ref, data_type) if ref.ndim == 3 else ref
        else:
            erp_ref = _compute_erp(bundle.data, data_type)

        return float(np.corrcoef(erp_out.ravel(), erp_ref.ravel())[0, 1])


class ERPPeakSNRMetric:
    """Peak-to-baseline SNR of the ERP extracted from the denoiser output (in dB).

    Computes ``20 * log10(peak_amplitude / baseline_rms)`` on the grand-average
    ERP, where *baseline* is a pre-stimulus window and *peak* is searched over
    the post-baseline portion (or a custom window).

    Parameters
    ----------
    baseline_window : (t_start_s, t_end_s), optional
        Baseline period in seconds relative to epoch onset.  Falls back to
        ``bundle.metadata['baseline_window']``, then to the first 20 % of
        the epoch.
    peak_window : (t_start_s, t_end_s), optional
        Restrict the peak search to this window.  If ``None``, the entire
        post-baseline region is searched.
    """

    name = "erp_peak_snr_dB"

    def __init__(
        self,
        baseline_window: tuple[float, float] | None = None,
        peak_window: tuple[float, float] | None = None,
    ) -> None:
        self.baseline_window = baseline_window
        self.peak_window = peak_window

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        sfreq = bundle.sfreq
        data_type = bundle.metadata.get("data_type", "single_subject")
        erp = _compute_erp(output, data_type)  # (n_channels, n_times)
        n_times = erp.shape[1]

        # Resolve baseline window
        bw = self.baseline_window or bundle.metadata.get("baseline_window")
        if bw is not None:
            bl_start = max(0, int(bw[0] * sfreq))
            bl_end = min(n_times, int(bw[1] * sfreq))
        else:
            bl_start, bl_end = 0, max(1, n_times // 5)

        # Resolve peak window
        if self.peak_window is not None:
            pk_start = max(0, int(self.peak_window[0] * sfreq))
            pk_end = min(n_times, int(self.peak_window[1] * sfreq))
        else:
            pk_start, pk_end = bl_end, n_times

        baseline_rms = float(np.sqrt(np.mean(erp[:, bl_start:bl_end] ** 2)))
        peak_amplitude = float(np.max(np.abs(erp[:, pk_start:pk_end])))

        if baseline_rms < 1e-30:
            return np.inf
        return float(20.0 * np.log10(peak_amplitude / baseline_rms))


class TrialReproducibilityMetric:
    """ERP reproducibility via split-half (single-subject) or leave-one-out (group).

    *Single-subject* (``data_type='single_subject'``,
    ``data`` shape ``(n_channels, n_times, n_epochs)``):
        Epochs are randomly split in half ``n_splits`` times.  An ERP is
        computed from each half, and Pearson r between them is averaged
        across splits.  High r → the response is consistent across trials.

    *Group* (``data_type='group'``,
    ``data`` shape ``(n_subjects, n_channels, n_times)``):
        Leave-one-out cross-subject ERP correlation: for each subject,
        correlate their evoked with the grand average computed from all
        *other* subjects.  High r → individual ERPs are consistent with
        the group template.

    Parameters
    ----------
    n_splits : int
        Number of random splits used in single-subject mode.  Default 20.
    rng_seed : int
        Seed for reproducibility.  Default 0.
    """

    name = "trial_reproducibility"

    def __init__(self, n_splits: int = 20, rng_seed: int = 0) -> None:
        self.n_splits = n_splits
        self.rng_seed = rng_seed

    def __call__(self, bundle: DataBundle, output: np.ndarray) -> float:
        data_type = bundle.metadata.get("data_type", "single_subject")
        rng = np.random.default_rng(self.rng_seed)

        if data_type == "group":
            return self._group_reproducibility(output)
        return self._split_half(output, rng)

    def _split_half(self, data: np.ndarray, rng: np.random.Generator) -> float:
        """Split-half correlation for (n_channels, n_times, n_epochs) data."""
        if data.ndim != 3:
            raise ValueError(
                "TrialReproducibilityMetric requires 3-D data "
                "(n_channels, n_times, n_epochs) for single-subject mode."
            )
        n_ch, n_times, n_epochs = data.shape
        if n_epochs < 4:
            raise ValueError(f"Need at least 4 epochs for split-half, got {n_epochs}.")

        correlations = []
        for _ in range(self.n_splits):
            idx = rng.permutation(n_epochs)
            half = n_epochs // 2
            erp_a = data[:, :, idx[:half]].mean(axis=2)
            erp_b = data[:, :, idx[half : 2 * half]].mean(axis=2)
            r = float(np.corrcoef(erp_a.ravel(), erp_b.ravel())[0, 1])
            correlations.append(r)

        return float(np.mean(correlations))

    def _group_reproducibility(self, data: np.ndarray) -> float:
        """LOO cross-subject ERP correlation for (n_subjects, n_channels, n_times) data."""
        if data.ndim != 3:
            raise ValueError(
                "TrialReproducibilityMetric requires 3-D data "
                "(n_subjects, n_channels, n_times) for group mode."
            )
        n_subjects = data.shape[0]
        if n_subjects < 3:
            raise ValueError(
                f"Need at least 3 subjects for LOO reproducibility, got {n_subjects}."
            )

        correlations = []
        for i in range(n_subjects):
            grand_avg = np.delete(data, i, axis=0).mean(axis=0)  # (n_ch, n_times)
            r = float(np.corrcoef(data[i].ravel(), grand_avg.ravel())[0, 1])
            correlations.append(r)

        return float(np.mean(correlations))


# ---------------------------------------------------------------------------
# 6.  MoBI edge-case data generators
# ---------------------------------------------------------------------------
# Each generator returns a DataBundle ready to be passed to
# run_edge_case_test().  All data is synthetic but is designed to mimic
# the pathological properties that arise in real MoBI recordings.


def make_flat_channel_data(
    n_channels: int = 64,
    n_times: int = 5000,
    sfreq: float = 250.0,
    n_flat: int = 4,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Data with a subset of completely flat (dead/disconnected) channels.

    Flat channels produce rank deficiency and can destabilise covariance
    inversion inside DSS.

    Parameters
    ----------
    n_flat : int
        Number of channels to set to exactly zero.
    """
    rng = rng or np.random.default_rng(0)
    data = rng.standard_normal((n_channels, n_times))
    flat_idx = rng.choice(n_channels, size=n_flat, replace=False)
    data[flat_idx] = 0.0
    return DataBundle(
        data=data,
        description=f"flat_channel - {n_flat}/{n_channels} dead channels",
        sfreq=sfreq,
        metadata={"flat_channel_indices": flat_idx.tolist()},
    )


def make_motion_artifact_data(
    n_channels: int = 32,
    n_times: int = 5000,
    sfreq: float = 250.0,
    n_steps: int = 5,
    step_amplitude: float = 10.0,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Data with abrupt DC-shift motion artifacts on a subset of channels.

    Step-function jumps are a hallmark of cable/electrode movement in
    ambulatory EEG.  They make the data non-stationary and can dominate
    the baseline covariance.
    """
    rng = rng or np.random.default_rng(1)
    signal = rng.standard_normal((n_channels, n_times))

    # Inject step artifacts on half the channels
    artifact_channels = n_channels // 2
    step_times = rng.integers(100, n_times - 100, size=n_steps)
    artifact = np.zeros((artifact_channels, n_times))
    for t in step_times:
        artifact[:, t:] += step_amplitude * rng.standard_normal(artifact_channels)[:, None]

    data = signal.copy()
    data[:artifact_channels] += artifact
    return DataBundle(
        data=data,
        description=f"motion_artifact - {n_steps} step jumps (amplitude={step_amplitude})",
        sfreq=sfreq,
        metadata={"step_times": step_times.tolist()},
    )


def make_muscle_noise_data(
    n_channels: int = 32,
    n_times: int = 5000,
    sfreq: float = 500.0,
    muscle_band: tuple[float, float] = (70.0, 150.0),
    muscle_snr: float = 5.0,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Data with broadband high-frequency muscle (EMG) contamination.

    Muscle noise is spectrally flat above ~70 Hz and can swamp the EEG
    signal in ambulatory recordings.
    """
    from scipy.signal import butter, sosfiltfilt

    rng = rng or np.random.default_rng(2)
    signal = rng.standard_normal((n_channels, n_times))

    # Broadband noise filtered to the muscle band
    nyq = sfreq / 2.0
    sos = butter(4, [muscle_band[0] / nyq, muscle_band[1] / nyq], btype="band", output="sos")
    raw_noise = rng.standard_normal((n_channels, n_times))
    muscle = sosfiltfilt(sos, raw_noise, axis=1)

    # Scale to desired SNR
    sig_rms = np.sqrt(np.mean(signal**2))
    muscle_rms = np.sqrt(np.mean(muscle**2)) + 1e-15
    muscle = muscle * (sig_rms / muscle_rms) * (1.0 / muscle_snr)

    data = signal + muscle
    return DataBundle(
        data=data,
        description=f"muscle_noise - band {muscle_band} Hz, SNR={muscle_snr}",
        sfreq=sfreq,
        metadata={"muscle_band": muscle_band, "muscle_snr": muscle_snr},
    )


def make_line_noise_data(
    n_channels: int = 32,
    n_times: int = 5000,
    sfreq: float = 1000.0,
    line_freq: float = 50.0,
    n_harmonics: int = 3,
    line_amplitude: float = 2.0,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Data with sinusoidal line-noise and harmonics added to all channels.

    Line-noise amplitude varies per channel to simulate realistic
    capacitive coupling differences.
    """
    rng = rng or np.random.default_rng(3)
    t = np.arange(n_times) / sfreq
    signal = rng.standard_normal((n_channels, n_times))

    # Add harmonic line noise
    line_noise = np.zeros((n_channels, n_times))
    for h in range(1, n_harmonics + 1):
        freq = line_freq * h
        if freq >= sfreq / 2:
            break
        amplitude = line_amplitude / h  # Decreasing harmonics
        phase = rng.uniform(0, 2 * np.pi, (n_channels, 1))
        channel_scale = rng.uniform(0.5, 1.5, (n_channels, 1))
        line_noise += amplitude * channel_scale * np.sin(2 * np.pi * freq * t + phase)

    data = signal + line_noise
    return DataBundle(
        data=data,
        description=f"line_noise - {line_freq} Hz + {n_harmonics} harmonics",
        sfreq=sfreq,
        metadata={"line_freq": line_freq, "n_harmonics": n_harmonics},
    )


def make_rank_deficient_data(
    n_channels: int = 64,
    n_times: int = 5000,
    sfreq: float = 250.0,
    n_independent_sources: int = 20,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Data generated from fewer independent sources than channels.

    Rank deficiency is common in EEG after average referencing or when
    using dense arrays with correlated electrodes.  It causes numerical
    issues in matrix inversion inside DSS.
    """
    rng = rng or np.random.default_rng(4)
    sources = rng.standard_normal((n_independent_sources, n_times))
    # Random mixing matrix maps sources to channels
    mixing = rng.standard_normal((n_channels, n_independent_sources))
    data = mixing @ sources
    # Small sensor noise to prevent exact rank deficiency
    data += rng.standard_normal((n_channels, n_times)) * 0.01
    return DataBundle(
        data=data,
        description=(
            f"rank_deficient - {n_independent_sources} sources, "
            f"{n_channels} channels (rank={n_independent_sources})"
        ),
        sfreq=sfreq,
        metadata={"n_independent_sources": n_independent_sources},
    )


def make_short_segment_data(
    n_channels: int = 32,
    n_times: int = 100,
    sfreq: float = 250.0,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Very short data segment — fewer samples than channels.

    Short segments make covariance estimation unreliable and can trigger
    edge effects in IIR/FIR filters.
    """
    rng = rng or np.random.default_rng(5)
    data = rng.standard_normal((n_channels, n_times))
    return DataBundle(
        data=data,
        description=f"short_segment - {n_times} samples, {n_channels} channels",
        sfreq=sfreq,
    )


def make_low_snr_data(
    n_channels: int = 32,
    n_times: int = 5000,
    sfreq: float = 250.0,
    signal_freq: float = 10.0,
    snr_db: float = -10.0,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Narrowband signal buried under broadband noise at a low SNR.

    Parameters
    ----------
    snr_db : float
        Signal-to-noise ratio in dB.  Negative values mean noise > signal.
    """
    rng = rng or np.random.default_rng(6)
    t = np.arange(n_times) / sfreq

    # Clean signal: narrow-band oscillation
    phases = rng.uniform(0, 2 * np.pi, (n_channels, 1))
    clean = np.sin(2 * np.pi * signal_freq * t + phases)

    # Noise
    noise = rng.standard_normal((n_channels, n_times))
    noise_scale = 10 ** (-snr_db / 20.0)  # convert dB to linear
    data = clean + noise_scale * noise

    return DataBundle(
        data=data,
        description=f"low_snr - {snr_db} dB SNR, {signal_freq} Hz signal",
        sfreq=sfreq,
        clean_reference=clean,
        metadata={"signal_freq": signal_freq, "snr_db": snr_db},
    )


def make_nonstationary_noise_data(
    n_channels: int = 32,
    n_times: int = 10000,
    sfreq: float = 250.0,
    n_segments: int = 4,
    rng: np.random.Generator | None = None,
) -> DataBundle:
    """Data with noise variance that changes abruptly between segments.

    Non-stationarity invalidates the stationary noise assumption made by
    many denoisers and is a realistic property of outdoor MoBI recordings.
    """
    rng = rng or np.random.default_rng(7)
    data = np.zeros((n_channels, n_times))
    seg_len = n_times // n_segments
    noise_scales = rng.uniform(0.1, 5.0, n_segments)

    for i, scale in enumerate(noise_scales):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else n_times
        data[:, start:end] = scale * rng.standard_normal((n_channels, end - start))

    return DataBundle(
        data=data,
        description=f"nonstationary - {n_segments} segments with varying noise (σ={noise_scales.round(2).tolist()})",
        sfreq=sfreq,
        metadata={"noise_scales": noise_scales.tolist()},
    )


def make_real_mobi_data(
    epochs_list: list,
    description: str = "",
    *,
    baseline_window: tuple[float, float] | None = None,
) -> DataBundle:
    """Build a :class:`DataBundle` from a list of MNE Epochs objects.

    This is the entry point for real-data validation.  The caller epochs
    their recordings (one ``mne.Epochs`` object per subject) and passes
    the list here.  The function handles the two relevant cases:

    **Single subject** (``len(epochs_list) == 1``):
        All trials from the single Epochs object are stacked into a
        ``(n_channels, n_times, n_epochs)`` array — the format expected
        by DSS with :class:`~mne_denoise.dss.denoisers.AverageBias`.

    **Group** (``len(epochs_list) > 1``):
        Each subject's trials are averaged into an evoked response, then
        stacked into ``(n_subjects, n_channels, n_times)`` — the format
        expected by Joint DSS / ``AverageBias(axis='datasets')``.

    The original Epochs objects are preserved in
    ``bundle.metadata['epochs_list']`` so that metrics such as
    :class:`TrialReproducibilityMetric` can access per-trial data when
    needed.

    Parameters
    ----------
    epochs_list : list of mne.Epochs
        One Epochs object per subject.  All must share the same channels,
        epoch duration, and sampling frequency.
    description : str
        Free-text label for this bundle.  Auto-generated if empty.
    baseline_window : (t_start_s, t_end_s), optional
        Pre-stimulus baseline window in seconds relative to epoch onset.
        Stored in ``metadata['baseline_window']`` for use by
        :class:`ERPPeakSNRMetric`.

    Returns
    -------
    DataBundle
    """
    if len(epochs_list) == 0:
        raise ValueError("epochs_list must contain at least one Epochs object.")

    first = epochs_list[0]
    sfreq = float(first.info["sfreq"])
    n_channels = len(first.ch_names)
    n_times = len(first.times)

    for i, ep in enumerate(epochs_list[1:], start=1):
        if float(ep.info["sfreq"]) != sfreq:
            raise ValueError(
                f"Subject {i} sfreq ({ep.info['sfreq']}) differs from subject 0 ({sfreq})."
            )
        if len(ep.ch_names) != n_channels:
            raise ValueError(
                f"Subject {i} has {len(ep.ch_names)} channels, expected {n_channels}."
            )
        if len(ep.times) != n_times:
            raise ValueError(
                f"Subject {i} epoch length ({len(ep.times)}) differs from subject 0 ({n_times})."
            )

    n_subjects = len(epochs_list)
    mode = "single_subject" if n_subjects == 1 else f"{n_subjects}_subjects"
    desc = description or f"real_mobi - {mode}"

    meta: dict = {"epochs_list": epochs_list}
    if baseline_window is not None:
        meta["baseline_window"] = baseline_window

    if n_subjects == 1:
        # MNE Epochs convention: (n_epochs, n_channels, n_times)
        # DSS convention:        (n_channels, n_times, n_epochs)
        raw = epochs_list[0].get_data()  # (n_epochs, n_ch, n_times)
        data = np.transpose(raw, (1, 2, 0))  # → (n_ch, n_times, n_epochs)
        meta["data_type"] = "single_subject"
        meta["n_epochs"] = data.shape[2]
    else:
        # Evoked per subject → (n_subjects, n_channels, n_times)
        data = np.stack([ep.average().data for ep in epochs_list], axis=0)
        meta["data_type"] = "group"
        meta["n_subjects"] = n_subjects
        meta["n_epochs_per_subject"] = [len(ep) for ep in epochs_list]

    return DataBundle(
        data=data,
        description=desc,
        sfreq=sfreq,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 7.  Private helpers
# ---------------------------------------------------------------------------


def _is_numerically_valid(arr: np.ndarray) -> bool:
    """Return True if arr contains no NaN or Inf values."""
    return bool(np.isfinite(arr).all())


def _snr_db(signal: np.ndarray, reference: np.ndarray) -> float:
    """Compute SNR in dB: 10 * log10(signal_power / noise_power).

    Noise is estimated as the residual ``signal - reference``.
    """
    noise = signal - reference
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean(noise**2))
    if noise_power < 1e-30:
        return np.inf
    return 10.0 * np.log10(signal_power / noise_power)


def _band_power(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    """Mean power of *data* in the frequency band *band* (in Hz).

    Uses Welch's method on the first two dimensions (channels × time).
    Works for 2-D ``(n_ch, n_times)`` data; for 3-D data it flattens
    the last axis first.
    """
    from scipy.signal import welch

    if data.ndim == 3:
        n_ch, n_times, n_epochs = data.shape
        data_2d = data.reshape(n_ch, -1)
    else:
        data_2d = data

    freqs, psd = welch(data_2d, fs=sfreq, axis=1)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0
    return float(np.mean(psd[:, band_mask]))


def _compute_erp(data: np.ndarray, data_type: str = "single_subject") -> np.ndarray:
    """Average *data* across trials or subjects to produce an ERP.

    Parameters
    ----------
    data : ndarray
        ``(n_channels, n_times, n_epochs)`` for single-subject, or
        ``(n_subjects, n_channels, n_times)`` for group data.
    data_type : str
        ``'single_subject'`` or ``'group'``.

    Returns
    -------
    erp : ndarray, shape (n_channels, n_times)
    """
    if data.ndim == 2:
        return data  # Already a 2-D ERP
    if data_type == "group":
        return data.mean(axis=0)  # (n_subjects, n_ch, n_times) → (n_ch, n_times)
    return data.mean(axis=2)  # (n_ch, n_times, n_epochs) → (n_ch, n_times)


# ---------------------------------------------------------------------------
# 8.  Batch runner
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    """Aggregated results from :func:`run_batch_tests`.

    Attributes
    ----------
    results : list of TestResult
        One entry per test, in run order.
    """

    results: list[TestResult]

    @property
    def n_passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def n_failed(self) -> int:
        return len(self.results) - self.n_passed

    @property
    def pass_rate(self) -> float:
        return self.n_passed / len(self.results) if self.results else 0.0

    def failed(self) -> list[TestResult]:
        """Return only the failed :class:`TestResult` entries."""
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        """Return a human-readable aligned summary table."""
        header = f"{'STATUS':<6}  {'METRIC':>28}  {'VALUE':>10}  {'TIME':>7}  DESCRIPTION"
        sep = "-" * 90
        lines = [header, sep]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            val = f"{r.metric_value:.4g}" if r.metric_value is not None else "N/A"
            exc = f"  [{type(r.exception).__name__}]" if r.exception else ""
            lines.append(
                f"{status:<6}  {r.metric_name:>28}  {val:>10}  "
                f"{r.elapsed_s:>6.3f}s  {r.description}{exc}"
            )
        lines += [
            sep,
            f"Passed {self.n_passed}/{len(self.results)} ({self.pass_rate:.0%})  "
            f"| total {sum(r.elapsed_s for r in self.results):.2f}s",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def run_batch_tests(
    test_specs: list[tuple[DataBundle, Callable, Metric]],
    *,
    expected_ranges: list[tuple[float, float] | None] | None = None,
    verbose: bool = True,
) -> BatchResult:
    """Run multiple edge-case tests and aggregate the results.

    Parameters
    ----------
    test_specs : list of (DataBundle, method, Metric)
        Each entry is exactly the positional arguments you would pass to
        :func:`run_edge_case_test`.
    expected_ranges : list of (low, high) or None, optional
        Per-test pass/fail thresholds, parallel to *test_specs*.
        Omit the list (or pass ``None`` for individual entries) to
        require only numerical validity.
    verbose : bool
        Print each :class:`TestResult` as it completes.  Default ``True``.

    Returns
    -------
    BatchResult

    Examples
    --------
    >>> specs = [
    ...     (make_flat_channel_data(), bias.apply, NumericalValidityMetric()),
    ...     (make_line_noise_data(),   bias.apply, BandPowerRatioMetric((49, 51))),
    ... ]
    >>> batch = run_batch_tests(specs, expected_ranges=[None, (1.5, None)])
    >>> print(batch.summary())
    """
    if expected_ranges is None:
        expected_ranges = [None] * len(test_specs)
    elif len(expected_ranges) != len(test_specs):
        raise ValueError(
            f"expected_ranges has {len(expected_ranges)} entries but "
            f"test_specs has {len(test_specs)}."
        )

    results = []
    for (bundle, method, metric), er in zip(test_specs, expected_ranges):
        result = run_edge_case_test(bundle, method, metric, expected_range=er)
        results.append(result)
        if verbose:
            print(result)

    batch = BatchResult(results=results)
    if verbose:
        print(f"\n{batch.n_passed}/{len(results)} passed ({batch.pass_rate:.0%})")
    return batch
