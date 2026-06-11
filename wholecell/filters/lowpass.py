"""
lowpass.py
----------
Low-pass filtering utilities for electrophysiology traces.

All filtering is zero-phase (forward-backward Butterworth) to avoid
introducing phase distortion. The filter is applied per-analysis — the
raw data on the Recording is never modified.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def apply_lowpass(
    voltage: np.ndarray,
    sampling_rate_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter to a voltage trace.

    Parameters
    ----------
    voltage : np.ndarray
        Input voltage trace (mV), shape (n_samples,).
    sampling_rate_hz : float
        Sampling rate of the recording (Hz).
    cutoff_hz : float
        Filter cutoff frequency (Hz). Must be less than sampling_rate_hz / 2.
    order : int
        Butterworth filter order. Default 4.

    Returns
    -------
    np.ndarray
        Filtered voltage array, same shape as input.

    Raises
    ------
    ValueError
        If cutoff_hz >= sampling_rate_hz / 2 (Nyquist violation).

    Notes
    -----
    Uses ``scipy.signal.sosfiltfilt`` (second-order sections, zero-phase)
    which is numerically more stable than direct-form filtfilt for higher
    orders.
    """
    nyquist_hz = sampling_rate_hz / 2.0

    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"Low-pass cutoff ({cutoff_hz:.1f} Hz) must be less than the "
            f"Nyquist frequency ({nyquist_hz:.1f} Hz) for a sampling rate "
            f"of {sampling_rate_hz:.1f} Hz."
        )

    sos = signal.butter(order, cutoff_hz, btype="low", fs=sampling_rate_hz, output="sos")
    return signal.sosfiltfilt(sos, voltage).astype(voltage.dtype)


def suggest_cutoff(sampling_rate_hz: float, purpose: str = "spike") -> float:
    """Return a sensible default cutoff frequency for common use cases.

    Parameters
    ----------
    sampling_rate_hz : float
    purpose : str
        ``"spike"``   → 5000 Hz (preserves fast AP kinetics)
        ``"passive"`` → 1000 Hz (removes high-freq noise for Rin/tau)
        ``"sag"``     → 500 Hz  (smooth enough for slow Ih kinetics)

    Returns
    -------
    float
        Suggested cutoff in Hz, capped at Nyquist - 1 kHz.

    Notes
    -----
    These are sensible defaults, not mandates. Users should inspect filtered
    traces before committing to a cutoff.
    """
    presets = {
        "spike": 5000.0,
        "passive": 1000.0,
        "sag": 500.0,
    }
    if purpose not in presets:
        raise ValueError(
            f"Unknown purpose '{purpose}'. Valid: {list(presets.keys())}"
        )

    cutoff = presets[purpose]
    nyquist = sampling_rate_hz / 2.0
    return min(cutoff, nyquist - 1000.0)
