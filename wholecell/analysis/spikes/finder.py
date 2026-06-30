"""
finder.py
---------
Collection-level spike detection runner. Called by Cell.find_spikes.

Responsibilities
~~~~~~~~~~~~~~~~
- Iterate over all sweeps in a SweepCollection
- Dispatch to the chosen SpikeFinder backend
- Return ALL detected spikes, each annotated with the epoch they fall in
- Epoch-based filtering is intentionally NOT done here — it is the caller's
  responsibility (e.g. run_fi_analysis filters to the stimulus epoch)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wholecell.core.sweep_collection import SweepCollection, SweepRef
from wholecell.analysis.spikes.base import build_spike_finder, SpikeDetection
from wholecell.filters.lowpass import apply_lowpass

_DETECTION_LOWPASS_HZ = 2000.0
# Window (±samples) to search in raw voltage when refining the peak index.
# The lowpass filter rounds the sharp AP peak, shifting filtered argmax by
# 1-2 samples vs the true raw maximum.  3 samples = 0.15 ms at 20 kHz.
_PEAK_REFINE_SAMPLES = 3


def run_spike_detection(
    collection: SweepCollection,
    epoch_index: int,
    backend: str = "derivative",
    lowpass_hz: float | None = None,
    **spike_finder_kwargs: Any,
) -> dict:
    """Detect spikes in all sweeps of a SweepCollection.

    All spikes found in each sweep are returned regardless of epoch.
    Each spike dict includes ``epoch_at_threshold`` (which epoch the threshold
    crossing falls in) and ``latency_to_epoch_onset_ms`` (time from that
    epoch's start).  Downstream callers can filter by epoch as needed.

    Parameters
    ----------
    collection : SweepCollection
    epoch_index : int
        Zero-based epoch index for the reference current injection window.
        Used only to compute ``current_injection_pA`` per sweep — NOT used
        to filter spikes. Fails loudly if epoch cannot be parsed.
    backend : str
        ``"derivative"`` or ``"ipfx"``.
    lowpass_hz : float or None
        Lowpass filter applied to voltage before spike detection.
    **spike_finder_kwargs
        Passed to the spike finder backend constructor.

    Returns
    -------
    dict with keys:
        - ``"backend"`` (str)
        - ``"backend_params"`` (dict): all detection parameters, for audit log
        - ``"epoch_index"`` (int): the reference epoch index
        - ``"per_sweep"`` (list of dict): one entry per sweep, each with:
            - ``"filename"`` (str)
            - ``"sweep_index"`` (int)
            - ``"display_label"`` (str)
            - ``"epoch_index"`` (int)
            - ``"epoch_start_s"`` (float) — reference epoch start
            - ``"epoch_end_s"`` (float) — reference epoch end
            - ``"current_injection_pA"`` (float) — mean current in reference epoch
            - ``"n_spikes"`` (int) — total spikes detected in sweep
            - ``"spikes"`` (list of dict): one dict per spike with all
              SpikeDetection fields plus:
                ``epoch_at_threshold`` (int or None),
                ``latency_to_epoch_onset_ms`` (float, NaN if outside all epochs),
                ``sweep_current_injection_pA`` (float),
                ``current_at_threshold_pA`` (float),
                ``slow_ahp_voltage_mV`` (float),
                ``slow_ahp_time_s`` (float)

    Raises
    ------
    RuntimeError
        If epoch parsing fails for any sweep. Fails loudly.
    """
    finder = build_spike_finder(backend, **spike_finder_kwargs)

    per_sweep = []

    for ref in collection.sweeps:
        sweep_result = _detect_in_sweep(
            collection, ref, epoch_index, finder, lowpass_hz
        )
        per_sweep.append(sweep_result)

    return {
        "backend": backend,
        "backend_params": finder.params,
        "epoch_index": epoch_index,
        "lowpass_hz": lowpass_hz,
        "per_sweep": per_sweep,
    }


def _detect_in_sweep(
    collection: SweepCollection,
    ref: SweepRef,
    epoch_index: int,
    finder,
    lowpass_hz: float | None,
) -> dict:
    """Run spike detection for one sweep and annotate spikes with epoch info.

    Parameters
    ----------
    collection : SweepCollection
    ref : SweepRef
    epoch_index : int
        Reference epoch for current amplitude measurement only.
    finder : SpikeFinder
    lowpass_hz : float or None

    Returns
    -------
    dict
        Per-sweep result dict (see run_spike_detection docstring).
    """
    time, voltage_raw, current = collection.get_sweep_arrays(ref)

    rec = collection._recordings[ref.filename]

    # Build a filtered copy for detection logic only (dV/dt, peak/trough search).
    # voltage_raw is kept intact so that feature values (peak, threshold, trough
    # voltages) are read from the unfiltered trace — the 2 kHz filter attenuates
    # the peaks of fast APs and must not bias the stored voltage measurements.
    nyquist = (1.0 / (time[1] - time[0])) / 2.0
    if _DETECTION_LOWPASS_HZ < nyquist:
        voltage_det = apply_lowpass(voltage_raw, nyquist * 2.0, _DETECTION_LOWPASS_HZ)
    else:
        voltage_det = voltage_raw

    epoch = rec.get_epoch(ref.sweep_index, epoch_index)
    epoch_start_s = epoch.start_time_s
    epoch_end_s = epoch.end_time_s

    if np.all(np.isnan(current)):
        try:
            current = rec.get_command_waveform(ref.sweep_index)
        except Exception:
            pass

    current_pA = _epoch_mean_current(current, epoch.start_sample, epoch.end_sample)

    all_spikes = finder.detect(time, voltage_det, current)

    # Refine peak index and overwrite all voltage values from the raw trace.
    # The 2 kHz filter rounds the sharp AP peak, shifting the filtered argmax
    # by 1-2 samples vs the true raw maximum.  Re-find the peak in a small
    # neighborhood of voltage_raw, then read all voltages from that raw trace.
    for sp in all_spikes:
        lo = max(0, sp.peak_index - _PEAK_REFINE_SAMPLES)
        hi = min(len(voltage_raw), sp.peak_index + _PEAK_REFINE_SAMPLES + 1)
        sp.peak_index = lo + int(np.argmax(voltage_raw[lo:hi]))
        sp.peak_time_s = float(time[sp.peak_index])
        sp.peak_voltage_mV = float(voltage_raw[sp.peak_index])
        sp.threshold_voltage_mV = float(voltage_raw[sp.threshold_index])
        sp.trough_voltage_mV = float(voltage_raw[sp.trough_index])

    spike_dicts = [
        _spike_to_dict(sp, ref, i)
        for i, sp in enumerate(all_spikes)
    ]

    for sd, sp in zip(spike_dicts, all_spikes):
        idx = min(sp.threshold_index, len(current) - 1)
        sd["current_at_threshold_pA"] = float(current[idx])

    # Slow AHP: minimum voltage between each spike's trough and the next
    # spike's threshold (or the reference epoch end for the last spike).
    for i, sd in enumerate(spike_dicts):
        start = all_spikes[i].trough_index
        if i + 1 < len(all_spikes):
            end = all_spikes[i + 1].threshold_index
        else:
            end = epoch.end_sample
        end = min(end, len(voltage_raw))
        if end > start:
            rel = int(np.argmin(voltage_raw[start:end]))
            ahp_idx = start + rel
            sd["slow_ahp_voltage_mV"] = float(voltage_raw[ahp_idx])
            sd["slow_ahp_time_s"] = float(time[ahp_idx])
        else:
            sd["slow_ahp_voltage_mV"] = float("nan")
            sd["slow_ahp_time_s"] = float("nan")

    # Annotate each spike with its epoch and latency to that epoch's onset.
    all_epochs = rec.get_epochs(ref.sweep_index)
    for sd, sp in zip(spike_dicts, all_spikes):
        ep_info = _find_epoch_at_time(all_epochs, sp.threshold_time_s)
        if ep_info is not None:
            sd["epoch_at_threshold"] = ep_info.epoch_index
            sd["latency_to_epoch_onset_ms"] = (
                sp.threshold_time_s - ep_info.start_time_s
            ) * 1000.0
        else:
            sd["epoch_at_threshold"] = None
            sd["latency_to_epoch_onset_ms"] = float("nan")
        sd["sweep_current_injection_pA"] = current_pA

    return {
        "filename": ref.filename,
        "sweep_index": ref.sweep_index,
        "display_label": ref.display_label,
        "epoch_index": epoch_index,
        "epoch_start_s": epoch_start_s,
        "epoch_end_s": epoch_end_s,
        "current_injection_pA": current_pA,
        "n_spikes": len(spike_dicts),
        "spikes": spike_dicts,
    }


def _spike_to_dict(
    spike: SpikeDetection,
    ref: SweepRef,
    spike_index_in_sweep: int,
) -> dict:
    """Convert a SpikeDetection to a flat dict for export."""
    return {
        "filename": ref.filename,
        "sweep_index": ref.sweep_index,
        "spike_index_in_sweep": spike_index_in_sweep,
        "display_label": ref.display_label,
        "peak_time_s": spike.peak_time_s,
        "peak_voltage_mV": spike.peak_voltage_mV,
        "threshold_time_s": spike.threshold_time_s,
        "threshold_voltage_mV": spike.threshold_voltage_mV,
        "trough_time_s": spike.trough_time_s,
        "trough_voltage_mV": spike.trough_voltage_mV,
        "backend": spike.backend,
    }


def _find_epoch_at_time(epochs: list, time_s: float):
    """Return the EpochInfo whose window contains time_s, or None."""
    for ep in epochs:
        if ep.start_time_s <= time_s < ep.end_time_s:
            return ep
    return None


def _epoch_mean_current(
    current: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> float:
    """Return mean current over the epoch window, ignoring NaN."""
    segment = current[start_sample:end_sample]
    valid = segment[~np.isnan(segment)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")
