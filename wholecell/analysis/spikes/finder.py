"""
finder.py
---------
Collection-level spike detection runner. Called by Cell.find_spikes.

Responsibilities
~~~~~~~~~~~~~~~~
- Iterate over all sweeps in a SweepCollection
- Slice voltage/current to the user-specified epoch window
- Dispatch to the chosen SpikeFinder backend
- Filter detections to only those within the epoch window (excludes
  spontaneous spikes outside the current injection epoch)
- Return a result dict ready for Cell._store_result
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wholecell.core.sweep_collection import SweepCollection, SweepRef
from wholecell.analysis.spikes.base import build_spike_finder, SpikeDetection


def run_spike_detection(
    collection: SweepCollection,
    epoch_index: int,
    backend: str = "derivative",
    lowpass_hz: float | None = None,
    **spike_finder_kwargs: Any,
) -> dict:
    """Detect spikes in a SweepCollection within a specific epoch.

    Only spikes whose threshold crossing occurs within the epoch time window
    are included in the results. This excludes spontaneous spikes that happen
    outside the current injection window.

    Parameters
    ----------
    collection : SweepCollection
    epoch_index : int
        Zero-based epoch index for the current injection window. Fails
        loudly if epoch cannot be parsed from any sweep.
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
        - ``"per_sweep"`` (list of dict): one entry per sweep, each with:
            - ``"filename"`` (str)
            - ``"sweep_index"`` (int)
            - ``"display_label"`` (str)
            - ``"epoch_index"`` (int)
            - ``"epoch_start_s"`` (float)
            - ``"epoch_end_s"`` (float)
            - ``"current_injection_pA"`` (float)
            - ``"n_spikes"`` (int)
            - ``"spikes"`` (list of dict): one dict per spike, with all
              SpikeDetection fields as keys. ``filename`` and ``sweep_index``
              are included in every spike dict for downstream export.

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
    """Run spike detection for one sweep and filter to the epoch window.

    Parameters
    ----------
    collection : SweepCollection
    ref : SweepRef
    epoch_index : int
    finder : SpikeFinder
    lowpass_hz : float or None

    Returns
    -------
    dict
        Per-sweep result dict (see run_spike_detection docstring).
    """
    # Get full sweep arrays (voltage filtered if lowpass_hz is set)
    time, voltage, current = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)

    # Get epoch boundaries — fails loudly if epochs cannot be parsed
    rec = collection._recordings[ref.filename]
    epoch = rec.get_epoch(ref.sweep_index, epoch_index)
    epoch_start_s = epoch.start_time_s
    epoch_end_s = epoch.end_time_s

    # Determine current injection amplitude from the epoch
    current_pA = _epoch_mean_current(current, epoch.start_sample, epoch.end_sample)

    # Run detection on the FULL sweep (backend sees complete trace)
    # Filtering to epoch window happens below — this mirrors IPFX behaviour
    # and avoids edge effects from slicing before detection.
    all_spikes = finder.detect(time, voltage, current)

    # Filter: only keep spikes whose threshold crossing is within the epoch
    epoch_spikes = [
        sp for sp in all_spikes
        if epoch_start_s <= sp.threshold_time_s < epoch_end_s
    ]

    spike_dicts = [
        _spike_to_dict(sp, ref, i)
        for i, sp in enumerate(epoch_spikes)
    ]

    # Current injection at each spike's threshold sample.
    # Falls back to the epoch command level when the recorded current channel
    # is absent (single-channel ABF files return all-NaN for current).
    for sd, sp in zip(spike_dicts, epoch_spikes):
        idx = min(sp.threshold_index, len(current) - 1)
        val = float(current[idx])
        sd["current_at_threshold_pA"] = val if not np.isnan(val) else float(epoch.level)

    # Slow AHP: minimum voltage between each spike and the next spike's
    # threshold (or the epoch end for the last spike).
    for i, sd in enumerate(spike_dicts):
        start = epoch_spikes[i].trough_index
        if i + 1 < len(epoch_spikes):
            end = epoch_spikes[i + 1].threshold_index
        else:
            end = epoch.end_sample
        end = min(end, len(voltage))
        if end > start:
            rel = int(np.argmin(voltage[start:end]))
            ahp_idx = start + rel
            sd["slow_ahp_voltage_mV"] = float(voltage[ahp_idx])
            sd["slow_ahp_time_s"] = float(time[ahp_idx])
        else:
            sd["slow_ahp_voltage_mV"] = float("nan")
            sd["slow_ahp_time_s"] = float("nan")

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
    """Convert a SpikeDetection to a flat dict for export.

    ``filename`` and ``sweep_index`` are always present as separate atomic
    fields so downstream users never need to parse compound identifiers.
    ``display_label`` is included for GUI use only.
    """
    return {
        # Identity — always separate, never compound
        "filename": ref.filename,
        "sweep_index": ref.sweep_index,
        "spike_index_in_sweep": spike_index_in_sweep,
        "display_label": ref.display_label,
        # Detection outputs
        "peak_time_s": spike.peak_time_s,
        "peak_voltage_mV": spike.peak_voltage_mV,
        "threshold_time_s": spike.threshold_time_s,
        "threshold_voltage_mV": spike.threshold_voltage_mV,
        "trough_time_s": spike.trough_time_s,
        "trough_voltage_mV": spike.trough_voltage_mV,
        "backend": spike.backend,
    }


def _epoch_mean_current(
    current: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> float:
    """Return mean current over the epoch window, ignoring NaN."""
    segment = current[start_sample:end_sample]
    valid = segment[~np.isnan(segment)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")
