"""
features.py
-----------
Per-spike feature extraction: threshold voltage, peak, trough, half-width,
AHP depth, and adaptation metrics.

These features are computed from the raw (or lightly filtered) voltage trace
around each detected spike. They are distinct from detection — detection
finds spikes, this module measures their shapes.

The output of run_feature_extraction feeds:
  - Cell.export_spike_table() → CSV with filename, sweep_index, and features
  - GUI spike shape overlay (threshold, peak, and trough markers on traces)
  - Phase plane plots (dV/dt vs V)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wholecell.core.sweep_collection import SweepCollection, SweepRef


# ---------------------------------------------------------------------------
# Public entry point (called by Cell.extract_spike_features)
# ---------------------------------------------------------------------------

def run_feature_extraction(
    collection: SweepCollection,
    spike_result: dict,
    lowpass_hz: float | None = None,
) -> dict:
    """Extract per-spike features for all spikes in a detection result.

    Parameters
    ----------
    collection : SweepCollection
    spike_result : dict
        Output of ``run_spike_detection`` (the ``"data"`` field of the
        timestamped result stored on Cell).
    lowpass_hz : float or None
        Lowpass filter applied to voltage before feature extraction. Applied
        independently of any filter used during detection.

    Returns
    -------
    dict with keys:
        - ``"spike_table"`` (list of dict): one dict per spike, suitable for
          pd.DataFrame construction. Columns:

          Always present:
            ``filename``, ``sweep_index``, ``spike_index_in_sweep``
            ``display_label`` (GUI label, not a key)

          Detection carry-overs:
            ``peak_time_s``, ``peak_voltage_mV``
            ``threshold_time_s``, ``threshold_voltage_mV``
            ``trough_time_s``, ``trough_voltage_mV``

          Extracted features:
            ``height_mV``             spike height (peak - threshold)
            ``half_width_ms``         width at half-maximum amplitude
            ``ahp_depth_mV``          after-hyperpolarisation depth
                                      (threshold_voltage - trough_voltage)
            ``rise_time_ms``          threshold to peak
            ``decay_time_ms``         peak to trough
            ``upstroke_mVms``         max dV/dt during upstroke
            ``downstroke_mVms``       min dV/dt during downstroke

        - ``"cell_level"`` (dict): across-cell summary scalars:
            ``first_spike_threshold_mV``, ``first_spike_peak_mV``,
            ``mean_half_width_ms``, and adaptation index.
    """
    spike_table: list[dict] = []

    for sweep_data in spike_result["data"]["per_sweep"]:
        ref = SweepRef(
            filename=sweep_data["filename"],
            sweep_index=sweep_data["sweep_index"],
            display_label=sweep_data["display_label"],
        )
        time, voltage, _ = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)

        for spike_dict in sweep_data["spikes"]:
            features = extract_single_spike_features(
                time=time,
                voltage=voltage,
                peak_index=_time_to_index(time, spike_dict["peak_time_s"]),
                threshold_index=_time_to_index(time, spike_dict["threshold_time_s"]),
                trough_index=_time_to_index(time, spike_dict["trough_time_s"]),
            )
            row = {**spike_dict, **features}
            spike_table.append(row)

    cell_level = _compute_cell_level_features(spike_table)

    return {
        "spike_table": spike_table,
        "cell_level": cell_level,
    }


# ---------------------------------------------------------------------------
# Single-spike feature extraction
# ---------------------------------------------------------------------------

def extract_single_spike_features(
    time: np.ndarray,
    voltage: np.ndarray,
    peak_index: int,
    threshold_index: int,
    trough_index: int,
    upstroke_search_window_ms: float = 2.0,
    downstroke_search_window_ms: float = 5.0,
) -> dict:
    """Compute shape features for one action potential.

    Parameters
    ----------
    time : np.ndarray
        Full sweep time array (s).
    voltage : np.ndarray
        Full sweep voltage array (mV).
    peak_index : int
        Sample index of the spike peak.
    threshold_index : int
        Sample index of the spike threshold (dV/dt crossing).
    trough_index : int
        Sample index of the fast trough.
    upstroke_search_window_ms : float
        Window around the upstroke to search for max dV/dt (ms). Default 2.
    downstroke_search_window_ms : float
        Window around the downstroke to search for min dV/dt (ms). Default 5.

    Returns
    -------
    dict with keys: ``height_mV``, ``half_width_ms``, ``ahp_depth_mV``,
    ``rise_time_ms``, ``decay_time_ms``, ``upstroke_mVms``,
    ``downstroke_mVms``.

    Notes
    -----
    All features return NaN if computation fails (e.g. index out of bounds,
    division by zero). This is intentional — bad sweeps should not crash
    the pipeline.

    TODO: implement each feature.
    """
    dt_s = time[1] - time[0] if len(time) > 1 else 1e-5
    sampling_rate_hz = 1.0 / dt_s

    peak_v = voltage[peak_index]
    threshold_v = voltage[threshold_index]
    trough_v = voltage[trough_index]

    height_mV = float(peak_v - threshold_v)
    ahp_depth_mV = float(threshold_v - trough_v)

    rise_time_ms = float((time[peak_index] - time[threshold_index]) * 1000.0)
    decay_time_ms = float((time[trough_index] - time[peak_index]) * 1000.0)

    half_width_ms = _estimate_half_width(
        time, voltage, threshold_index, peak_index, trough_index, threshold_v, peak_v
    )

    dvdt_mVms = np.gradient(voltage, time) / 1000.0
    upstroke_mVms = _max_dvdt_in_window(
        dvdt_mVms, threshold_index, peak_index
    )
    downstroke_mVms = _min_dvdt_in_window(
        dvdt_mVms, peak_index, trough_index
    )

    return {
        "height_mV": height_mV,
        "half_width_ms": half_width_ms,
        "ahp_depth_mV": ahp_depth_mV,
        "rise_time_ms": rise_time_ms,
        "decay_time_ms": decay_time_ms,
        "upstroke_mVms": upstroke_mVms,
        "downstroke_mVms": downstroke_mVms,
    }


# ---------------------------------------------------------------------------
# Feature helpers (stubs)
# ---------------------------------------------------------------------------

def _estimate_half_width(
    time: np.ndarray,
    voltage: np.ndarray,
    threshold_index: int,
    peak_index: int,
    trough_index: int,
    threshold_voltage: float,
    peak_voltage: float,
) -> float:
    """Estimate spike width at half-maximum amplitude (ms).

    Half-maximum is defined as: threshold_voltage + (peak_voltage - threshold_voltage) / 2.

    Finds the two samples that cross this level — one on the rising phase
    (between threshold and peak) and one on the falling phase (between peak
    and trough) — and interpolates for sub-sample precision.

    Returns NaN if the half-max crossing cannot be found.
    """
    half_max = threshold_voltage + (peak_voltage - threshold_voltage) / 2.0

    # Rising crossing: scan threshold_index → peak_index for upward crossing
    rise_idx = None
    for k in range(threshold_index, peak_index):
        if voltage[k] <= half_max <= voltage[k + 1]:
            # Linear interpolation
            frac = (half_max - voltage[k]) / (voltage[k + 1] - voltage[k])
            rise_idx = k + frac
            break

    # Falling crossing: scan peak_index → trough_index for downward crossing
    fall_idx = None
    for k in range(peak_index, min(trough_index, len(voltage) - 2)):
        if voltage[k] >= half_max >= voltage[k + 1]:
            frac = (voltage[k] - half_max) / (voltage[k] - voltage[k + 1])
            fall_idx = k + frac
            break

    if rise_idx is None or fall_idx is None:
        return float("nan")

    dt_s = time[1] - time[0] if len(time) > 1 else 1e-5
    return float((fall_idx - rise_idx) * dt_s * 1000.0)


def _max_dvdt_in_window(
    dvdt: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Return maximum dV/dt (mV/ms) between start and end indices."""
    if start >= end or end > len(dvdt):
        return float("nan")
    return float(np.max(dvdt[start:end]))


def _min_dvdt_in_window(
    dvdt: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Return minimum dV/dt (mV/ms) between start and end indices."""
    if start >= end or end > len(dvdt):
        return float("nan")
    return float(np.min(dvdt[start:end]))


def _time_to_index(time: np.ndarray, target_s: float) -> int:
    """Return the sample index closest to target_s in a time array."""
    return int(np.argmin(np.abs(time - target_s)))


# ---------------------------------------------------------------------------
# Cell-level feature summary
# ---------------------------------------------------------------------------

def _compute_cell_level_features(spike_table: list[dict]) -> dict:
    """Compute across-cell summary features from the spike table.

    Parameters
    ----------
    spike_table : list of dict

    Returns
    -------
    dict with keys:
        ``first_spike_threshold_mV``, ``first_spike_peak_mV``,
        ``mean_half_width_ms``, ``adaptation_index``.

    Notes
    -----
    First spike features use the spike with the earliest ``threshold_time_s``
    across all sweeps.

    Adaptation index = (ISI_last - ISI_first) / (ISI_last + ISI_first)
    computed within the sweep with the most spikes.

    TODO: implement adaptation index calculation.
    """
    if not spike_table:
        return {}

    # Sort by threshold time to find first spike
    sorted_spikes = sorted(spike_table, key=lambda r: r.get("threshold_time_s", float("inf")))
    first = sorted_spikes[0]

    cell = {
        "first_spike_threshold_mV": first.get("threshold_voltage_mV", float("nan")),
        "first_spike_peak_mV": first.get("peak_voltage_mV", float("nan")),
        "first_spike_height_mV": first.get("height_mV", float("nan")),
    }

    half_widths = [r["half_width_ms"] for r in spike_table
                   if r.get("half_width_ms") is not None and
                   not np.isnan(r["half_width_ms"])]
    if half_widths:
        cell["mean_half_width_ms"] = float(np.mean(half_widths))

    # Adaptation index from the sweep with the most spikes
    by_sweep: dict[tuple, list] = {}
    for row in spike_table:
        key = (row.get("filename"), row.get("sweep_index"))
        by_sweep.setdefault(key, []).append(row)

    most_spikes = max(by_sweep.values(), key=len, default=[])
    if len(most_spikes) >= 3:
        # Sort by threshold time within sweep
        sweep_spikes = sorted(most_spikes, key=lambda r: r.get("threshold_time_s", 0.0))
        times = [r["threshold_time_s"] for r in sweep_spikes]
        isis = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        isi_first, isi_last = isis[0], isis[-1]
        denom = isi_last + isi_first
        if denom > 0:
            cell["adaptation_index"] = float((isi_last - isi_first) / denom)

    return cell
