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


# Columns excluded from auto-collect when building first_spike_* cell features.
# These are identity/metadata fields, not numeric measurements.
_SPIKE_IDENTITY_COLUMNS = frozenset({
    "filename",
    "sweep_index",
    "spike_index_in_sweep",
    "display_label",
    "backend",
    "epoch_at_threshold",
})


# ---------------------------------------------------------------------------
# Public entry point (called by Cell.extract_spike_features)
# ---------------------------------------------------------------------------

def run_feature_extraction(
    collection: SweepCollection,
    spike_result: dict,
    lowpass_hz: float | None = None,
    stimulus_epoch_index: int | None = None,
) -> dict:
    """Extract per-spike features for all spikes in a detection result.

    Parameters
    ----------
    collection : SweepCollection
    spike_result : dict
        Output of ``run_spike_detection`` (the ``"data"`` field of the
        timestamped result stored on Cell).
    lowpass_hz : float or None
        Lowpass filter applied to voltage before feature extraction.
    stimulus_epoch_index : int or None
        If provided, cell-level summary features (rheobase, first-AP
        features, adaptation index) are computed only from spikes in this
        epoch. Pass the same epoch_index used for spike detection /
        F-I analysis.

    Returns
    -------
    dict with keys:
        - ``"spike_table"`` (list of dict): one dict per spike, suitable for
          pd.DataFrame construction. All spikes are included (no epoch
          filtering). Columns include all detection carry-overs, extracted
          shape features, plus ``epoch_at_threshold`` and
          ``latency_to_epoch_onset_ms`` from the finder step.

        - ``"cell_level"`` (dict): across-cell summary scalars including
          ``first_spike_*`` fields auto-collected from all numeric spike-table
          columns (excluding identity columns) for the first evoked AP, plus
          ``adaptation_index`` and ``mean_half_width_ms``.
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

    cell_level = _compute_cell_level_features(
        spike_table, stimulus_epoch_index=stimulus_epoch_index
    )

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

def _compute_cell_level_features(
    spike_table: list[dict],
    stimulus_epoch_index: int | None = None,
) -> dict:
    """Compute across-cell summary features from the spike table.

    When ``stimulus_epoch_index`` is provided, first-AP features and rheobase
    are determined from spikes in that epoch only:
      1. Filter to stimulus epoch spikes.
      2. Find the rheobase sweep (minimum sweep_current_injection_pA with ≥1 spike).
      3. Find the first evoked AP (earliest threshold_time_s in that sweep).
      4. Auto-collect all numeric fields from that spike, prefixed with
         ``first_spike_``. Identity columns (see _SPIKE_IDENTITY_COLUMNS) are
         excluded. This auto-collection picks up future spike-table columns
         automatically without requiring code changes here.

    Adaptation index is computed from the sweep (within the stimulus epoch,
    if specified) that has the most spikes.

    Parameters
    ----------
    spike_table : list of dict
    stimulus_epoch_index : int or None
    """
    if not spike_table:
        return {}

    cell: dict = {}

    # Determine the working set for first-spike and adaptation metrics
    if stimulus_epoch_index is not None:
        epoch_spikes = [
            r for r in spike_table
            if r.get("epoch_at_threshold") == stimulus_epoch_index
        ]
    else:
        epoch_spikes = spike_table

    # First evoked AP: first spike at rheobase current
    if epoch_spikes:
        # Group by sweep to find rheobase (minimum current with ≥1 spike)
        by_sweep: dict[tuple, list] = {}
        for row in epoch_spikes:
            key = (row.get("filename"), row.get("sweep_index"))
            by_sweep.setdefault(key, []).append(row)

        # Representative current for each sweep (use sweep_current_injection_pA
        # if present, otherwise fall back to current_at_threshold_pA)
        def _sweep_current(rows: list) -> float:
            v = rows[0].get("sweep_current_injection_pA",
                            rows[0].get("current_at_threshold_pA", float("nan")))
            return float(v) if v is not None else float("nan")

        rheobase_key = min(
            by_sweep.keys(),
            key=lambda k: _sweep_current(by_sweep[k]),
        )
        rheobase_spikes = sorted(
            by_sweep[rheobase_key],
            key=lambda r: r.get("threshold_time_s", float("inf")),
        )
        first = rheobase_spikes[0]

        # Auto-collect all numeric fields from the first spike
        for col, val in first.items():
            if col in _SPIKE_IDENTITY_COLUMNS:
                continue
            try:
                cell[f"first_spike_{col}"] = float(val) if val is not None else float("nan")
            except (TypeError, ValueError):
                pass  # skip non-numeric fields

        # Explicit alias for clarity (already included above but kept for discoverability)
        if "latency_to_epoch_onset_ms" in first:
            cell["first_spike_latency_ms"] = cell.get(
                "first_spike_latency_to_epoch_onset_ms", float("nan")
            )

    # Mean half-width across all spikes (not epoch-filtered — shape summary)
    half_widths = [
        r["half_width_ms"] for r in spike_table
        if r.get("half_width_ms") is not None and not np.isnan(r["half_width_ms"])
    ]
    if half_widths:
        cell["mean_half_width_ms"] = float(np.mean(half_widths))

    # Adaptation index from the sweep with the most spikes (within epoch if specified)
    by_sweep_adapt: dict[tuple, list] = {}
    for row in epoch_spikes:
        key = (row.get("filename"), row.get("sweep_index"))
        by_sweep_adapt.setdefault(key, []).append(row)

    most_spikes = max(by_sweep_adapt.values(), key=len, default=[])
    if len(most_spikes) >= 3:
        sweep_spikes = sorted(most_spikes, key=lambda r: r.get("peak_time_s", 0.0))
        times = [r["peak_time_s"] for r in sweep_spikes]
        isis = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        isi_first, isi_last = isis[0], isis[-1]
        denom = isi_last + isi_first
        if denom > 0:
            cell["adaptation_index"] = float((isi_last - isi_first) / denom)

    return cell
