"""
features.py
-----------
Per-spike shape measurements: height, half-width, AHP depth, rise/decay times,
and up/downstroke velocities, plus cell-level summaries built from them.

The per-spike measurements are computed at detection time — ``finder.py`` calls
``extract_single_spike_features`` for every spike it finds, so every spike dict
in a ``run_spike_detection`` result already carries the fields listed in
``SPIKE_SHAPE_FIELDS``. This module therefore does the measuring but not the
iterating; ``run_feature_extraction`` only assembles what the finder produced.

The output feeds:
  - Every exported spike table (GUI Export > Spike Table, batch_intrinsic)
    via ``spike_table_dataframe``
  - ``run_ramp_analysis`` first-AP features
  - GUI spike shape overlay (threshold, peak, and trough markers on traces)
  - Phase plane plots (dV/dt vs V)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from wholecell.core.sweep_collection import SweepCollection, SweepRef


# Shape features measured for every detected spike. Single source of truth for
# these column names — finder.py, ramp.py and the export helpers all read it.
SPIKE_SHAPE_FIELDS = (
    "height_mV",
    "half_width_ms",
    "ahp_depth_mV",
    "rise_time_ms",
    "decay_time_ms",
    "upstroke_mVms",
    "downstroke_mVms",
)

# Column order for every exported spike table. Detection fields first (in the
# order finder.py builds them), then the shape features.
SPIKE_TABLE_COLUMNS = [
    "filename",
    "sweep_index",
    "spike_index_in_sweep",
    "peak_time_s",
    "peak_voltage_mV",
    "threshold_time_s",
    "threshold_voltage_mV",
    "trough_time_s",
    "trough_voltage_mV",
    "backend",
    "current_at_threshold_pA",
    "slow_ahp_voltage_mV",
    "slow_ahp_time_s",
    "epoch_at_threshold",
    "latency_to_epoch_onset_ms",
    "sweep_current_injection_pA",
    *SPIKE_SHAPE_FIELDS,
]


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
# Spike table assembly / export
# ---------------------------------------------------------------------------

def spike_table_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the canonical spike-table DataFrame from per-spike dicts.

    Shared by the GUI export (``trace_viewer._on_export_spike_table``) and the
    batch script so both write identical columns in identical order.

    ``display_label`` is dropped (it duplicates filename/sweep_index), rows are
    sorted by sweep then spike order, and columns are reindexed to
    ``SPIKE_TABLE_COLUMNS``. Any column not listed there is appended at the end
    rather than dropped, so a new finder field still exports.

    Parameters
    ----------
    rows : list of dict
        Per-spike dicts, e.g. the concatenation of every
        ``result["per_sweep"][i]["spikes"]``.

    Returns
    -------
    pd.DataFrame
    """
    if not rows:
        return pd.DataFrame(columns=SPIKE_TABLE_COLUMNS)

    df = (
        pd.DataFrame(rows)
        .drop(columns=["display_label"], errors="ignore")
        .sort_values(["filename", "sweep_index", "spike_index_in_sweep"])
        .reset_index(drop=True)
    )
    ordered = [c for c in SPIKE_TABLE_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in SPIKE_TABLE_COLUMNS]
    return df[ordered + extra]


def ensure_shape_features(
    collection: SweepCollection,
    sweep_data: dict,
    lowpass_hz: float | None = None,
) -> list[dict]:
    """Return one sweep's spike dicts, backfilling shape features if absent.

    Spike dicts from the current finder already carry every field in
    ``SPIKE_SHAPE_FIELDS``. Results loaded from a session JSON saved before
    those were added do not, so they are measured here from the trace. The
    common case (nothing missing) returns the input list untouched and reads
    no data.

    Parameters
    ----------
    collection : SweepCollection
        Must contain the sweep referenced by ``sweep_data``. If the sweep
        cannot be read, the spikes are returned unmodified.
    sweep_data : dict
        One entry of a ``run_spike_detection`` result's ``"per_sweep"`` list.
    lowpass_hz : float or None
        Applied only on the backfill path.

    Returns
    -------
    list of dict
    """
    spikes = sweep_data.get("spikes", [])
    if all(f in sp for sp in spikes for f in SPIKE_SHAPE_FIELDS):
        return spikes

    ref = SweepRef(
        filename=sweep_data["filename"],
        sweep_index=sweep_data["sweep_index"],
        display_label=sweep_data.get("display_label", ""),
    )
    try:
        time, voltage, _ = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)
    except Exception:
        return spikes

    dvdt_mVms = np.gradient(voltage, time) / 1000.0

    out: list[dict] = []
    for sp in spikes:
        if all(f in sp for f in SPIKE_SHAPE_FIELDS):
            out.append(sp)
            continue
        features = extract_single_spike_features(
            time=time,
            voltage=voltage,
            peak_index=_time_to_index(time, sp["peak_time_s"]),
            threshold_index=_time_to_index(time, sp["threshold_time_s"]),
            trough_index=_time_to_index(time, sp["trough_time_s"]),
            dvdt_mVms=dvdt_mVms,
        )
        out.append({**sp, **features})
    return out


# ---------------------------------------------------------------------------
# Public entry point (called by Cell.extract_spike_features)
# ---------------------------------------------------------------------------

def run_feature_extraction(
    collection: SweepCollection,
    spike_result: dict,
    lowpass_hz: float | None = None,
    stimulus_epoch_index: int | None = None,
) -> dict:
    """Assemble the per-spike feature table for a detection result.

    Shape features are measured by ``finder.py`` at detection time, so this is
    a flatten-and-summarise pass rather than a second measurement pass.

    Parameters
    ----------
    collection : SweepCollection
    spike_result : dict
        Output of ``run_spike_detection`` (the ``"data"`` field of the
        timestamped result stored on Cell).
    lowpass_hz : float or None
        Only used to backfill spike results saved before shape features were
        computed at detection time (see ``ensure_shape_features``).
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
          filtering). Columns are ``SPIKE_TABLE_COLUMNS``.

        - ``"cell_level"`` (dict): across-cell summary scalars including
          ``first_spike_*`` fields auto-collected from all numeric spike-table
          columns (excluding identity columns) for the first evoked AP, plus
          ``adaptation_index`` and ``mean_half_width_ms``.
    """
    spike_table: list[dict] = []

    for sweep_data in spike_result["data"]["per_sweep"]:
        spike_table.extend(
            ensure_shape_features(collection, sweep_data, lowpass_hz)
        )

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
    dvdt_mVms: np.ndarray | None = None,
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
    dvdt_mVms : np.ndarray or None
        Precomputed dV/dt (mV/ms) for the whole sweep. Computing it here costs
        a full-sweep pass per spike, so callers measuring many spikes in one
        sweep should compute it once and pass it in. Defaults to computing it
        from ``voltage`` and ``time``.

    Returns
    -------
    dict with keys: ``height_mV``, ``half_width_ms``, ``ahp_depth_mV``,
    ``rise_time_ms``, ``decay_time_ms``, ``upstroke_mVms``,
    ``downstroke_mVms`` — i.e. ``SPIKE_SHAPE_FIELDS``.

    Notes
    -----
    All features return NaN if computation fails (e.g. index out of bounds,
    division by zero). This is intentional — bad sweeps should not crash
    the pipeline.
    """
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

    if dvdt_mVms is None:
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
# Feature helpers
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
