"""
fi_curve.py
-----------
Frequency-current (F-I) curve construction and fitting.

This module takes spike detection results (from finder.py) and per-sweep
epoch information to build the F-I curve: injected current amplitude vs.
mean firing rate (and/or spike count) per sweep.

Outputs
~~~~~~~
- Per-sweep: current_injection_pA, n_spikes, mean_firing_rate_hz,
  instantaneous_rates (list), first_isi_ms, last_isi_ms
- Cell-level: rheobase_pA, fi_slope_hz_per_pA (linear fit above rheobase),
  max_firing_rate_hz, the full F-I curve as parallel lists

The full F-I curve is stored as lists (not a fixed-width table) to
accommodate cells with different numbers of current steps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wholecell.core.sweep_collection import SweepCollection


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_fi_analysis(
    collection: SweepCollection,
    epoch_index: int,
    spike_result: dict,
) -> dict:
    """Build F-I curve from spike detection results.

    Parameters
    ----------
    collection : SweepCollection
    epoch_index : int
        Used to retrieve epoch duration and current step amplitudes.
        Fails loudly if epoch cannot be parsed.
    spike_result : dict
        Output of ``run_spike_detection`` (the ``"data"`` field from Cell).
        Must have been run on the same collection.

    Returns
    -------
    dict with keys:
        - ``"per_sweep"`` (list of dict): one entry per sweep with:
            ``filename``, ``sweep_index``, ``display_label``,
            ``current_injection_pA``, ``epoch_duration_s``, ``n_spikes``,
            ``mean_firing_rate_hz``, ``instantaneous_rates_hz`` (list),
            ``first_isi_ms``, ``last_isi_ms``

        - ``"fi_curve"`` (dict): parallel lists for plotting:
            ``current_pA`` (list), ``mean_rate_hz`` (list),
            ``n_spikes`` (list)

        - ``"cell_level"`` (dict):
            ``rheobase_pA``, ``fi_slope_hz_per_pA``, ``max_firing_rate_hz``,
            ``n_steps_analyzed``

    Raises
    ------
    RuntimeError
        If epoch duration cannot be determined for any sweep.
    """
    per_sweep = []

    for sweep_data in spike_result["data"]["per_sweep"]:
        row = _build_sweep_row(collection, sweep_data, epoch_index)
        per_sweep.append(row)

    # Sort by current amplitude for clean F-I curve
    per_sweep.sort(key=lambda r: r["current_injection_pA"])

    fi_curve = {
        "current_pA": [r["current_injection_pA"] for r in per_sweep],
        "mean_rate_hz": [r["mean_firing_rate_hz"] for r in per_sweep],
        "n_spikes": [r["n_spikes"] for r in per_sweep],
    }

    cell_level = _compute_cell_level_fi(per_sweep)

    return {
        "per_sweep": per_sweep,
        "fi_curve": fi_curve,
        "cell_level": cell_level,
    }


# ---------------------------------------------------------------------------
# Per-sweep row builder
# ---------------------------------------------------------------------------

def _build_sweep_row(
    collection: SweepCollection,
    sweep_data: dict,
    epoch_index: int,
) -> dict:
    """Build a per-sweep F-I row from spike detection data.

    Parameters
    ----------
    collection : SweepCollection
    sweep_data : dict
        One element of spike_result["data"]["per_sweep"].
    epoch_index : int

    Returns
    -------
    dict
    """
    from wholecell.core.sweep_collection import SweepRef

    ref = SweepRef(
        filename=sweep_data["filename"],
        sweep_index=sweep_data["sweep_index"],
        display_label=sweep_data["display_label"],
    )

    # Epoch duration from the recording
    rec = collection._recordings[ref.filename]
    epoch = rec.get_epoch(ref.sweep_index, epoch_index)
    epoch_duration_s = epoch.end_time_s - epoch.start_time_s

    spike_times = [sp["threshold_time_s"] for sp in sweep_data["spikes"]]
    n_spikes = len(spike_times)

    mean_rate_hz = (n_spikes / epoch_duration_s) if epoch_duration_s > 0 else 0.0

    isis_ms, inst_rates_hz = _compute_isis(spike_times)

    return {
        "filename": ref.filename,
        "sweep_index": ref.sweep_index,
        "display_label": ref.display_label,
        "current_injection_pA": sweep_data["current_injection_pA"],
        "epoch_duration_s": epoch_duration_s,
        "n_spikes": n_spikes,
        "mean_firing_rate_hz": float(mean_rate_hz),
        "instantaneous_rates_hz": inst_rates_hz,
        "first_isi_ms": float(isis_ms[0]) if isis_ms else float("nan"),
        "last_isi_ms": float(isis_ms[-1]) if isis_ms else float("nan"),
    }


# ---------------------------------------------------------------------------
# Cell-level F-I summary
# ---------------------------------------------------------------------------

def _compute_cell_level_fi(per_sweep: list[dict]) -> dict:
    """Estimate rheobase, F-I slope, and max firing rate.

    Parameters
    ----------
    per_sweep : list of dict
        Sorted by current_injection_pA.

    Returns
    -------
    dict with keys: rheobase_pA, fi_slope_hz_per_pA, max_firing_rate_hz,
    n_steps_analyzed.

    Notes
    -----
    Rheobase: smallest current injection at which at least one spike occurred.

    F-I slope: linear regression of mean_rate_hz vs current_injection_pA
    for sweeps above rheobase. Stored as Hz/pA.

    TODO: implement linear regression; handle non-monotonic F-I curves
    (common in some cell types). Consider offering both full-range and
    linear-range slope estimates.
    """
    n = len(per_sweep)

    rheobase_pA = float("nan")
    for row in per_sweep:
        if row["n_spikes"] > 0:
            rheobase_pA = float(row["current_injection_pA"])
            break

    max_rate = max((r["mean_firing_rate_hz"] for r in per_sweep), default=float("nan"))

    # TODO: linear regression for F-I slope above rheobase

    return {
        "rheobase_pA": rheobase_pA,
        "fi_slope_hz_per_pA": float("nan"),  # TODO
        "max_firing_rate_hz": float(max_rate),
        "n_steps_analyzed": n,
    }


# ---------------------------------------------------------------------------
# ISI helpers
# ---------------------------------------------------------------------------

def _compute_isis(
    spike_times_s: list[float],
) -> tuple[list[float], list[float]]:
    """Compute inter-spike intervals and instantaneous firing rates.

    Parameters
    ----------
    spike_times_s : list of float
        Spike threshold times in seconds, ordered chronologically.

    Returns
    -------
    isis_ms : list of float
        Inter-spike intervals in milliseconds.
    instantaneous_rates_hz : list of float
        Instantaneous firing rate for each ISI (1 / ISI in seconds).
    """
    if len(spike_times_s) < 2:
        return [], []

    isis_s = [
        spike_times_s[i + 1] - spike_times_s[i]
        for i in range(len(spike_times_s) - 1)
    ]
    isis_ms = [isi * 1000.0 for isi in isis_s]
    rates_hz = [1.0 / isi for isi in isis_s if isi > 0]

    return isis_ms, rates_hz
