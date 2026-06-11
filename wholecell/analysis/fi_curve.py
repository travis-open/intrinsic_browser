"""
fi_curve.py
-----------
Frequency-current (F-I) curve construction and fitting.

Takes spike detection results (from finder.py) and constructs the F-I curve:
injected current amplitude vs. mean firing rate and peak instantaneous rate
per sweep, restricted to spikes within the specified stimulus epoch.

Outputs
~~~~~~~
- Per-sweep: current_injection_pA, epoch_duration_s, n_spikes,
  mean_firing_rate_hz, peak_instantaneous_rate_hz, mean_isi_ms, min_isi_ms,
  instantaneous_rates_hz (list), first_isi_ms, last_isi_ms
- Cell-level: rheobase_pA, fi_slope_hz_per_pA, fi_slope_r2,
  fi_slope_n_points, max_firing_rate_hz, max_peak_instantaneous_rate_hz,
  n_steps_analyzed
- fi_curve: parallel lists for plotting (current_pA, mean_rate_hz,
  peak_rate_hz, n_spikes)
"""

from __future__ import annotations

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

    Only spikes with ``epoch_at_threshold == epoch_index`` are counted.
    This correctly handles the new all-spikes detection output where spikes
    outside the stimulus epoch are annotated but not filtered.

    Parameters
    ----------
    collection : SweepCollection
    epoch_index : int
        The stimulus epoch to analyse. Spikes in other epochs are ignored.
    spike_result : dict
        Output of ``run_spike_detection`` (the ``"data"`` field from Cell).

    Returns
    -------
    dict with keys:
        - ``"per_sweep"`` (list of dict): one entry per sweep
        - ``"fi_curve"`` (dict): parallel lists for plotting
        - ``"cell_level"`` (dict): scalar summaries

    Raises
    ------
    RuntimeError
        If epoch duration cannot be determined for any sweep.
    """
    per_sweep = []

    for sweep_data in spike_result["data"]["per_sweep"]:
        row = _build_sweep_row(collection, sweep_data, epoch_index)
        per_sweep.append(row)

    per_sweep.sort(key=lambda r: r["current_injection_pA"])

    fi_curve = {
        "current_pA": [r["current_injection_pA"] for r in per_sweep],
        "mean_rate_hz": [r["mean_firing_rate_hz"] for r in per_sweep],
        "peak_rate_hz": [r["peak_instantaneous_rate_hz"] for r in per_sweep],
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

    Filters to spikes in the specified stimulus epoch only.
    """
    from wholecell.core.sweep_collection import SweepRef

    ref = SweepRef(
        filename=sweep_data["filename"],
        sweep_index=sweep_data["sweep_index"],
        display_label=sweep_data["display_label"],
    )

    rec = collection._recordings[ref.filename]
    epoch = rec.get_epoch(ref.sweep_index, epoch_index)
    epoch_duration_s = epoch.end_time_s - epoch.start_time_s

    # Filter to spikes in the stimulus epoch only
    epoch_spikes = [
        sp for sp in sweep_data["spikes"]
        if sp.get("epoch_at_threshold") == epoch_index
    ]

    spike_times = [sp["threshold_time_s"] for sp in epoch_spikes]
    n_spikes = len(spike_times)

    mean_rate_hz = (n_spikes / epoch_duration_s) if epoch_duration_s > 0 else 0.0

    isis_ms, inst_rates_hz = _compute_isis(spike_times)

    mean_isi_ms = float(np.mean(isis_ms)) if isis_ms else float("nan")
    min_isi_ms = float(np.min(isis_ms)) if isis_ms else float("nan")
    peak_inst_rate_hz = (1000.0 / min_isi_ms) if not np.isnan(min_isi_ms) else float("nan")

    return {
        "filename": ref.filename,
        "sweep_index": ref.sweep_index,
        "display_label": ref.display_label,
        "current_injection_pA": sweep_data["current_injection_pA"],
        "epoch_duration_s": epoch_duration_s,
        "n_spikes": n_spikes,
        "mean_firing_rate_hz": float(mean_rate_hz),
        "peak_instantaneous_rate_hz": peak_inst_rate_hz,
        "mean_isi_ms": mean_isi_ms,
        "min_isi_ms": min_isi_ms,
        "instantaneous_rates_hz": inst_rates_hz,
        "first_isi_ms": float(isis_ms[0]) if isis_ms else float("nan"),
        "last_isi_ms": float(isis_ms[-1]) if isis_ms else float("nan"),
    }


# ---------------------------------------------------------------------------
# Cell-level F-I summary
# ---------------------------------------------------------------------------

def _compute_cell_level_fi(per_sweep: list[dict]) -> dict:
    """Estimate rheobase, F-I slope, and max firing rates.

    F-I slope fitting: linear regression over the ascending linear portion,
    defined as sweeps from rheobase up to (and including) the first sweep
    where mean_firing_rate_hz >= 80% of the maximum rate. This avoids
    plateau/saturation at high current steps which would bias the slope down.

    Returns NaN for slope/R² when fewer than 2 points are available.
    """
    n = len(per_sweep)

    rheobase_pA = float("nan")
    for row in per_sweep:
        if row["n_spikes"] > 0:
            rheobase_pA = float(row["current_injection_pA"])
            break

    supra = [r for r in per_sweep if r["n_spikes"] > 0]
    max_rate = max((r["mean_firing_rate_hz"] for r in supra), default=float("nan"))
    max_peak_inst = max(
        (r["peak_instantaneous_rate_hz"] for r in supra
         if not np.isnan(r["peak_instantaneous_rate_hz"])),
        default=float("nan"),
    )

    fi_slope_hz_per_pA = float("nan")
    fi_slope_r2 = float("nan")
    fi_slope_n_points = 0

    if len(supra) >= 2 and not np.isnan(max_rate) and max_rate > 0:
        rate_threshold = 0.8 * max_rate
        fit_rows = []
        for row in supra:
            fit_rows.append(row)
            if row["mean_firing_rate_hz"] >= rate_threshold:
                break

        if len(fit_rows) >= 2:
            currents = np.array([r["current_injection_pA"] for r in fit_rows])
            rates = np.array([r["mean_firing_rate_hz"] for r in fit_rows])
            coeffs = np.polyfit(currents, rates, 1)
            fi_slope_hz_per_pA = float(coeffs[0])
            fi_slope_n_points = len(fit_rows)

            # R² = 1 - SS_res / SS_tot
            predicted = np.polyval(coeffs, currents)
            ss_res = float(np.sum((rates - predicted) ** 2))
            ss_tot = float(np.sum((rates - np.mean(rates)) ** 2))
            fi_slope_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "rheobase_pA": rheobase_pA,
        "max_firing_rate_hz": float(max_rate) if not np.isnan(max_rate) else float("nan"),
        "max_peak_instantaneous_rate_hz": float(max_peak_inst),
        "fi_slope_hz_per_pA": fi_slope_hz_per_pA,
        "fi_slope_r2": fi_slope_r2,
        "fi_slope_n_points": fi_slope_n_points,
        "n_steps_analyzed": n,
    }


# ---------------------------------------------------------------------------
# ISI helpers
# ---------------------------------------------------------------------------

def _compute_isis(
    spike_times_s: list[float],
) -> tuple[list[float], list[float]]:
    """Compute inter-spike intervals and instantaneous firing rates.

    Returns
    -------
    isis_ms : list of float
    instantaneous_rates_hz : list of float
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
