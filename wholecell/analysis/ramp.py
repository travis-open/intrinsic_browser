"""
ramp.py
-------
First-AP feature extraction for current ramp protocols.

Workflow:
  1. User runs "Find Spikes" on ramp sweeps (existing button).
  2. User clicks "Analyze Ramp APs" — this module is called.
  3. For each sweep, the first spike in the ramp epoch is identified and its
     measurements are read off the detection result.
  4. Features are averaged across sweeps and saved as ramp_evoked_APs.

Shape features (half-width, height, ...) are measured for every spike at
detection time by ``finder.py``, so this module carries them rather than
re-measuring the first AP. That keeps the ramp numbers identical to the ones
in the exported spike table, and keeps ``height_mV`` consistent with the
``peak_voltage_mV``/``threshold_voltage_mV`` carried alongside it.
"""

from __future__ import annotations

import numpy as np

from wholecell.core.sweep_collection import SweepCollection
from wholecell.analysis.spikes.features import (
    SPIKE_SHAPE_FIELDS,
    ensure_shape_features,
)

# Fields carried directly from the spike detection dict (no re-computation).
_CARRY_FIELDS = (
    "threshold_voltage_mV",
    "peak_voltage_mV",
    "trough_voltage_mV",
    "current_at_threshold_pA",
    "latency_to_epoch_onset_ms",
    "slow_ahp_voltage_mV",
    "slow_ahp_time_s",
    *SPIKE_SHAPE_FIELDS,
)

# Identity/metadata fields excluded from mean/std aggregation.
_IDENTITY_FIELDS = frozenset({
    "filename",
    "sweep_index",
    "display_label",
    "backend",
    "spike_index_in_sweep",
    "epoch_at_threshold",
})


def run_ramp_analysis(
    collection: SweepCollection,
    epoch_index: int,
    spike_result_entry: dict,
    lowpass_hz: float | None = None,
) -> dict:
    """Extract first-AP features from ramp-evoked spikes across sweeps.

    Parameters
    ----------
    collection : SweepCollection
        Sweeps to analyze. Sweeps referenced in spike_result_entry but absent
        from collection are silently skipped.
    epoch_index : int
        The ramp epoch index (same value used for spike detection / F-I).
        Only spikes whose threshold falls in this epoch are considered.
    spike_result_entry : dict
        Full timestamped entry from ``cell.results["spikes"][-1]``
        (i.e. the dict with "timestamp", "params", "data" keys).
    lowpass_hz : float or None
        Only used when the spike result predates detection-time shape
        features and they have to be backfilled from the trace. Shape features
        from a current detection run are measured on the raw voltage and this
        parameter does not affect them.

    Returns
    -------
    dict with keys:
        - ``"epoch_index"`` (int)
        - ``"per_sweep"`` (list of dict): one entry per sweep that had a
          spike in the ramp epoch, with all AP features plus identity fields.
        - ``"cell_level"`` (dict): mean and std for every numeric per-sweep
          field, plus ``"n_sweeps_analyzed"``.
    """
    spike_data = spike_result_entry.get("data", spike_result_entry)
    per_sweep_spikes = spike_data.get("per_sweep", [])

    collection_keys = {
        (r.filename, r.sweep_index) for r in collection.sweeps
    }

    per_sweep_results: list[dict] = []

    for sweep_sd in per_sweep_spikes:
        fname = sweep_sd["filename"]
        sw_idx = sweep_sd["sweep_index"]

        if (fname, sw_idx) not in collection_keys:
            continue

        # Backfills shape features for spike results saved before they were
        # measured at detection time; a no-op for current results.
        spikes = ensure_shape_features(collection, sweep_sd, lowpass_hz)

        epoch_spikes = [
            sp for sp in spikes
            if sp.get("epoch_at_threshold") == epoch_index
        ]
        if not epoch_spikes:
            continue

        first_spike = min(epoch_spikes, key=lambda s: s.get("threshold_time_s", float("inf")))

        display_label = sweep_sd.get("display_label", f"{fname}[{sw_idx}]")
        row: dict = {
            "filename": fname,
            "sweep_index": sw_idx,
            "display_label": display_label,
        }
        for field in _CARRY_FIELDS:
            row[field] = first_spike.get(field, float("nan"))

        per_sweep_results.append(row)

    cell_level = _aggregate(per_sweep_results)

    return {
        "epoch_index": epoch_index,
        "per_sweep": per_sweep_results,
        "cell_level": cell_level,
    }


def _aggregate(per_sweep: list[dict]) -> dict:
    """Compute mean ± std across sweeps for all numeric fields."""
    cell: dict = {"n_sweeps_analyzed": len(per_sweep)}

    if not per_sweep:
        return cell

    numeric_fields = [
        k for k in per_sweep[0]
        if k not in _IDENTITY_FIELDS
    ]

    for field in numeric_fields:
        vals = []
        for row in per_sweep:
            v = row.get(field)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            vals.append(fv)

        arr = np.array(vals, dtype=float)
        cell[f"mean_{field}"] = float(np.nanmean(arr))
        cell[f"std_{field}"] = float(np.nanstd(arr))

    return cell
