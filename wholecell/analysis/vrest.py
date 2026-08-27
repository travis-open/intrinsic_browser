"""
vrest.py
--------
Resting membrane potential estimation from one or more sweeps.

For sweeps with current injection (step or ramp protocols), only the
pre-injection baseline period (epoch 0) is used. For sweeps with no
meaningful current injection (gap-free recordings, free-run protocols),
the full sweep is averaged.
"""

from __future__ import annotations

import numpy as np

from wholecell.core.sweep_collection import SweepCollection, SweepRef

# Current range (max - min) must exceed this to be considered an injection.
_INJECTION_THRESHOLD_PA = 5.0


def run_vrest_analysis(
    collection: SweepCollection,
    lowpass_hz: float | None = None,
) -> dict:
    """Estimate resting membrane potential from a set of sweeps.

    Parameters
    ----------
    collection : SweepCollection
        Sweeps to analyze (typically the checked resting-potential sweeps).
    lowpass_hz : float or None
        Optional lowpass filter applied to voltage before averaging.

    Returns
    -------
    dict with keys:
        - ``"per_sweep"`` (list of dict): one entry per sweep with
          ``filename``, ``sweep_index``, ``display_label``, and
          ``v_rest_mV`` (the mean voltage for that sweep's baseline window).
        - ``"cell_level"`` (dict): ``v_rest_mV`` (mean across sweeps),
          ``v_rest_std_mV``, ``n_sweeps_analyzed``, and
          ``initial_voltage_mV`` (mean voltage during the first epoch of the
          first sweep; falls back to the full first sweep when it has no
          epoch table).
    """
    per_sweep: list[dict] = []

    for ref in collection.sweeps:
        v_mean = _baseline_voltage(collection, ref, lowpass_hz)
        per_sweep.append({
            "filename": ref.filename,
            "sweep_index": ref.sweep_index,
            "display_label": ref.display_label,
            "v_rest_mV": v_mean,
        })

    vals = [r["v_rest_mV"] for r in per_sweep
            if not np.isnan(r["v_rest_mV"])]

    cell_level: dict = {"n_sweeps_analyzed": len(per_sweep)}
    if vals:
        cell_level["v_rest_mV"] = float(np.mean(vals))
        cell_level["v_rest_std_mV"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    else:
        cell_level["v_rest_mV"] = float("nan")
        cell_level["v_rest_std_mV"] = float("nan")

    if collection.sweeps:
        cell_level["initial_voltage_mV"] = _initial_voltage(
            collection, collection.sweeps[0], lowpass_hz
        )
    else:
        cell_level["initial_voltage_mV"] = float("nan")

    return {
        "per_sweep": per_sweep,
        "cell_level": cell_level,
    }


def _has_current_injection(
    collection: SweepCollection,
    ref: SweepRef,
) -> bool:
    """Return True if the sweep contains meaningful current injection.

    Checks the recorded current channel first (range > threshold). If the
    current channel is all NaN, falls back to checking epoch command levels.
    """
    _, _, current = collection.get_sweep_arrays(ref)
    valid = current[~np.isnan(current)]
    if len(valid) > 0:
        return float(np.max(valid) - np.min(valid)) > _INJECTION_THRESHOLD_PA

    # No recorded current — check epoch command levels
    try:
        rec = collection._recordings[ref.filename]
        epochs = rec.get_epochs(ref.sweep_index)
        return any(abs(ep.level) > _INJECTION_THRESHOLD_PA for ep in epochs)
    except Exception:
        return False


def _initial_voltage(
    collection: SweepCollection,
    ref: SweepRef,
    lowpass_hz: float | None,
) -> float:
    """Return the mean voltage during the first epoch of ``ref``.

    Uses epoch 0 (the pre-injection holding period). If the sweep has no
    epoch table, falls back to averaging the full sweep.
    """
    try:
        _, v, _ = collection.get_epoch_arrays(ref, 0, lowpass_hz=lowpass_hz)
        valid = v[~np.isnan(v)]
        if len(valid) > 0:
            return float(np.mean(valid))
    except Exception:
        pass

    _, v, _ = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)
    valid = v[~np.isnan(v)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


def _baseline_voltage(
    collection: SweepCollection,
    ref: SweepRef,
    lowpass_hz: float | None,
) -> float:
    """Return mean voltage for the appropriate baseline window.

    If the sweep has no current injection, averages the full sweep.
    If current injection is present, uses epoch 0 (the pre-injection
    holding period) as the baseline window.
    """
    if not _has_current_injection(collection, ref):
        _, v, _ = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)
        valid = v[~np.isnan(v)]
        return float(np.mean(valid)) if len(valid) > 0 else float("nan")

    # Injection present — use epoch 0 as the pre-injection baseline
    try:
        _, v, _ = collection.get_epoch_arrays(ref, 0, lowpass_hz=lowpass_hz)
        valid = v[~np.isnan(v)]
        if len(valid) > 0:
            return float(np.mean(valid))
    except Exception:
        pass

    # Fallback: full sweep
    _, v, _ = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)
    valid = v[~np.isnan(v)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")
