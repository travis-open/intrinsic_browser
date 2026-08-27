"""
vrest.py
--------
Resting membrane potential estimation from one or more sweeps.

For sweeps with current injection (step or ramp protocols), only the
pre-injection baseline period (epoch 0) is used. For sweeps with no
meaningful current injection (gap-free recordings, free-run protocols),
the full sweep is averaged.

Optionally (``detect_aps=True``, the default) action potentials are also
detected across the full length of every sweep, yielding per-sweep AP counts
and interspike-interval statistics (mean ISI and its coefficient of
variation) computed from spike peak times.
"""

from __future__ import annotations

import numpy as np

from wholecell.core.sweep_collection import SweepCollection, SweepRef

# Current range (max - min) must exceed this to be considered an injection.
_INJECTION_THRESHOLD_PA = 5.0


def run_vrest_analysis(
    collection: SweepCollection,
    lowpass_hz: float | None = None,
    detect_aps: bool = True,
    dvdt_detection_mVms: float = 20.0,
    peak_search_window_ms: float = 20.0,
) -> dict:
    """Estimate resting membrane potential from a set of sweeps.

    Parameters
    ----------
    collection : SweepCollection
        Sweeps to analyze (typically the checked resting-potential sweeps).
    lowpass_hz : float or None
        Optional lowpass filter applied to voltage before averaging.
    detect_aps : bool
        If True (default), also detect action potentials across the full
        length of every sweep and report per-sweep AP / interspike-interval
        metrics. Detection reuses ``run_spike_detection`` (the same path as
        the "Find Spikes" button), so it applies the standard internal
        2 kHz detection lowpass and reads peak voltages from the raw trace.
    dvdt_detection_mVms : float
        dV/dt threshold (mV/ms) for AP detection. Pass the value currently
        in use for the session. Ignored when ``detect_aps`` is False.
    peak_search_window_ms : float
        Peak search window (ms) for AP detection. Ignored when
        ``detect_aps`` is False.

    Returns
    -------
    dict with keys:
        - ``"per_sweep"`` (list of dict): one entry per sweep with
          ``filename``, ``sweep_index``, ``display_label``,
          ``v_rest_mV`` (the mean voltage for that sweep's baseline window),
          ``sweep_length_s``, and — when ``detect_aps`` — ``ap_detected``
          (bool), ``n_aps`` (int), ``mean_isi_s`` (mean interspike interval
          from spike peak times, NaN if < 2 APs), and ``isi_cv`` (sample-SD
          coefficient of variation of the ISIs, NaN if < 2 ISIs).
        - ``"cell_level"`` (dict): ``v_rest_mV`` (mean across sweeps),
          ``v_rest_std_mV``, ``n_sweeps_analyzed``, ``initial_voltage_mV``
          (mean voltage during the first epoch of the first sweep; falls back
          to the full first sweep when it has no epoch table), and — when
          ``detect_aps`` — ``ap_detected`` (True if any sweep had >= 1 AP),
          ``n_aps_total`` (summed over sweeps), ``initial_mean_isi_s`` and
          ``initial_isi_cv`` (the first sweep's ``mean_isi_s`` / ``isi_cv``).
    """
    ap_map: dict | None = None
    ap_error: str | None = None
    if detect_aps and collection.sweeps:
        ap_map, ap_error = _detect_sweep_aps(
            collection, dvdt_detection_mVms, peak_search_window_ms
        )

    per_sweep: list[dict] = []

    for ref in collection.sweeps:
        entry = {
            "filename": ref.filename,
            "sweep_index": ref.sweep_index,
            "display_label": ref.display_label,
            "v_rest_mV": _baseline_voltage(collection, ref, lowpass_hz),
            "sweep_length_s": _sweep_length_s(collection, ref),
        }
        if detect_aps:
            peaks = None if ap_map is None else ap_map.get(
                (ref.filename, ref.sweep_index), []
            )
            entry.update(_ap_metrics(peaks))
        per_sweep.append(entry)

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

    if detect_aps:
        _add_cell_level_ap_metrics(cell_level, per_sweep, ap_map, ap_error)

    return {
        "per_sweep": per_sweep,
        "cell_level": cell_level,
    }


def _sweep_length_s(collection: SweepCollection, ref: SweepRef) -> float:
    """Return the sweep duration in seconds from its time axis."""
    try:
        t, _, _ = collection.get_sweep_arrays(ref)
    except Exception:
        return float("nan")
    if len(t) < 2:
        return float("nan")
    return float(len(t) * (t[1] - t[0]))


def _detect_sweep_aps(
    collection: SweepCollection,
    dvdt_detection_mVms: float,
    peak_search_window_ms: float,
) -> tuple[dict | None, str | None]:
    """Detect APs across the full length of every sweep.

    Reuses ``run_spike_detection`` (the "Find Spikes" path). ``epoch_index=0``
    is passed only because that runner requires a reference epoch for its
    current-amplitude annotation; it does not restrict the spike search, which
    always spans the whole sweep.

    Returns
    -------
    (ap_map, error)
        ``ap_map`` maps ``(filename, sweep_index)`` to a list of spike peak
        times (s), or ``None`` if detection failed. ``error`` is the failure
        message (or ``None``).
    """
    from wholecell.analysis.spikes.finder import run_spike_detection

    try:
        result = run_spike_detection(
            collection,
            epoch_index=0,
            backend="derivative",
            dvdt_detection_mVms=dvdt_detection_mVms,
            peak_search_window_ms=peak_search_window_ms,
        )
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, report reason
        return None, f"{type(exc).__name__}: {exc}"

    ap_map = {
        (sw["filename"], sw["sweep_index"]):
            [s["peak_time_s"] for s in sw.get("spikes", [])]
        for sw in result.get("per_sweep", [])
    }
    return ap_map, None


def _ap_metrics(peak_times: list[float] | None) -> dict:
    """Per-sweep AP / interspike-interval metrics from spike peak times."""
    if peak_times is None:
        return {
            "ap_detected": None,
            "n_aps": None,
            "mean_isi_s": float("nan"),
            "isi_cv": float("nan"),
        }

    n = len(peak_times)
    out = {
        "ap_detected": n >= 1,
        "n_aps": n,
        "mean_isi_s": float("nan"),
        "isi_cv": float("nan"),
    }
    if n >= 2:
        isi = np.diff(np.sort(np.asarray(peak_times, dtype=float)))
        mean_isi = float(np.mean(isi))
        out["mean_isi_s"] = mean_isi
        if len(isi) >= 2 and mean_isi != 0.0:
            out["isi_cv"] = float(np.std(isi, ddof=1) / mean_isi)
    return out


def _add_cell_level_ap_metrics(
    cell_level: dict,
    per_sweep: list[dict],
    ap_map: dict | None,
    ap_error: str | None,
) -> None:
    """Roll per-sweep AP metrics up to the cell level (in place)."""
    if ap_map is None:
        cell_level["ap_detected"] = None
        cell_level["n_aps_total"] = None
        cell_level["initial_mean_isi_s"] = float("nan")
        cell_level["initial_isi_cv"] = float("nan")
        if ap_error:
            cell_level["ap_detection_error"] = ap_error
        return

    cell_level["ap_detected"] = bool(any(e.get("ap_detected") for e in per_sweep))
    cell_level["n_aps_total"] = int(sum(e.get("n_aps") or 0 for e in per_sweep))
    first = per_sweep[0] if per_sweep else {}
    cell_level["initial_mean_isi_s"] = first.get("mean_isi_s", float("nan"))
    cell_level["initial_isi_cv"] = first.get("isi_cv", float("nan"))


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
