"""
passive.py
----------
Passive membrane property analysis: input resistance, membrane time constant,
and Ih sag characterisation.

All functions operate on a SweepCollection and return plain dicts. The Cell
object wraps these in timestamped result entries.

Filtering is per-analysis: callers pass lowpass_hz and it is applied only
within this module via SweepCollection.get_sweep_arrays.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wholecell.core.sweep_collection import SweepCollection, SweepRef


# ---------------------------------------------------------------------------
# Public entry point (called by Cell.analyze_passive)
# ---------------------------------------------------------------------------

def run_passive_analysis(
    collection: SweepCollection,
    epoch_index: int,
    measures: list[str] | None = None,
    lowpass_hz: float | None = None,
) -> dict:
    """Run passive membrane property analysis on a SweepCollection.

    Parameters
    ----------
    collection : SweepCollection
    epoch_index : int
        Zero-based epoch index identifying the current step window.
        Fails loudly if epochs cannot be parsed.
    measures : list of str or None
        Subset of: ``"input_resistance"``, ``"time_constant"``,
        ``"sag_ratio"``, ``"sag_amplitude"``, ``"sag_kinetics"``.
        Defaults to all.
    lowpass_hz : float or None
        Lowpass filter applied to voltage for this analysis only.

    Returns
    -------
    dict with keys:
        - ``"per_sweep"`` : list of per-sweep result dicts, each containing
          ``filename``, ``sweep_index``, and whichever measures were computed.
        - ``"cell_level"`` : dict of across-sweep summary scalars, e.g.
          mean Rin, tau from pooled fit, etc.

    Raises
    ------
    RuntimeError
        If epoch parsing fails for any sweep. Fails loudly.
    """
    all_measures = {
        "input_resistance",
        "time_constant",
        "sag_ratio",
        "sag_amplitude",
        "sag_kinetics",
    }
    if measures is None:
        measures = list(all_measures)
    else:
        unknown = set(measures) - all_measures
        if unknown:
            raise ValueError(f"Unknown measures: {unknown}. Valid: {all_measures}")

    per_sweep = []
    for ref in collection.sweeps:
        row = _analyze_one_sweep(
            collection, ref, epoch_index, measures, lowpass_hz
        )
        per_sweep.append(row)

    cell_level = _compute_cell_level(per_sweep, measures)

    return {
        "per_sweep": per_sweep,
        "cell_level": cell_level,
    }


# ---------------------------------------------------------------------------
# Per-sweep analysis
# ---------------------------------------------------------------------------

def _analyze_one_sweep(
    collection: SweepCollection,
    ref: SweepRef,
    epoch_index: int,
    measures: list[str],
    lowpass_hz: float | None,
) -> dict:
    """Compute requested passive measures for one sweep.

    Parameters
    ----------
    collection : SweepCollection
    ref : SweepRef
    epoch_index : int
    measures : list of str
    lowpass_hz : float or None

    Returns
    -------
    dict with keys: filename, sweep_index, display_label, and all measures.
    """
    # Retrieve full sweep for baseline and epoch window
    time, voltage, current = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)
    ep_time, ep_voltage, ep_current = collection.get_epoch_arrays(
        ref, epoch_index, lowpass_hz=lowpass_hz
    )

    # Baseline: voltage in the epoch immediately before the step
    # (epoch_index - 1, or first 50 ms of step epoch if no prior epoch)
    baseline_voltage = _estimate_baseline(
        collection, ref, epoch_index, lowpass_hz
    )

    # Use recorded current channel; fall back to epoch command level if absent
    step_current_pA = _mean_current(ep_current)
    if np.isnan(step_current_pA):
        rec = collection._recordings[ref.filename]
        epoch = rec.get_epoch(ref.sweep_index, epoch_index)
        step_current_pA = epoch.level  # pA command from protocol

    row: dict = {
        "filename": ref.filename,
        "sweep_index": ref.sweep_index,
        "display_label": ref.display_label,
        "epoch_index": epoch_index,
        "current_injection_pA": step_current_pA,
        "baseline_voltage_mV": baseline_voltage,
    }

    if "input_resistance" in measures:
        row["input_resistance_MOhm"] = estimate_input_resistance(
            baseline_voltage, ep_voltage, step_current_pA
        )

    if "time_constant" in measures:
        tau, fit_time, fit_voltage, fit_predicted = fit_time_constant(
            ep_time, ep_voltage, baseline_voltage
        )
        row["time_constant_ms"] = tau
        # Store fit arrays for GUI overlay — kept in result dict
        row["_tau_fit"] = {
            "time": fit_time.tolist() if fit_time is not None else None,
            "voltage": fit_voltage.tolist() if fit_voltage is not None else None,
            "predicted": fit_predicted.tolist() if fit_predicted is not None else None,
        }

    if any(m in measures for m in ("sag_ratio", "sag_amplitude", "sag_kinetics")):
        sag = estimate_sag(ep_time, ep_voltage, baseline_voltage)
        if "sag_ratio" in measures:
            row["sag_ratio"] = sag["sag_ratio"]
        if "sag_amplitude" in measures:
            row["sag_amplitude_mV"] = sag["sag_amplitude_mV"]
        if "sag_kinetics" in measures:
            row["sag_tau_ms"] = sag.get("sag_tau_ms", float("nan"))

    return row


# ---------------------------------------------------------------------------
# Core estimators (to be fully implemented)
# ---------------------------------------------------------------------------

def estimate_input_resistance(
    baseline_voltage_mV: float,
    epoch_voltage: np.ndarray,
    step_current_pA: float,
    steady_state_window_fraction: float = 0.2,
) -> float:
    """Estimate input resistance (MΩ) from a single current step sweep.

    Uses the steady-state voltage deflection at the end of the step.

    Parameters
    ----------
    baseline_voltage_mV : float
        Mean voltage immediately before the step onset (mV).
    epoch_voltage : np.ndarray
        Voltage trace during the current step epoch (mV).
    step_current_pA : float
        Injected current amplitude (pA). Sign is preserved.
    steady_state_window_fraction : float
        Fraction of the epoch duration used to estimate the steady-state
        voltage (taken from the end of the epoch). Default: last 20%.

    Returns
    -------
    float
        Input resistance in MΩ. Returns NaN if current is zero or if
        estimation fails.

    Notes
    -----
    R_in = ΔV / ΔI  (converted to MΩ: mV / pA = GΩ → × 1000 = MΩ)

    TODO: implement steady-state estimation with spike exclusion for
    depolarizing steps.
    """
    if step_current_pA == 0 or np.isnan(step_current_pA):
        return float("nan")

    n = len(epoch_voltage)
    ss_start = int(n * (1.0 - steady_state_window_fraction))
    steady_state_mV = float(np.mean(epoch_voltage[ss_start:]))

    delta_v_mV = steady_state_mV - baseline_voltage_mV
    delta_i_pA = step_current_pA

    # ΔV(mV) / ΔI(pA) = GΩ; × 1000 = MΩ
    rin_MOhm = (delta_v_mV / delta_i_pA) * 1000.0
    return float(rin_MOhm)


def fit_time_constant(
    epoch_time: np.ndarray,
    epoch_voltage: np.ndarray,
    baseline_voltage_mV: float,
    fit_window_fraction: float = 0.4,
) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Fit a single-exponential decay to the onset of a voltage step.

    The fit is performed on the rising/falling edge immediately after step
    onset (first ``fit_window_fraction`` of the epoch).

    Parameters
    ----------
    epoch_time : np.ndarray
        Time array for the epoch, starting at 0 (s).
    epoch_voltage : np.ndarray
        Voltage trace for the epoch (mV).
    baseline_voltage_mV : float
        Pre-step voltage used as the asymptote (mV).
    fit_window_fraction : float
        Fraction of epoch used for the fit. Default: first 40%.

    Returns
    -------
    tau_ms : float
        Membrane time constant in milliseconds. NaN if fit fails.
    fit_time : np.ndarray or None
        Time array for the fit window (for GUI overlay).
    fit_voltage : np.ndarray or None
        Observed voltage in the fit window.
    fit_predicted : np.ndarray or None
        Model-predicted voltage in the fit window.

    Notes
    -----
    Model: V(t) = V_ss + (V_0 - V_ss) * exp(-t / tau)
    Time is re-zeroed to fit-window onset before fitting.
    """
    from scipy.optimize import curve_fit

    n = len(epoch_time)
    if n < 10:
        return float("nan"), None, None, None

    n_fit = max(10, int(n * fit_window_fraction))
    t_win = epoch_time[:n_fit].copy()
    v_win = epoch_voltage[:n_fit].copy()

    # Re-zero time so t=0 at fit onset
    t0 = t_win[0]
    t_win = t_win - t0

    v0_guess = float(v_win[0])
    v_ss_guess = float(np.mean(v_win[-max(1, n_fit // 5):]))
    # Estimate tau from 63% of the total deflection
    target = v0_guess + 0.632 * (v_ss_guess - v0_guess)
    crossings = np.where(
        (v_win[:-1] - target) * (v_win[1:] - target) <= 0
    )[0]
    tau_guess = float(t_win[crossings[0]]) if len(crossings) else 0.01

    def _exp_model(t, v_ss, v0, tau):
        return v_ss + (v0 - v_ss) * np.exp(-t / tau)

    try:
        p0 = [v_ss_guess, v0_guess, max(tau_guess, 1e-4)]
        bounds = (
            [-np.inf, -np.inf, 1e-5],
            [np.inf,  np.inf,  10.0],
        )
        popt, _ = curve_fit(
            _exp_model, t_win, v_win,
            p0=p0, bounds=bounds, maxfev=5000,
        )
        tau_s = float(popt[2])
        tau_ms = tau_s * 1000.0
        fit_predicted = _exp_model(t_win, *popt)
        # Return time in original (non-re-zeroed) coordinates for GUI
        return tau_ms, t_win + t0, v_win, fit_predicted
    except Exception:
        return float("nan"), None, None, None


def estimate_sag(
    epoch_time: np.ndarray,
    epoch_voltage: np.ndarray,
    baseline_voltage_mV: float,
    peak_window_ms: float = 50.0,
) -> dict:
    """Estimate Ih sag properties from a hyperpolarizing step.

    Parameters
    ----------
    epoch_time : np.ndarray
        Time array for the epoch (s).
    epoch_voltage : np.ndarray
        Voltage trace for the epoch (mV).
    baseline_voltage_mV : float
        Pre-step baseline voltage (mV).
    peak_window_ms : float
        Time window after step onset within which to search for the
        hyperpolarisation peak (ms). Default: 50 ms.

    Returns
    -------
    dict with keys:
        - ``"sag_ratio"`` (float): (V_peak - V_ss) / (V_peak - V_baseline).
          0 = no sag, 1 = full recovery. NaN if not a hyperpolarizing step.
        - ``"sag_amplitude_mV"`` (float): V_ss - V_peak (positive = sag).
        - ``"sag_tau_ms"`` (float): time constant of sag relaxation (NaN if
          not computed).

    Notes
    -----
    Sag is only meaningful for hyperpolarizing steps. Returns NaN values
    for depolarizing sweeps.

    Sag ratio = (V_ss - V_peak) / (V_baseline - V_peak), bounded [0, 1].
    A ratio of 0 means no sag (peak equals steady state); 1 means full
    return to baseline.
    """
    nan_result = {
        "sag_ratio": float("nan"),
        "sag_amplitude_mV": float("nan"),
        "sag_tau_ms": float("nan"),
    }

    if len(epoch_voltage) < 10:
        return nan_result

    # Determine step direction from mean voltage relative to baseline
    mean_step_v = float(np.mean(epoch_voltage))
    if mean_step_v >= baseline_voltage_mV:
        # Depolarizing step — sag not meaningful
        return nan_result

    n = len(epoch_voltage)
    # Steady-state window: last 20%
    ss_start = int(n * 0.80)

    # Peak hyperpolarisation: global minimum over the first 80% of the epoch.
    # A fixed short window (e.g. 50 ms) misses the trough for slow cells.
    # Excluding the last 20% keeps the sag peak distinct from steady state.
    search_end = max(5, ss_start)
    peak_idx = int(np.argmin(epoch_voltage[:search_end]))
    v_peak = float(epoch_voltage[peak_idx])

    # Steady-state = mean of last 20% of epoch
    v_ss = float(np.mean(epoch_voltage[ss_start:]))

    deflection = baseline_voltage_mV - v_peak  # positive for hyperpolarisation
    if deflection <= 0:
        return nan_result

    sag_amplitude_mV = v_ss - v_peak  # positive = voltage recovered toward baseline
    sag_ratio = sag_amplitude_mV / deflection
    # Clamp to [0, 1] — floating-point noise can push slightly outside
    sag_ratio = float(np.clip(sag_ratio, 0.0, 1.0))

    # Optional: fit exponential to the sag relaxation (peak → steady state)
    sag_tau_ms = float("nan")
    try:
        from scipy.optimize import curve_fit

        t_sag = epoch_time[peak_idx:ss_start].copy()
        v_sag = epoch_voltage[peak_idx:ss_start].copy()
        if len(t_sag) > 10:
            t_sag = t_sag - t_sag[0]

            def _exp_model(t, v_ss_fit, amp, tau):
                return v_ss_fit + amp * np.exp(-t / tau)

            p0 = [v_ss, v_peak - v_ss, 0.05]
            popt, _ = curve_fit(
                _exp_model, t_sag, v_sag,
                p0=p0,
                bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, 10.0]),
                maxfev=3000,
            )
            sag_tau_ms = float(popt[2]) * 1000.0
    except Exception:
        pass

    return {
        "sag_ratio": sag_ratio,
        "sag_amplitude_mV": sag_amplitude_mV,
        "sag_tau_ms": sag_tau_ms,
    }


# ---------------------------------------------------------------------------
# Cell-level summary
# ---------------------------------------------------------------------------

def _compute_cell_level(per_sweep: list[dict], measures: list[str]) -> dict:
    """Aggregate per-sweep results into cell-level scalars.

    Parameters
    ----------
    per_sweep : list of dict
    measures : list of str

    Returns
    -------
    dict
        Cell-level summary (mean ± SD across sweeps for each measure,
        plus the linear fit slope for Rin across step amplitudes).

    TODO: implement linear regression for Rin (ΔV vs ΔI across sweeps).
    """
    cell = {}

    if "input_resistance" in measures:
        rins = [r.get("input_resistance_MOhm") for r in per_sweep
                if r.get("input_resistance_MOhm") is not None and
                not np.isnan(r["input_resistance_MOhm"])]
        if rins:
            cell["mean_input_resistance_MOhm"] = float(np.mean(rins))
            cell["std_input_resistance_MOhm"] = float(np.std(rins))
            cell["n_sweeps_rin"] = len(rins)

    if "time_constant" in measures:
        taus = [r.get("time_constant_ms") for r in per_sweep
                if r.get("time_constant_ms") is not None and
                not np.isnan(r["time_constant_ms"])]
        if taus:
            cell["mean_time_constant_ms"] = float(np.mean(taus))
            cell["std_time_constant_ms"] = float(np.std(taus))
            cell["n_sweeps_tau"] = len(taus)

    if "sag_ratio" in measures:
        sags = [r.get("sag_ratio") for r in per_sweep
                if r.get("sag_ratio") is not None and
                not np.isnan(r["sag_ratio"])]
        if sags:
            cell["mean_sag_ratio"] = float(np.mean(sags))
            cell["std_sag_ratio"] = float(np.std(sags))

    if "sag_amplitude" in measures:
        amps = [r.get("sag_amplitude_mV") for r in per_sweep
                if r.get("sag_amplitude_mV") is not None and
                not np.isnan(r["sag_amplitude_mV"])]
        if amps:
            cell["mean_sag_amplitude_mV"] = float(np.mean(amps))

    return cell


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_baseline(
    collection: SweepCollection,
    ref: SweepRef,
    step_epoch_index: int,
    lowpass_hz: float | None,
) -> float:
    """Return mean baseline voltage from the epoch before the step.

    Falls back to the first 50 ms of the step epoch if there is no prior
    epoch.
    """
    if step_epoch_index > 0:
        try:
            _, v, _ = collection.get_epoch_arrays(
                ref, step_epoch_index - 1, lowpass_hz=lowpass_hz
            )
            return float(np.mean(v))
        except (IndexError, RuntimeError):
            pass

    # Fallback: first 50 ms of the step epoch itself
    _, v, _ = collection.get_epoch_arrays(ref, step_epoch_index, lowpass_hz=lowpass_hz)
    rec = collection._recordings[ref.filename]
    n_50ms = int(0.05 * rec.sampling_rate_hz)
    return float(np.mean(v[:n_50ms])) if len(v) >= n_50ms else float(np.mean(v))


def _mean_current(current_array: np.ndarray) -> float:
    """Return mean of a current array, ignoring NaN."""
    valid = current_array[~np.isnan(current_array)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")
