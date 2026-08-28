"""
ahp.py
------
Afterhyperpolarization (AHP) following a depolarizing current step.

Runs on the same current-step protocols used to build the F-I curve. For each
depolarizing sweep the pre-step epoch supplies a baseline voltage, and the
minimum voltage is searched in two windows that both start at current offset:

- **mAHP** (medium) -- minimum within 100 ms of current off
- **sAHP** (slow)   -- minimum within 1 s of current off

The windows overlap by design, so ``sahp_voltage_mV <= mahp_voltage_mV`` always
holds. Both minima are searched on a low-pass filtered trace regardless of any
display filter setting, so a single noise sample cannot define the AHP.

Sweeps whose step current is not depolarizing (delta from the holding level
<= ``min_step_current_pA``) are skipped -- an AHP is only meaningful after a
depolarization.

Outputs
~~~~~~~
- Per-sweep: baseline_voltage_mV, current_injection_pA, step_current_delta_pA,
  current_off_time_s, and for each of mAHP/sAHP the raw voltage, absolute time,
  time since current off, delta from baseline, and the window length actually
  searched (shorter than requested when the sweep ends early).
- Cell-level: mean/SD and maximum delta for each measure, the current at which
  the maximum occurred, sweep counts, and the parameters used.
- ahp_curve: parallel lists for plotting delta vs. step current.
"""

from __future__ import annotations

import numpy as np

from wholecell.analysis.spikes.finder import _epoch_mean_current
from wholecell.core.sweep_collection import SweepCollection, SweepRef


# Windows both start exactly at current offset (no blanking delay), so any
# offset transient is part of the measurement by design.
_DEFAULT_LOWPASS_HZ = 2000.0


def run_ahp_analysis(
    collection: SweepCollection,
    epoch_index: int,
    lowpass_hz: float | None = _DEFAULT_LOWPASS_HZ,
    mahp_window_ms: float = 100.0,
    sahp_window_ms: float = 1000.0,
    min_step_current_pA: float = 0.0,
) -> dict:
    """Measure medium and slow AHP after a depolarizing current step.

    Parameters
    ----------
    collection : SweepCollection
        Sweeps to analyze (typically the checked current-step sweeps).
    epoch_index : int
        Zero-based index of the current step epoch. Its end is the current-off
        boundary; the epoch immediately before it supplies the baseline.
    lowpass_hz : float or None
        Cutoff applied to voltage before searching for the minima. Filtering is
        the default because a single noise sample would otherwise define the
        AHP; pass None only to inspect the unfiltered behaviour.
    mahp_window_ms, sahp_window_ms : float
        Search window lengths (ms) measured from current off. Both windows
        start at current off, so the sAHP window contains the mAHP window.
    min_step_current_pA : float
        A sweep is analyzed only when ``step_current_delta_pA`` exceeds this.
        The default of 0.0 keeps every depolarizing step.

    Returns
    -------
    dict with keys:
        - ``"per_sweep"`` (list of dict): one entry per analyzed sweep, sorted
          ascending by ``step_current_delta_pA``.
        - ``"ahp_curve"`` (dict): parallel lists for plotting.
        - ``"cell_level"`` (dict): scalar summaries and the parameters used.

    Raises
    ------
    RuntimeError
        If ``epoch_index`` does not exist on any sweep in the collection.
    """
    per_sweep: list[dict] = []
    n_skipped = 0
    n_missing_epoch = 0

    for ref in collection.sweeps:
        row = _build_sweep_row(
            collection, ref, epoch_index, lowpass_hz,
            mahp_window_ms, sahp_window_ms,
        )
        if row is None:
            n_missing_epoch += 1
            continue
        if not _is_depolarizing(row["step_current_delta_pA"], min_step_current_pA):
            n_skipped += 1
            continue
        per_sweep.append(row)

    if n_missing_epoch and n_missing_epoch == len(collection.sweeps):
        raise RuntimeError(
            f"Epoch {epoch_index} does not exist on any of the "
            f"{len(collection.sweeps)} sweeps in collection "
            f"'{collection.name}'."
        )

    per_sweep.sort(key=lambda r: r["step_current_delta_pA"])

    ahp_curve = {
        "step_current_pA": [r["step_current_delta_pA"] for r in per_sweep],
        "mahp_delta_mV": [r["mahp_delta_mV"] for r in per_sweep],
        "sahp_delta_mV": [r["sahp_delta_mV"] for r in per_sweep],
        "mahp_voltage_mV": [r["mahp_voltage_mV"] for r in per_sweep],
        "sahp_voltage_mV": [r["sahp_voltage_mV"] for r in per_sweep],
        "mahp_time_from_off_ms": [r["mahp_time_from_off_ms"] for r in per_sweep],
        "sahp_time_from_off_ms": [r["sahp_time_from_off_ms"] for r in per_sweep],
    }

    cell_level = _compute_cell_level_ahp(per_sweep)
    cell_level["n_sweeps_skipped_nondepolarizing"] = n_skipped
    cell_level["n_sweeps_missing_epoch"] = n_missing_epoch
    cell_level["epoch_index"] = epoch_index
    cell_level["lowpass_hz"] = lowpass_hz
    cell_level["mahp_window_ms"] = mahp_window_ms
    cell_level["sahp_window_ms"] = sahp_window_ms

    return {
        "per_sweep": per_sweep,
        "ahp_curve": ahp_curve,
        "cell_level": cell_level,
    }


# ---------------------------------------------------------------------------
# Per-sweep row builder
# ---------------------------------------------------------------------------

def _build_sweep_row(
    collection: SweepCollection,
    ref: SweepRef,
    epoch_index: int,
    lowpass_hz: float | None,
    mahp_window_ms: float,
    sahp_window_ms: float,
) -> dict | None:
    """Build one per-sweep AHP row, or None if the step epoch is missing."""
    from wholecell.core.cell import _compute_sweep_baseline

    rec = collection._recordings[ref.filename]

    try:
        epoch = rec.get_epoch(ref.sweep_index, epoch_index)
    except (IndexError, RuntimeError):
        return None

    time, voltage, current = collection.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)

    # Baseline reads the unfiltered trace via the shared helper -- averaging a
    # whole epoch already removes the noise a filter would.
    baseline_mV = _compute_sweep_baseline(rec, ref.sweep_index, epoch_index)

    if np.all(np.isnan(current)):
        try:
            current = rec.get_command_waveform(ref.sweep_index)
        except Exception:
            pass

    step_pA = _epoch_mean_current(current, epoch.start_sample, epoch.end_sample)
    hold_pA = _holding_current(rec, current, ref.sweep_index, epoch_index)
    delta_pA = step_pA - hold_pA

    row = {
        "filename": ref.filename,
        "sweep_index": ref.sweep_index,
        "display_label": ref.display_label,
        "baseline_voltage_mV": baseline_mV,
        "current_injection_pA": step_pA,
        "step_current_delta_pA": delta_pA,
        "current_off_time_s": float(epoch.end_time_s),
    }

    truncated = False
    for prefix, window_ms in (("mahp", mahp_window_ms), ("sahp", sahp_window_ms)):
        measured = _window_minimum(
            time, voltage, epoch.end_sample, float(epoch.end_time_s),
            baseline_mV, window_ms, rec.sampling_rate_hz,
        )
        truncated = truncated or measured.pop("_truncated")
        row.update({f"{prefix}_{k}": v for k, v in measured.items()})

    row["window_truncated"] = truncated
    return row


def _window_minimum(
    time: np.ndarray,
    voltage: np.ndarray,
    off_sample: int,
    off_time_s: float,
    baseline_mV: float,
    window_ms: float,
    sampling_rate_hz: float,
) -> dict:
    """Minimum voltage in ``window_ms`` starting at ``off_sample``.

    The window is clipped to the end of the sweep; ``window_ms`` in the result
    is the length actually searched, which flags truncation on short sweeps.
    """
    n_requested = int(round(window_ms / 1000.0 * sampling_rate_hz))
    end = min(off_sample + n_requested, len(voltage))

    if end <= off_sample:
        return {
            "voltage_mV": float("nan"),
            "time_s": float("nan"),
            "time_from_off_ms": float("nan"),
            "delta_mV": float("nan"),
            "window_ms": 0.0,
            "_truncated": True,
        }

    idx = off_sample + int(np.argmin(voltage[off_sample:end]))
    voltage_mV = float(voltage[idx])
    time_s = float(time[idx])
    actual_ms = (end - off_sample) / sampling_rate_hz * 1000.0

    return {
        "voltage_mV": voltage_mV,
        "time_s": time_s,
        "time_from_off_ms": (time_s - off_time_s) * 1000.0,
        "delta_mV": voltage_mV - baseline_mV,
        "window_ms": actual_ms,
        "_truncated": end < off_sample + n_requested,
    }


# ---------------------------------------------------------------------------
# Current helpers
# ---------------------------------------------------------------------------

def _holding_current(
    rec,
    current: np.ndarray,
    sweep_index: int,
    step_epoch_index: int,
) -> float:
    """Mean current over the epoch before the step (the holding level).

    Falls back to 0.0 when there is no preceding epoch, which makes
    ``step_current_delta_pA`` equal the absolute step level.
    """
    if step_epoch_index <= 0:
        return 0.0
    try:
        ep = rec.get_epoch(sweep_index, step_epoch_index - 1)
    except (IndexError, RuntimeError):
        return 0.0
    return _epoch_mean_current(current, ep.start_sample, ep.end_sample)


def _is_depolarizing(delta_pA: float, min_step_current_pA: float) -> bool:
    """True when the step is a depolarization worth measuring an AHP after."""
    return not np.isnan(delta_pA) and delta_pA > min_step_current_pA


# ---------------------------------------------------------------------------
# Cell-level summary
# ---------------------------------------------------------------------------

def _compute_cell_level_ahp(per_sweep: list[dict]) -> dict:
    """Mean/SD and peak AHP across the analyzed sweeps."""
    cell_level: dict = {
        "n_sweeps_analyzed": len(per_sweep),
        "any_window_truncated": any(r["window_truncated"] for r in per_sweep),
    }

    for prefix in ("mahp", "sahp"):
        key = f"{prefix}_delta_mV"
        pairs = [
            (r[key], r["step_current_delta_pA"])
            for r in per_sweep
            if not np.isnan(r[key])
        ]

        if not pairs:
            cell_level[f"mean_{key}"] = float("nan")
            cell_level[f"std_{key}"] = float("nan")
            cell_level[f"max_{key}"] = float("nan")
            cell_level[f"current_at_max_{prefix}_pA"] = float("nan")
            continue

        deltas = [d for d, _ in pairs]
        cell_level[f"mean_{key}"] = float(np.mean(deltas))
        cell_level[f"std_{key}"] = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

        # "Max" AHP is the largest hyperpolarization, i.e. the most negative delta.
        peak_delta, peak_current = min(pairs, key=lambda p: p[0])
        cell_level[f"max_{key}"] = float(peak_delta)
        cell_level[f"current_at_max_{prefix}_pA"] = float(peak_current)

    return cell_level
