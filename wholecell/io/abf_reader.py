"""
abf_reader.py
-------------
Utilities for working with pyabf beyond what Recording exposes directly.

This module contains:
  - Helper functions for inspecting ABF headers (protocol name, channel
    labels, units, clamp mode)
  - Epoch auto-detection heuristics (identifying which epoch is the step,
    which is the baseline)
  - Convenience functions for building SweepRef lists from an ABF file
    (e.g. "give me all sweeps where the step amplitude is negative")

These are used by the GUI sweep selector and by interactive scripts. They
are NOT part of the Recording class because they are higher-level utilities
that interpret protocol intent rather than just exposing raw data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# ABF file inspection
# ---------------------------------------------------------------------------

def get_abf_info(filepath: str | Path) -> dict:
    """Return a summary dict of ABF file metadata.

    Useful for quick inspection before loading a Recording.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    dict with keys:
        - ``filename`` (str)
        - ``n_sweeps`` (int)
        - ``sampling_rate_hz`` (float)
        - ``sweep_duration_s`` (float)
        - ``n_channels`` (int)
        - ``channel_names`` (list of str)
        - ``channel_units`` (list of str)
        - ``clamp_mode`` (str): ``"current_clamp"`` or ``"voltage_clamp"``
        - ``protocol_name`` (str): from ABF header if available
        - ``n_epochs`` (int): number of epochs in the protocol

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If pyabf cannot open the file.
    """
    import pyabf

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ABF file not found: {filepath}")

    try:
        abf = pyabf.ABF(str(filepath), loadData=False)
    except Exception as exc:
        raise RuntimeError(f"pyabf could not open '{filepath}': {exc}") from exc

    clamp_mode = _infer_clamp_mode(abf)

    return {
        "filename": filepath.stem,
        "n_sweeps": abf.sweepCount,
        "sampling_rate_hz": abf.dataRate,
        "sweep_duration_s": abf.sweepLengthSec,
        "n_channels": abf.channelCount,
        "channel_names": list(abf.adcNames),
        "channel_units": list(abf.adcUnits),
        "clamp_mode": clamp_mode,
        "protocol_name": getattr(abf, "protocol", "unknown"),
        "n_epochs": _count_epochs(abf),
    }


def get_step_amplitudes(
    filepath: str | Path,
    epoch_index: int,
) -> list[float]:
    """Return the current injection amplitude for each sweep at a given epoch.

    Works whether the protocol uses a per-sweep delta or a fixed amplitude.

    Parameters
    ----------
    filepath : str or Path
    epoch_index : int
        Zero-based epoch index to inspect.

    Returns
    -------
    list of float
        Step amplitude (pA or mV) for each sweep. Length equals n_sweeps.

    Raises
    ------
    RuntimeError
        If epochs cannot be parsed.
    """
    import pyabf

    filepath = Path(filepath)
    abf = pyabf.ABF(str(filepath), loadData=False)

    amplitudes: list[float] = []
    for sweep_idx in range(abf.sweepCount):
        abf.setSweep(sweep_idx)
        amp = _get_epoch_level(abf, sweep_idx, epoch_index, filepath.stem)
        amplitudes.append(amp)

    return amplitudes


# ---------------------------------------------------------------------------
# Epoch heuristics
# ---------------------------------------------------------------------------

def find_baseline_epoch(filepath: str | Path, sweep_index: int = 0) -> int:
    """Heuristically identify the baseline (pre-stimulus) epoch index.

    Returns the first epoch whose absolute level is smaller than all other
    epochs (typically the pre-step holding period).

    Parameters
    ----------
    filepath : str or Path
    sweep_index : int
        Sweep to inspect (default: first sweep).

    Returns
    -------
    int
        Zero-based index of the likely baseline epoch.

    Notes
    -----
    Heuristic. GUI should allow override.
    """
    import pyabf

    abf = pyabf.ABF(str(filepath), loadData=False)
    levels = _sweep_epoch_levels(abf, sweep_index, Path(filepath).stem)
    if not levels:
        return 0

    # Prefer an epoch with |level| < 1 (true zero-current baseline)
    for i, lv in enumerate(levels):
        if abs(lv) < 1.0:
            return i

    # Otherwise return the epoch with the smallest absolute level
    return int(min(range(len(levels)), key=lambda i: abs(levels[i])))


def find_step_epoch(filepath: str | Path, sweep_index: int = 0) -> int:
    """Heuristically identify the current step epoch index.

    Priority order:
    1. Epoch whose level varies across sweeps (delta != 0 in epochTable).
    2. Epoch with maximum absolute level (fixed-amplitude protocols).
    3. Index 1 as a last resort.

    Parameters
    ----------
    filepath : str or Path
    sweep_index : int

    Returns
    -------
    int
        Zero-based index of the likely current step epoch.

    Notes
    -----
    Heuristic. GUI should allow override.
    """
    import pyabf

    filepath = Path(filepath)
    abf = pyabf.ABF(str(filepath), loadData=False)

    # --- Try epochTable for deltaLevel (stepping protocols) ---
    try:
        abf.setSweep(sweep_index)
        epochs = list(abf.epochTable.epochs)
        for i, ep in enumerate(epochs):
            if ep.deltaLevel != 0.0:
                return i
        # epochTable present but no delta — fall through to level-based check
        levels_et = [ep.level for ep in epochs]
        best = int(max(range(len(levels_et)), key=lambda i: abs(levels_et[i])))
        if abs(levels_et[best]) > 0:
            return best
    except AttributeError:
        pass

    # --- Fallback: compare first vs last sweep via sweepEpochs ---
    levels_first = _sweep_epoch_levels(abf, 0, filepath.stem)
    if not levels_first:
        return 1

    if abf.sweepCount > 1:
        levels_last = _sweep_epoch_levels(abf, abf.sweepCount - 1, filepath.stem)
        deltas = [abs(levels_last[i] - levels_first[i]) for i in range(len(levels_first))]
        if max(deltas) > 0:
            return int(max(range(len(deltas)), key=lambda i: deltas[i]))

    # Fixed amplitude: pick epoch with largest absolute level
    best = int(max(range(len(levels_first)), key=lambda i: abs(levels_first[i])))
    return best if abs(levels_first[best]) > 0 else min(1, len(levels_first) - 1)


# ---------------------------------------------------------------------------
# SweepRef construction helpers
# ---------------------------------------------------------------------------

def all_sweep_refs(
    filepath: str | Path,
) -> list[dict]:
    """Return a list of sweep ref dicts for every sweep in an ABF file.

    Intended for populating the GUI sweep selector.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    list of dict, each with keys: ``filename``, ``sweep_index``.
    """
    import pyabf

    filepath = Path(filepath)
    abf = pyabf.ABF(str(filepath), loadData=False)
    stem = filepath.stem
    return [
        {"filename": stem, "sweep_index": i}
        for i in range(abf.sweepCount)
    ]


def sweeps_by_polarity(
    filepath: str | Path,
    epoch_index: int,
    polarity: str = "negative",
) -> list[dict]:
    """Return sweep ref dicts filtered by step polarity.

    Useful for quickly selecting only hyperpolarizing or depolarizing sweeps.

    Parameters
    ----------
    filepath : str or Path
    epoch_index : int
        Epoch to check for amplitude polarity.
    polarity : str
        ``"negative"`` for hyperpolarizing (current < 0),
        ``"positive"`` for depolarizing (current > 0),
        ``"zero"`` for no-step sweeps.

    Returns
    -------
    list of dict with keys: ``filename``, ``sweep_index``.
    """
    amplitudes = get_step_amplitudes(filepath, epoch_index)
    stem = Path(filepath).stem

    refs = []
    for i, amp in enumerate(amplitudes):
        if polarity == "negative" and amp < 0:
            refs.append({"filename": stem, "sweep_index": i})
        elif polarity == "positive" and amp > 0:
            refs.append({"filename": stem, "sweep_index": i})
        elif polarity == "zero" and amp == 0:
            refs.append({"filename": stem, "sweep_index": i})

    return refs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sweep_epoch_levels(abf, sweep_index: int, filename_stem: str) -> list[float]:
    """Return command levels for all epochs in one sweep via sweepEpochs."""
    try:
        abf.setSweep(sweep_index)
        return [float(lv) for lv in abf.sweepEpochs.levels]
    except AttributeError:
        return []


def _get_epoch_level(abf, sweep_index: int, epoch_index: int, filename_stem: str) -> float:
    """Return the command level for one epoch of one sweep.

    Tries epochTable (with delta) first, then sweepEpochs.
    """
    try:
        abf.setSweep(sweep_index)
        epochs = list(abf.epochTable.epochs)
        ep = epochs[epoch_index]
        return float(ep.level + ep.deltaLevel * sweep_index)
    except (AttributeError, IndexError):
        pass
    try:
        abf.setSweep(sweep_index)
        return float(abf.sweepEpochs.levels[epoch_index])
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(
            f"Cannot read level for sweep {sweep_index}, epoch {epoch_index} "
            f"of '{filename_stem}': {exc}"
        ) from exc


def _infer_clamp_mode(abf) -> str:
    """Return 'current_clamp' or 'voltage_clamp' from ABF units."""
    try:
        units = abf.adcUnits[0].lower()
        if "mv" in units or "v" in units:
            return "current_clamp"
        elif "pa" in units or "na" in units or "a" in units:
            return "voltage_clamp"
    except (AttributeError, IndexError):
        pass
    return "unknown"


def _count_epochs(abf) -> int:
    """Return the number of epochs in the protocol, or 0 if not available."""
    try:
        abf.setSweep(0)
        return len(list(abf.epochTable.epochs))
    except Exception:
        return 0
