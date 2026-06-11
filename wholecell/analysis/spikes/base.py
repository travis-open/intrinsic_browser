"""
base.py
-------
Abstract base class for spike finders, plus the strategy dispatch function
used by Cell.find_spikes.

Adding a new backend
~~~~~~~~~~~~~~~~~~~~
1. Subclass SpikeFinder and implement ``detect()``.
2. Register it in the ``BACKENDS`` dict at the bottom of this file.

The interface contract
~~~~~~~~~~~~~~~~~~~~~~
All spike finders accept (time, voltage, current) as numpy arrays and return
a list of SpikeDetection objects. They are unaware of files, sweeps, or
collections — that context is added by the caller (finder.py).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Spike detection result — per-spike container
# ---------------------------------------------------------------------------

@dataclass
class SpikeDetection:
    """Minimal spike detection result produced by all backends.

    Feature extraction (threshold voltage, half-width, AHP, etc.) is
    done downstream in features.py. This object captures only what is
    needed for detection and epoch filtering.

    Attributes
    ----------
    peak_index : int
        Sample index of the action potential peak (max voltage) within
        the full sweep array.
    peak_time_s : float
        Time of the peak in seconds from sweep onset.
    peak_voltage_mV : float
        Voltage at the peak (mV).
    threshold_index : int
        Sample index of the action potential threshold (dV/dt crossing).
    threshold_time_s : float
        Time of threshold crossing (s).
    threshold_voltage_mV : float
        Voltage at threshold (mV).
    trough_index : int
        Sample index of the fast trough (first minimum after peak).
    trough_time_s : float
        Time of trough (s).
    trough_voltage_mV : float
        Voltage at trough (mV).
    backend : str
        Name of the spike finder backend that produced this detection.
    """

    peak_index: int
    peak_time_s: float
    peak_voltage_mV: float
    threshold_index: int
    threshold_time_s: float
    threshold_voltage_mV: float
    trough_index: int
    trough_time_s: float
    trough_voltage_mV: float
    backend: str = "unknown"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SpikeFinder(abc.ABC):
    """Abstract base class for spike detection backends.

    All subclasses must implement ``detect()``. Parameters specific to each
    backend are passed via ``__init__``.

    The strategy pattern allows ``Cell.find_spikes`` to accept a backend
    string and dispatch to the appropriate finder without knowing its
    internals.
    """

    @abc.abstractmethod
    def detect(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
    ) -> list[SpikeDetection]:
        """Detect action potentials in a single sweep.

        Parameters
        ----------
        time : np.ndarray
            Time array (s), shape (n_samples,).
        voltage : np.ndarray
            Voltage trace (mV), shape (n_samples,). Pre-filtered if the
            caller requested lowpass filtering.
        current : np.ndarray
            Injected current (pA), shape (n_samples,). May contain NaN
            if no current channel was recorded.

        Returns
        -------
        list of SpikeDetection
            One entry per detected action potential, ordered by peak_time_s.
            Empty list if no spikes are detected.
        """
        ...

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]:
        """Return a serialisable dict of all detection parameters.

        Used for session audit log and result provenance. Must include
        all parameters that affect detection output.
        """
        ...

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Short string identifier for this backend, e.g. 'derivative'."""
        ...


# ---------------------------------------------------------------------------
# Backend registry and factory
# ---------------------------------------------------------------------------

def build_spike_finder(backend: str, **kwargs: Any) -> SpikeFinder:
    """Instantiate a spike finder by backend name.

    Parameters
    ----------
    backend : str
        One of: ``"derivative"``, ``"ipfx"``.
    **kwargs
        Passed to the backend's ``__init__``.

    Returns
    -------
    SpikeFinder

    Raises
    ------
    ValueError
        If backend is not recognised.
    ImportError
        If backend requires an optional dependency that is not installed.
    """
    backend = backend.lower().strip()

    if backend == "derivative":
        from wholecell.analysis.spikes.derivative import DerivativeSpikeFinder
        return DerivativeSpikeFinder(**kwargs)

    elif backend == "ipfx":
        try:
            from wholecell.analysis.spikes.ipfx_backend import IpfxSpikeFinder
        except ImportError as exc:
            raise ImportError(
                "The 'ipfx' backend requires the ipfx package. "
                "Install it with: pip install ipfx\n"
                "Alternatively use backend='derivative' which has no extra dependencies."
            ) from exc
        return IpfxSpikeFinder(**kwargs)

    else:
        raise ValueError(
            f"Unknown spike finder backend: '{backend}'. "
            "Valid options: 'derivative', 'ipfx'."
        )
