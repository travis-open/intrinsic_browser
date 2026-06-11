"""
derivative.py
-------------
Built-in spike finder based on dV/dt threshold crossing.

Algorithm overview
~~~~~~~~~~~~~~~~~~
1. Compute dV/dt (mV/ms) using numpy gradient.
2. Find samples where dV/dt crosses the threshold (default: 20 mV/ms).
3. For each threshold crossing:
   a. Walk forward to find the peak (maximum voltage within a search window).
   b. Walk forward from the peak to find the fast trough (first local
      minimum, bounded by the next threshold crossing or a max window).
   c. The threshold crossing sample is the action potential threshold.
4. Apply a refractory period: reject any detection whose threshold crossing
   occurs within ``refractory_ms`` of the previous detection's peak.

This algorithm is intentionally similar to the core of IPFX's spike
detection so that results are comparable when switching backends.

Parameters exposed to the user
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dvdt_threshold_mVms : float
    dV/dt threshold in mV/ms. Default 20 mV/ms (IPFX default).
refractory_ms : float
    Minimum time between consecutive spike threshold crossings (ms).
    Default 2 ms.
peak_search_window_ms : float
    Window after threshold crossing within which to search for the
    voltage peak (ms). Default 10 ms.
trough_search_window_ms : float
    Window after the peak within which to search for the fast trough (ms).
    Default 100 ms.
min_peak_voltage_mV : float
    Minimum voltage for a valid spike peak (mV). Default -20 mV.
    Rejects small depolarisations that are not true action potentials.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wholecell.analysis.spikes.base import SpikeFinder, SpikeDetection


class DerivativeSpikeFinder(SpikeFinder):
    """Spike finder based on dV/dt threshold crossing.

    Parameters
    ----------
    dvdt_threshold_mVms : float
        dV/dt threshold (mV/ms). Default 20.
    refractory_ms : float
        Refractory period (ms). Default 2.
    peak_search_window_ms : float
        Window to search for peak after threshold crossing (ms). Default 10.
    trough_search_window_ms : float
        Window to search for fast trough after peak (ms). Default 100.
    min_peak_voltage_mV : float
        Minimum voltage for a valid spike peak (mV). Default -20.
    """

    def __init__(
        self,
        dvdt_threshold_mVms: float = 20.0,
        refractory_ms: float = 2.0,
        peak_search_window_ms: float = 10.0,
        trough_search_window_ms: float = 100.0,
        min_peak_voltage_mV: float = -20.0,
    ) -> None:
        self.dvdt_threshold_mVms = dvdt_threshold_mVms
        self.refractory_ms = refractory_ms
        self.peak_search_window_ms = peak_search_window_ms
        self.trough_search_window_ms = trough_search_window_ms
        self.min_peak_voltage_mV = min_peak_voltage_mV

    @property
    def backend_name(self) -> str:
        return "derivative"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "dvdt_threshold_mVms": self.dvdt_threshold_mVms,
            "refractory_ms": self.refractory_ms,
            "peak_search_window_ms": self.peak_search_window_ms,
            "trough_search_window_ms": self.trough_search_window_ms,
            "min_peak_voltage_mV": self.min_peak_voltage_mV,
        }

    def detect(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
    ) -> list[SpikeDetection]:
        """Detect spikes via dV/dt threshold crossing.

        Parameters
        ----------
        time : np.ndarray, shape (n,), units: seconds
        voltage : np.ndarray, shape (n,), units: mV
        current : np.ndarray, shape (n,), units: pA

        Returns
        -------
        list of SpikeDetection, sorted by peak_time_s.
        """
        if len(time) < 2:
            return []

        dt_s = time[1] - time[0]
        sampling_rate_hz = 1.0 / dt_s

        # Compute dV/dt in mV/ms
        dvdt_mVms = np.gradient(voltage, time) / 1000.0

        # Convert time windows from ms to samples
        refractory_samples = int(self.refractory_ms * 1e-3 * sampling_rate_hz)
        peak_window_samples = int(self.peak_search_window_ms * 1e-3 * sampling_rate_hz)
        trough_window_samples = int(self.trough_search_window_ms * 1e-3 * sampling_rate_hz)

        # Find upward threshold crossings (dV/dt goes from below to above threshold)
        above = dvdt_mVms >= self.dvdt_threshold_mVms
        crossings = np.where(~above[:-1] & above[1:])[0] + 1  # rising edge indices

        spikes: list[SpikeDetection] = []
        last_peak_index = -refractory_samples  # no refractory constraint initially

        for threshold_idx in crossings:
            # Refractory period check
            if threshold_idx - last_peak_index < refractory_samples:
                continue

            # Find peak: max voltage in window after threshold crossing
            peak_end = min(threshold_idx + peak_window_samples, len(voltage))
            peak_idx = threshold_idx + int(np.argmax(voltage[threshold_idx:peak_end]))

            if voltage[peak_idx] < self.min_peak_voltage_mV:
                continue

            # Find fast trough: first local minimum after peak
            trough_end = min(peak_idx + trough_window_samples, len(voltage))
            trough_idx = self._find_trough(voltage, peak_idx, trough_end)

            last_peak_index = peak_idx

            spikes.append(SpikeDetection(
                peak_index=int(peak_idx),
                peak_time_s=float(time[peak_idx]),
                peak_voltage_mV=float(voltage[peak_idx]),
                threshold_index=int(threshold_idx),
                threshold_time_s=float(time[threshold_idx]),
                threshold_voltage_mV=float(voltage[threshold_idx]),
                trough_index=int(trough_idx),
                trough_time_s=float(time[trough_idx]),
                trough_voltage_mV=float(voltage[trough_idx]),
                backend=self.backend_name,
            ))

        return spikes

    @staticmethod
    def _find_trough(
        voltage: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> int:
        """Find the fast trough (first local min) after the spike peak.

        Walks forward from start_idx looking for the first sample where
        voltage starts to increase again. Falls back to argmin in the
        window if no local minimum is found before end_idx.

        Parameters
        ----------
        voltage : np.ndarray
        start_idx : int
            Index just after the peak.
        end_idx : int
            Maximum index to search (exclusive).

        Returns
        -------
        int
            Index of the trough within voltage.
        """
        for i in range(start_idx, end_idx - 1):
            if voltage[i + 1] > voltage[i]:
                return i
        # Fallback: absolute minimum in window
        window = voltage[start_idx:end_idx]
        return start_idx + int(np.argmin(window))
