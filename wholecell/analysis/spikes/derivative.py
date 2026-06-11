"""
derivative.py
-------------
Built-in spike finder based on dV/dt threshold crossing.

Algorithm overview
~~~~~~~~~~~~~~~~~~
1. Compute dV/dt (mV/ms) using numpy gradient.
2. Find samples where dV/dt crosses the detection threshold (default: 20 mV/ms)
   upward; these are candidate AP onset markers used only to anchor the search.
3. For each candidate:
   a. Walk forward to find the peak (maximum voltage within a search window).
   b. Walk backward from the detection crossing to find where dV/dt first
      crosses 5 % of the sweep-wide maximum dV/dt upward; that sample defines
      the voltage threshold reported for the AP.
   c. Walk forward from the peak to find the fast trough (first local minimum).
4. Apply a refractory period: reject any detection whose candidate crossing
   occurs within ``refractory_ms`` of the previous detection's peak.

The "voltage threshold" is therefore the membrane potential at the point on
the rising phase where dV/dt first reaches ``dvdt_threshold_pct`` % of the
largest dV/dt seen anywhere in the sweep.  This is more robust than a fixed
mV/ms cutoff because it scales with the AP amplitude and recording conditions.

Parameters exposed to the user
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dvdt_detection_mVms : float
    dV/dt crossing level used to *detect* APs (mV/ms). Default 20 mV/ms.
dvdt_threshold_pct : float
    Percentage of sweep-maximum dV/dt used to define the voltage threshold
    of each AP. Default 5 %.
refractory_ms : float
    Minimum time between consecutive spike threshold crossings (ms).
    Default 2 ms.
peak_search_window_ms : float
    Window after detection crossing within which to search for the voltage
    peak (ms). Default 10 ms.
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
    dvdt_detection_mVms : float
        dV/dt level used to *detect* AP candidates (mV/ms). Default 20.
    dvdt_threshold_pct : float
        Percentage of sweep-maximum dV/dt used to define the voltage
        threshold of each AP. Default 5.
    refractory_ms : float
        Refractory period (ms). Default 2.
    peak_search_window_ms : float
        Window to search for peak after detection crossing (ms). Default 10.
    trough_search_window_ms : float
        Window to search for fast trough after peak (ms). Default 100.
    min_peak_voltage_mV : float
        Minimum voltage for a valid spike peak (mV). Default -20.
    """

    def __init__(
        self,
        dvdt_detection_mVms: float = 20.0,
        dvdt_threshold_pct: float = 5.0,
        refractory_ms: float = 2.0,
        peak_search_window_ms: float = 10.0,
        trough_search_window_ms: float = 100.0,
        min_peak_voltage_mV: float = -20.0,
    ) -> None:
        self.dvdt_detection_mVms = dvdt_detection_mVms
        self.dvdt_threshold_pct = dvdt_threshold_pct
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
            "dvdt_detection_mVms": self.dvdt_detection_mVms,
            "dvdt_threshold_pct": self.dvdt_threshold_pct,
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
        """Detect spikes and determine voltage threshold via 5 % dV/dt criterion.

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

        # Sweep-level max dV/dt — used to set the voltage-threshold criterion
        sweep_max_dvdt = float(np.max(dvdt_mVms))
        dvdt_5pct = (self.dvdt_threshold_pct / 100.0) * sweep_max_dvdt

        # Convert time windows from ms to samples
        refractory_samples = int(self.refractory_ms * 1e-3 * sampling_rate_hz)
        peak_window_samples = int(self.peak_search_window_ms * 1e-3 * sampling_rate_hz)
        trough_window_samples = int(self.trough_search_window_ms * 1e-3 * sampling_rate_hz)

        # Find upward crossings of the detection level — anchor points only
        above_det = dvdt_mVms >= self.dvdt_detection_mVms
        crossings = np.where(~above_det[:-1] & above_det[1:])[0] + 1

        spikes: list[SpikeDetection] = []
        last_peak_index = -refractory_samples

        for detect_idx in crossings:
            # Refractory period check
            if detect_idx - last_peak_index < refractory_samples:
                continue

            # Find peak: max voltage in window after detection crossing
            peak_end = min(detect_idx + peak_window_samples, len(voltage))
            peak_idx = detect_idx + int(np.argmax(voltage[detect_idx:peak_end]))

            if voltage[peak_idx] < self.min_peak_voltage_mV:
                continue

            # Find fast trough: first local minimum after peak
            trough_end = min(peak_idx + trough_window_samples, len(voltage))
            trough_idx = self._find_trough(voltage, peak_idx, trough_end)

            last_peak_index = peak_idx

            # Voltage threshold: scan backward from detect_idx to find where
            # dV/dt first rises above dvdt_5pct (i.e. the 5 % rising edge)
            threshold_idx = self._find_voltage_threshold(
                dvdt_mVms, detect_idx, refractory_samples, dvdt_5pct
            )

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
    def _find_voltage_threshold(
        dvdt: np.ndarray,
        detect_idx: int,
        refractory_samples: int,
        dvdt_5pct: float,
    ) -> int:
        """Return the sample index of the AP voltage threshold.

        Scans backward from ``detect_idx`` (the 20 mV/ms crossing) and
        returns the first sample (going forward) where dV/dt crosses
        ``dvdt_5pct`` upward.  Falls back to ``detect_idx`` if no such
        crossing is found within the refractory window.
        """
        search_start = max(0, detect_idx - refractory_samples)
        # Walk backward until dV/dt drops below the 5 % level
        for j in range(detect_idx, search_start - 1, -1):
            if dvdt[j] < dvdt_5pct:
                return j + 1  # first sample at/above 5 % going forward
        return search_start

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
