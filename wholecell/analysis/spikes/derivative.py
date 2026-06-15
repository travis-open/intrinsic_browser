"""
derivative.py
-------------
Built-in spike finder based on dV/dt threshold crossing with local-maximum thresholding.

Algorithm overview
~~~~~~~~~~~~~~~~~~
1. Compute dV/dt (mV/ms) using numpy gradient.
2. Find samples where dV/dt crosses the detection threshold (default: 20 mV/ms)
   upward; these are candidate AP onset markers used only to anchor the search.
3. For each candidate:
   a. Compute the local maximum dV/dt in a window around the detection crossing
      (default ±10 ms). This adapts the threshold criterion to each AP.
   b. Walk backward from the detection crossing to find where dV/dt first
      crosses 5 % of the local maximum dV/dt upward; that sample defines the
      voltage threshold reported for the AP. Falls back to 5% of sweep-maximum
      if local max is too small (< 50 mV/ms) to avoid noise amplification.
   c. Walk forward to find the peak (maximum voltage within a search window,
      default 20 ms to handle artifacts).
   d. Walk forward from the peak to find the fast trough (first local minimum).
4. Apply a refractory period: reject any detection whose candidate crossing
   occurs within ``refractory_ms`` of the previous detection's peak.

The local-maximum approach makes threshold detection robust to current-injection
artifacts: when current onset creates a dV/dt bump at the detection crossing,
the local max is small, so the threshold criterion uses 5% of that local peak,
correctly skipping the artifact and landing on the true AP rising phase.

Parameters exposed to the user
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dvdt_detection_mVms : float
    dV/dt crossing level used to *detect* APs (mV/ms). Default 20 mV/ms.
dvdt_threshold_pct : float
    Percentage of dV/dt (local or global) used to define the voltage threshold
    of each AP. Default 5 %.
refractory_ms : float
    Minimum time between consecutive spike threshold crossings (ms).
    Default 2 ms.
peak_search_window_ms : float
    Window after detection crossing within which to search for the voltage
    peak (ms). Default 20 ms (increased to handle artifacts far from threshold).
trough_search_window_ms : float
    Window after the peak within which to search for the fast trough (ms).
    Default 100 ms.
min_peak_voltage_mV : float
    Minimum voltage for a valid spike peak (mV). Default -20 mV.
    Rejects small depolarisations that are not true action potentials.
dvdt_local_window_ms : float
    Window around each detection crossing (±) to compute local maximum dV/dt
    for threshold criterion. Default 10 ms.
min_local_dvdt_for_fallback : float
    Minimum local dV/dt to use local criterion; otherwise fall back to global
    sweep-maximum criterion. Prevents noise amplification. Default 50 mV/ms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wholecell.analysis.spikes.base import SpikeFinder, SpikeDetection


class DerivativeSpikeFinder(SpikeFinder):
    """Spike finder based on dV/dt threshold crossing with local-maximum thresholding.

    Parameters
    ----------
    dvdt_detection_mVms : float
        dV/dt level used to *detect* AP candidates (mV/ms). Default 20.
    dvdt_threshold_pct : float
        Percentage of dV/dt used to define the voltage threshold of each AP.
        Applied to local maximum dV/dt (see dvdt_local_window_ms). Default 5.
    refractory_ms : float
        Refractory period (ms). Default 2.
    peak_search_window_ms : float
        Window to search for peak after detection crossing (ms). Default 20.
    trough_search_window_ms : float
        Window to search for fast trough after peak (ms). Default 100.
    min_peak_voltage_mV : float
        Minimum voltage for a valid spike peak (mV). Default -20.
    dvdt_local_window_ms : float
        Window around each detection crossing (±) to compute local maximum
        dV/dt for threshold criterion. Helps avoid artifact confusion. Default 10.
    min_local_dvdt_for_fallback : float
        Minimum local dV/dt value to use local criterion; otherwise fall back
        to global sweep-maximum criterion. Prevents noise amplification. Default 50.
    """

    def __init__(
        self,
        dvdt_detection_mVms: float = 20.0,
        dvdt_threshold_pct: float = 5.0,
        refractory_ms: float = 2.0,
        peak_search_window_ms: float = 20.0,
        trough_search_window_ms: float = 100.0,
        min_peak_voltage_mV: float = -20.0,
        dvdt_local_window_ms: float = 10.0,
        min_local_dvdt_for_fallback: float = 50.0,
        min_rise_samples: int = 3,
    ) -> None:
        self.dvdt_detection_mVms = dvdt_detection_mVms
        self.dvdt_threshold_pct = dvdt_threshold_pct
        self.refractory_ms = refractory_ms
        self.peak_search_window_ms = peak_search_window_ms
        self.trough_search_window_ms = trough_search_window_ms
        self.min_peak_voltage_mV = min_peak_voltage_mV
        self.dvdt_local_window_ms = dvdt_local_window_ms
        self.min_local_dvdt_for_fallback = min_local_dvdt_for_fallback
        self.min_rise_samples = min_rise_samples

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
            "dvdt_local_window_ms": self.dvdt_local_window_ms,
            "min_local_dvdt_for_fallback": self.min_local_dvdt_for_fallback,
            "min_rise_samples": self.min_rise_samples,
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

        # Sweep-level max dV/dt — fallback for low-amplitude APs
        sweep_max_dvdt = float(np.max(dvdt_mVms))
        dvdt_5pct_global = (self.dvdt_threshold_pct / 100.0) * sweep_max_dvdt

        # Convert time windows from ms to samples
        refractory_samples = int(self.refractory_ms * 1e-3 * sampling_rate_hz)
        peak_window_samples = int(self.peak_search_window_ms * 1e-3 * sampling_rate_hz)
        trough_window_samples = int(self.trough_search_window_ms * 1e-3 * sampling_rate_hz)
        local_window_samples = int(self.dvdt_local_window_ms * 1e-3 * sampling_rate_hz)

        # Find upward crossings of the detection level — anchor points only
        above_det = dvdt_mVms >= self.dvdt_detection_mVms
        crossings = np.where(~above_det[:-1] & above_det[1:])[0] + 1

        spikes: list[SpikeDetection] = []
        last_peak_index = -refractory_samples

        for detect_idx in crossings:
            # Refractory period check
            if detect_idx - last_peak_index < refractory_samples:
                continue

            # Compute local max dV/dt around this detection crossing
            local_search_start = max(0, detect_idx - local_window_samples)
            local_search_end = min(detect_idx + local_window_samples, len(dvdt_mVms))
            local_max_dvdt = float(np.max(dvdt_mVms[local_search_start:local_search_end]))

            # Decide whether to use local or global 5% criterion
            if local_max_dvdt >= self.min_local_dvdt_for_fallback:
                dvdt_5pct = (self.dvdt_threshold_pct / 100.0) * local_max_dvdt
            else:
                dvdt_5pct = dvdt_5pct_global

            # Find peak: max voltage in window after detection crossing
            peak_end = min(detect_idx + peak_window_samples, len(voltage))
            peak_idx = detect_idx + int(np.argmax(voltage[detect_idx:peak_end]))

            if voltage[peak_idx] < self.min_peak_voltage_mV:
                continue

            # Find fast trough: first local minimum after peak
            trough_end = min(peak_idx + trough_window_samples, len(voltage))
            trough_idx = self._find_trough(voltage, peak_idx, trough_end, self.min_rise_samples)

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
        min_rise_samples: int = 3,
    ) -> int:
        """Find the fast trough (first sustained local min) after the spike peak.

        Requires voltage to rise for ``min_rise_samples`` consecutive samples
        before declaring a trough, preventing single noisy upward ticks on the
        downstroke from triggering a premature return.

        Parameters
        ----------
        voltage : np.ndarray
        start_idx : int
            Index just after the peak.
        end_idx : int
            Maximum index to search (exclusive).
        min_rise_samples : int
            Number of consecutive rising samples required to confirm a trough.

        Returns
        -------
        int
            Index of the trough within voltage.
        """
        consecutive_rises = 0
        for i in range(start_idx, end_idx - 1):
            if voltage[i + 1] > voltage[i]:
                consecutive_rises += 1
                if consecutive_rises >= min_rise_samples:
                    return i + 1 - min_rise_samples
            else:
                consecutive_rises = 0
        # Fallback: absolute minimum in window
        window = voltage[start_idx:end_idx]
        return start_idx + int(np.argmin(window))
