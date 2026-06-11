"""
ipfx_backend.py
---------------
Optional spike finder backend that wraps the Allen Institute IPFX library.

This module is only imported if the user selects backend='ipfx' AND ipfx is
installed. If ipfx is not installed, build_spike_finder() raises an
ImportError with clear installation instructions.

IPFX spike detection uses the same underlying dV/dt threshold approach as
our native derivative finder, but includes additional heuristics for
handling burst spikes, noise, and edge effects that have been validated
against a large cortical cell dataset.

Usage
-----
The IPFX backend is useful when:
  - Direct comparison with IPFX-based analyses is required.
  - The built-in derivative finder produces too many or too few detections.

Note: IPFX requires numpy arrays only — no NWB files or IPFX data objects
are constructed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wholecell.analysis.spikes.base import SpikeFinder, SpikeDetection


class IpfxSpikeFinder(SpikeFinder):
    """Spike finder that wraps IPFX's spike detection utilities.

    Parameters
    ----------
    dvdt_threshold_mVms : float
        dV/dt threshold passed to IPFX spike detector (mV/ms). Default 20.
    filter_frequency_hz : float or None
        Internal filter frequency passed to IPFX. If None, no additional
        filtering is applied by IPFX (pre-filtering via lowpass_hz in
        Cell.find_spikes is preferred). Default None.
    min_peak_voltage_mV : float
        Minimum peak voltage for a valid spike (mV). Default -20.

    Notes
    -----
    IPFX's ``SpikeDetector`` class is instantiated fresh per sweep call to
    avoid state leakage between sweeps.
    """

    def __init__(
        self,
        dvdt_threshold_mVms: float = 20.0,
        filter_frequency_hz: float | None = None,
        min_peak_voltage_mV: float = -20.0,
    ) -> None:
        self.dvdt_threshold_mVms = dvdt_threshold_mVms
        self.filter_frequency_hz = filter_frequency_hz
        self.min_peak_voltage_mV = min_peak_voltage_mV

        # Verify ipfx is importable at construction time
        self._check_ipfx()

    @property
    def backend_name(self) -> str:
        return "ipfx"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "dvdt_threshold_mVms": self.dvdt_threshold_mVms,
            "filter_frequency_hz": self.filter_frequency_hz,
            "min_peak_voltage_mV": self.min_peak_voltage_mV,
        }

    def detect(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
    ) -> list[SpikeDetection]:
        """Detect spikes using IPFX's SpikeDetector.

        Parameters
        ----------
        time : np.ndarray, units: seconds
        voltage : np.ndarray, units: mV
        current : np.ndarray, units: pA

        Returns
        -------
        list of SpikeDetection

        Notes
        -----
        IPFX returns spike features as a dict of arrays (one value per
        spike). We map those arrays back to SpikeDetection objects so the
        rest of the pipeline sees a consistent interface.

        TODO: implement once IPFX import paths are confirmed. The key
        functions to call are:
            - ``ipfx.spike_detector.detect_putative_spikes``
            - ``ipfx.spike_features.spike_feature_extractor`` (or equivalent)
        """
        raise NotImplementedError(
            "IpfxSpikeFinder.detect: implementation pending.\n"
            "Will call ipfx.spike_detector.detect_putative_spikes with "
            "(time, voltage, current) arrays and map results to SpikeDetection objects."
        )

    @staticmethod
    def _check_ipfx() -> None:
        try:
            import ipfx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'ipfx' spike finder backend requires the ipfx package.\n"
                "Install it with: pip install ipfx\n"
                "Or use backend='derivative' (no extra dependencies)."
            ) from exc
