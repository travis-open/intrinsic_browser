"""
recording.py
------------
Wraps a single ABF file. Responsible for:
  - Loading raw data via pyabf
  - Epoch detection and parsing
  - Per-sweep QC metric computation
  - Exposing sweep data as numpy arrays (time, voltage, current)
  - Optional per-analysis low-pass filtering (filtering is NOT stored on the
    Recording itself; callers request filtered copies)

A Recording is the lowest-level data object. It knows nothing about analysis
results or cross-file sweep collections — those live on Cell.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EpochInfo:
    """Structured description of one ABF epoch within a sweep.

    Attributes
    ----------
    epoch_index : int
        Zero-based index of this epoch within the sweep.
    epoch_type : str
        ABF epoch type string, e.g. "Step", "Ramp", "Pulse".
    start_sample : int
        Sample index (within the sweep) at which this epoch begins.
    end_sample : int
        Sample index (within the sweep) at which this epoch ends (exclusive).
    start_time_s : float
        Time in seconds from sweep onset at which this epoch begins.
    end_time_s : float
        Time in seconds from sweep onset at which this epoch ends.
    level : float
        Command level for this epoch (pA for current clamp, mV for voltage
        clamp). For step epochs this is the absolute amplitude; for ramps the
        starting level.
    delta_level : float
        Per-sweep increment applied to ``level`` across the sweep family.
        Will be 0.0 for protocols that do not step the command.
    """

    epoch_index: int
    epoch_type: str
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    level: float
    delta_level: float


@dataclass
class SweepQCMetrics:
    """First-pass quality control metrics computed from a single sweep.

    These are estimated from the *first active epoch* (typically the
    pre-stimulus baseline) and are intended to help users decide which sweeps
    to include before running any analysis.

    Attributes
    ----------
    filename : str
        Stem of the source ABF file (no directory, no extension).
    sweep_index : int
        Zero-based sweep index within the ABF file.
    starting_voltage_mV : float
        Mean voltage during the baseline epoch (mV). NaN if unavailable.
    rms_noise_mV : float
        RMS of the voltage signal during the baseline epoch (mV).
    holding_current_pA : float or None
        Mean holding current if a current channel is present in the file,
        otherwise None.
    baseline_epoch_index : int
        Which epoch was used for baseline estimation.
    notes : str
        Any warnings or issues detected during QC estimation.
    """

    filename: str
    sweep_index: int
    starting_voltage_mV: float
    rms_noise_mV: float
    holding_current_pA: Optional[float]
    baseline_epoch_index: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class Recording:
    """Wraps a single ABF file and exposes its sweeps for analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the .abf file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If pyabf cannot open the file, or if epoch information cannot be
        parsed (fail-loud policy).

    Examples
    --------
    >>> rec = Recording("cell_01_steps.abf")
    >>> t, v, i = rec.get_sweep_arrays(sweep_index=0)
    >>> epochs = rec.get_epochs(sweep_index=0)
    >>> qc = rec.qc_metrics[0]
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath).resolve()
        if not self.filepath.exists():
            raise FileNotFoundError(f"ABF file not found: {self.filepath}")

        self._abf = None          # pyabf.ABF instance, loaded lazily
        self._qc_metrics: list[SweepQCMetrics] = []
        self._epochs_cache: dict[int, list[EpochInfo]] = {}

        self._load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def filename(self) -> str:
        """Stem of the ABF filename (no directory, no extension)."""
        return self.filepath.stem

    @property
    def filename_full(self) -> str:
        """Full filename including extension, no directory."""
        return self.filepath.name

    @property
    def n_sweeps(self) -> int:
        """Number of sweeps in this ABF file."""
        return self._abf.sweepCount

    @property
    def sampling_rate_hz(self) -> float:
        """Sampling rate in Hz."""
        return self._abf.dataRate

    @property
    def qc_metrics(self) -> list[SweepQCMetrics]:
        """Per-sweep QC metrics. Computed once during loading."""
        return self._qc_metrics

    @property
    def sweep_duration_s(self) -> float:
        """Duration of a single sweep in seconds."""
        return self._abf.sweepLengthSec

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    def get_sweep_arrays(
        self,
        sweep_index: int,
        lowpass_hz: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (time, voltage, current) arrays for one sweep.

        Parameters
        ----------
        sweep_index : int
            Zero-based sweep index.
        lowpass_hz : float or None
            If provided, apply a zero-phase Butterworth low-pass filter at
            this cutoff frequency (Hz) to the voltage trace before returning.
            The current trace is never filtered. Raw data on the Recording is
            unmodified.

        Returns
        -------
        time : np.ndarray, shape (n_samples,)
            Time in seconds from sweep onset.
        voltage : np.ndarray, shape (n_samples,)
            Membrane voltage in mV (filtered if lowpass_hz is given).
        current : np.ndarray, shape (n_samples,)
            Injected current in pA. Will be an array of NaN if no current
            channel is present in the file.

        Raises
        ------
        IndexError
            If sweep_index is out of range.
        """
        self._validate_sweep_index(sweep_index)

        self._abf.setSweep(sweep_index, channel=0)
        time = self._abf.sweepX.copy()
        voltage = self._abf.sweepY.copy()

        current = self._get_current_array(sweep_index)

        if lowpass_hz is not None:
            from wholecell.filters.lowpass import apply_lowpass
            voltage = apply_lowpass(voltage, self.sampling_rate_hz, lowpass_hz)

        return time, voltage, current

    def get_derivative(
        self,
        sweep_index: int,
        lowpass_hz: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (time, dV/dt) for one sweep in units of mV/ms.

        Parameters
        ----------
        sweep_index : int
            Zero-based sweep index.
        lowpass_hz : float or None
            Lowpass cutoff applied to voltage before differentiation.

        Returns
        -------
        time : np.ndarray
            Time array aligned to the derivative (same length as voltage,
            using central differences).
        dvdt : np.ndarray
            dV/dt in mV/ms.
        """
        time, voltage, _ = self.get_sweep_arrays(sweep_index, lowpass_hz=lowpass_hz)
        dvdt = np.gradient(voltage, time) / 1000.0  # convert to mV/ms
        return time, dvdt

    # ------------------------------------------------------------------
    # Epoch access
    # ------------------------------------------------------------------

    def get_epochs(self, sweep_index: int) -> list[EpochInfo]:
        """Return the list of epochs for one sweep.

        Epochs are parsed from the ABF header on first access and cached.

        Parameters
        ----------
        sweep_index : int
            Zero-based sweep index.

        Returns
        -------
        list of EpochInfo
            Ordered by epoch_index.

        Raises
        ------
        RuntimeError
            If epoch information cannot be extracted from the ABF header.
            Fails loudly rather than returning empty or partial data.
        """
        self._validate_sweep_index(sweep_index)
        if sweep_index not in self._epochs_cache:
            self._epochs_cache[sweep_index] = self._parse_epochs(sweep_index)
        return self._epochs_cache[sweep_index]

    def get_epoch(self, sweep_index: int, epoch_index: int) -> EpochInfo:
        """Return a single epoch by index.

        Parameters
        ----------
        sweep_index : int
        epoch_index : int
            Zero-based epoch index within the sweep.

        Raises
        ------
        IndexError
            If epoch_index is out of range for this sweep.
        RuntimeError
            If epochs cannot be parsed.
        """
        epochs = self.get_epochs(sweep_index)
        if epoch_index < 0 or epoch_index >= len(epochs):
            raise IndexError(
                f"epoch_index {epoch_index} out of range — sweep {sweep_index} "
                f"of '{self.filename}' has {len(epochs)} epochs (0–{len(epochs)-1})."
            )
        return epochs[epoch_index]

    def get_command_waveform(self, sweep_index: int) -> np.ndarray:
        """Return the synthesised DAC command waveform for one sweep.

        Uses pyabf's ``sweepC``, which correctly handles all epoch types
        (Step, Ramp, Pulse, etc.) by reading the protocol's epoch table.
        This is the right source to use when no recorded current channel is
        present in the file.

        Parameters
        ----------
        sweep_index : int

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Command waveform in the DAC units (pA for current clamp).
        """
        self._validate_sweep_index(sweep_index)
        self._abf.setSweep(sweep_index)
        return self._abf.sweepC.copy()

    def get_epoch_arrays(
        self,
        sweep_index: int,
        epoch_index: int,
        lowpass_hz: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (time, voltage, current) sliced to one epoch window.

        Parameters
        ----------
        sweep_index : int
        epoch_index : int
        lowpass_hz : float or None
            Passed through to get_sweep_arrays.

        Returns
        -------
        time, voltage, current : np.ndarray
            Slices of the full sweep arrays corresponding to the epoch window.
        """
        epoch = self.get_epoch(sweep_index, epoch_index)
        time, voltage, current = self.get_sweep_arrays(sweep_index, lowpass_hz=lowpass_hz)
        sl = slice(epoch.start_sample, epoch.end_sample)
        return time[sl], voltage[sl], current[sl]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise Recording identity to a dict (for session JSON).

        Note: raw data is not serialised — only the filepath and metadata
        needed to reload the file.
        """
        return {
            "filepath": str(self.filepath),
            "filename": self.filename,
            "n_sweeps": self.n_sweeps,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sweep_duration_s": self.sweep_duration_s,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Open the ABF file and compute QC metrics for all sweeps."""
        try:
            import pyabf
        except ImportError as exc:
            raise ImportError(
                "pyabf is required to load ABF files. "
                "Install it with: pip install pyabf"
            ) from exc

        try:
            self._abf = pyabf.ABF(str(self.filepath), loadData=True)
        except Exception as exc:
            raise RuntimeError(
                f"pyabf could not open '{self.filepath}': {exc}"
            ) from exc

        self._qc_metrics = [
            self._compute_qc(sweep_index)
            for sweep_index in range(self.n_sweeps)
        ]

    def _compute_qc(self, sweep_index: int) -> SweepQCMetrics:
        """Compute QC metrics for one sweep.

        Uses the first available epoch as the baseline window. If epochs
        cannot be parsed a graceful fallback uses the first 100 ms.
        """
        notes = ""
        baseline_epoch_index = 0

        try:
            epochs = self.get_epochs(sweep_index)
            epoch = epochs[0]
            start, end = epoch.start_sample, epoch.end_sample
        except (RuntimeError, IndexError):
            notes = "Epoch parsing failed; baseline estimated from first 100 ms."
            n_samples_100ms = int(0.1 * self.sampling_rate_hz)
            start, end = 0, n_samples_100ms

        self._abf.setSweep(sweep_index, channel=0)
        baseline_v = self._abf.sweepY[start:end]

        starting_voltage = float(np.mean(baseline_v)) if len(baseline_v) > 0 else float("nan")
        rms_noise = float(np.std(baseline_v)) if len(baseline_v) > 0 else float("nan")

        holding_current = self._get_holding_current(sweep_index, start, end)

        return SweepQCMetrics(
            filename=self.filename,
            sweep_index=sweep_index,
            starting_voltage_mV=starting_voltage,
            rms_noise_mV=rms_noise,
            holding_current_pA=holding_current,
            baseline_epoch_index=baseline_epoch_index,
            notes=notes,
        )

    def _get_current_array(self, sweep_index: int) -> np.ndarray:
        """Return the current channel array, or NaN array if unavailable."""
        if self._abf.channelCount > 1:
            try:
                self._abf.setSweep(sweep_index, channel=1)
                return self._abf.sweepY.copy()
            except Exception:
                pass
        n = int(self.sweep_duration_s * self.sampling_rate_hz)
        return np.full(n, np.nan)

    def _get_holding_current(
        self,
        sweep_index: int,
        start_sample: int,
        end_sample: int,
    ) -> Optional[float]:
        """Return mean holding current over a sample window, or None."""
        if self._abf.channelCount > 1:
            try:
                self._abf.setSweep(sweep_index, channel=1)
                segment = self._abf.sweepY[start_sample:end_sample]
                if len(segment) > 0:
                    return float(np.mean(segment))
            except Exception:
                pass
        return None

    def _parse_epochs(self, sweep_index: int) -> list[EpochInfo]:
        """Parse epoch information for one sweep.

        Tries ``epochTable`` first (present on most ABF2 files with a full DAC
        waveform editor entry); falls back to ``sweepEpochs`` which is always
        populated by pyabf after ``setSweep``.

        Raises
        ------
        RuntimeError
            If no epoch information can be found.
        """
        self._abf.setSweep(sweep_index)

        # --- Primary path: epochTable (has deltaLevel per epoch) ---
        try:
            epoch_table = self._abf.epochTable
            epochs: list[EpochInfo] = []
            for i, ep in enumerate(epoch_table.epochs):
                start_s = ep.p1 / self.sampling_rate_hz
                end_s = ep.p2 / self.sampling_rate_hz
                epochs.append(EpochInfo(
                    epoch_index=i,
                    epoch_type=ep.epochType,
                    start_sample=ep.p1,
                    end_sample=ep.p2,
                    start_time_s=start_s,
                    end_time_s=end_s,
                    level=ep.level,
                    delta_level=ep.deltaLevel,
                ))
            if epochs:
                return epochs
        except AttributeError:
            pass

        # --- Fallback: sweepEpochs (always available, no deltaLevel) ---
        try:
            se = self._abf.sweepEpochs
            epochs = []
            for i, (p1, p2, level, ep_type) in enumerate(
                zip(se.p1s, se.p2s, se.levels, se.types)
            ):
                epochs.append(EpochInfo(
                    epoch_index=i,
                    epoch_type=ep_type,
                    start_sample=p1,
                    end_sample=p2,
                    start_time_s=p1 / self.sampling_rate_hz,
                    end_time_s=p2 / self.sampling_rate_hz,
                    level=float(level),
                    delta_level=0.0,
                ))
            if epochs:
                return epochs
        except AttributeError:
            pass

        raise RuntimeError(
            f"Cannot parse epochs for sweep {sweep_index} of '{self.filename}'. "
            "Neither epochTable nor sweepEpochs is available. "
            "Verify that the ABF file was recorded with a protocol that defines epochs."
        )

    def _validate_sweep_index(self, sweep_index: int) -> None:
        if sweep_index < 0 or sweep_index >= self.n_sweeps:
            raise IndexError(
                f"sweep_index {sweep_index} out of range — "
                f"'{self.filename}' has {self.n_sweeps} sweeps (0–{self.n_sweeps - 1})."
            )

    def __repr__(self) -> str:
        return (
            f"Recording('{self.filename}', "
            f"{self.n_sweeps} sweeps, "
            f"{self.sampling_rate_hz:.0f} Hz)"
        )
