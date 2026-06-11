"""
sweep_collection.py
-------------------
A named, user-defined selection of sweeps, potentially spanning multiple
ABF files. This is the unit of analysis: callers pass a SweepCollection to
analysis functions rather than raw Recording objects.

Key design principles
~~~~~~~~~~~~~~~~~~~~~
- Sweep identity is always stored as two separate atomic fields:
  ``filename`` (str, stem only) and ``sweep_index`` (int). A compound
  display label is available but is never used as the primary key.
- A SweepCollection is immutable once constructed. Analysis functions
  receive it read-only.
- The collection holds references to the parent Recording objects so
  analysis code can call ``get_sweep_arrays`` without re-opening files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

if TYPE_CHECKING:
    from wholecell.core.recording import Recording, SweepQCMetrics


# ---------------------------------------------------------------------------
# Sweep reference — atomic identity unit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepRef:
    """Uniquely identifies one sweep across all loaded recordings.

    Attributes
    ----------
    filename : str
        Stem of the source ABF file (no directory, no extension). Matches
        ``Recording.filename``.
    sweep_index : int
        Zero-based sweep index within the file.
    display_label : str
        Human-readable label used in the GUI and as an optional column in
        output tables. Never relied upon as a key.

    Notes
    -----
    ``filename`` and ``sweep_index`` are the only fields a downstream user
    should ever need to reload data:

    >>> row = df[df.sweep_index == 3]
    >>> abf = pyabf.ABF(row.filename.iloc[0] + ".abf")
    """

    filename: str
    sweep_index: int
    display_label: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.display_label:
            object.__setattr__(
                self, "display_label", f"{self.filename}__s{self.sweep_index:03d}"
            )

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict for session persistence."""
        return {
            "filename": self.filename,
            "sweep_index": self.sweep_index,
            "display_label": self.display_label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SweepRef":
        return cls(
            filename=d["filename"],
            sweep_index=d["sweep_index"],
            display_label=d.get("display_label", ""),
        )


# ---------------------------------------------------------------------------
# SweepCollection
# ---------------------------------------------------------------------------

class SweepCollection:
    """A named, ordered selection of sweeps for a specific analysis purpose.

    Parameters
    ----------
    name : str
        User-assigned label for this collection, e.g. ``"current_steps"``
        or ``"hyperpolarizing_steps"``. Must be unique within a Cell.
    sweeps : list of SweepRef
        Ordered sweep references. Order is preserved and matters for F-I
        curves and adaptation analysis.
    recordings : dict mapping filename stem → Recording
        The loaded Recording objects that back the SweepRefs. This allows
        the collection to fetch data arrays without re-opening files.
    notes : str, optional
        Free-text annotation from the user (captured in session JSON).

    Examples
    --------
    >>> sc = SweepCollection(
    ...     name="current_steps",
    ...     sweeps=[
    ...         SweepRef("cell_01_steps", 0),
    ...         SweepRef("cell_01_steps", 1),
    ...         SweepRef("cell_01_steps_2", 0),
    ...     ],
    ...     recordings={"cell_01_steps": rec1, "cell_01_steps_2": rec2},
    ... )
    >>> t, v, i = sc.get_sweep_arrays(SweepRef("cell_01_steps", 1))
    """

    def __init__(
        self,
        name: str,
        sweeps: list[SweepRef],
        recordings: dict[str, "Recording"],
        notes: str = "",
    ) -> None:
        self.name = name
        self._sweeps = list(sweeps)
        self._recordings = recordings
        self.notes = notes

        self._validate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sweeps(self) -> list[SweepRef]:
        """Ordered list of SweepRef objects in this collection."""
        return list(self._sweeps)

    @property
    def n_sweeps(self) -> int:
        return len(self._sweeps)

    @property
    def filenames(self) -> list[str]:
        """Unique filenames (stems) represented in this collection."""
        seen: list[str] = []
        for ref in self._sweeps:
            if ref.filename not in seen:
                seen.append(ref.filename)
        return seen

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_sweep_arrays(
        self,
        ref: SweepRef,
        lowpass_hz: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (time, voltage, current) arrays for one sweep.

        Parameters
        ----------
        ref : SweepRef
            Must be a member of this collection.
        lowpass_hz : float or None
            If provided, voltage is low-pass filtered before return.

        Returns
        -------
        time, voltage, current : np.ndarray
        """
        rec = self._get_recording(ref)
        return rec.get_sweep_arrays(ref.sweep_index, lowpass_hz=lowpass_hz)

    def get_derivative(
        self,
        ref: SweepRef,
        lowpass_hz: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (time, dV/dt) in mV/ms for one sweep.

        Parameters
        ----------
        ref : SweepRef
        lowpass_hz : float or None
            Lowpass applied to voltage before differentiation.

        Returns
        -------
        time, dvdt : np.ndarray
        """
        rec = self._get_recording(ref)
        return rec.get_derivative(ref.sweep_index, lowpass_hz=lowpass_hz)

    def get_epoch_arrays(
        self,
        ref: SweepRef,
        epoch_index: int,
        lowpass_hz: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (time, voltage, current) sliced to one epoch window.

        Parameters
        ----------
        ref : SweepRef
        epoch_index : int
            Zero-based epoch index within the sweep.
        lowpass_hz : float or None

        Returns
        -------
        time, voltage, current : np.ndarray
        """
        rec = self._get_recording(ref)
        return rec.get_epoch_arrays(ref.sweep_index, epoch_index, lowpass_hz=lowpass_hz)

    def get_qc_metrics(self, ref: SweepRef) -> "SweepQCMetrics":
        """Return QC metrics for one sweep.

        Parameters
        ----------
        ref : SweepRef

        Returns
        -------
        SweepQCMetrics
        """
        rec = self._get_recording(ref)
        return rec.qc_metrics[ref.sweep_index]

    def all_qc_metrics(self) -> list["SweepQCMetrics"]:
        """Return QC metrics for all sweeps in order."""
        return [self.get_qc_metrics(ref) for ref in self._sweeps]

    def iter_sweep_arrays(
        self,
        lowpass_hz: Optional[float] = None,
    ) -> Iterator[tuple[SweepRef, np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over (ref, time, voltage, current) for all sweeps.

        Useful for analysis loops:

        >>> for ref, t, v, i in sc.iter_sweep_arrays(lowpass_hz=2000.0):
        ...     # run per-sweep analysis
        ...     pass
        """
        for ref in self._sweeps:
            t, v, i = self.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)
            yield ref, t, v, i

    # ------------------------------------------------------------------
    # Exclusion support
    # ------------------------------------------------------------------

    def exclude_sweeps(self, refs: list[SweepRef]) -> "SweepCollection":
        """Return a new SweepCollection with the specified sweeps removed.

        The original collection is not modified (immutable pattern).

        Parameters
        ----------
        refs : list of SweepRef
            Sweeps to exclude.

        Returns
        -------
        SweepCollection
            New collection with the same name and recordings but the
            specified sweeps removed.
        """
        exclude_set = set(refs)
        remaining = [r for r in self._sweeps if r not in exclude_set]
        return SweepCollection(
            name=self.name,
            sweeps=remaining,
            recordings=self._recordings,
            notes=self.notes,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict for session persistence.

        Note: Recording objects are not serialised — only filepaths needed
        to reload them.
        """
        return {
            "name": self.name,
            "notes": self.notes,
            "sweeps": [ref.to_dict() for ref in self._sweeps],
            "filenames": self.filenames,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        recordings: dict[str, "Recording"],
    ) -> "SweepCollection":
        """Reconstruct from session JSON dict.

        Parameters
        ----------
        d : dict
            Output of ``to_dict()``.
        recordings : dict
            Mapping of filename stem → Recording, already loaded by Cell.
        """
        sweeps = [SweepRef.from_dict(s) for s in d["sweeps"]]
        return cls(
            name=d["name"],
            sweeps=sweeps,
            recordings=recordings,
            notes=d.get("notes", ""),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_recording(self, ref: SweepRef) -> "Recording":
        if ref.filename not in self._recordings:
            raise KeyError(
                f"No recording loaded for filename '{ref.filename}'. "
                f"Available recordings: {list(self._recordings.keys())}"
            )
        return self._recordings[ref.filename]

    def _validate(self) -> None:
        """Check that all SweepRefs point to loaded recordings."""
        missing = [
            ref for ref in self._sweeps
            if ref.filename not in self._recordings
        ]
        if missing:
            raise ValueError(
                f"SweepCollection '{self.name}': the following sweeps reference "
                f"recordings that are not loaded: "
                f"{[r.filename for r in missing]}. "
                f"Add the corresponding ABF files to the Cell before creating "
                f"this collection."
            )

    def __len__(self) -> int:
        return self.n_sweeps

    def __iter__(self) -> Iterator[SweepRef]:
        return iter(self._sweeps)

    def __repr__(self) -> str:
        return (
            f"SweepCollection(name='{self.name}', "
            f"{self.n_sweeps} sweeps across {len(self.filenames)} file(s))"
        )
