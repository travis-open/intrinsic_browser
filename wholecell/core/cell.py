"""
cell.py
-------
Cell is the top-level interactive analysis object. One Cell instance
represents one recording session (one biological cell), which may span
multiple ABF files.

Responsibilities
~~~~~~~~~~~~~~~~
- Owns and manages loaded Recording objects
- Creates and stores named SweepCollections
- Runs analyses and stores timestamped results
- Manages session persistence (save/load JSON)
- Provides a clean public API for interactive use and GUI calls

Typical interactive workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> cell = Cell(cell_id="cell_01")
>>> cell.add_recording("data/cell_01_steps.abf")
>>> cell.add_recording("data/cell_01_steps2.abf")
>>> cell.print_qc_table("cell_01_steps")
>>> sc = cell.create_sweep_collection(
...     name="current_steps",
...     sweeps=[
...         {"filename": "cell_01_steps",  "sweep_index": 0},
...         {"filename": "cell_01_steps",  "sweep_index": 1},
...         {"filename": "cell_01_steps2", "sweep_index": 0},
...     ],
... )
>>> result = cell.analyze_passive(
...     collection_name="current_steps",
...     epoch_index=1,
...     lowpass_hz=2000.0,
... )
>>> cell.save_session("cell_01_session.json")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from wholecell.core.recording import Recording
from wholecell.core.sweep_collection import SweepCollection, SweepRef


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

class Cell:
    """Top-level interactive analysis object for one patch-clamp recording.

    Parameters
    ----------
    cell_id : str
        A short human-readable identifier for this cell, used in output
        filenames and the session JSON. E.g. ``"cell_01"`` or
        ``"230915_cell2"``.
    output_dir : str or Path, optional
        Directory where results (CSVs, JSON) will be saved. Defaults to
        the current working directory. Created if it does not exist.
    notes : str, optional
        Free-text notes about this cell (genotype, age, condition, etc.).
        Stored in the session JSON.

    Attributes
    ----------
    recordings : dict[str, Recording]
        Loaded recordings keyed by filename stem.
    collections : dict[str, SweepCollection]
        Named sweep collections created by the user.
    results : dict[str, list[dict]]
        Analysis results keyed by result type (e.g. ``"passive"``,
        ``"spikes"``). Each value is a list of timestamped result dicts,
        most recent last.
    """

    def __init__(
        self,
        cell_id: str,
        output_dir: str | Path = ".",
        notes: str = "",
    ) -> None:
        self.cell_id = cell_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.notes = notes

        self.recordings: dict[str, Recording] = {}
        self.collections: dict[str, SweepCollection] = {}
        self.results: dict[str, list[dict]] = {}

        # Audit log: every significant action recorded here for traceability
        self._audit_log: list[dict] = []

    # ------------------------------------------------------------------
    # Recording management
    # ------------------------------------------------------------------

    def add_recording(self, filepath: str | Path) -> Recording:
        """Load an ABF file and register it with this Cell.

        Parameters
        ----------
        filepath : str or Path
            Path to the .abf file.

        Returns
        -------
        Recording
            The newly loaded Recording object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If a recording with the same filename stem is already loaded.
        RuntimeError
            If pyabf cannot open the file.

        Examples
        --------
        >>> rec = cell.add_recording("cell_01_steps.abf")
        >>> print(rec)
        Recording('cell_01_steps', 12 sweeps, 20000 Hz)
        """
        rec = Recording(filepath)
        if rec.filename in self.recordings:
            raise ValueError(
                f"A recording named '{rec.filename}' is already loaded. "
                "If you have two files with the same stem, rename one before loading."
            )
        self.recordings[rec.filename] = rec
        self._log("add_recording", {"filepath": str(rec.filepath), "filename": rec.filename})
        return rec

    def remove_recording(self, filename: str) -> None:
        """Unload a recording by filename stem.

        Parameters
        ----------
        filename : str
            Stem of the ABF file to remove (no extension).

        Raises
        ------
        KeyError
            If no recording with that name is loaded.
        ValueError
            If any existing SweepCollection references this recording.
        """
        if filename not in self.recordings:
            raise KeyError(f"No recording named '{filename}' is loaded.")

        dependent = [
            name for name, sc in self.collections.items()
            if filename in sc.filenames
        ]
        if dependent:
            raise ValueError(
                f"Cannot remove '{filename}': it is referenced by sweep "
                f"collection(s): {dependent}. Remove or update those collections first."
            )

        del self.recordings[filename]
        self._log("remove_recording", {"filename": filename})

    # ------------------------------------------------------------------
    # QC inspection
    # ------------------------------------------------------------------

    def get_qc_table(self, filename: str) -> pd.DataFrame:
        """Return a DataFrame of per-sweep QC metrics for one recording.

        Columns: filename, sweep_index, starting_voltage_mV, rms_noise_mV,
        holding_current_pA (NaN if unavailable), notes.

        Parameters
        ----------
        filename : str
            Stem of the recording to inspect.

        Returns
        -------
        pd.DataFrame
        """
        if filename not in self.recordings:
            raise KeyError(f"No recording named '{filename}' is loaded.")
        metrics = self.recordings[filename].qc_metrics
        rows = []
        for m in metrics:
            rows.append({
                "filename": m.filename,
                "sweep_index": m.sweep_index,
                "starting_voltage_mV": m.starting_voltage_mV,
                "rms_noise_mV": m.rms_noise_mV,
                "holding_current_pA": m.holding_current_pA,
                "notes": m.notes,
            })
        return pd.DataFrame(rows)

    def print_qc_table(self, filename: str) -> None:
        """Print a formatted QC table to stdout.

        Intended for interactive use in a notebook or script.

        Parameters
        ----------
        filename : str
            Stem of the recording to inspect.
        """
        df = self.get_qc_table(filename)
        print(f"\nQC metrics — {filename}")
        print("=" * 60)
        print(df.to_string(index=False, float_format="{:.3f}".format))
        print()

    # ------------------------------------------------------------------
    # Sweep collection management
    # ------------------------------------------------------------------

    def create_sweep_collection(
        self,
        name: str,
        sweeps: list[dict | SweepRef],
        notes: str = "",
        overwrite: bool = False,
    ) -> SweepCollection:
        """Create a named SweepCollection from a list of sweep references.

        Parameters
        ----------
        name : str
            Unique name for this collection within the Cell.
        sweeps : list of dict or SweepRef
            Each dict must have keys ``"filename"`` (str, stem) and
            ``"sweep_index"`` (int). Optionally ``"display_label"`` (str).
            Alternatively pass SweepRef objects directly.
        notes : str, optional
            Free-text annotation stored in the session JSON.
        overwrite : bool
            If True, replace an existing collection with this name.
            If False (default), raise ValueError if the name already exists.

        Returns
        -------
        SweepCollection

        Examples
        --------
        >>> sc = cell.create_sweep_collection(
        ...     name="hyperpolarizing",
        ...     sweeps=[
        ...         {"filename": "cell_01_steps", "sweep_index": 0},
        ...         {"filename": "cell_01_steps", "sweep_index": 1},
        ...     ],
        ... )
        """
        if name in self.collections and not overwrite:
            raise ValueError(
                f"A sweep collection named '{name}' already exists. "
                "Pass overwrite=True to replace it."
            )

        refs: list[SweepRef] = []
        for item in sweeps:
            if isinstance(item, SweepRef):
                refs.append(item)
            elif isinstance(item, dict):
                refs.append(SweepRef(
                    filename=item["filename"],
                    sweep_index=item["sweep_index"],
                    display_label=item.get("display_label", ""),
                ))
            else:
                raise TypeError(
                    f"Each sweep must be a dict or SweepRef, got {type(item)}."
                )

        sc = SweepCollection(
            name=name,
            sweeps=refs,
            recordings=self.recordings,
            notes=notes,
        )
        self.collections[name] = sc
        self._log("create_sweep_collection", {
            "name": name,
            "n_sweeps": len(refs),
            "sweeps": [r.to_dict() for r in refs],
            "notes": notes,
        })
        return sc

    def get_collection(self, name: str) -> SweepCollection:
        """Retrieve a named SweepCollection.

        Raises
        ------
        KeyError
            If no collection with that name exists.
        """
        if name not in self.collections:
            raise KeyError(
                f"No sweep collection named '{name}'. "
                f"Available: {list(self.collections.keys())}"
            )
        return self.collections[name]

    # ------------------------------------------------------------------
    # Analysis entry points (stubs — implementations in analysis modules)
    # ------------------------------------------------------------------

    def analyze_passive(
        self,
        collection_name: str,
        epoch_index: int,
        measures: list[str] | None = None,
        lowpass_hz: float | None = None,
    ) -> dict:
        """Estimate passive membrane properties from a current step collection.

        Runs the requested measures (input resistance, membrane time constant,
        sag ratio, sag amplitude, sag kinetics) on all sweeps in the named
        collection and stores the results with a timestamp.

        Parameters
        ----------
        collection_name : str
            Name of the SweepCollection to analyze.
        epoch_index : int
            Zero-based epoch index identifying the current step window.
            Fails loudly if epochs cannot be parsed (see Recording.get_epochs).
        measures : list of str, optional
            Subset of measures to compute. Valid values:
            ``"input_resistance"``, ``"time_constant"``, ``"sag_ratio"``,
            ``"sag_amplitude"``, ``"sag_kinetics"``.
            Defaults to all available measures.
        lowpass_hz : float or None
            Low-pass filter cutoff for this analysis only. Does not affect
            raw data or other analyses.

        Returns
        -------
        dict
            Timestamped result dict. Also appended to
            ``self.results["passive"]``.

        Raises
        ------
        RuntimeError
            If epoch information cannot be parsed from any sweep.
        KeyError
            If collection_name does not exist.

        Notes
        -----
        Implementation in ``wholecell.analysis.passive``.
        """
        from wholecell.analysis.passive import run_passive_analysis
        sc = self.get_collection(collection_name)
        params = {
            "collection_name": collection_name,
            "epoch_index": epoch_index,
            "measures": measures,
            "lowpass_hz": lowpass_hz,
        }
        result = run_passive_analysis(sc, epoch_index, measures=measures, lowpass_hz=lowpass_hz)
        return self._store_result("passive", result, params)

    def find_spikes(
        self,
        collection_name: str,
        epoch_index: int,
        backend: str = "derivative",
        lowpass_hz: float | None = None,
        **spike_finder_kwargs: Any,
    ) -> dict:
        """Detect action potentials in the specified epoch of all sweeps.

        Parameters
        ----------
        collection_name : str
        epoch_index : int
            Zero-based epoch index for the current injection window.
            Only spikes occurring within this epoch are counted.
            Fails loudly if epoch cannot be parsed.
        backend : str
            Spike finder backend: ``"derivative"`` (built-in, default) or
            ``"ipfx"`` (requires ipfx to be installed).
        lowpass_hz : float or None
            Low-pass filter cutoff for spike detection only.
        **spike_finder_kwargs
            Passed to the spike finder backend (e.g. ``dvdt_threshold=20.0``
            for the derivative finder).

        Returns
        -------
        dict
            Timestamped result dict containing per-spike data. Also appended
            to ``self.results["spikes"]``.

        Raises
        ------
        RuntimeError
            If epoch parsing fails.
        ImportError
            If backend is ``"ipfx"`` and ipfx is not installed.

        Notes
        -----
        Implementation in ``wholecell.analysis.spikes.derivative`` and
        ``wholecell.analysis.spikes.ipfx_backend``.
        """
        from wholecell.analysis.spikes.finder import run_spike_detection
        sc = self.get_collection(collection_name)
        params = {
            "collection_name": collection_name,
            "epoch_index": epoch_index,
            "backend": backend,
            "lowpass_hz": lowpass_hz,
            **spike_finder_kwargs,
        }
        result = run_spike_detection(
            sc, epoch_index,
            backend=backend,
            lowpass_hz=lowpass_hz,
            **spike_finder_kwargs,
        )
        return self._store_result("spikes", result, params)

    def analyze_fi_curve(
        self,
        collection_name: str,
        epoch_index: int,
        spike_result_timestamp: str | None = None,
    ) -> dict:
        """Build an F-I curve from previously detected spikes.

        Requires ``find_spikes`` to have been run on the same collection.

        Parameters
        ----------
        collection_name : str
        epoch_index : int
            Used to determine current injection amplitude per sweep from
            the epoch table. Fails loudly if epoch cannot be parsed.
        spike_result_timestamp : str or None
            If None, uses the most recent spike detection result for this
            collection. Pass a timestamp string to reference a specific
            prior result.

        Returns
        -------
        dict
            Timestamped result dict containing: per-sweep spike counts,
            current amplitudes, mean firing rates, the full F-I curve as
            parallel lists, rheobase estimate, and F-I slope(s).
            Also appended to ``self.results["fi_curve"]``.

        Notes
        -----
        Implementation in ``wholecell.analysis.fi_curve``.
        """
        from wholecell.analysis.fi_curve import run_fi_analysis
        sc = self.get_collection(collection_name)
        spikes = self._get_latest_result("spikes")
        params = {
            "collection_name": collection_name,
            "epoch_index": epoch_index,
            "spike_result_timestamp": spike_result_timestamp,
        }
        result = run_fi_analysis(sc, epoch_index, spikes)
        return self._store_result("fi_curve", result, params)

    def extract_spike_features(
        self,
        collection_name: str,
        spike_result_timestamp: str | None = None,
        lowpass_hz: float | None = None,
    ) -> dict:
        """Extract per-spike features (threshold, peak, trough, width, AHP).

        Parameters
        ----------
        collection_name : str
        spike_result_timestamp : str or None
            Reference a specific prior spike detection result, or None to
            use the most recent.
        lowpass_hz : float or None
            Low-pass filter applied to voltage before feature extraction.

        Returns
        -------
        dict
            Timestamped result dict. The ``"spike_table"`` key contains a
            list of dicts (one per spike) with columns:

            - ``filename`` (str) — stem of source ABF file
            - ``sweep_index`` (int)
            - ``spike_index_in_sweep`` (int)
            - ``peak_time_s``, ``peak_voltage_mV``
            - ``threshold_time_s``, ``threshold_voltage_mV``
            - ``trough_time_s``, ``trough_voltage_mV``
            - ``half_width_ms``
            - ``ahp_depth_mV``
            - ``display_label`` (str) — for GUI use only, not a key

        Notes
        -----
        Implementation in ``wholecell.analysis.spikes.features``.
        """
        from wholecell.analysis.spikes.features import run_feature_extraction
        sc = self.get_collection(collection_name)
        spikes = self._get_latest_result("spikes")
        params = {
            "collection_name": collection_name,
            "spike_result_timestamp": spike_result_timestamp,
            "lowpass_hz": lowpass_hz,
        }
        result = run_feature_extraction(sc, spikes, lowpass_hz=lowpass_hz)
        return self._store_result("spike_features", result, params)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_spike_table(
        self,
        filepath: str | Path | None = None,
        result_timestamp: str | None = None,
    ) -> pd.DataFrame:
        """Export the spike feature table as a CSV.

        Parameters
        ----------
        filepath : str or Path or None
            Output CSV path. Defaults to
            ``{output_dir}/{cell_id}_spikes_{timestamp}.csv``.
        result_timestamp : str or None
            Reference a specific prior spike feature result, or None for
            the most recent.

        Returns
        -------
        pd.DataFrame
            The spike table with columns: filename, sweep_index,
            spike_index_in_sweep, and all spike features.
        """
        result = self._get_latest_result("spike_features")
        df = pd.DataFrame(result["spike_table"])

        if filepath is None:
            ts = result["timestamp"].replace(":", "").replace("-", "").replace(" ", "_")
            filepath = self.output_dir / f"{self.cell_id}_spikes_{ts}.csv"

        df.to_csv(filepath, index=False)
        self._log("export_spike_table", {"filepath": str(filepath)})
        print(f"Spike table saved: {filepath}")
        return df

    def export_sweep_summary(
        self,
        collection_name: str,
        filepath: str | Path | None = None,
    ) -> pd.DataFrame:
        """Export the per-sweep summary table as a CSV.

        Columns include: filename, sweep_index, n_spikes, max_firing_rate_hz,
        current_injection_pA, mean_voltage_mV, and others populated by
        analysis runs.

        Parameters
        ----------
        collection_name : str
        filepath : str or Path or None

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError(
            "export_sweep_summary: implementation pending analysis modules."
        )

    def export_cell_summary(
        self,
        filepath: str | Path | None = None,
    ) -> dict:
        """Export cell-level summary scalars and curves to JSON.

        The JSON structure is intentionally flexible: fixed scalars (Rin,
        rheobase, first spike threshold) live at the top level; variable-
        length data (full F-I curve, adaptation curves) live under named
        keys. This accommodates cells with different numbers of current steps.

        Parameters
        ----------
        filepath : str or Path or None
            Defaults to ``{output_dir}/{cell_id}_cell_summary.json``.

        Returns
        -------
        dict
            The exported summary dict.
        """
        raise NotImplementedError(
            "export_cell_summary: implementation pending analysis modules."
        )

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def save_session(self, filepath: str | Path | None = None) -> Path:
        """Save the full analysis session to a JSON file.

        The session captures: loaded file paths, all sweep collections,
        all results (timestamped), analysis parameters, and the audit log.
        This provides a complete audit trail of every decision made.

        Parameters
        ----------
        filepath : str or Path or None
            Defaults to ``{output_dir}/{cell_id}_session.json``.

        Returns
        -------
        Path
            Path to the saved session file.
        """
        if filepath is None:
            filepath = self.output_dir / f"{self.cell_id}_session.json"
        filepath = Path(filepath)

        session = {
            "cell_id": self.cell_id,
            "notes": self.notes,
            "saved_at": _timestamp(),
            "recordings": {
                name: rec.to_dict()
                for name, rec in self.recordings.items()
            },
            "collections": {
                name: sc.to_dict()
                for name, sc in self.collections.items()
            },
            "results": self.results,
            "audit_log": self._audit_log,
        }

        with open(filepath, "w") as f:
            json.dump(session, f, indent=2, default=str)

        print(f"Session saved: {filepath}")
        return filepath

    @classmethod
    def load_session(cls, filepath: str | Path) -> "Cell":
        """Reconstruct a Cell from a saved session JSON file.

        ABF files are re-opened from their original paths. If a file has
        moved, a FileNotFoundError is raised with the missing path.

        Parameters
        ----------
        filepath : str or Path
            Path to the session JSON file.

        Returns
        -------
        Cell
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            session = json.load(f)

        cell = cls(
            cell_id=session["cell_id"],
            notes=session.get("notes", ""),
        )

        for name, rec_dict in session["recordings"].items():
            cell.add_recording(rec_dict["filepath"])

        for name, sc_dict in session["collections"].items():
            cell.collections[name] = SweepCollection.from_dict(
                sc_dict, recordings=cell.recordings
            )

        cell.results = session.get("results", {})
        cell._audit_log = session.get("audit_log", [])

        print(f"Session loaded: {filepath}")
        return cell

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_result(
        self,
        result_type: str,
        result_data: dict,
        params: dict,
    ) -> dict:
        """Wrap a result dict with a timestamp and store it."""
        entry = {
            "timestamp": _timestamp(),
            "params": params,
            "data": result_data,
        }
        if result_type not in self.results:
            self.results[result_type] = []
        self.results[result_type].append(entry)
        self._log(f"analyze_{result_type}", {"params": params, "timestamp": entry["timestamp"]})
        return entry

    def _get_latest_result(self, result_type: str) -> dict:
        """Return the most recent result of a given type.

        Raises
        ------
        KeyError
            If no results of that type exist yet.
        """
        if result_type not in self.results or not self.results[result_type]:
            raise KeyError(
                f"No '{result_type}' results found. "
                f"Run the corresponding analysis first."
            )
        return self.results[result_type][-1]

    def _log(self, action: str, details: dict) -> None:
        """Append an entry to the audit log."""
        self._audit_log.append({
            "timestamp": _timestamp(),
            "action": action,
            "details": details,
        })

    def __repr__(self) -> str:
        return (
            f"Cell(id='{self.cell_id}', "
            f"{len(self.recordings)} recording(s), "
            f"{len(self.collections)} collection(s))"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """ISO 8601 timestamp string for result versioning."""
    return datetime.now().isoformat(timespec="seconds")
