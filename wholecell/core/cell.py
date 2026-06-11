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

import numpy as np
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
        """Detect action potentials in all sweeps of the named collection.

        All spikes found anywhere in each sweep are returned. Each spike is
        annotated with ``epoch_at_threshold`` (which epoch it falls in) and
        ``latency_to_epoch_onset_ms``. ``epoch_index`` is used only to
        measure ``current_injection_pA`` per sweep — not to filter spikes.
        Downstream callers such as ``analyze_fi_curve`` and
        ``extract_spike_features`` can filter to a specific epoch.

        Parameters
        ----------
        collection_name : str
        epoch_index : int
            Zero-based reference epoch index. Used to determine injected
            current amplitude per sweep. Fails loudly if it cannot be parsed.
        backend : str
            Spike finder backend: ``"derivative"`` (built-in, default) or
            ``"ipfx"`` (requires ipfx to be installed).
        lowpass_hz : float or None
            Low-pass filter cutoff for spike detection only.
        **spike_finder_kwargs
            Passed to the spike finder backend.

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
        Only spikes in ``epoch_index`` are counted (filtered by
        ``epoch_at_threshold``).

        Parameters
        ----------
        collection_name : str
        epoch_index : int
            Stimulus epoch to analyse. Spikes outside this epoch are ignored.
            Also used to retrieve epoch duration for rate calculation.
        spike_result_timestamp : str or None
            If None, uses the most recent spike detection result.

        Returns
        -------
        dict
            Timestamped result dict with per-sweep metrics, F-I curve lists,
            rheobase, slope (Hz/pA), R², peak instantaneous rates, and ISIs.
            Also appended to ``self.results["fi_curve"]``.
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

    def plot_fi_curve(
        self,
        collection_name: str | None = None,
        fi_result_timestamp: str | None = None,
        block: bool = False,
    ) -> None:
        """Open the F-I curve viewer popup.

        Parameters
        ----------
        collection_name : str or None
            Ignored (kept for API symmetry). Uses the most recent fi_curve
            result regardless of which collection it came from.
        fi_result_timestamp : str or None
            Reference a specific prior fi_curve result, or None for the
            most recent.
        block : bool
            If True, block until the window is closed. If False (default),
            show non-blocking.
        """
        from wholecell.gui.fi_viewer import FICurveViewer
        fi_entry = self._get_latest_result("fi_curve")
        viewer = FICurveViewer(
            fi_result=fi_entry["data"],
            title=f"{self.cell_id} — F-I Curve",
        )
        if block:
            viewer.exec()
        else:
            viewer.show()

    def extract_spike_features(
        self,
        collection_name: str,
        spike_result_timestamp: str | None = None,
        lowpass_hz: float | None = None,
        stimulus_epoch_index: int | None = None,
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
        stimulus_epoch_index : int or None
            If provided, cell-level summary features (first-AP features,
            rheobase, adaptation index) are computed only from spikes in
            this epoch. Should match the epoch_index used in find_spikes /
            analyze_fi_curve.

        Returns
        -------
        dict
            Timestamped result dict. The ``"spike_table"`` key contains all
            detected spikes (no epoch filtering) with columns including all
            detection fields, shape features, ``epoch_at_threshold``, and
            ``latency_to_epoch_onset_ms``. The ``"cell_level"`` key has
            first-spike features auto-collected from the rheobase AP.
        """
        from wholecell.analysis.spikes.features import run_feature_extraction
        sc = self.get_collection(collection_name)
        spikes = self._get_latest_result("spikes")
        params = {
            "collection_name": collection_name,
            "spike_result_timestamp": spike_result_timestamp,
            "lowpass_hz": lowpass_hz,
            "stimulus_epoch_index": stimulus_epoch_index,
        }
        result = run_feature_extraction(
            sc, spikes,
            lowpass_hz=lowpass_hz,
            stimulus_epoch_index=stimulus_epoch_index,
        )
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
        step_epoch_index: int,
        filepath: str | Path | None = None,
    ) -> pd.DataFrame:
        """Export the per-sweep summary table as a CSV.

        Always includes baseline_voltage_mV and step_current_pA for every
        sweep (computed from the recording). Passive and spike metrics are
        merged where available; columns are NaN-filled if not yet computed.

        Parameters
        ----------
        collection_name : str
        step_epoch_index : int
            Zero-based epoch index of the current step, used to compute
            baseline (epoch before the step) and step current (sweepC mean
            during the step).
        filepath : str or Path or None
            Defaults to ``{output_dir}/{cell_id}_sweep_summary_{ts}.csv``.

        Returns
        -------
        pd.DataFrame
        """
        sc = self.get_collection(collection_name)

        # Build base rows — always computed regardless of analysis state
        rows: dict[tuple[str, int], dict] = {}
        for ref in sc.sweeps:
            rec = self.recordings[ref.filename]
            rows[(ref.filename, ref.sweep_index)] = {
                "filename": ref.filename,
                "sweep_index": ref.sweep_index,
                "display_label": ref.display_label,
                "step_current_pA": _compute_sweep_step_current(
                    rec, ref.sweep_index, step_epoch_index
                ),
                "baseline_voltage_mV": _compute_sweep_baseline(
                    rec, ref.sweep_index, step_epoch_index
                ),
            }

        # Merge passive results (most recent entry wins per sweep)
        for entry in self.results.get("passive", []):
            data = entry.get("data", {})
            if data.get("type") == "averaged_passive":
                # GUI-averaged result: same metrics assigned to all source sweeps
                metrics = {k: data[k] for k in (
                    "input_resistance_MOhm", "time_constant_ms",
                    "sag_ratio", "sag_amplitude_mV", "sag_tau_ms",
                ) if k in data}
                for sw in data.get("source_sweeps", []):
                    key = (sw["filename"], sw["sweep_index"])
                    if key in rows:
                        rows[key].update(metrics)
            else:
                # Per-sweep result from Cell.analyze_passive
                for pr in data.get("per_sweep", []):
                    key = (pr["filename"], pr["sweep_index"])
                    if key in rows:
                        rows[key].update({k: pr[k] for k in (
                            "input_resistance_MOhm", "time_constant_ms",
                            "sag_ratio", "sag_amplitude_mV", "sag_tau_ms",
                        ) if k in pr})

        # Merge spike results (most recent entry wins per sweep)
        for entry in self.results.get("spikes", []):
            data = entry.get("data", {})
            for sr in data.get("per_sweep", []):
                key = (sr["filename"], sr["sweep_index"])
                if key in rows:
                    rows[key]["n_spikes"] = sr.get("n_spikes", len(sr.get("spikes", [])))
                    rows[key]["mean_firing_rate_hz"] = sr.get("mean_firing_rate_hz", float("nan"))

        # Assemble in collection order
        df = pd.DataFrame([rows[(ref.filename, ref.sweep_index)] for ref in sc.sweeps])

        col_order = [
            "filename", "sweep_index", "display_label",
            "step_current_pA", "baseline_voltage_mV",
            "input_resistance_MOhm", "time_constant_ms",
            "sag_ratio", "sag_amplitude_mV", "sag_tau_ms",
            "n_spikes", "mean_firing_rate_hz",
        ]
        present = [c for c in col_order if c in df.columns]
        extra = [c for c in df.columns if c not in col_order]
        df = df[present + extra]

        if filepath is None:
            ts = _timestamp().replace(":", "").replace("-", "").replace("T", "_")
            filepath = self.output_dir / f"{self.cell_id}_sweep_summary_{ts}.csv"

        df.to_csv(filepath, index=False)
        self._log("export_sweep_summary", {"filepath": str(filepath), "collection": collection_name})
        print(f"Sweep summary saved: {filepath}")
        return df

    def export_cell_summary(
        self,
        filepath: str | Path | None = None,
    ) -> dict:
        """Export cell-level summary scalars and curves to JSON.

        Writes whatever analysis results are available. Missing sections are
        omitted rather than raising an error. Variable-length data (F-I curve,
        adaptation curves) lives under named keys; fixed scalars at top level.

        Parameters
        ----------
        filepath : str or Path or None
            Defaults to ``{output_dir}/{cell_id}_cell_summary.json``.

        Returns
        -------
        dict
            The exported summary dict.
        """
        summary: dict = {
            "cell_id": self.cell_id,
            "notes": self.notes,
            "exported_at": _timestamp(),
        }

        # Passive section
        if self.results.get("passive"):
            data = self.results["passive"][-1]["data"]
            if data.get("type") == "averaged_passive":
                summary["passive"] = {k: v for k, v in data.items()
                                       if k not in ("type", "_tau_fit")}
            else:
                summary["passive"] = data.get("cell_level", {})
                summary["passive"]["source"] = "per_sweep_average"

        # Spike features section
        if self.results.get("spike_features"):
            data = self.results["spike_features"][-1]["data"]
            spike_table = data.get("spike_table", [])
            if spike_table:
                thresholds = [s["threshold_voltage_mV"] for s in spike_table
                              if not np.isnan(s.get("threshold_voltage_mV", float("nan")))]
                half_widths = [s["half_width_ms"] for s in spike_table
                               if not np.isnan(s.get("half_width_ms", float("nan")))]
                first = spike_table[0]
                summary["spikes"] = {
                    "n_spikes_total": len(spike_table),
                    "first_spike_threshold_mV": first.get("threshold_voltage_mV"),
                    "first_spike_half_width_ms": first.get("half_width_ms"),
                    "mean_threshold_mV": float(np.mean(thresholds)) if thresholds else None,
                    "mean_half_width_ms": float(np.mean(half_widths)) if half_widths else None,
                }

        # F-I curve section
        if self.results.get("fi_curve"):
            summary["fi_curve"] = self.results["fi_curve"][-1]["data"]

        if filepath is None:
            filepath = self.output_dir / f"{self.cell_id}_cell_summary.json"
        filepath = Path(filepath)

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self._log("export_cell_summary", {"filepath": str(filepath)})
        print(f"Cell summary saved: {filepath}")
        return summary

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


def _compute_sweep_baseline(rec, sweep_index: int, step_epoch_index: int) -> float:
    """Return mean baseline voltage from the epoch immediately before the step.

    Falls back to the first 50 ms of the step epoch if no prior epoch exists.
    """
    if step_epoch_index > 0:
        try:
            ep = rec.get_epoch(sweep_index, step_epoch_index - 1)
            _, v, _ = rec.get_sweep_arrays(sweep_index)
            seg = v[ep.start_sample:ep.end_sample]
            return float(np.mean(seg)) if len(seg) > 0 else float("nan")
        except (IndexError, RuntimeError):
            pass
    try:
        ep = rec.get_epoch(sweep_index, step_epoch_index)
        _, v, _ = rec.get_sweep_arrays(sweep_index)
        n_50ms = int(0.05 * rec.sampling_rate_hz)
        seg = v[ep.start_sample:ep.start_sample + n_50ms]
        return float(np.mean(seg)) if len(seg) > 0 else float("nan")
    except Exception:
        return float("nan")


def _compute_sweep_step_current(rec, sweep_index: int, step_epoch_index: int) -> float:
    """Return mean command current during the step epoch (from sweepC)."""
    try:
        cmd = rec.get_command_waveform(sweep_index)
        ep = rec.get_epoch(sweep_index, step_epoch_index)
        window = cmd[ep.start_sample:ep.end_sample]
        return float(np.mean(window)) if len(window) > 0 else float("nan")
    except Exception:
        return float("nan")
