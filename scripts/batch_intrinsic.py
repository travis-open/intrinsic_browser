"""
batch_intrinsic.py
------------------
Apply a consistent intrinsic-properties analysis to a list of cells described in
a spreadsheet, reproducing the trace_viewer GUI buttons for each protocol file.

Per cell row, for every protocol column that names a single existing ABF file:

    small_steps_file  ->  Find Spikes  +  F-I Curve
    free_run_file     ->  Measure V_rest
    ramp_file         ->  Find Spikes (same params)  +  Analyze Ramp APs
    sagIh_file        ->  Analyze Each Sweep (Passive)
    hyperpol_file     ->  Average & Analyze Subthreshold

Outputs (into --output-dir, default <repo>/test_outputs):
    <cell_id>_spikes.csv         detection-only spike table for small_steps (GUI-faithful)
    <cell_id>_cell_summary.json  Cell.export_cell_summary — all analysis sections
    batch_summary.csv            one row per cell: status + curated cell-level features

This mirrors the GUI: derivative spike backend, filter off (lowpass_hz=None),
step/ramp epoch from find_step_epoch.

Sheet snapshot
--------------
``scripts/cells.csv`` is a CSV export of
https://docs.google.com/spreadsheets/d/1ja0srW9ut32qgVTGo8S9QUov1KxRvyKNUySP9Rx-uOM
(tab gid=0). Refresh it with:
    https://docs.google.com/spreadsheets/d/1ja0srW9ut32qgVTGo8S9QUov1KxRvyKNUySP9Rx-uOM/export?format=csv&gid=0

Running
-------
Must run inside the activated ``intrinsic_props`` conda env (bare ``python.exe``
from the env dir fails to put MKL's DLLs on PATH and numpy LAPACK crashes):

    conda run --no-capture-output -n intrinsic_props python scripts/batch_intrinsic.py

or ``conda activate intrinsic_props`` first, then ``python scripts/batch_intrinsic.py``.

Examples
--------
    # Test run: first 3 cells, all protocols, dvdt 3.0, peak window 10.0
    conda run --no-capture-output -n intrinsic_props python scripts/batch_intrinsic.py

    # Only the current-step protocols, explicit cells
    ... python scripts/batch_intrinsic.py --protocols small_steps,sagIh,hyperpol \
        --cells 20260607_cell7_JMT,20260610_cell5_JMT

    # Everything in the sheet
    ... python scripts/batch_intrinsic.py --limit 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from wholecell.core.cell import Cell
from wholecell.io.abf_reader import find_step_epoch
from wholecell.analysis.vrest import run_vrest_analysis
from wholecell.analysis.ramp import run_ramp_analysis
from wholecell.analysis.passive import (
    estimate_input_resistance,
    fit_time_constant,
    estimate_sag,
)


# protocol name -> sheet column. Order = execution order (small_steps first: it
# feeds the F-I curve and its spikes result must be captured before ramp
# detection appends another "spikes" entry).
PROTOCOL_COLUMN = {
    "small_steps": "small_steps_file",
    "ramp": "ramp_file",
    "sagIh": "sagIh_file",
    "hyperpol": "hyperpol_file",
    "free_run": "free_run_file",
}
ALL_PROTOCOLS = list(PROTOCOL_COLUMN)

# Column order of the GUI spike-table export (trace_viewer._on_export_spike_table),
# i.e. run_spike_detection per-spike dicts minus "display_label".
SPIKE_TABLE_COLUMNS = [
    "filename",
    "sweep_index",
    "spike_index_in_sweep",
    "peak_time_s",
    "peak_voltage_mV",
    "threshold_time_s",
    "threshold_voltage_mV",
    "trough_time_s",
    "trough_voltage_mV",
    "backend",
    "current_at_threshold_pA",
    "slow_ahp_voltage_mV",
    "slow_ahp_time_s",
    "epoch_at_threshold",
    "latency_to_epoch_onset_ms",
    "sweep_current_injection_pA",
]

IDENTITY_COLUMNS = [
    "cell_id",
    "small_steps_file",
    "n_sweeps",
    "epoch_index",
    "n_spikes_total",
    "protocols_run",
    "protocol_status",
    "spikes_csv",
    "summary_json",
    "status",
]

# (output column, summary section, key within that section)
# sections: fi = fi_curve.cell_level, vrest = v_rest (flat), sag_passive =
# passive_range (flat cell_level), avg_passive = passive_repeated_step (flat),
# ramp = ramp_evoked_APs.cell_level
CELL_LEVEL_COLUMNS = [
    ("fi__rheobase_pA", "fi", "rheobase_pA"),
    ("fi__max_firing_rate_hz", "fi", "max_firing_rate_hz"),
    ("fi__current_at_max_firing_pA", "fi", "current_at_max_firing_pA"),
    ("fi__dep_block_current_pA", "fi", "dep_block_current_pA"),
    ("fi__fi_slope_hz_per_pA", "fi", "fi_slope_hz_per_pA"),
    ("fi__fi_slope_r2", "fi", "fi_slope_r2"),
    ("vrest__v_rest_mV", "vrest", "v_rest_mV"),
    ("vrest__v_rest_std_mV", "vrest", "v_rest_std_mV"),
    ("vrest__initial_voltage_mV", "vrest", "initial_voltage_mV"),
    ("vrest__ap_detected", "vrest", "ap_detected"),
    ("vrest__n_aps_total", "vrest", "n_aps_total"),
    ("vrest__initial_mean_isi_s", "vrest", "initial_mean_isi_s"),
    ("vrest__initial_isi_cv", "vrest", "initial_isi_cv"),
    ("sag_passive__mean_input_resistance_MOhm", "sag_passive", "mean_input_resistance_MOhm"),
    ("sag_passive__mean_time_constant_ms", "sag_passive", "mean_time_constant_ms"),
    ("sag_passive__mean_time_constant_r2", "sag_passive", "mean_time_constant_r2"),
    ("sag_passive__sag_ratio", "sag_passive", "sag_ratio"),
    ("sag_passive__sag_amplitude_mV", "sag_passive", "sag_amplitude_mV"),
    ("sag_passive__sag_tau_ms", "sag_passive", "sag_tau_ms"),
    ("avg_passive__step_current_pA", "avg_passive", "step_current_pA"),
    ("avg_passive__input_resistance_MOhm", "avg_passive", "input_resistance_MOhm"),
    ("avg_passive__time_constant_ms", "avg_passive", "time_constant_ms"),
    ("avg_passive__time_constant_r2", "avg_passive", "time_constant_r2"),
    ("avg_passive__sag_ratio", "avg_passive", "sag_ratio"),
    ("ramp__mean_threshold_voltage_mV", "ramp", "mean_threshold_voltage_mV"),
    ("ramp__mean_peak_voltage_mV", "ramp", "mean_peak_voltage_mV"),
    ("ramp__mean_half_width_ms", "ramp", "mean_half_width_ms"),
    ("ramp__mean_current_at_threshold_pA", "ramp", "mean_current_at_threshold_pA"),
    ("ramp__n_sweeps_analyzed", "ramp", "n_sweeps_analyzed"),
]

BATCH_SUMMARY_COLUMNS = IDENTITY_COLUMNS + [c[0] for c in CELL_LEVEL_COLUMNS]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def _looks_like_multi_file(value: str) -> bool:
    """True if a ``*_file`` cell names more than one ABF (e.g. apamin pairs)."""
    return "[" in value or "," in value


def safe_find_step_epoch(path: Path, fallback: int = 1) -> int:
    """``find_step_epoch`` with the GUI's fallback."""
    try:
        return int(find_step_epoch(str(path)))
    except Exception:
        return fallback


def _all_sweeps(name: str, rec) -> list[dict]:
    return [{"filename": name, "sweep_index": i} for i in range(rec.n_sweeps)]


def build_spike_table(spike_result_data: dict) -> pd.DataFrame:
    """Replicate trace_viewer._on_export_spike_table against a stored result.

    ``spike_result_data`` is a ``cell.results["spikes"][k]["data"]`` dict.
    """
    rows: list[dict] = []
    for sweep in spike_result_data.get("per_sweep", []):
        rows.extend(sweep.get("spikes", []))

    if not rows:
        return pd.DataFrame(columns=SPIKE_TABLE_COLUMNS)

    df = (
        pd.DataFrame(rows)
        .drop(columns=["display_label"], errors="ignore")
        .sort_values(["filename", "sweep_index", "spike_index_in_sweep"])
        .reset_index(drop=True)
    )
    ordered = [c for c in SPIKE_TABLE_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in SPIKE_TABLE_COLUMNS]
    return df[ordered + extra]


def flatten_cell_level(summary: dict) -> dict:
    """Pull the curated cell-level scalars out of a Cell.export_cell_summary dict."""
    src = {
        "fi": summary.get("fi_curve", {}).get("cell_level", {}) or {},
        "vrest": summary.get("v_rest", {}) or {},
        "sag_passive": summary.get("passive_range", {}) or {},
        "avg_passive": summary.get("passive_repeated_step", {}) or {},
        "ramp": summary.get("ramp_evoked_APs", {}).get("cell_level", {}) or {},
    }
    out: dict = {}
    for col, section, key in CELL_LEVEL_COLUMNS:
        val = src.get(section, {}).get(key, "")
        out[col] = "" if val is None else val
    return out


# ---------------------------------------------------------------------------
# Average & Analyze Subthreshold — ported from trace_viewer._run_average_analysis
# ---------------------------------------------------------------------------

def _avg_baseline(rec, sweep_index: int, epoch_index: int,
                  t: np.ndarray, v: np.ndarray) -> float:
    """Baseline voltage for the averaged trace (trace_viewer._estimate_avg_baseline)."""
    if epoch_index > 0:
        try:
            ep_pre = rec.get_epoch(sweep_index, epoch_index - 1)
            return float(np.mean(v[ep_pre.start_sample:ep_pre.end_sample]))
        except (IndexError, RuntimeError):
            pass
    ep = rec.get_epoch(sweep_index, epoch_index)
    n_50ms = int(0.05 * rec.sampling_rate_hz)
    return float(np.mean(v[ep.start_sample:ep.start_sample + n_50ms]))


def run_average_subthreshold(cell: Cell, name: str, epoch_index: int,
                             lowpass_hz: float | None = None) -> dict:
    """Port of trace_viewer._run_average_analysis for one all-sweeps collection.

    Requires every sweep to share the same step level and duration in
    ``epoch_index`` (as the GUI's ``_validate_step_command`` enforces).
    Returns the ``result_data`` dict stored under ``passive_repeated_step``.
    """
    col = cell.collections[name]
    rec = cell.recordings[name]
    refs = list(col.sweeps)
    if not refs:
        raise ValueError("collection has no sweeps")

    levels, durations = [], []
    for ref in refs:
        ep = rec.get_epoch(ref.sweep_index, epoch_index)
        levels.append(ep.level)
        durations.append(ep.end_sample - ep.start_sample)

    ref_level, ref_dur = levels[0], durations[0]
    bad_level = [refs[i].sweep_index for i, lv in enumerate(levels)
                 if abs(lv - ref_level) > 0.5]
    bad_dur = [refs[i].sweep_index for i, d in enumerate(durations) if d != ref_dur]
    problems = []
    if bad_level:
        problems.append(
            f"step amplitude differs in sweeps {bad_level} (expected {ref_level:.1f} pA)")
    if bad_dur:
        problems.append(f"step duration differs in sweeps {bad_dur}")
    if problems:
        raise ValueError("; ".join(problems))

    step_current_pA = float(levels[0])

    arrays: list[np.ndarray] = []
    t_ref: np.ndarray | None = None
    for ref in refs:
        t, v, _ = col.get_sweep_arrays(ref, lowpass_hz=lowpass_hz)
        arrays.append(v)
        if t_ref is None:
            t_ref = t
    avg_v = np.mean(np.stack(arrays, axis=0), axis=0)

    ep = rec.get_epoch(refs[0].sweep_index, epoch_index)
    sl = slice(ep.start_sample, ep.end_sample)
    ep_time, ep_voltage = t_ref[sl], avg_v[sl]

    baseline = _avg_baseline(rec, refs[0].sweep_index, epoch_index, t_ref, avg_v)

    rin = estimate_input_resistance(baseline, ep_voltage, step_current_pA)
    tau_ms, _, _, _, tau_r2 = fit_time_constant(ep_time, ep_voltage, baseline)
    sag = estimate_sag(ep_time, ep_voltage, baseline)

    return {
        "type": "averaged_passive",
        "source_sweeps": [{"filename": r.filename, "sweep_index": r.sweep_index}
                          for r in refs],
        "n_sweeps_averaged": len(refs),
        "step_current_pA": step_current_pA,
        "baseline_voltage_mV": float(baseline),
        "input_resistance_MOhm": float(rin),
        "time_constant_ms": float(tau_ms),
        "time_constant_r2": float(tau_r2),
        "sag_ratio": float(sag["sag_ratio"]),
        "sag_amplitude_mV": float(sag["sag_amplitude_mV"]),
        "sag_tau_ms": float(sag["sag_tau_ms"]),
    }


# ---------------------------------------------------------------------------
# per-cell processing
# ---------------------------------------------------------------------------

def _warn_missing(cell_id: str, protocol: str, folder: str, filename: str) -> None:
    print(
        f"WARNING  {cell_id}  {protocol}  file not found:\n"
        f"         folder: {folder}\n"
        f"         file:   {filename}",
        file=sys.stderr,
    )


def process_cell(row: pd.Series, protocols: list[str], out_dir: Path, args,
                 missing_accum: list[tuple]) -> dict:
    cell_id = str(row["cell_id"]).strip()
    abf_folder = "" if _blank(row.get("abf_folder")) else str(row["abf_folder"]).strip()
    notes = "" if _blank(row.get("notes")) else str(row["notes"]).strip()

    status_row = {k: "" for k in BATCH_SUMMARY_COLUMNS}
    status_row["cell_id"] = cell_id
    status_row["small_steps_file"] = (
        "" if _blank(row.get("small_steps_file")) else str(row["small_steps_file"]).strip()
    )

    def resolve(protocol: str) -> tuple[Path | None, str | None]:
        """Return (path, reason). path is None when the protocol can't run."""
        raw = row.get(PROTOCOL_COLUMN[protocol])
        val = "" if _blank(raw) else str(raw).strip()
        if not val:
            return None, "skipped"
        if _looks_like_multi_file(val):
            return None, "multi-file, skipped"
        if not abf_folder:
            _warn_missing(cell_id, protocol, "<blank abf_folder>", val)
            missing_accum.append((cell_id, protocol, "<blank abf_folder>", val))
            return None, "no abf_folder"
        p = Path(abf_folder) / val
        if not p.exists():
            _warn_missing(cell_id, protocol, abf_folder, val)
            missing_accum.append((cell_id, protocol, abf_folder, val))
            return None, "file not found"
        return p, None

    cell = Cell(cell_id=cell_id, output_dir=out_dir, notes=notes)
    ran: list[str] = []
    notes_by_protocol: dict[str, str] = {}
    small_steps_spike_data: dict | None = None

    # ---- small_steps: Find Spikes + F-I Curve -------------------------------
    if "small_steps" in protocols:
        p, reason = resolve("small_steps")
        if p is None:
            status_row["protocol_status"] = f"small_steps: {reason}"
            status_row["status"] = f"error: small_steps {reason}"
            return status_row
        try:
            rec = cell.add_recording(p)
            name = rec.filename
            epoch = args.epoch_index if args.epoch_index is not None else safe_find_step_epoch(p)
            cell.create_sweep_collection(name, _all_sweeps(name, rec))
            cell.find_spikes(name, epoch, dvdt_detection_mVms=args.dvdt,
                             peak_search_window_ms=args.peak_window, lowpass_hz=args.lowpass)
            small_steps_spike_data = cell.results["spikes"][-1]["data"]
            cell.analyze_fi_curve(name, epoch)
            status_row["n_sweeps"] = rec.n_sweeps
            status_row["epoch_index"] = epoch
            ran.append("small_steps")
        except Exception as exc:
            traceback.print_exc()
            status_row["protocol_status"] = f"small_steps: error: {exc}"
            status_row["status"] = f"error: small_steps: {exc}"
            return status_row

    # ---- ramp: Find Spikes (same params) + Analyze Ramp APs ---------------
    if "ramp" in protocols:
        p, reason = resolve("ramp")
        if p is None:
            notes_by_protocol["ramp"] = reason
        else:
            try:
                rec = cell.add_recording(p)
                name = rec.filename
                epoch = safe_find_step_epoch(p)
                cell.create_sweep_collection(name, _all_sweeps(name, rec))
                cell.find_spikes(name, epoch, dvdt_detection_mVms=args.dvdt,
                                 peak_search_window_ms=args.peak_window,
                                 lowpass_hz=args.lowpass)
                result = run_ramp_analysis(
                    cell.collections[name], epoch, cell.results["spikes"][-1],
                    lowpass_hz=args.lowpass,
                )
                cell._store_result("ramp_evoked_APs", result, {
                    "collection_name": name, "epoch_index": epoch,
                    "source": "batch_intrinsic",
                })
                ran.append("ramp")
            except Exception as exc:
                traceback.print_exc()
                notes_by_protocol["ramp"] = f"error: {exc}"

    # ---- sagIh: Analyze Each Sweep (Passive) -----------------------------
    if "sagIh" in protocols:
        p, reason = resolve("sagIh")
        if p is None:
            notes_by_protocol["sagIh"] = reason
        else:
            try:
                rec = cell.add_recording(p)
                name = rec.filename
                epoch = safe_find_step_epoch(p)
                cell.create_sweep_collection(name, _all_sweeps(name, rec))
                cell.analyze_passive(name, epoch)  # stores "passive_range"
                ran.append("sagIh")
            except Exception as exc:
                traceback.print_exc()
                notes_by_protocol["sagIh"] = f"error: {exc}"

    # ---- hyperpol: Average & Analyze Subthreshold -----------------------
    if "hyperpol" in protocols:
        p, reason = resolve("hyperpol")
        if p is None:
            notes_by_protocol["hyperpol"] = reason
        else:
            try:
                rec = cell.add_recording(p)
                name = rec.filename
                epoch = safe_find_step_epoch(p)
                cell.create_sweep_collection(name, _all_sweeps(name, rec))
                result = run_average_subthreshold(cell, name, epoch, lowpass_hz=args.lowpass)
                cell._store_result("passive_repeated_step", result, {
                    "collection_name": name, "epoch_index": epoch,
                    "n_sweeps_averaged": result["n_sweeps_averaged"],
                    "source": "batch_intrinsic",
                })
                ran.append("hyperpol")
            except Exception as exc:
                traceback.print_exc()
                notes_by_protocol["hyperpol"] = f"error: {exc}"

    # ---- free_run: Measure V_rest --------------------------------------
    if "free_run" in protocols:
        p, reason = resolve("free_run")
        if p is None:
            notes_by_protocol["free_run"] = reason
        else:
            try:
                rec = cell.add_recording(p)
                name = rec.filename
                cell.create_sweep_collection(name, _all_sweeps(name, rec))
                result = run_vrest_analysis(
                    cell.collections[name], lowpass_hz=args.lowpass,
                    dvdt_detection_mVms=args.dvdt,
                    peak_search_window_ms=args.peak_window,
                )
                cell._store_result("v_rest", result, {
                    "collection_name": name, "source": "batch_intrinsic",
                })
                ran.append("free_run")
            except Exception as exc:
                traceback.print_exc()
                notes_by_protocol["free_run"] = f"error: {exc}"

    # ---- exports -------------------------------------------------------
    if small_steps_spike_data is not None:
        spike_df = build_spike_table(small_steps_spike_data)
        spikes_csv = out_dir / f"{cell_id}_spikes.csv"
        spike_df.to_csv(spikes_csv, index=False)
        status_row["n_spikes_total"] = len(spike_df)
        status_row["spikes_csv"] = str(spikes_csv)

    summary_json = out_dir / f"{cell_id}_cell_summary.json"
    summary = cell.export_cell_summary(filepath=summary_json)
    status_row["summary_json"] = str(summary_json)
    status_row.update(flatten_cell_level(summary))

    status_row["protocols_run"] = ",".join(ran)
    status_row["protocol_status"] = ", ".join(
        f"{k}: {v}" for k, v in notes_by_protocol.items()
    )
    status_row["status"] = "ok" if ran else "error: no protocol produced a result"
    return status_row


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def select_rows(sheet: pd.DataFrame, cells: list[str] | None, limit: int) -> pd.DataFrame:
    if "cell_id" not in sheet.columns:
        raise SystemExit("Sheet has no 'cell_id' column.")

    if cells:
        missing = [c for c in cells if c not in set(sheet["cell_id"])]
        if missing:
            raise SystemExit(f"cell_id(s) not found in sheet: {missing}")
        rows = sheet[sheet["cell_id"].isin(cells)].copy()
        rows["_order"] = rows["cell_id"].map({c: i for i, c in enumerate(cells)})
        return rows.sort_values("_order").drop(columns="_order")

    if "small_steps_file" not in sheet.columns:
        raise SystemExit("Sheet has no 'small_steps_file' column for default row selection.")
    has_file = sheet[~sheet["small_steps_file"].map(_blank)].copy()
    if limit and limit > 0:
        return has_file.head(limit)
    return has_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, default=REPO_ROOT / "scripts" / "cells.csv",
                        help="CSV snapshot of the cell sheet (default: scripts/cells.csv)")
    parser.add_argument("--protocols", default=",".join(ALL_PROTOCOLS),
                        help="Comma list from: " + ",".join(ALL_PROTOCOLS) + " (default: all)")
    parser.add_argument("--cells", default="",
                        help="Comma-separated cell_id list. Overrides --limit.")
    parser.add_argument("--limit", type=int, default=3,
                        help="First N rows with a non-empty small_steps_file (default: 3; 0 = no limit)")
    parser.add_argument("--dvdt", type=float, default=3.0,
                        help="dV/dt detection threshold, mV/ms (default: 3.0)")
    parser.add_argument("--peak-window", type=float, default=10.0,
                        help="Peak search window, ms (default: 10.0)")
    parser.add_argument("--lowpass", type=float, default=None,
                        help="Lowpass cutoff Hz for analyses (default: none, matches GUI)")
    parser.add_argument("--epoch-index", type=int, default=None,
                        help="Force the small_steps step epoch (default: find_step_epoch)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "test_outputs",
                        help="Where to write outputs (default: <repo>/test_outputs)")
    args = parser.parse_args(argv)

    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    unknown = [p for p in protocols if p not in ALL_PROTOCOLS]
    if unknown:
        raise SystemExit(f"Unknown --protocols: {unknown}. Valid: {ALL_PROTOCOLS}")

    if not args.sheet.exists():
        raise SystemExit(f"Sheet not found: {args.sheet}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = pd.read_csv(args.sheet, dtype=str)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()] or None
    rows = select_rows(sheet, cells, args.limit)
    if rows.empty:
        raise SystemExit("No matching cells to process.")

    print(f"Batch intrinsic analysis  |  protocols={protocols}  "
          f"dvdt={args.dvdt} mV/ms  peak_window={args.peak_window} ms")
    print(f"Output: {out_dir}")
    print(f"Cells: {list(rows['cell_id'])}\n")

    results: list[dict] = []
    missing_accum: list[tuple] = []
    for _, row in rows.iterrows():
        cell_id = str(row["cell_id"]).strip()
        print(f"--- {cell_id} ---")
        try:
            status_row = process_cell(row, protocols, out_dir, args, missing_accum)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            traceback.print_exc()
            status_row = {k: "" for k in BATCH_SUMMARY_COLUMNS}
            status_row["cell_id"] = cell_id
            status_row["status"] = f"error: {exc}"
        results.append(status_row)
        extra = ""
        if status_row["status"] == "ok":
            extra = (f"  | ran: {status_row['protocols_run']}"
                     f"  spikes={status_row['n_spikes_total']}")
            if status_row["protocol_status"]:
                extra += f"  ({status_row['protocol_status']})"
        print(f"    {status_row['status']}{extra}")

    summary_df = pd.DataFrame(results, columns=BATCH_SUMMARY_COLUMNS)
    batch_csv = out_dir / "batch_summary.csv"
    summary_df.to_csv(batch_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n=== batch summary ===")
    print(summary_df[["cell_id", "n_sweeps", "epoch_index", "n_spikes_total",
                      "protocols_run", "status"]].to_string(index=False))

    if missing_accum:
        print(f"\n=== {len(missing_accum)} file(s) not found ===")
        for cid, proto, folder, fname in missing_accum:
            print(f"  {cid}  {proto}:  {folder}\\{fname}")

    print(f"\nWrote {batch_csv}")
    n_ok = int((summary_df["status"] == "ok").sum())
    n_err = int(summary_df["status"].str.startswith("error").sum())
    print(f"{n_ok} ok, {n_err} error  ({len(missing_accum)} missing file warnings)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
