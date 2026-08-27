"""
batch_intrinsic.py
------------------
Apply a consistent intrinsic-properties analysis to a list of cells described in
a spreadsheet, reproducing the trace_viewer GUI sequence:

    Find Spikes  ->  F-I Curve  ->  Export Spike Table (CSV)  ->  Export Cell Summary (JSON)

For each selected cell it loads one ABF file (named by a chosen ``*_file`` column),
builds an all-sweeps collection, detects spikes with the derivative backend,
builds the F-I curve, then writes ``<cell_id>_spikes.csv`` and
``<cell_id>_cell_summary.json`` to the output directory.

This mirrors the GUI exactly: spike detection only (no feature extraction), GUI
default filter off (``lowpass_hz=None``), step epoch from ``find_step_epoch``.

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
    # Test run: small_steps_file, first 3 cells, dvdt 3.0, peak window 10.0
    conda run --no-capture-output -n intrinsic_props python scripts/batch_intrinsic.py

    # A different protocol column and explicit cells
    ... python scripts/batch_intrinsic.py --file-column ramp_file \
        --cells 20260607_cell7_JMT,20260610_cell5_JMT

    # All cells that have a small_steps_file
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

import pandas as pd

from wholecell.core.cell import Cell
from wholecell.io.abf_reader import find_step_epoch


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

BATCH_SUMMARY_COLUMNS = [
    "cell_id",
    "file",
    "n_sweeps",
    "epoch_index",
    "n_spikes_total",
    "rheobase_pA",
    "max_firing_rate_hz",
    "fi_slope_hz_per_pA",
    "spikes_csv",
    "summary_json",
    "status",
]


def _looks_like_multi_file(value: str) -> bool:
    """True if a ``*_file`` cell names more than one ABF (e.g. apamin pairs)."""
    return "[" in value or "," in value


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def safe_find_step_epoch(path: Path) -> int:
    """``find_step_epoch`` with the GUI's fallback to epoch index 1."""
    try:
        return int(find_step_epoch(str(path)))
    except Exception:
        return 1


def build_spike_table(spike_result_data: dict) -> pd.DataFrame:
    """Replicate trace_viewer._on_export_spike_table against a stored result.

    ``spike_result_data`` is ``cell.results["spikes"][-1]["data"]``.
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
    # Stable, GUI-matching column order; keep any unexpected extras at the end.
    ordered = [c for c in SPIKE_TABLE_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in SPIKE_TABLE_COLUMNS]
    return df[ordered + extra]


def select_rows(sheet: pd.DataFrame, file_column: str, cells: list[str] | None,
                limit: int) -> pd.DataFrame:
    if "cell_id" not in sheet.columns:
        raise SystemExit("Sheet has no 'cell_id' column.")
    if file_column not in sheet.columns:
        raise SystemExit(
            f"Sheet has no '{file_column}' column. "
            f"Available *_file columns: {[c for c in sheet.columns if c.endswith('_file')]}"
        )

    if cells:
        missing = [c for c in cells if c not in set(sheet["cell_id"])]
        if missing:
            raise SystemExit(f"cell_id(s) not found in sheet: {missing}")
        rows = sheet[sheet["cell_id"].isin(cells)].copy()
        # preserve the order the user asked for
        rows["_order"] = rows["cell_id"].map({c: i for i, c in enumerate(cells)})
        return rows.sort_values("_order").drop(columns="_order")

    has_file = sheet[~sheet[file_column].map(_blank)].copy()
    if limit and limit > 0:
        return has_file.head(limit)
    return has_file


def process_cell(row: pd.Series, file_column: str, out_dir: Path, args) -> dict:
    cell_id = str(row["cell_id"]).strip()
    file_value = str(row[file_column]).strip()
    status_row = {k: "" for k in BATCH_SUMMARY_COLUMNS}
    status_row["cell_id"] = cell_id
    status_row["file"] = file_value

    if _blank(row.get("abf_folder")):
        status_row["status"] = "skip: no abf_folder"
        return status_row
    if _blank(file_value):
        status_row["status"] = f"skip: no {file_column}"
        return status_row
    if _looks_like_multi_file(file_value):
        status_row["status"] = f"skip: {file_column} names multiple files"
        return status_row

    abf_path = Path(str(row["abf_folder"]).strip()) / file_value
    if not abf_path.exists():
        status_row["status"] = f"skip: file not found ({abf_path})"
        return status_row

    notes = "" if _blank(row.get("notes")) else str(row["notes"]).strip()
    cell = Cell(cell_id=cell_id, output_dir=out_dir, notes=notes)
    rec = cell.add_recording(abf_path)
    name = rec.filename

    epoch_index = args.epoch_index if args.epoch_index is not None else safe_find_step_epoch(abf_path)

    cell.create_sweep_collection(
        name,
        [{"filename": name, "sweep_index": i} for i in range(rec.n_sweeps)],
    )

    # --- Find Spikes (GUI: derivative backend, filter off) ---
    cell.find_spikes(
        name,
        epoch_index,
        dvdt_detection_mVms=args.dvdt,
        peak_search_window_ms=args.peak_window,
        lowpass_hz=args.lowpass,
    )

    # --- F-I Curve ---
    cell.analyze_fi_curve(name, epoch_index)

    # --- Export: spike table CSV (GUI-faithful, detection fields only) ---
    spike_data = cell.results["spikes"][-1]["data"]
    spike_df = build_spike_table(spike_data)
    spikes_csv = out_dir / f"{cell_id}_spikes.csv"
    spike_df.to_csv(spikes_csv, index=False)

    # --- Export: cell summary JSON ---
    summary_json = out_dir / f"{cell_id}_cell_summary.json"
    summary = cell.export_cell_summary(filepath=summary_json)

    fi_cell_level = summary.get("fi_curve", {}).get("cell_level", {})
    status_row.update({
        "n_sweeps": rec.n_sweeps,
        "epoch_index": epoch_index,
        "n_spikes_total": len(spike_df),
        "rheobase_pA": fi_cell_level.get("rheobase_pA", ""),
        "max_firing_rate_hz": fi_cell_level.get("max_firing_rate_hz", ""),
        "fi_slope_hz_per_pA": fi_cell_level.get("fi_slope_hz_per_pA", ""),
        "spikes_csv": str(spikes_csv),
        "summary_json": str(summary_json),
        "status": "ok",
    })
    return status_row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, default=REPO_ROOT / "scripts" / "cells.csv",
                        help="CSV snapshot of the cell sheet (default: scripts/cells.csv)")
    parser.add_argument("--file-column", default="small_steps_file",
                        help="Which *_file column to analyze (default: small_steps_file)")
    parser.add_argument("--cells", default="",
                        help="Comma-separated cell_id list. Overrides --limit.")
    parser.add_argument("--limit", type=int, default=3,
                        help="Process the first N rows with a non-empty file column "
                             "(default: 3; 0 = no limit)")
    parser.add_argument("--dvdt", type=float, default=3.0,
                        help="dV/dt detection threshold, mV/ms (default: 3.0)")
    parser.add_argument("--peak-window", type=float, default=10.0,
                        help="Peak search window, ms (default: 10.0)")
    parser.add_argument("--lowpass", type=float, default=None,
                        help="Lowpass cutoff Hz for detection (default: none, matches GUI)")
    parser.add_argument("--epoch-index", type=int, default=None,
                        help="Force a step epoch index (default: per-file find_step_epoch)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "test_outputs",
                        help="Where to write outputs (default: <repo>/test_outputs)")
    args = parser.parse_args(argv)

    if not args.sheet.exists():
        raise SystemExit(f"Sheet not found: {args.sheet}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = pd.read_csv(args.sheet, dtype=str)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()] or None
    rows = select_rows(sheet, args.file_column, cells, args.limit)

    if rows.empty:
        raise SystemExit("No matching cells to process.")

    print(f"Batch intrinsic analysis  |  column={args.file_column}  "
          f"dvdt={args.dvdt} mV/ms  peak_window={args.peak_window} ms")
    print(f"Output: {out_dir}")
    print(f"Cells: {list(rows['cell_id'])}\n")

    results: list[dict] = []
    for _, row in rows.iterrows():
        cell_id = str(row["cell_id"]).strip()
        print(f"--- {cell_id} ---")
        try:
            status_row = process_cell(row, args.file_column, out_dir, args)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            traceback.print_exc()
            status_row = {k: "" for k in BATCH_SUMMARY_COLUMNS}
            status_row["cell_id"] = cell_id
            status_row["status"] = f"error: {exc}"
        results.append(status_row)
        print(f"    {status_row['status']}"
              + (f"  | spikes={status_row['n_spikes_total']}"
                 f"  rheobase={status_row['rheobase_pA']}"
                 f"  max_rate={status_row['max_firing_rate_hz']}"
                 if status_row["status"] == "ok" else ""))

    summary_df = pd.DataFrame(results, columns=BATCH_SUMMARY_COLUMNS)
    batch_csv = out_dir / "batch_summary.csv"
    summary_df.to_csv(batch_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n=== batch summary ===")
    print(summary_df[["cell_id", "n_sweeps", "epoch_index", "n_spikes_total",
                      "rheobase_pA", "max_firing_rate_hz", "status"]].to_string(index=False))
    print(f"\nWrote {batch_csv}")

    n_ok = int((summary_df["status"] == "ok").sum())
    n_err = int(summary_df["status"].str.startswith("error").sum())
    print(f"{n_ok} ok, {n_err} error, {len(summary_df) - n_ok - n_err} skipped")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
