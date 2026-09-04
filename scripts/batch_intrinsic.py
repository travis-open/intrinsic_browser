"""
batch_intrinsic.py
------------------
Apply a consistent intrinsic-properties analysis to a list of recordings
described in a long manifest, reproducing the trace_viewer GUI buttons for each
protocol file.

The block model
---------------
The unit of analysis is a **block**: ``(cell_id, seq)`` -- the set of protocol
files acquired together at one timepoint, labelled by a free-text
``condition``. A cell measured at baseline and again in apamin has two blocks
and produces two output rows.

``seq`` orders blocks within a cell (0, 1, 2 ...); ``condition`` names them. Two
apamin blocks during a wash-in are ``seq=1`` and ``seq=2``, both
``condition="apamin"``. Nothing in this file enumerates conditions or drug
names, so a new drug -- or a third, or a time course -- needs no code change.

Each block gets its own ``Cell`` object holding at most one file per protocol,
so ``Cell``'s last-wins ``results[...][-1]`` semantics stay harmless. Blank
``seq``/``condition`` collapse to a single ``baseline`` block per cell, which is
the old one-row-per-cell behaviour exactly.

Per block, for every protocol row naming an existing ABF file:

    small_steps  ->  Find Spikes  +  F-I Curve  +  Analyze AHP
    ramp         ->  Find Spikes (same params)  +  Analyze Ramp APs
    sagIh        ->  Analyze Each Sweep (Passive)
    hyperpol     ->  Average & Analyze Subthreshold
    free_run     ->  Measure V_rest

Outputs (into --output-dir, default <repo>/test_outputs):
    <block>_spikes_<protocol>.csv  per-spike table (GUI-faithful), one per
                                   protocol that ran Find Spikes
    <block>_cell_summary.json      Cell.export_cell_summary -- all sections,
                                   stamped with the block's metadata
    batch_summary.csv              LONG: one row per block, same cell-level
                                   feature columns as before

where ``<block>`` is ``{cell_id}__{condition}_s{seq}``. Downstream code should
read output paths from ``batch_summary.csv`` rather than reconstructing them.

This mirrors the GUI: derivative spike backend, filter off (lowpass_hz=None),
step/ramp epoch from find_step_epoch.

Manifest
--------
A long CSV, one row per ABF file, with columns:

    cell_id, abf_folder, abf_file, protocol, condition, seq,
    drug, concentration, exclude, notes

``condition`` defaults to ``baseline`` and ``seq`` to ``0`` when blank;
``drug``/``concentration``/``notes`` are uninterpreted passthrough; a non-blank
``exclude`` drops that recording. Generate one from the old wide cell sheet
with ``scripts/sheet_to_long.py``.

Cell- and animal-level metadata (mouse_id, treatment, sex, cell type) live in
their own tables and are joined on ``cell_id`` / ``mouse_id`` downstream.

Running
-------
Must run inside the activated ``intrinsic_props`` conda env (bare ``python.exe``
from the env dir fails to put MKL's DLLs on PATH and numpy LAPACK crashes):

    conda run --no-capture-output -n intrinsic_props python scripts/batch_intrinsic.py

or ``conda activate intrinsic_props`` first, then ``python scripts/batch_intrinsic.py``.

Examples
--------
    # Check the manifest without loading any data
    ... python scripts/batch_intrinsic.py --validate-only --limit 0

    # Test run: first 3 blocks, all protocols
    ... python scripts/batch_intrinsic.py

    # Baseline only, explicit cells (reproduces the pre-block-model output)
    ... python scripts/batch_intrinsic.py --conditions baseline \
        --cells 20260607_cell7_JMT,20260610_cell5_JMT

    # Everything in the manifest
    ... python scripts/batch_intrinsic.py --limit 0
"""

from __future__ import annotations

import argparse
import csv
import re
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
from wholecell.analysis.spikes.features import spike_table_dataframe
from wholecell.analysis.passive import (
    estimate_input_resistance,
    fit_time_constant,
    estimate_sag,
)


MANIFEST_COLUMNS = [
    "cell_id",
    "abf_folder",
    "abf_file",
    "protocol",
    "condition",
    "seq",
    "drug",
    "concentration",
    "exclude",
    "notes",
]

DEFAULT_CONDITION = "baseline"

IDENTITY_COLUMNS = [
    "cell_id",
    "condition",
    "seq",
    "drug",
    "concentration",
    "abf_files",
    "recorded_at",
    "minutes_from_first",
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
    ("ahp__mean_mahp_delta_mV", "ahp", "mean_mahp_delta_mV"),
    ("ahp__max_mahp_delta_mV", "ahp", "max_mahp_delta_mV"),
    ("ahp__current_at_max_mahp_pA", "ahp", "current_at_max_mahp_pA"),
    ("ahp__mean_sahp_delta_mV", "ahp", "mean_sahp_delta_mV"),
    ("ahp__max_sahp_delta_mV", "ahp", "max_sahp_delta_mV"),
    ("ahp__current_at_max_sahp_pA", "ahp", "current_at_max_sahp_pA"),
    ("ahp__n_sweeps_analyzed", "ahp", "n_sweeps_analyzed"),
    ("ahp__any_window_truncated", "ahp", "any_window_truncated"),
]

BATCH_SUMMARY_COLUMNS = IDENTITY_COLUMNS + [c[0] for c in CELL_LEVEL_COLUMNS]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def _text(value) -> str:
    return "" if _blank(value) else str(value).strip()


def _safe_label(value: str) -> str:
    """Filesystem-safe form of a condition label.

    Conditions are free text, so ``apamin+ttx`` and ``5 uM`` are both plausible.
    Collapse anything that is not alphanumeric, dash or dot to an underscore.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "na"


def block_label(cell_id: str, condition: str, seq: int) -> str:
    """Output-file stem identifying one block."""
    return f"{cell_id}__{_safe_label(condition)}_s{seq}"


def safe_find_step_epoch(path: Path, fallback: int = 1) -> int:
    """``find_step_epoch`` with the GUI's fallback."""
    try:
        return int(find_step_epoch(str(path)))
    except Exception:
        return fallback


def _all_sweeps(name: str, rec) -> list[dict]:
    return [{"filename": name, "sweep_index": i} for i in range(rec.n_sweeps)]


def peek_recorded_at(path: Path):
    """Acquisition time from an ABF header without loading sample data.

    Used only by ``--validate-only``, where reading every sweep of 600+ files
    to check ordering would defeat the point.
    """
    try:
        import pyabf

        return pyabf.ABF(str(path), loadData=False).abfDateTime
    except Exception:
        return None


def build_spike_table(spike_result_data: dict) -> pd.DataFrame:
    """Flatten a stored detection result into the shared spike-table schema.

    ``spike_result_data`` is a ``cell.results["spikes"][k]["data"]`` dict.
    Column order comes from ``spike_table_dataframe``, the same helper the GUI
    export uses, so both write identical CSVs.
    """
    rows: list[dict] = []
    for sweep in spike_result_data.get("per_sweep", []):
        rows.extend(sweep.get("spikes", []))
    return spike_table_dataframe(rows)


def flatten_cell_level(summary: dict) -> dict:
    """Pull the curated cell-level scalars out of a Cell.export_cell_summary dict."""
    src = {
        "fi": summary.get("fi_curve", {}).get("cell_level", {}) or {},
        "vrest": summary.get("v_rest", {}) or {},
        "sag_passive": summary.get("passive_range", {}) or {},
        "avg_passive": summary.get("passive_repeated_step", {}) or {},
        "ramp": summary.get("ramp_evoked_APs", {}).get("cell_level", {}) or {},
        "ahp": summary.get("ahp", {}) or {},
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
# protocol runners
#
# Each runner loads one ABF into the block's Cell, runs the analyses the GUI
# would run for that protocol, and returns the labels to record in
# ``protocols_run``. ``ctx`` carries the CLI args plus the two side-channels a
# runner may write to: detected spikes (one CSV per protocol) and the status
# fields that only small_steps knows.
# ---------------------------------------------------------------------------

class BlockContext:
    """Mutable scratch space shared by the runners within one block."""

    def __init__(self, args) -> None:
        self.args = args
        self.spike_data: dict[str, dict] = {}   # protocol -> detection result
        self.stats: dict[str, object] = {}      # n_sweeps / epoch_index
        self.errors: dict[str, str] = {}        # sub-analysis failures (e.g. ahp)


def _prepare(cell: Cell, path: Path):
    """Load an ABF and give it an all-sweeps collection. Returns (name, rec)."""
    rec = cell.add_recording(path)
    name = rec.filename
    cell.create_sweep_collection(name, _all_sweeps(name, rec))
    return name, rec


def _run_small_steps(cell: Cell, path: Path, ctx: BlockContext) -> list[str]:
    """Find Spikes + F-I Curve + Analyze AHP on a current-step family."""
    args = ctx.args
    name, rec = _prepare(cell, path)
    epoch = args.epoch_index if args.epoch_index is not None else safe_find_step_epoch(path)
    cell.find_spikes(name, epoch, dvdt_detection_mVms=args.dvdt,
                     peak_search_window_ms=args.peak_window, lowpass_hz=args.lowpass)
    ctx.spike_data["small_steps"] = cell.results["spikes"][-1]["data"]
    cell.analyze_fi_curve(name, epoch)
    ctx.stats["n_sweeps"] = rec.n_sweeps
    ctx.stats["epoch_index"] = epoch
    ran = ["small_steps"]

    # AHP: same current-step collection / epoch as the F-I curve, no spike
    # detection required (trace_viewer._run_ahp_analysis). Own try/except so an
    # AHP failure doesn't undo the F-I result.
    try:
        cell.analyze_ahp(name, epoch)  # GUI defaults: 2 kHz, 100/1000 ms
        ran.append("ahp")
    except Exception as exc:
        traceback.print_exc()
        ctx.errors["ahp"] = f"error: {exc}"
    return ran


def _run_ramp(cell: Cell, path: Path, ctx: BlockContext) -> list[str]:
    """Find Spikes (same params) + Analyze Ramp APs."""
    args = ctx.args
    name, _ = _prepare(cell, path)
    epoch = safe_find_step_epoch(path)
    cell.find_spikes(name, epoch, dvdt_detection_mVms=args.dvdt,
                     peak_search_window_ms=args.peak_window,
                     lowpass_hz=args.lowpass)
    ctx.spike_data["ramp"] = cell.results["spikes"][-1]["data"]
    result = run_ramp_analysis(
        cell.collections[name], epoch, cell.results["spikes"][-1],
        lowpass_hz=args.lowpass,
    )
    cell._store_result("ramp_evoked_APs", result, {
        "collection_name": name, "epoch_index": epoch,
        "source": "batch_intrinsic",
    })
    return ["ramp"]


def _run_sag_ih(cell: Cell, path: Path, ctx: BlockContext) -> list[str]:
    """Analyze Each Sweep (Passive) -> stores "passive_range"."""
    name, _ = _prepare(cell, path)
    epoch = safe_find_step_epoch(path)
    cell.analyze_passive(name, epoch)
    return ["sagIh"]


def _run_hyperpol(cell: Cell, path: Path, ctx: BlockContext) -> list[str]:
    """Average & Analyze Subthreshold -> stores "passive_repeated_step"."""
    name, _ = _prepare(cell, path)
    epoch = safe_find_step_epoch(path)
    result = run_average_subthreshold(cell, name, epoch, lowpass_hz=ctx.args.lowpass)
    cell._store_result("passive_repeated_step", result, {
        "collection_name": name, "epoch_index": epoch,
        "n_sweeps_averaged": result["n_sweeps_averaged"],
        "source": "batch_intrinsic",
    })
    return ["hyperpol"]


def _run_free_run(cell: Cell, path: Path, ctx: BlockContext) -> list[str]:
    """Measure V_rest on a free-running (no command) file."""
    args = ctx.args
    name, _ = _prepare(cell, path)
    result = run_vrest_analysis(
        cell.collections[name], lowpass_hz=args.lowpass,
        dvdt_detection_mVms=args.dvdt,
        peak_search_window_ms=args.peak_window,
    )
    if result.get("spike_detection") is not None:
        ctx.spike_data["free_run"] = result["spike_detection"]
    cell._store_result("v_rest", result, {
        "collection_name": name, "source": "batch_intrinsic",
    })
    return ["free_run"]


# Key order is the execution order within a block, and it is load-bearing:
# small_steps must precede ramp, because analyze_fi_curve reads
# results["spikes"][-1] and ramp detection appends another "spikes" entry.
PROTOCOL_RUNNERS = {
    "small_steps": _run_small_steps,
    "ramp": _run_ramp,
    "sagIh": _run_sag_ih,
    "hyperpol": _run_hyperpol,
    "free_run": _run_free_run,
}
ALL_PROTOCOLS = list(PROTOCOL_RUNNERS)
PROTOCOL_RANK = {p: i for i, p in enumerate(ALL_PROTOCOLS)}


# ---------------------------------------------------------------------------
# manifest loading and block grouping
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> pd.DataFrame:
    """Read the long manifest, apply defaults, and drop excluded rows."""
    df = pd.read_csv(path, dtype=str)

    required = ["cell_id", "abf_file", "protocol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Manifest {path} is missing required column(s): {missing}\n"
            f"Expected the long schema: {MANIFEST_COLUMNS}\n"
            "Convert a wide cell sheet with scripts/sheet_to_long.py."
        )
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for col in ("cell_id", "abf_folder", "abf_file", "protocol", "drug",
                "concentration", "exclude", "notes"):
        df[col] = df[col].map(_text)
    df["condition"] = df["condition"].map(lambda v: _text(v) or DEFAULT_CONDITION)
    df["seq"] = df["seq"].map(lambda v: int(float(_text(v))) if _text(v) else 0)

    df = df[df["cell_id"] != ""]
    excluded = df["exclude"] != ""
    if excluded.any():
        print(f"Manifest: {int(excluded.sum())} row(s) skipped via 'exclude'.")
    return df[~excluded].reset_index(drop=True)


def group_blocks(df: pd.DataFrame) -> list[tuple[tuple[str, int], pd.DataFrame]]:
    """Group manifest rows into blocks, each sorted into execution order.

    Returns ``[((cell_id, seq), rows), ...]`` ordered by cell then seq.
    """
    blocks: list[tuple[tuple[str, int], pd.DataFrame]] = []
    for key, rows in df.groupby(["cell_id", "seq"], sort=True):
        rows = rows.assign(
            _rank=rows["protocol"].map(lambda p: PROTOCOL_RANK.get(p, 99))
        ).sort_values(["_rank", "abf_file"]).drop(columns="_rank")
        blocks.append((key, rows.reset_index(drop=True)))
    blocks.sort(key=lambda kv: (kv[0][0], kv[0][1]))
    return blocks


def block_problems(rows: pd.DataFrame) -> list[str]:
    """Manifest errors that make a block ambiguous. Reported, never raised."""
    problems: list[str] = []

    conditions = sorted(set(rows["condition"]))
    if len(conditions) > 1:
        problems.append(
            f"condition is not constant within the block: {conditions} "
            "(rows sharing a seq must share a condition)"
        )

    # Only protocols we can actually analyze matter here. A repeated SK_VC is
    # just an unanalyzed pair; a repeated small_steps means a mis-entered seq.
    analyzable = rows[rows["protocol"].isin(PROTOCOL_RUNNERS)]
    counts = analyzable["protocol"].value_counts()
    for protocol, n in counts[counts > 1].items():
        files = list(analyzable.loc[analyzable["protocol"] == protocol, "abf_file"])
        problems.append(
            f"{protocol} appears {n}x in one block ({', '.join(files)}) -- "
            "give the repeats different seq values"
        )
    return problems


# ---------------------------------------------------------------------------
# per-block processing
# ---------------------------------------------------------------------------

def _warn_missing(cell_id: str, protocol: str, folder: str, filename: str) -> None:
    print(
        f"WARNING  {cell_id}  {protocol}  file not found:\n"
        f"         folder: {folder}\n"
        f"         file:   {filename}",
        file=sys.stderr,
    )


def resolve_path(row: pd.Series, missing_accum: list[tuple]) -> tuple[Path | None, str | None]:
    """Return (path, reason). ``path`` is None when the row can't be analyzed."""
    cell_id, protocol = row["cell_id"], row["protocol"]
    abf_file, abf_folder = row["abf_file"], row["abf_folder"]
    if not abf_file:
        return None, "no abf_file"
    if not abf_folder:
        _warn_missing(cell_id, protocol, "<blank abf_folder>", abf_file)
        missing_accum.append((cell_id, protocol, "<blank abf_folder>", abf_file))
        return None, "no abf_folder"
    p = Path(abf_folder) / abf_file
    if not p.exists():
        _warn_missing(cell_id, protocol, abf_folder, abf_file)
        missing_accum.append((cell_id, protocol, abf_folder, abf_file))
        return None, "file not found"
    return p, None


def process_block(key: tuple[str, int], rows: pd.DataFrame, protocols: list[str],
                  out_dir: Path, args, missing_accum: list[tuple]) -> dict:
    cell_id, seq = key
    condition = rows["condition"].iloc[0]
    label = block_label(cell_id, condition, seq)

    status_row = {k: "" for k in BATCH_SUMMARY_COLUMNS}
    status_row["cell_id"] = cell_id
    status_row["condition"] = condition
    status_row["seq"] = seq
    status_row["drug"] = next((v for v in rows["drug"] if v), "")
    status_row["concentration"] = next((v for v in rows["concentration"] if v), "")
    status_row["abf_files"] = ";".join(
        f"{r['protocol']}={r['abf_file']}" for _, r in rows.iterrows()
    )

    notes_by_protocol: dict[str, str] = {}
    # Two different reasons a block can produce nothing, kept apart so a run
    # that simply filtered a block out does not look like a failure.
    has_runner = any(p in PROTOCOL_RUNNERS for p in rows["protocol"])
    was_requested = any(
        p in PROTOCOL_RUNNERS and p in protocols for p in rows["protocol"]
    )

    problems = block_problems(rows)
    if problems:
        joined = "; ".join(problems)
        print(f"    manifest problem: {joined}", file=sys.stderr)
        status_row["protocol_status"] = joined
        status_row["status"] = f"error: {joined}"
        return status_row

    notes = next((v for v in rows["notes"] if v), "")
    cell = Cell(cell_id=cell_id, output_dir=out_dir, notes=notes,
                metadata={"condition": condition, "seq": seq,
                          "drug": status_row["drug"],
                          "concentration": status_row["concentration"],
                          "abf_files": status_row["abf_files"]})
    ctx = BlockContext(args)
    ran: list[str] = []

    for _, row in rows.iterrows():
        protocol = row["protocol"]
        runner = PROTOCOL_RUNNERS.get(protocol)
        if runner is None:
            notes_by_protocol[protocol] = "no analyzer"
            continue
        if protocol not in protocols:
            notes_by_protocol[protocol] = "not requested"
            continue
        path, reason = resolve_path(row, missing_accum)
        if path is None:
            notes_by_protocol[protocol] = reason
            continue
        try:
            ran.extend(runner(cell, path, ctx))
        except Exception as exc:  # noqa: BLE001 - one protocol must not sink the block
            traceback.print_exc()
            notes_by_protocol[protocol] = f"error: {exc}"

    notes_by_protocol.update(ctx.errors)

    # Earliest acquisition time across the files that actually loaded. This is
    # the block's position on the experiment clock.
    times = [r.recorded_at for r in cell.recordings.values() if r.recorded_at is not None]
    if times:
        status_row["recorded_at"] = min(times).isoformat()

    # ---- exports -------------------------------------------------------
    # One spike table per protocol that ran Find Spikes. n_spikes_total keeps
    # its established meaning (small_steps count) so the feature columns stay
    # comparable with earlier runs.
    written_csvs: list[str] = []
    for protocol, spike_data in ctx.spike_data.items():
        spike_df = build_spike_table(spike_data)
        spikes_csv = out_dir / f"{label}_spikes_{protocol}.csv"
        spike_df.to_csv(spikes_csv, index=False)
        written_csvs.append(str(spikes_csv))
        if protocol == "small_steps":
            status_row["n_spikes_total"] = len(spike_df)
    if written_csvs:
        status_row["spikes_csv"] = ";".join(written_csvs)

    summary_json = out_dir / f"{label}_cell_summary.json"
    summary = cell.export_cell_summary(filepath=summary_json)
    status_row["summary_json"] = str(summary_json)
    status_row.update(flatten_cell_level(summary))

    status_row["n_sweeps"] = ctx.stats.get("n_sweeps", "")
    status_row["epoch_index"] = ctx.stats.get("epoch_index", "")
    status_row["protocols_run"] = ",".join(ran)
    status_row["protocol_status"] = ", ".join(
        f"{k}: {v}" for k, v in notes_by_protocol.items()
    )
    if ran:
        status_row["status"] = "ok"
    elif not has_runner:
        # Every row here is a protocol we have no analyzer for (today, the
        # voltage-clamp SK_VC files). Nothing went wrong -- there was simply
        # nothing to analyze, and calling that an error would bury the blocks
        # that really did fail.
        status_row["status"] = "skipped: no analyzable protocol"
    elif not was_requested:
        status_row["status"] = "skipped: no requested protocol in this block"
    else:
        status_row["status"] = "error: no protocol produced a result"
    return status_row


def add_elapsed_minutes(rows: list[dict]) -> list[str]:
    """Fill ``minutes_from_first`` and flag seq/clock disagreements.

    The manifest's ``seq`` is authoritative for ordering; the ABF clock is the
    independent check on it. A cell whose blocks were acquired in a different
    order than the manifest claims is almost always a data-entry error, so it
    is worth saying out loud -- but not worth refusing to analyze.
    """
    warnings: list[str] = []
    by_cell: dict[str, list[dict]] = {}
    for row in rows:
        by_cell.setdefault(row["cell_id"], []).append(row)

    for cell_id, cell_rows in by_cell.items():
        ordered = sorted(cell_rows, key=lambda r: r["seq"] if r["seq"] != "" else 0)
        timed = [r for r in ordered if r["recorded_at"]]
        if not timed:
            continue
        ref = pd.Timestamp(timed[0]["recorded_at"])
        for row in timed:
            delta = (pd.Timestamp(row["recorded_at"]) - ref).total_seconds() / 60.0
            row["minutes_from_first"] = round(delta, 2)
        clock_order = sorted(timed, key=lambda r: r["recorded_at"])
        if [r["seq"] for r in clock_order] != [r["seq"] for r in timed]:
            warnings.append(
                f"{cell_id}: seq order {[r['seq'] for r in timed]} disagrees with "
                f"acquisition order {[r['seq'] for r in clock_order]}"
            )
    return warnings


# ---------------------------------------------------------------------------
# validation mode
# ---------------------------------------------------------------------------

def validate(blocks, protocols: list[str]) -> int:
    """Check the manifest without loading sample data. Returns an exit code."""
    n_errors = n_warnings = 0
    missing: list[str] = []
    no_analyzer: dict[str, int] = {}

    for (cell_id, seq), rows in blocks:
        condition = rows["condition"].iloc[0]
        prefix = f"{cell_id}  seq={seq}  ({condition})"
        for msg in block_problems(rows):
            print(f"ERROR    {prefix}: {msg}")
            n_errors += 1
        for _, row in rows.iterrows():
            protocol = row["protocol"]
            if protocol not in PROTOCOL_RUNNERS:
                no_analyzer[protocol] = no_analyzer.get(protocol, 0) + 1
                continue
            if protocol not in protocols:
                continue
            folder, name = row["abf_folder"], row["abf_file"]
            if not folder or not (Path(folder) / name).exists():
                missing.append(f"{prefix}  {protocol}: {folder}\\{name}")

    # seq vs. acquisition order, from headers only
    for cell_id, cell_blocks in _by_cell(blocks).items():
        stamped = []
        for (_, seq), rows in cell_blocks:
            times = []
            for _, row in rows.iterrows():
                if row["protocol"] not in PROTOCOL_RUNNERS:
                    continue
                p = Path(row["abf_folder"]) / row["abf_file"] if row["abf_folder"] else None
                if p is not None and p.exists():
                    t = peek_recorded_at(p)
                    if t is not None:
                        times.append(t)
            if times:
                stamped.append((seq, min(times)))
        if len(stamped) > 1:
            if [s for s, _ in stamped] != [s for s, _ in sorted(stamped, key=lambda x: x[1])]:
                print(f"WARNING  {cell_id}: seq order disagrees with ABF acquisition order "
                      f"({[(s, t.isoformat()) for s, t in stamped]})")
                n_warnings += 1

    if missing:
        print(f"\n=== {len(missing)} file(s) not found ===")
        for line in missing:
            print(f"  {line}")
    if no_analyzer:
        print("\n=== protocols with no analyzer (rows ignored) ===")
        for protocol, n in sorted(no_analyzer.items()):
            print(f"  {protocol}: {n} row(s)")

    n_blocks = len(blocks)
    n_cells = len({c for (c, _), _ in blocks})
    print(f"\n{n_blocks} block(s) across {n_cells} cell(s)  |  "
          f"{n_errors} error(s), {n_warnings} warning(s), {len(missing)} missing file(s)")
    return 1 if n_errors else 0


def _by_cell(blocks) -> dict:
    out: dict = {}
    for key, rows in blocks:
        out.setdefault(key[0], []).append((key, rows))
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def select_blocks(blocks, cells: list[str] | None, conditions: list[str] | None,
                  limit: int):
    """Filter blocks by cell_id / condition, then apply --limit."""
    if cells:
        known = {key[0] for key, _ in blocks}
        unknown = [c for c in cells if c not in known]
        if unknown:
            raise SystemExit(f"cell_id(s) not found in manifest: {unknown}")
        order = {c: i for i, c in enumerate(cells)}
        blocks = [b for b in blocks if b[0][0] in order]
        blocks.sort(key=lambda kv: (order[kv[0][0]], kv[0][1]))
    if conditions:
        wanted = set(conditions)
        blocks = [b for b in blocks if b[1]["condition"].iloc[0] in wanted]
        if not blocks:
            raise SystemExit(f"No blocks match --conditions {conditions}")
    if limit and limit > 0:
        blocks = blocks[:limit]
    return blocks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, default=REPO_ROOT / "scripts" / "recordings.csv",
                        help="Long recordings manifest (default: scripts/recordings.csv)")
    parser.add_argument("--protocols", default=",".join(ALL_PROTOCOLS),
                        help="Comma list from: " + ",".join(ALL_PROTOCOLS) + " (default: all)")
    parser.add_argument("--cells", default="",
                        help="Comma-separated cell_id list. Overrides --limit ordering.")
    parser.add_argument("--conditions", default="",
                        help="Comma-separated condition list (e.g. baseline,apamin)")
    parser.add_argument("--limit", type=int, default=3,
                        help="First N blocks (default: 3; 0 = no limit)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Check the manifest and file paths without analyzing")
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
        raise SystemExit(
            f"Manifest not found: {args.sheet}\n"
            "Generate one from the wide cell sheet:\n"
            "    python scripts/sheet_to_long.py --sheet scripts/cells_forsberg.csv"
        )

    manifest = load_manifest(args.sheet)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()] or None
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()] or None
    blocks = select_blocks(group_blocks(manifest), cells, conditions, args.limit)
    if not blocks:
        raise SystemExit("No matching blocks to process.")

    if args.validate_only:
        print(f"Validating {args.sheet}  |  protocols={protocols}\n")
        return validate(blocks, protocols)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch intrinsic analysis  |  protocols={protocols}  "
          f"dvdt={args.dvdt} mV/ms  peak_window={args.peak_window} ms")
    print(f"Output: {out_dir}")
    print(f"Blocks: {len(blocks)} across {len({k[0] for k, _ in blocks})} cell(s)\n")

    results: list[dict] = []
    missing_accum: list[tuple] = []
    for key, rows in blocks:
        cell_id, seq = key
        condition = rows["condition"].iloc[0]
        print(f"--- {cell_id}  seq={seq}  ({condition}) ---")
        try:
            status_row = process_block(key, rows, protocols, out_dir, args, missing_accum)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            traceback.print_exc()
            status_row = {k: "" for k in BATCH_SUMMARY_COLUMNS}
            status_row["cell_id"] = cell_id
            status_row["condition"] = condition
            status_row["seq"] = seq
            status_row["status"] = f"error: {exc}"
        results.append(status_row)
        extra = ""
        if status_row["status"] == "ok":
            extra = (f"  | ran: {status_row['protocols_run']}"
                     f"  spikes={status_row['n_spikes_total']}")
            if status_row["protocol_status"]:
                extra += f"  ({status_row['protocol_status']})"
        print(f"    {status_row['status']}{extra}")

    order_warnings = add_elapsed_minutes(results)

    summary_df = pd.DataFrame(results, columns=BATCH_SUMMARY_COLUMNS)
    batch_csv = out_dir / "batch_summary.csv"
    summary_df.to_csv(batch_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n=== batch summary ===")
    print(summary_df[["cell_id", "condition", "seq", "minutes_from_first",
                      "n_sweeps", "n_spikes_total", "protocols_run",
                      "status"]].to_string(index=False))

    if order_warnings:
        print(f"\n=== {len(order_warnings)} cell(s) where seq disagrees with the ABF clock ===")
        for line in order_warnings:
            print(f"  {line}")

    if missing_accum:
        print(f"\n=== {len(missing_accum)} file(s) not found ===")
        for cid, proto, folder, fname in missing_accum:
            print(f"  {cid}  {proto}:  {folder}\\{fname}")

    print(f"\nWrote {batch_csv}")
    n_ok = int((summary_df["status"] == "ok").sum())
    n_skip = int(summary_df["status"].str.startswith("skipped").sum())
    n_err = int(summary_df["status"].str.startswith("error").sum())
    print(f"{n_ok} ok, {n_skip} skipped, {n_err} error  "
          f"({len(missing_accum)} missing file warnings)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
