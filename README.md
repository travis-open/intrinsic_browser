# intrinsic_props

Interactive analysis of whole-cell patch-clamp electrophysiology data. Designed for measuring intrinsic membrane properties from current-clamp recordings: input resistance, membrane time constant, Ih sag, action potential shape, and firing properties (F-I curves, rheobase, spike adaptation).

Data is read from Axon Binary Format (ABF) files. Results are saved as human-readable JSON and CSV files.

---

## Requirements

- [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda

That's the only prerequisite. Everything else is installed automatically.

**Don't have conda yet?**
1. Go to [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/)
2. Download the installer for your operating system
3. Run the installer and accept the defaults
4. Restart your terminal

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/travis-open/intrinsic_browser.git
cd intrinsic_browser

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate intrinsic_props

# 4. Install the package
pip install -e .
```

You only need to do this once. After that, skip to **Usage** below.

---

## Usage

### Activating the environment

Every time you open a new terminal, activate the environment before running scripts:

```bash
conda activate intrinsic_props
```

### Interactive scripting

```python
from wholecell.core.cell import Cell

cell = Cell(cell_id="cell_01", output_dir="results/cell_01")
cell.add_recording("data/cell_01_steps.abf")

# Inspect sweep quality before analysis
cell.print_qc_table("cell_01_steps")

# Select sweeps for analysis
sc = cell.create_sweep_collection(
    name="current_steps",
    sweeps=[
        {"filename": "cell_01_steps", "sweep_index": 0},
        {"filename": "cell_01_steps", "sweep_index": 1},
        {"filename": "cell_01_steps", "sweep_index": 2},
    ],
)

# Estimate passive properties from epoch 1 (the current step)
cell.analyze_passive(
    collection_name="current_steps",
    epoch_index=1,
    measures=["input_resistance", "time_constant", "sag_ratio"],
    lowpass_hz=1000.0,
)

# Detect spikes
cell.find_spikes(
    collection_name="current_steps",
    epoch_index=1,
    backend="derivative",
    dvdt_threshold_mVms=20.0,
)

# Extract spike shape features
cell.extract_spike_features(collection_name="current_steps")

# Build F-I curve
cell.analyze_fi_curve(collection_name="current_steps", epoch_index=1)

# Export results
cell.export_spike_table()

# Save session (captures all analysis decisions for reproducibility)
cell.save_session()
```

### Trace viewer (GUI)

```bash
python -m wholecell.gui.trace_viewer
```

Opens a directory picker if run with no arguments. You can also pass a path directly:

```bash
python -m wholecell.gui.trace_viewer path/to/recording.abf
python -m wholecell.gui.trace_viewer path/to/directory
python -m wholecell.gui.trace_viewer path/to/cell_session.json
```

Optional flags: `--cell-id`, `--lowpass <Hz>`, `--epoch <index>`.

### Conditions in the viewer

The **Condition** box under the collection selector labels the active collection —
`control`, `apamin`, `wash`, anything you type. It is free text: no drug list to
maintain, no code change for a new one.

1. Pick a collection (one is auto-created per ABF file), type its condition.
2. Run the analyses as usual. Results are recorded against that label.
3. **Export ▸ Cell Summary (JSON)** — with two or more conditions labelled it asks for
   a *directory* and writes one summary per condition, in exactly the layout
   `scripts/batch_intrinsic.py` produces. A cell analyzed by hand and one analyzed by
   the batch are interchangeable downstream.

The label lives on the collection, so it survives **Save Session**, and re-labelling a
collection re-files results that were already analyzed — a mislabelled condition can be
fixed without re-running anything.

Two things follow the active condition rather than the cell: the analysis "done"
checkboxes, and the spike results the F-I curve is built from. Detecting spikes on the
apamin file no longer feeds a control F-I curve.

Once a second condition has been analyzed, the **F-I** and **AHP** popups overlay all of
them on one axes with a legend, so a drug effect is visible at the rig.

> Without any condition set, everything behaves exactly as before: one summary file, one
> set of curves.

---

## Batch analysis and repeated measurements

`scripts/batch_intrinsic.py` is the headless mirror of the GUI buttons, applied
over a manifest of recordings.

### The block model

The unit of analysis is a **block**: `(cell_id, seq)` — the set of protocol files
acquired together at one timepoint, labelled by a free-text `condition`. A cell
measured at baseline and again after a drug has two blocks and produces two rows
of output.

- `seq` orders blocks within a cell (0, 1, 2 …). `condition` names them. Two
  apamin blocks during a wash-in are `seq=1` and `seq=2`, both
  `condition="apamin"`.
- `condition` is never parsed or enumerated in code, so a new drug — or a third,
  or a time course — needs no code change. Drug identity is data, not schema.
- Each block gets its own `Cell` object holding at most one file per protocol.
- Blank `seq`/`condition` collapse to a single `baseline` block per cell, which
  is the one-row-per-cell behaviour of the pre-block-model pipeline exactly.

### The manifest

A long CSV, **one row per ABF file**:

| column | required | meaning |
|--------|----------|---------|
| `cell_id` | yes | joins to the cells / animals tables |
| `abf_folder` | yes | directory holding the file |
| `abf_file` | yes | one filename — no lists |
| `protocol` | yes | `small_steps`, `ramp`, `sagIh`, `hyperpol`, `free_run` |
| `condition` | no | free text; blank → `baseline` |
| `seq` | no | block index within the cell; blank → `0` |
| `drug` | no | uninterpreted passthrough |
| `concentration` | no | uninterpreted passthrough |
| `exclude` | no | non-blank drops that recording |
| `notes` | no | free text |

Cell- and animal-level metadata (`mouse_id`, `treatment`, `sex`, cell type) stay
in their own tables and are joined on `cell_id` / `mouse_id` downstream.

`scripts/sheet_to_long.py` converts the older wide cell sheet (one column per
protocol) into this schema, mapping the `*_apamin_*` columns to
`condition=apamin, seq=1`.

```bash
python scripts/sheet_to_long.py --sheet scripts/cells_forsberg.csv --out scripts/recordings.csv
python scripts/batch_intrinsic.py --validate-only --limit 0
python scripts/batch_intrinsic.py --limit 0
```

`--validate-only` checks the manifest and every file path without loading sample
data: it reports blocks whose `condition` is inconsistent, blocks naming the same
protocol twice (a mis-entered `seq`), protocols with no analyzer, missing files,
and cells whose `seq` order disagrees with the ABF acquisition clock.

Useful flags: `--cells`, `--conditions`, `--protocols`, `--limit` (blocks, not
cells), `--output-dir`.

### Batch outputs

| File | Contents |
|------|----------|
| `batch_summary.csv` | **long** — one row per block, with `condition`, `seq`, `recorded_at`, `minutes_from_first`, and the cell-level feature columns (`fi__*`, `vrest__*`, `sag_passive__*`, `avg_passive__*`, `ramp__*`, `ahp__*`) |
| `{block}_cell_summary.json` | all analysis sections for one block, stamped with a `metadata` stanza naming its condition and seq |
| `{block}_spikes_{protocol}.csv` | per-spike table, one per protocol that ran spike detection |

where `{block}` is `{cell_id}__{condition}_s{seq}`. Read output paths from
`batch_summary.csv` rather than reconstructing them.

The GUI writes the same per-block summaries (see
[Conditions in the viewer](#conditions-in-the-viewer)), so hand-analyzed and
batch-analyzed cells can be aggregated together.

Because the feature columns are the same regardless of how many conditions a cell
has, a paired comparison is a pivot:

```python
import pandas as pd

df = pd.read_csv("test_outputs/long_full/batch_summary.csv")
wide = df.pivot(index="cell_id", columns="condition", values="ahp__mean_mahp_delta_mV")
wide["delta"] = wide["apamin"] - wide["baseline"]
```

---

## Output files

These are the files the **interactive / GUI** workflow writes into `output_dir`.
The batch pipeline writes a different, block-keyed set — see
[Batch outputs](#batch-outputs) above.

| File | Contents |
|------|----------|
| `{cell_id}_session.json` | Full session: loaded files, sweep selections, all results with timestamps, audit log of every analysis decision |
| `{cell_id}_spikes_{timestamp}.csv` | Per-spike table with `filename`, `sweep_index`, and all shape features |
| `{cell_id}_sweep_summary_{timestamp}.csv` | Per-sweep summary: spike count, firing rate, current injection amplitude |
| `{cell_id}_cell_summary.json` | Cell-level scalars (Rin, rheobase, first spike threshold) and full F-I curve |

`filename` and `sweep_index` are always saved as separate columns so that any sweep can be reloaded directly:

```python
import pyabf, pandas as pd

spikes = pd.read_csv("results/cell_01/cell_01_spikes.csv")
row = spikes[spikes.sweep_index == 3].iloc[0]
abf = pyabf.ABF(row.filename + ".abf")
```

---

## Spike detection backends

Two backends are available:

**`derivative`** (default, no extra dependencies)
Detects spikes based on dV/dt threshold crossing. Comparable to the IPFX algorithm.

```python
cell.find_spikes(collection_name="current_steps", epoch_index=1,
                 backend="derivative", dvdt_threshold_mVms=20.0)
```

**`ipfx`** (optional, Allen Institute)
Uses the [IPFX](https://github.com/AllenInstitute/ipfx) library. Install separately if needed:

```bash
pip install ipfx
```

```python
cell.find_spikes(collection_name="current_steps", epoch_index=1,
                 backend="ipfx")
```

---

## Viewer settings

The interactive viewer reads user preferences from `~/.wholecell/settings.json` (`C:\Users\<you>\.wholecell\settings.json` on Windows).  The file is created automatically the first time a spinbox value is changed in the viewer, or you can create it manually.

```json
{
  "default_data_directory": "C:\\Users\\tahage\\Box\\CreedLabBoxDrive",
  "lowpass_hz": 2000.0,
  "dvdt_threshold_mv_per_ms": 5.0,
  "peak_window_ms": 20.0
}
```

| Key | What it controls | Default |
|-----|-----------------|---------|
| `default_data_directory` | Starting folder for all file-open dialogs | OS default |
| `lowpass_hz` | Lowpass filter cutoff (toggled with **F**) | `2000.0` |
| `dvdt_threshold_mv_per_ms` | dV/dt spike-detection threshold | `20.0` |
| `peak_window_ms` | Search window for spike peak | `20.0` |

Settings are per-user and scoped to the machine, so each person on a shared setup can point at their own data directory without affecting anyone else.  Spinbox changes in the viewer are saved immediately, so your preferred thresholds carry over between sessions automatically.

---

## Troubleshooting

**`conda: command not found`**
Run `conda init` in your terminal, then restart it before trying again.

**`conda activate` doesn't change my prompt (Windows)**
Use Anaconda Prompt instead of Command Prompt or PowerShell, or run `conda init powershell` once and restart.

**Updating the environment after a code update**
If `environment.yml` has changed after a `git pull`:

```bash
conda env update -f environment.yml --prune
```

**Removing the environment and starting fresh**

```bash
conda deactivate
conda env remove -n intrinsic_props
conda env create -f environment.yml
```

---

## Project structure

```
intrinsic_props/
├── environment.yml         # conda environment specification
├── pyproject.toml          # package metadata
├── README.md
├── scripts/
│   ├── batch_intrinsic.py      # headless batch driver (block model)
│   ├── sheet_to_long.py        # wide cell sheet -> long recordings manifest
│   ├── first_last_spike.py     # per-sweep first/last AP timing from summaries
│   └── recordings.csv          # generated long manifest
└── wholecell/              # importable package
    ├── config.py               # user settings (~/.wholecell/settings.json)
    ├── core/
    │   ├── cell.py             # top-level analysis object
    │   ├── recording.py        # single ABF file wrapper
    │   └── sweep_collection.py # named, multi-file sweep selections
    ├── analysis/
    │   ├── passive.py          # Rin, tau, sag
    │   ├── fi_curve.py         # F-I curve, rheobase, slope
    │   ├── ahp.py              # post-step mAHP / sAHP
    │   ├── ramp.py             # ramp-evoked AP features
    │   ├── vrest.py            # resting Vm, spontaneous firing
    │   └── spikes/
    │       ├── base.py         # SpikeFinder interface
    │       ├── derivative.py   # built-in dV/dt finder
    │       ├── ipfx_backend.py # optional IPFX wrapper
    │       ├── finder.py       # collection-level spike detection
    │       └── features.py     # spike shape feature extraction
    ├── filters/
    │   └── lowpass.py          # zero-phase Butterworth filter
    ├── gui/
    │   ├── trace_viewer.py     # interactive PyQt/pyqtgraph viewer
    │   ├── fi_viewer.py        # F-I curve popup
    │   └── ahp_viewer.py       # AHP popup
    └── io/
        └── abf_reader.py       # ABF inspection and epoch utilities
```

---

*Built with [Claude Code](https://claude.com/claude-code).*
