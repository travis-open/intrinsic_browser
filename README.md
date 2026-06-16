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
git clone https://github.com/yourlab/intrinsic_props.git
cd intrinsic_props

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

---

## Output files

For each cell, analysis produces the following files in the specified `output_dir`:

| File | Contents |
|------|----------|
| `{cell_id}_session.json` | Full session: loaded files, sweep selections, all results with timestamps, audit log of every analysis decision |
| `{cell_id}_spikes_{timestamp}.csv` | Per-spike table with `filename`, `sweep_index`, and all shape features |
| `{cell_id}_sweeps_{timestamp}.csv` | Per-sweep summary: spike count, firing rate, current injection amplitude |
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
└── wholecell/              # importable package
    ├── core/
    │   ├── cell.py             # top-level analysis object
    │   ├── recording.py        # single ABF file wrapper
    │   └── sweep_collection.py # named, multi-file sweep selections
    ├── analysis/
    │   ├── passive.py          # Rin, tau, sag
    │   ├── fi_curve.py         # F-I curve, rheobase, slope
    │   └── spikes/
    │       ├── base.py         # SpikeFinder interface
    │       ├── derivative.py   # built-in dV/dt finder
    │       ├── ipfx_backend.py # optional IPFX wrapper
    │       ├── finder.py       # collection-level spike detection
    │       └── features.py     # spike shape feature extraction
    ├── filters/
    │   └── lowpass.py          # zero-phase Butterworth filter
    └── io/
        └── abf_reader.py       # ABF inspection and epoch utilities
```
