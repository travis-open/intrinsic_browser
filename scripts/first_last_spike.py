"""Extract first/last AP timing per fi_curve sweep from cell_summary.json files.

Each summary describes one *block* -- one cell at one timepoint under one
condition -- so ``cell_id`` alone does not identify a row. ``condition`` and
``seq`` come from the summary's ``metadata`` stanza (blank for summaries
written before the block model) and together with ``cell_id`` form the key.

Usage:
    python scripts/first_last_spike.py test_outputs/long_full -o first_last_spike.csv
"""

import argparse
import csv
import json
from pathlib import Path


def rows_for_cell(summary_path):
    with open(summary_path) as fh:
        summary = json.load(fh)

    cell_id = summary.get("cell_id", Path(summary_path).stem)
    metadata = summary.get("metadata", {}) or {}
    condition = metadata.get("condition", "")
    seq = metadata.get("seq", "")
    for sweep in summary.get("fi_curve", {}).get("per_sweep", []):
        times = sweep.get("spike_times_from_onset_ms") or []
        first = times[0] if times else None
        last = times[-1] if times else None
        yield {
            "cell_id": cell_id,
            "condition": condition,
            "seq": seq,
            "filename": sweep["filename"],
            "sweep_index": sweep["sweep_index"],
            "display_label": sweep["display_label"],
            "current_injection_pA": sweep["current_injection_pA"],
            "epoch_duration_s": sweep["epoch_duration_s"],
            "n_spikes": sweep["n_spikes"],
            # ms from step onset; last_spike_ms is also the "latency to last AP"
            "first_spike_ms": first,
            "last_spike_ms": last,
            # how long before the step ends the cell stopped firing
            "last_spike_to_offset_ms": (
                None if last is None else sweep["epoch_duration_s"] * 1000.0 - last
            ),
            "spike_train_duration_ms": (
                None if first is None else last - first
            ),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="cell_summary.json file, or a directory of them")
    ap.add_argument("-o", "--out", default="first_last_spike.csv")
    args = ap.parse_args()

    path = Path(args.path)
    files = (
        sorted(path.glob("*_cell_summary.json")) if path.is_dir() else [path]
    )

    rows = [row for f in files for row in rows_for_cell(f)]
    if not rows:
        print("no fi_curve sweeps found")
        return

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} sweeps from {len(files)} block(s) -> {args.out}")


if __name__ == "__main__":
    main()
