"""
sheet_to_long.py
----------------
Convert the wide cell sheet (one row per cell, one column per protocol) into
the long ``recordings`` manifest (one row per ABF file) that
``batch_intrinsic.py`` consumes.

The wide layout has no slot for a repeated measurement: ``ramp_file`` holds one
filename, so a cell measured before and after a drug cannot be expressed. The
long layout keys every recording by ``(cell_id, seq)`` -- the *block* -- and
labels it with a free-text ``condition``. Drug identity is data, never schema,
so a new drug (or a third, or a wash-in time course) needs no code change here
and none in the batch driver.

What the conversion does
------------------------
* Each populated ``*_file`` column becomes one row, mapped through
  ``WIDE_COLUMN_MAP`` to a ``(protocol, condition, seq)`` triple. The existing
  ``*_apamin_*`` columns -- which no code has ever read -- become
  ``condition=apamin, seq=1`` rows.
* Bracketed multi-file cells (``"[a.abf, b.abf]"``, sometimes with trailing
  whitespace) are *expanded* into one row each. The old pipeline dropped these
  outright; here nothing is silently lost, and a genuine duplicate protocol
  within a block is reported by the batch driver's validation.
* ``SK_VC`` rows are emitted even though no analyzer exists for them. The batch
  reports them as ``no analyzer`` -- visible, rather than quietly absent.

Cell- and animal-level metadata are deliberately *not* folded in: they stay in
their own tables (``cells``, ``mice``) and are joined by key downstream.

Usage
-----
    conda run --no-capture-output -n intrinsic_props python scripts/sheet_to_long.py \
        --sheet scripts/cells_forsberg.csv --out scripts/recordings.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd


# wide column -> (protocol, condition, seq)
# seq orders blocks within a cell; condition names them. Both are free to grow:
# a wash column would be ("ramp", "wash", 2), a second drug ("ramp", "ttx", 3).
WIDE_COLUMN_MAP = {
    "free_run_file": ("free_run", "baseline", 0),
    "hyperpol_file": ("hyperpol", "baseline", 0),
    "small_steps_file": ("small_steps", "baseline", 0),
    "ramp_file": ("ramp", "baseline", 0),
    "sagIh_file": ("sagIh", "baseline", 0),
    "SK_VC_file": ("SK_VC", "baseline", 0),
    "small_steps_apamin_file": ("small_steps", "apamin", 1),
    "ramp_apamin_file": ("ramp", "apamin", 1),
    "hyperpol_apamin_file": ("hyperpol", "apamin", 1),
    "sagIh_apamin_file": ("sagIh", "apamin", 1),
    "SK_VC_apamin_file": ("SK_VC", "apamin", 1),
}

# condition -> drug label written into the long sheet. Passthrough only; no
# code interprets it.
CONDITION_DRUG = {"baseline": "", "apamin": "apamin"}

# Display/sort order for protocols in the generated sheet. Purely cosmetic --
# the batch driver imposes its own execution order.
PROTOCOL_SORT = ["small_steps", "ramp", "sagIh", "hyperpol", "free_run", "SK_VC"]

LONG_COLUMNS = [
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


def _blank(value) -> bool:
    return (
        value is None
        or (isinstance(value, float) and pd.isna(value))
        or str(value).strip() == ""
    )


def split_files(value: str) -> list[str]:
    """Split a ``*_file`` cell into individual ABF filenames.

    Handles the bracketed-list convention used for apamin pairs,
    ``"[a.abf, b.abf]"``, including the trailing spaces that appear in the real
    sheet. A plain single filename comes back as a one-element list.
    """
    val = str(value).strip().strip("[]").strip()
    return [part.strip() for part in val.split(",") if part.strip()]


def wide_to_long(sheet: pd.DataFrame) -> pd.DataFrame:
    """Expand the wide cell sheet into one row per ABF file."""
    known = [c for c in WIDE_COLUMN_MAP if c in sheet.columns]
    if not known:
        raise SystemExit(
            f"Sheet has none of the known protocol columns: {sorted(WIDE_COLUMN_MAP)}"
        )

    rows: list[dict] = []
    for _, row in sheet.iterrows():
        cell_id = "" if _blank(row.get("cell_id")) else str(row["cell_id"]).strip()
        if not cell_id:
            continue
        abf_folder = (
            "" if _blank(row.get("abf_folder")) else str(row["abf_folder"]).strip()
        )

        # The wide sheet's per-cell `exclusion` is a QC verdict on the whole
        # cell; carry it into `notes` so it stays visible. It is deliberately
        # NOT copied into `exclude` -- dropping a whole cell silently on a note
        # like "sag" would be a surprise. Set `exclude` by hand to skip a row.
        cell_note = "" if _blank(row.get("notes")) else str(row["notes"]).strip()
        exclusion = (
            "" if _blank(row.get("exclusion")) else str(row["exclusion"]).strip()
        )
        parts = [p for p in (cell_note, f"exclusion: {exclusion}" if exclusion else "") if p]
        note = "; ".join(parts)

        for column in known:
            raw = row.get(column)
            if _blank(raw):
                continue
            protocol, condition, seq = WIDE_COLUMN_MAP[column]
            for abf_file in split_files(raw):
                rows.append(
                    {
                        "cell_id": cell_id,
                        "abf_folder": abf_folder,
                        "abf_file": abf_file,
                        "protocol": protocol,
                        "condition": condition,
                        "seq": seq,
                        "drug": CONDITION_DRUG.get(condition, ""),
                        "concentration": "",
                        "exclude": "",
                        "notes": note,
                    }
                )

    long_df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    if long_df.empty:
        return long_df

    rank = {p: i for i, p in enumerate(PROTOCOL_SORT)}
    long_df = (
        long_df.assign(_p=long_df["protocol"].map(lambda p: rank.get(p, 99)))
        .sort_values(["cell_id", "seq", "_p", "abf_file"])
        .drop(columns="_p")
    )
    return long_df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sheet",
        type=Path,
        default=REPO_ROOT / "scripts" / "cells_forsberg.csv",
        help="Wide cell sheet to convert (default: scripts/cells_forsberg.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "scripts" / "recordings.csv",
        help="Long manifest to write (default: scripts/recordings.csv)",
    )
    args = parser.parse_args(argv)

    if not args.sheet.exists():
        raise SystemExit(f"Sheet not found: {args.sheet}")

    sheet = pd.read_csv(args.sheet, dtype=str)
    long_df = wide_to_long(sheet)
    if long_df.empty:
        raise SystemExit("Conversion produced no rows -- is this the right sheet?")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)

    n_cells = long_df["cell_id"].nunique()
    n_blocks = long_df.groupby(["cell_id", "seq"]).ngroups
    print(f"Wrote {args.out}")
    print(f"  {len(long_df)} recordings  |  {n_cells} cells  |  {n_blocks} blocks")
    print("\n  by condition / protocol:")
    for (condition, protocol), n in long_df.groupby(["condition", "protocol"]).size().items():
        print(f"    {condition:>10}  {protocol:<12} {n}")

    counts = long_df.groupby(["cell_id", "seq", "protocol"]).size()
    dupes = counts[counts > 1]
    if not dupes.empty:
        print(
            f"\n  {len(dupes)} block(s) with a repeated protocol (expanded "
            "multi-file cells; batch validation will report these):"
        )
        for (cell_id, seq, protocol), n in dupes.items():
            print(f"    {cell_id}  seq={seq}  {protocol} x{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
