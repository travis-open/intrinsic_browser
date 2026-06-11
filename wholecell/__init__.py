"""
wholecell — interactive analysis of whole-cell patch-clamp electrophysiology data.

Typical usage
-------------
>>> from wholecell.core.cell import Cell
>>> cell = Cell(cell_id="cell_01")
>>> cell.add_recording("path/to/steps.abf")
>>> cell.add_recording("path/to/steps2.abf")
>>> sc = cell.create_sweep_collection(
...     name="current_steps",
...     sweeps=[{"filename": "steps.abf", "sweep_index": 0},
...             {"filename": "steps.abf", "sweep_index": 1}]
... )
>>> cell.save_session("session.json")
"""

__version__ = "0.1.0"
