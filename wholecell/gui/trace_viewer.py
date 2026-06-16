"""
trace_viewer.py
---------------
Interactive trace viewer for whole-cell current-clamp recordings.

Layout
------
  Left sidebar : Cell ID field, ABF file/sweep tree, collection management,
                 sweep list, analysis buttons, results display, Save Session
  Right panel  : voltage plot (+ optional dV/dt and current panels below)
  Menu bar     : File menu (Open Directory, Save Session) +
                 Export menu (Spike Table, Sweep Summary, Cell Summary)

Controls
--------
  ↑ / ↓          navigate cursor through the sweep list
  Space / Enter   toggle checkbox of the cursor sweep
  A               check all sweeps
  N               uncheck all sweeps
  F               toggle lowpass filter (2 kHz default) for all shown sweeps
  D               toggle dV/dt panel
  C               toggle current panel
  S               toggle spike markers
  Q / Escape      quit

The cursor sweep is always shown as a bright white trace regardless of its
checkbox state.  Checked sweeps are shown in colour (viridis-like cycle).

"Average & Analyze Subthreshold" averages all checked sweeps (same filter
applied) and runs passive analysis on the result.  All checked sweeps must
share the same step epoch amplitude and duration; a clear error is shown if
they do not.  Results are stored on the Cell object for export.

"Find Spikes" detects spikes in all checked sweeps and stores the result on
the Cell object.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _sweep_color(index: int, n_total: int):
    import pyqtgraph as pg
    return pg.intColor(index, hues=max(n_total, 1), minValue=180)


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class TraceViewer:
    """pyqtgraph-based viewer for whole-cell recordings, driven by a Cell.

    Parameters
    ----------
    cell : Cell
        The Cell object backing this viewer.  May have zero, one, or more
        recordings loaded.  If collections exist the first one is shown.
    step_epoch_index : int or None
        Zero-based epoch index of the current step epoch.  Auto-detected
        from the first recording when a collection is set if not provided.
    lowpass_hz : float or None
        Default lowpass filter cutoff toggled with F.
    """

    def __init__(
        self,
        cell,
        step_epoch_index: int | None = None,
        lowpass_hz: float | None = None,
    ) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets
        from wholecell.config import load_settings

        self._cell = cell
        self._current_collection = None
        self._step_epoch_index = step_epoch_index

        _cfg = load_settings()
        if lowpass_hz is None:
            lowpass_hz = _cfg.get("lowpass_hz", 2000.0)
        self._default_lowpass_hz = lowpass_hz
        self._cfg_dvdt = _cfg.get("dvdt_threshold_mv_per_ms", 20.0)
        self._cfg_peak_window = _cfg.get("peak_window_ms", 20.0)

        self._cursor = 0
        self._filter_on = False
        self._show_current = False
        self._show_dvdt = False
        self._show_spikes = False
        self._solo_mode = False

        # Analysis state — keyed by (filename, sweep_index)
        self._spike_data: dict[tuple[str, int], list] = {}
        self._tau_lookup: dict[tuple[str, int], dict] = {}
        self._avg_tau_fit: dict | None = None

        # Per-position curve sets (position = index into current collection)
        self._curves: dict[int, dict] = {}

        # ---- Qt application ----
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        # ---- Main window (QMainWindow for menu bar support) ----
        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle(
            f"Trace Viewer — {cell.cell_id}" if cell.cell_id else "Trace Viewer"
        )
        self._win.resize(1400, 780)
        self._win.keyPressEvent = self._on_key
        self._win.closeEvent = self._on_close_event

        # Central widget
        central = QtWidgets.QWidget()
        self._win.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ---- Menu bar ----
        self._build_menu_bar()

        # ---- Left sidebar ----
        left = QtWidgets.QWidget()
        left.setFixedWidth(240)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Cell ID
        id_row = QtWidgets.QHBoxLayout()
        id_lbl = QtWidgets.QLabel("Cell ID:")
        id_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self._cell_id_edit = QtWidgets.QLineEdit(cell.cell_id)
        self._cell_id_edit.setStyleSheet(
            "QLineEdit { background: #1a1a1a; color: #eee; font-size: 11px; "
            "border: 1px solid #444; padding: 2px; }"
        )
        self._cell_id_edit.editingFinished.connect(self._on_cell_id_changed)
        id_row.addWidget(id_lbl)
        id_row.addWidget(self._cell_id_edit)
        left_layout.addLayout(id_row)

        # File / sweep tree
        files_lbl = QtWidgets.QLabel("Files  (check sweeps to add to collection)")
        files_lbl.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        left_layout.addWidget(files_lbl)

        self._file_tree = QtWidgets.QTreeWidget()
        self._file_tree.setHeaderHidden(True)
        self._file_tree.setStyleSheet(
            "QTreeWidget { background: #111; color: #ddd; font-size: 11px; }"
            "QTreeWidget::item:selected { background: #2a2a4a; }"
            "QTreeWidget::indicator { width: 13px; height: 13px; }"
            "QTreeWidget::indicator:unchecked { border: 1px solid #888; background: #2a2a2a; }"
            "QTreeWidget::indicator:checked { border: 1px solid #aaa; background: #5588dd; }"
        )
        self._file_tree.setFixedHeight(160)
        self._file_tree.itemChanged.connect(self._on_tree_item_changed)
        left_layout.addWidget(self._file_tree)

        btn_add = QtWidgets.QPushButton("Add Checked to Collection")
        btn_add.setStyleSheet("font-size: 11px; padding: 3px;")
        btn_add.clicked.connect(self._on_add_to_collection)
        left_layout.addWidget(btn_add)

        # Collections selector
        coll_row = QtWidgets.QHBoxLayout()
        coll_lbl = QtWidgets.QLabel("Collection:")
        coll_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self._collections_combo = QtWidgets.QComboBox()
        self._collections_combo.setStyleSheet(
            "QComboBox { background: #1a1a1a; color: #ddd; font-size: 11px; "
            "border: 1px solid #444; }"
            "QComboBox QAbstractItemView { background: #1a1a1a; color: #ddd; "
            "selection-background-color: #2a2a4a; border: 1px solid #555; }"
        )
        self._collections_combo.currentTextChanged.connect(self._on_collection_changed)
        coll_row.addWidget(coll_lbl)
        coll_row.addWidget(self._collections_combo, stretch=1)
        left_layout.addLayout(coll_row)

        # Sweep list
        sweep_lbl = QtWidgets.QLabel("Sweeps  (Space = toggle)")
        sweep_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        left_layout.addWidget(sweep_lbl)

        self._list = QtWidgets.QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background: #1a1a1a; color: #ddd; font-size: 11px; }"
            "QListWidget::item:selected { background: #333; }"
            "QListWidget::indicator { width: 13px; height: 13px; }"
            "QListWidget::indicator:unchecked { border: 1px solid #888; background: #2a2a2a; }"
            "QListWidget::indicator:checked { border: 1px solid #aaa; background: #5588dd; }"
        )
        self._list.itemChanged.connect(self._on_item_changed)
        left_layout.addWidget(self._list, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("All")
        btn_none = QtWidgets.QPushButton("None")
        self._btn_solo = QtWidgets.QPushButton("Solo  (V)")
        for btn in (btn_all, btn_none, self._btn_solo):
            btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        btn_all.clicked.connect(self._check_all)
        btn_none.clicked.connect(self._check_none)
        self._btn_solo.clicked.connect(self._toggle_solo)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(self._btn_solo)
        left_layout.addLayout(btn_row)

        thresh_row = QtWidgets.QHBoxLayout()
        thresh_lbl = QtWidgets.QLabel("dV/dt thresh:")
        thresh_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self._dvdt_spin = QtWidgets.QDoubleSpinBox()
        self._dvdt_spin.setRange(1.0, 200.0)
        self._dvdt_spin.setSingleStep(1.0)
        self._dvdt_spin.setDecimals(1)
        self._dvdt_spin.setValue(self._cfg_dvdt)
        self._dvdt_spin.setSuffix(" mV/ms")
        self._dvdt_spin.setStyleSheet(
            "QDoubleSpinBox { background: #1a1a1a; color: #ddd; font-size: 11px; "
            "border: 1px solid #444; padding: 2px; }"
        )
        thresh_row.addWidget(thresh_lbl)
        thresh_row.addWidget(self._dvdt_spin, stretch=1)
        left_layout.addLayout(thresh_row)

        peak_row = QtWidgets.QHBoxLayout()
        peak_lbl = QtWidgets.QLabel("Peak window:")
        peak_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self._peak_window_spin = QtWidgets.QDoubleSpinBox()
        self._peak_window_spin.setRange(5.0, 100.0)
        self._peak_window_spin.setSingleStep(1.0)
        self._peak_window_spin.setDecimals(1)
        self._peak_window_spin.setValue(self._cfg_peak_window)
        self._peak_window_spin.setSuffix(" ms")
        self._peak_window_spin.setStyleSheet(
            "QDoubleSpinBox { background: #1a1a1a; color: #ddd; font-size: 11px; "
            "border: 1px solid #444; padding: 2px; }"
        )
        peak_row.addWidget(peak_lbl)
        peak_row.addWidget(self._peak_window_spin, stretch=1)
        left_layout.addLayout(peak_row)

        # Persist spinbox values whenever the user changes them
        from wholecell.config import update_setting
        self._dvdt_spin.valueChanged.connect(
            lambda v: update_setting("dvdt_threshold_mv_per_ms", v)
        )
        self._peak_window_spin.valueChanged.connect(
            lambda v: update_setting("peak_window_ms", v)
        )

        btn_spikes = QtWidgets.QPushButton("Find Spikes  (S = toggle)")
        btn_spikes.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #2a2a4a; color: #aaf;"
        )
        btn_spikes.clicked.connect(self._run_spike_detection)
        left_layout.addWidget(btn_spikes)

        def _make_status_checkbox(tooltip: str) -> QtWidgets.QCheckBox:
            cb = QtWidgets.QCheckBox()
            cb.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            cb.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            cb.setToolTip(tooltip)
            return cb

        btn_fi = QtWidgets.QPushButton("F-I Curve")
        btn_fi.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #2a3a4a; color: #8cf;"
        )
        btn_fi.clicked.connect(self._show_fi_curve)
        self._chk_fi = _make_status_checkbox("F-I Curve has been run for this cell")
        row_fi = QtWidgets.QHBoxLayout()
        row_fi.addWidget(btn_fi, stretch=1)
        row_fi.addWidget(self._chk_fi)
        left_layout.addLayout(row_fi)

        btn_ramp = QtWidgets.QPushButton("Analyze Ramp APs")
        btn_ramp.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #3a2a4a; color: #c8f;"
        )
        btn_ramp.clicked.connect(self._run_ramp_analysis)
        self._chk_ramp = _make_status_checkbox("Ramp AP analysis has been run for this cell")
        row_ramp = QtWidgets.QHBoxLayout()
        row_ramp.addWidget(btn_ramp, stretch=1)
        row_ramp.addWidget(self._chk_ramp)
        left_layout.addLayout(row_ramp)

        btn_avg = QtWidgets.QPushButton("Average && Analyze Subthreshold")
        btn_avg.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #2a4a2a; color: #8f8;"
        )
        btn_avg.clicked.connect(self._run_average_analysis)
        self._chk_avg = _make_status_checkbox("Average subthreshold analysis has been run for this cell")
        row_avg = QtWidgets.QHBoxLayout()
        row_avg.addWidget(btn_avg, stretch=1)
        row_avg.addWidget(self._chk_avg)
        left_layout.addLayout(row_avg)

        btn_passive = QtWidgets.QPushButton("Analyze Each Sweep (Passive)")
        btn_passive.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #2a4a3a; color: #8fb;"
        )
        btn_passive.clicked.connect(self._run_per_sweep_passive_analysis)
        self._chk_passive = _make_status_checkbox("Per-sweep passive analysis has been run for this cell")
        row_passive = QtWidgets.QHBoxLayout()
        row_passive.addWidget(btn_passive, stretch=1)
        row_passive.addWidget(self._chk_passive)
        left_layout.addLayout(row_passive)

        btn_vrest = QtWidgets.QPushButton("Measure V_rest")
        btn_vrest.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #2a3a2a; color: #af8;"
        )
        btn_vrest.clicked.connect(self._run_vrest_analysis)
        self._chk_vrest = _make_status_checkbox("Resting potential has been measured for this cell")
        row_vrest = QtWidgets.QHBoxLayout()
        row_vrest.addWidget(btn_vrest, stretch=1)
        row_vrest.addWidget(self._chk_vrest)
        left_layout.addLayout(row_vrest)

        btn_clear = QtWidgets.QPushButton("Clear Plot")
        btn_clear.setStyleSheet("font-size: 11px; padding: 3px;")
        btn_clear.clicked.connect(self._clear_plot)
        left_layout.addWidget(btn_clear)

        # Results display
        res_lbl = QtWidgets.QLabel("Results")
        res_lbl.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        left_layout.addWidget(res_lbl)

        self._results_box = QtWidgets.QTextEdit()
        self._results_box.setReadOnly(True)
        self._results_box.setFixedHeight(150)
        self._results_box.setStyleSheet(
            "QTextEdit { background: #111; color: #cfc; font-size: 11px; "
            "font-family: monospace; border: 1px solid #333; }"
        )
        self._results_box.setPlaceholderText("No analysis yet.")
        left_layout.addWidget(self._results_box)

        # Save Session button
        btn_save = QtWidgets.QPushButton("Save Session")
        btn_save.setStyleSheet("font-size: 11px; padding: 3px; background: #3a3a2a; color: #ff8;")
        btn_save.clicked.connect(self._on_save_session)
        left_layout.addWidget(btn_save)

        main_layout.addWidget(left)

        # ---- Right: pyqtgraph plots ----
        self._pg_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self._pg_widget, stretch=1)

        self._plot_v = self._pg_widget.addPlot(row=0, col=0)
        self._plot_v.setLabel("left", "Voltage", units="mV")
        self._plot_v.getAxis("left").enableAutoSIPrefix(False)
        self._plot_v.setLabel("bottom", "Time", units="s")
        self._plot_v.showGrid(x=True, y=True, alpha=0.25)

        self._plot_d = self._pg_widget.addPlot(row=1, col=0)
        self._plot_d.setLabel("left", "dV/dt", units="mV/ms")
        self._plot_d.getAxis("left").enableAutoSIPrefix(False)
        self._plot_d.setLabel("bottom", "Time", units="s")
        self._plot_d.showGrid(x=True, y=True, alpha=0.25)
        self._plot_d.setXLink(self._plot_v)
        self._plot_d.setVisible(False)

        self._plot_i = self._pg_widget.addPlot(row=2, col=0)
        self._plot_i.setLabel("left", "Current", units="pA")
        self._plot_i.getAxis("left").enableAutoSIPrefix(False)
        self._plot_i.setLabel("bottom", "Time", units="s")
        self._plot_i.showGrid(x=True, y=True, alpha=0.25)
        self._plot_i.setXLink(self._plot_v)
        self._plot_i.setVisible(False)

        self._status = pg.LabelItem(justify="left")
        self._pg_widget.addItem(self._status, row=3, col=0)

        # Average trace curves
        self._avg_curve_v = self._plot_v.plot(
            pen=self._make_pen("#0ff", width=3, alpha=220), name="avg"
        )
        self._avg_curve_i = self._plot_i.plot(
            pen=self._make_pen("#0ff", width=2, alpha=200)
        )
        self._avg_curve_d = self._plot_d.plot(
            pen=self._make_pen("#0ff", width=2, alpha=200)
        )
        self._avg_tau_curve = self._plot_v.plot(
            pen=self._make_pen("#ff0", width=2, alpha=200,
                               style=QtCore.Qt.PenStyle.DashLine)
        )

        # Spike marker scatter plots
        _sp = dict(pen=None, symbolPen=None)
        self._mk_threshold = self._plot_v.plot(
            symbol="t1", symbolSize=11, symbolBrush="#f80", **_sp)
        self._mk_peak = self._plot_v.plot(
            symbol="o",  symbolSize=10, symbolBrush="#4f4", **_sp)
        self._mk_trough = self._plot_v.plot(
            symbol="d",  symbolSize=10, symbolBrush="#f44", **_sp)
        self._mk_slow_ahp = self._plot_v.plot(
            symbol="s",  symbolSize=10, symbolBrush="#88f", **_sp)
        self._mk_upstroke = self._plot_d.plot(
            symbol="t",  symbolSize=11, symbolBrush="#0ff", **_sp)
        self._mk_downstroke = self._plot_d.plot(
            symbol="t1", symbolSize=11, symbolBrush="#f4f", **_sp)
        for mk in (self._mk_threshold, self._mk_peak, self._mk_trough,
                   self._mk_slow_ahp, self._mk_upstroke, self._mk_downstroke):
            mk.setZValue(20)

        # Auto-create a default collection for every loaded recording
        # so sweeps are visible immediately without manual tree interaction.
        self._auto_create_collections(list(self._cell.recordings.keys()))

        # Populate file tree and collections from Cell state
        self._build_file_tree()
        self._update_collections_combo()

        # Show first collection if any exist
        if self._cell.collections:
            first_name = next(iter(self._cell.collections))
            idx = self._collections_combo.findText(first_name)
            if idx >= 0:
                self._collections_combo.setCurrentIndex(idx)
            self._on_collection_changed(first_name)
        else:
            self._update_status()

        self._win.show()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        from pyqtgraph.Qt import QtWidgets

        mb = self._win.menuBar()
        mb.setStyleSheet("QMenuBar { background: #1a1a1a; color: #ddd; }")

        # File menu
        file_menu = mb.addMenu("File")
        act_session = file_menu.addAction("Open Session…")
        act_session.triggered.connect(self._on_open_session)
        act_new_cell = file_menu.addAction("New Cell…")
        act_new_cell.triggered.connect(self._on_new_cell)
        act_open = file_menu.addAction("Open Directory…")
        act_open.triggered.connect(self._on_open_directory)
        file_menu.addSeparator()
        act_save = file_menu.addAction("Save Session")
        act_save.triggered.connect(self._on_save_session)
        act_save.setShortcut("Ctrl+S")

        # Export menu
        exp_menu = mb.addMenu("Export")
        act_spikes = exp_menu.addAction("Spike Table (CSV)…")
        act_spikes.triggered.connect(self._on_export_spike_table)
        act_sweep = exp_menu.addAction("Sweep Summary (CSV)…")
        act_sweep.triggered.connect(self._on_export_sweep_summary)
        act_cell = exp_menu.addAction("Cell Summary (JSON)…")
        act_cell.triggered.connect(self._on_export_cell_summary)

    # ------------------------------------------------------------------
    # File tree management
    # ------------------------------------------------------------------

    def _build_file_tree(self) -> None:
        """Populate the file/sweep tree from currently loaded recordings."""
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._file_tree.blockSignals(True)
        self._file_tree.clear()

        for fname, rec in self._cell.recordings.items():
            file_item = QtWidgets.QTreeWidgetItem([fname])
            file_item.setFlags(
                file_item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            file_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            file_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, fname)

            for i in range(rec.n_sweeps):
                sweep_item = QtWidgets.QTreeWidgetItem([f"  Sweep {i:03d}"])
                sweep_item.setFlags(
                    sweep_item.flags()
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    | QtCore.Qt.ItemFlag.ItemIsEnabled
                )
                sweep_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                sweep_item.setData(
                    0, QtCore.Qt.ItemDataRole.UserRole, (fname, i)
                )
                file_item.addChild(sweep_item)

            self._file_tree.addTopLevelItem(file_item)
            file_item.setExpanded(False)

        self._file_tree.blockSignals(False)

    def _on_tree_item_changed(self, item, col: int) -> None:
        """Propagate parent check state to children and vice-versa."""
        from pyqtgraph.Qt import QtCore

        self._file_tree.blockSignals(True)
        state = item.checkState(0)
        if item.parent() is None:
            # Parent toggled — propagate to all children
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
        self._file_tree.blockSignals(False)

    def _auto_create_collections(self, stems: list[str]) -> None:
        """Create a default all-sweeps collection for each stem that lacks one."""
        for stem in stems:
            if stem not in self._cell.recordings:
                continue
            if stem in self._cell.collections:
                continue
            rec = self._cell.recordings[stem]
            sweeps = [{"filename": stem, "sweep_index": i}
                      for i in range(rec.n_sweeps)]
            try:
                self._cell.create_sweep_collection(stem, sweeps)
            except Exception:
                pass

    def _on_add_to_collection(self) -> None:
        """Collect checked sweep tree items and create a SweepCollection."""
        from pyqtgraph.Qt import QtCore, QtWidgets
        from wholecell.core.sweep_collection import SweepRef

        checked: list[SweepRef] = []
        root = self._file_tree.invisibleRootItem()
        for fi in range(root.childCount()):
            file_item = root.child(fi)
            fname = file_item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            for si in range(file_item.childCount()):
                sweep_item = file_item.child(si)
                if sweep_item.checkState(0) == QtCore.Qt.CheckState.Checked:
                    _, sweep_idx = sweep_item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    checked.append(SweepRef(fname, sweep_idx))

        if not checked:
            QtWidgets.QMessageBox.warning(
                self._win, "No sweeps selected",
                "Check at least one sweep in the file tree."
            )
            return

        # Determine collection name
        filenames = list(dict.fromkeys(ref.filename for ref in checked))
        if len(filenames) == 1:
            name = filenames[0]
        else:
            name, ok = QtWidgets.QInputDialog.getText(
                self._win, "Collection Name",
                "Sweeps span multiple files.\nEnter a name for this collection:"
            )
            if not ok or not name.strip():
                return
            name = name.strip()

        sweeps_dicts = [
            {"filename": r.filename, "sweep_index": r.sweep_index}
            for r in checked
        ]
        try:
            self._cell.create_sweep_collection(name, sweeps_dicts, overwrite=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._win, "Error creating collection", str(exc)
            )
            return

        self._update_collections_combo()
        idx = self._collections_combo.findText(name)
        if idx >= 0:
            self._collections_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Collections management
    # ------------------------------------------------------------------

    def _update_collections_combo(self) -> None:
        prev = self._collections_combo.currentText()
        self._collections_combo.blockSignals(True)
        self._collections_combo.clear()
        for name in self._cell.collections:
            self._collections_combo.addItem(name)
        idx = self._collections_combo.findText(prev)
        if idx >= 0:
            self._collections_combo.setCurrentIndex(idx)
        self._collections_combo.blockSignals(False)

    def _on_collection_changed(self, name: str) -> None:
        if not name or name not in self._cell.collections:
            return
        self._current_collection = self._cell.collections[name]
        self._update_axis_labels()

        # Re-detect step epoch every time the collection changes — different
        # files may have different protocols with the step at a different index.
        self._auto_detect_epoch()

        # Clear average trace and spike state so nothing from the previous
        # collection bleeds into the new one.
        self._clear_avg_curves()
        self._spike_data = {}
        self._update_spike_markers()

        self._cursor = 0
        self._populate_sweep_list()
        self._clear_curves()
        self._full_update()

    def _auto_detect_epoch(self) -> None:
        if self._current_collection is None:
            return
        try:
            from wholecell.io.abf_reader import find_step_epoch
            ref = self._current_collection.sweeps[0]
            rec = self._cell.recordings[ref.filename]
            self._step_epoch_index = find_step_epoch(rec.filepath)
        except Exception:
            self._step_epoch_index = 1

    # ------------------------------------------------------------------
    # Sweep list helpers
    # ------------------------------------------------------------------

    def _populate_sweep_list(self) -> None:
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._list.blockSignals(True)
        self._list.clear()
        if self._current_collection:
            for pos, ref in enumerate(self._current_collection.sweeps):
                self._list.addItem(self._list_item(pos, ref))
        self._list.blockSignals(False)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _list_item(self, pos: int, ref):
        from pyqtgraph.Qt import QtCore, QtWidgets
        item = QtWidgets.QListWidgetItem(ref.display_label)
        item.setFlags(
            item.flags()
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        item.setCheckState(QtCore.Qt.CheckState.Checked)
        return item

    def _ref_at(self, pos: int):
        """Return the SweepRef at position pos in the current collection."""
        return self._current_collection.sweeps[pos]

    def _checked_sweeps(self) -> list[int]:
        """Return list of checked positions in the sweep list."""
        from pyqtgraph.Qt import QtCore
        return [
            i for i in range(self._list.count())
            if self._list.item(i).checkState() == QtCore.Qt.CheckState.Checked
        ]

    # ------------------------------------------------------------------
    # Full refresh
    # ------------------------------------------------------------------

    def _clear_avg_curves(self) -> None:
        for curve in (self._avg_curve_v, self._avg_curve_i,
                      self._avg_curve_d, self._avg_tau_curve):
            curve.setData([], [])
        self._avg_tau_fit = None

    def _clear_plot(self) -> None:
        self._check_none()
        self._clear_avg_curves()
        self._results_box.clear()

    def _clear_curves(self) -> None:
        for pos in list(self._curves.keys()):
            self._remove_curves(pos)

    def _full_update(self) -> None:
        if self._current_collection is None:
            self._update_status()
            return

        lowpass = self._default_lowpass_hz if self._filter_on else None
        n = self._current_collection.n_sweeps
        to_show = (
            {self._cursor} if self._solo_mode else set(self._checked_sweeps())
        )

        for pos in list(self._curves.keys()):
            if pos not in to_show:
                self._remove_curves(pos)

        for pos in to_show:
            ref = self._ref_at(pos)
            color = "#fff" if self._solo_mode else _sweep_color(pos, n)
            pen = self._make_pen(color, 2 if self._solo_mode else 1, 200)

            try:
                t, v, _ = self._current_collection.get_sweep_arrays(
                    ref, lowpass_hz=lowpass
                )
            except Exception:
                continue

            if pos not in self._curves:
                self._curves[pos] = self._make_curves(pos)
            c = self._curves[pos]

            c["v"].setData(t, v)
            c["v"].setPen(pen)
            c["v"].setZValue(1)

            if self._show_current:
                t_i, i_arr = self._get_current_trace(pos)
                c["i"].setData(t_i, i_arr)
                c["i"].setPen(pen)
            else:
                c["i"].setData([], [])

            if self._show_dvdt:
                c["d"].setData(t, np.gradient(v, t) / 1000.0)
                c["d"].setPen(pen)
            else:
                c["d"].setData([], [])

            self._update_tau(pos, c)

        self._update_spike_markers()
        self._update_avg_tau()
        self._list.setCurrentRow(self._cursor)
        self._update_status()

    def _make_curves(self, pos: int) -> dict:
        from pyqtgraph.Qt import QtCore
        c: dict = {}
        c["v"] = self._plot_v.plot(name=f"pos{pos:03d}")
        c["i"] = self._plot_i.plot()
        c["d"] = self._plot_d.plot()
        c["tau"] = self._plot_v.plot(
            pen=self._make_pen("#ff0", 2, 200,
                               style=QtCore.Qt.PenStyle.DashLine)
        )
        return c

    def _remove_curves(self, pos: int) -> None:
        if pos not in self._curves:
            return
        for item in self._curves[pos].values():
            for plot in (self._plot_v, self._plot_i, self._plot_d):
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        del self._curves[pos]

    def _make_pen(self, color, width=1, alpha=255, style=None):
        import pyqtgraph as pg
        pen = pg.mkPen(color, width=width)
        if alpha < 255:
            c = pen.color()
            c.setAlpha(alpha)
            pen.setColor(c)
        if style is not None:
            pen.setStyle(style)
        return pen

    # ------------------------------------------------------------------
    # Spike detection
    # ------------------------------------------------------------------

    def _run_spike_detection(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        from wholecell.core.sweep_collection import SweepCollection, SweepRef
        from wholecell.analysis.spikes.finder import run_spike_detection

        if self._current_collection is None:
            self._results_box.setPlainText("No active collection.")
            return

        checked = self._checked_sweeps()
        if not checked:
            self._results_box.setPlainText("No sweeps checked for spike detection.")
            return

        refs = [self._ref_at(pos) for pos in checked]
        col = SweepCollection(
            name="_viewer_spike_temp",
            sweeps=refs,
            recordings=self._cell.recordings,
        )

        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        dvdt_thresh = self._dvdt_spin.value()
        peak_window = self._peak_window_spin.value()
        try:
            result = run_spike_detection(
                col, epoch_index=epoch_idx,
                dvdt_detection_mVms=dvdt_thresh,
                peak_search_window_ms=peak_window,
            )
        except Exception as exc:
            self._results_box.setPlainText(f"Spike detection error:\n{exc}")
            return

        # Store on Cell with filename provenance
        params = {
            "collection_name": self._current_collection.name,
            "epoch_index": epoch_idx,
            "dvdt_detection_mVms": dvdt_thresh,
            "peak_search_window_ms": peak_window,
            "source_sweeps": [{"filename": r.filename, "sweep_index": r.sweep_index}
                               for r in refs],
        }
        self._cell._store_result("spikes", result, params)

        # Update local spike data (keyed by (filename, sweep_index))
        for sd in result.get("per_sweep", []):
            key = (sd["filename"], sd["sweep_index"])
            self._spike_data[key] = sd.get("spikes", [])

        n_spikes = sum(len(v) for v in self._spike_data.values())
        self._results_box.setPlainText(
            f"Detected {n_spikes} spike(s) across {len(checked)} sweep(s).\n"
            "Press S to toggle markers."
        )
        self._show_spikes = True
        self._update_spike_markers()

    def _show_fi_curve(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        from wholecell.analysis.fi_curve import run_fi_analysis
        from wholecell.gui.fi_viewer import FICurveViewer

        if self._current_collection is None:
            self._results_box.setPlainText("No active collection.")
            return

        spikes_results = self._cell.results.get("spikes", [])
        if not spikes_results:
            self._results_box.setPlainText(
                "No spike detection results found.\nRun Find Spikes first."
            )
            return

        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        spike_entry = spikes_results[-1]
        try:
            fi_result = run_fi_analysis(
                self._current_collection, epoch_idx, spike_entry
            )
        except Exception as exc:
            self._results_box.setPlainText(f"F-I analysis error:\n{exc}")
            return

        self._cell._store_result("fi_curve", fi_result, {
            "collection_name": self._current_collection.name,
            "epoch_index": epoch_idx,
            "source": "trace_viewer",
        })

        cell_level = fi_result.get("cell_level", {})
        rheobase = cell_level.get("rheobase_pA", float("nan"))
        max_rate = cell_level.get("max_firing_rate_hz", float("nan"))
        slope = cell_level.get("fi_slope_hz_per_pA", float("nan"))

        msg_parts = []
        if rheobase == rheobase:  # not NaN
            msg_parts.append(f"Rheobase: {rheobase:.0f} pA")
        if max_rate == max_rate:
            msg_parts.append(f"Max rate: {max_rate:.1f} Hz")
        if slope == slope:
            msg_parts.append(f"F-I slope: {slope:.4f} Hz/pA")
        self._results_box.setPlainText("\n".join(msg_parts) if msg_parts else "F-I analysis complete.")
        self._refresh_analysis_checks()

        self._fi_viewer = FICurveViewer(
            fi_result=fi_result,
            title=f"{self._cell.cell_id} — F-I Curve",
        )
        self._fi_viewer.show()

    def _run_ramp_analysis(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        from wholecell.core.sweep_collection import SweepCollection, SweepRef
        from wholecell.analysis.ramp import run_ramp_analysis

        if self._current_collection is None:
            self._results_box.setPlainText("No active collection.")
            return

        spikes_results = self._cell.results.get("spikes", [])
        if not spikes_results:
            self._results_box.setPlainText(
                "No spike detection results found.\nRun Find Spikes first."
            )
            return

        checked = self._checked_sweeps()
        if not checked:
            self._results_box.setPlainText("No sweeps checked.")
            return

        refs = [self._ref_at(pos) for pos in checked]
        col = SweepCollection(
            name="_viewer_ramp_temp",
            sweeps=refs,
            recordings=self._cell.recordings,
        )

        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        lowpass = self._default_lowpass_hz if self._filter_on else None
        spike_entry = spikes_results[-1]

        try:
            result = run_ramp_analysis(col, epoch_idx, spike_entry, lowpass_hz=lowpass)
        except Exception as exc:
            self._results_box.setPlainText(f"Ramp AP analysis error:\n{exc}")
            return

        self._cell._store_result("ramp_evoked_APs", result, {
            "collection_name": self._current_collection.name,
            "epoch_index": epoch_idx,
            "lowpass_hz": lowpass,
            "n_sweeps": len(checked),
            "source": "ramp_analysis",
        })

        per_sweep = result.get("per_sweep", [])
        cell_level = result.get("cell_level", {})
        filt_str = f"LP {lowpass:.0f} Hz" if lowpass else "raw"

        hdr = (
            f"{'Sw':>3}  {'I_thr(pA)':>9}  {'Vthr(mV)':>8}  "
            f"{'Vpk(mV)':>8}  {'HW(ms)':>6}  {'Ht(mV)':>7}"
        )
        sep = "─" * len(hdr)
        rows_txt = [f"N={len(per_sweep)}  Filter={filt_str}", sep, hdr, sep]

        def _fmt(v, fmt=".1f"):
            try:
                return f"{float(v):{fmt}}" if float(v) == float(v) else "   NaN"
            except (TypeError, ValueError):
                return "   NaN"

        for r in per_sweep:
            rows_txt.append(
                f"{r.get('sweep_index', '?'):>3}  "
                f"{_fmt(r.get('current_at_threshold_pA')):>9}  "
                f"{_fmt(r.get('threshold_voltage_mV')):>8}  "
                f"{_fmt(r.get('peak_voltage_mV')):>8}  "
                f"{_fmt(r.get('half_width_ms')):>6}  "
                f"{_fmt(r.get('height_mV')):>7}"
            )

        rows_txt.append(sep)
        mean_thr = cell_level.get("mean_threshold_voltage_mV", float("nan"))
        std_thr = cell_level.get("std_threshold_voltage_mV", float("nan"))
        mean_i = cell_level.get("mean_current_at_threshold_pA", float("nan"))
        std_i = cell_level.get("std_current_at_threshold_pA", float("nan"))
        mean_hw = cell_level.get("mean_half_width_ms", float("nan"))
        std_hw = cell_level.get("std_half_width_ms", float("nan"))

        if mean_thr == mean_thr:
            rows_txt.append(f"Threshold  mean ± SD : {mean_thr:.1f} ± {std_thr:.1f} mV")
        if mean_i == mean_i:
            rows_txt.append(f"I at thr   mean ± SD : {mean_i:.1f} ± {std_i:.1f} pA")
        if mean_hw == mean_hw:
            rows_txt.append(f"Half-width mean ± SD : {mean_hw:.2f} ± {std_hw:.2f} ms")

        self._results_box.setPlainText("\n".join(rows_txt))
        self._refresh_analysis_checks()

    def _run_vrest_analysis(self) -> None:
        from wholecell.core.sweep_collection import SweepCollection, SweepRef
        from wholecell.analysis.vrest import run_vrest_analysis

        if self._current_collection is None:
            self._results_box.setPlainText("No active collection.")
            return

        checked = self._checked_sweeps()
        if not checked:
            self._results_box.setPlainText("No sweeps checked.")
            return

        refs = [self._ref_at(pos) for pos in checked]
        col = SweepCollection(
            name="_viewer_vrest_temp",
            sweeps=refs,
            recordings=self._cell.recordings,
        )

        lowpass = self._default_lowpass_hz if self._filter_on else None

        try:
            result = run_vrest_analysis(col, lowpass_hz=lowpass)
        except Exception as exc:
            self._results_box.setPlainText(f"V_rest analysis error:\n{exc}")
            return

        self._cell._store_result("v_rest", result, {
            "collection_name": self._current_collection.name,
            "lowpass_hz": lowpass,
            "n_sweeps": len(checked),
            "source": "vrest_analysis",
        })

        per_sweep = result.get("per_sweep", [])
        cell_level = result.get("cell_level", {})
        filt_str = f"LP {lowpass:.0f} Hz" if lowpass else "raw"

        hdr = f"{'Sw':>3}  {'V_rest (mV)':>11}"
        sep = "─" * len(hdr)
        rows_txt = [f"N={len(per_sweep)}  Filter={filt_str}", sep, hdr, sep]

        for r in per_sweep:
            v = r.get("v_rest_mV", float("nan"))
            v_str = f"{v:.2f}" if v == v else "   NaN"
            rows_txt.append(f"{r.get('sweep_index', '?'):>3}  {v_str:>11}")

        rows_txt.append(sep)
        v_mean = cell_level.get("v_rest_mV", float("nan"))
        v_std = cell_level.get("v_rest_std_mV", float("nan"))
        if v_mean == v_mean:
            rows_txt.append(f"V_rest  mean ± SD : {v_mean:.2f} ± {v_std:.2f} mV")

        self._results_box.setPlainText("\n".join(rows_txt))
        self._refresh_analysis_checks()

    def _update_spike_markers(self) -> None:
        visible = self._show_spikes and bool(self._spike_data)

        if not visible:
            for mk in (self._mk_threshold, self._mk_peak, self._mk_trough,
                       self._mk_slow_ahp, self._mk_upstroke, self._mk_downstroke):
                mk.setData([], [])
            return

        if self._current_collection is None:
            return

        checked = (
            {self._cursor} if self._solo_mode else set(self._checked_sweeps())
        )
        thresh_t, thresh_v = [], []
        peak_t, peak_v = [], []
        trough_t, trough_v = [], []
        slow_ahp_t, slow_ahp_v = [], []
        up_t, up_dvdt = [], []
        dn_t, dn_dvdt = [], []

        lowpass = self._default_lowpass_hz if self._filter_on else None

        for pos in checked:
            ref = self._ref_at(pos)
            key = (ref.filename, ref.sweep_index)
            spikes = self._spike_data.get(key)
            if not spikes:
                continue

            try:
                t_arr, v_arr, _ = self._current_collection.get_sweep_arrays(
                    ref, lowpass_hz=lowpass
                )
                dvdt = np.gradient(v_arr, t_arr) / 1000.0
            except Exception:
                t_arr = v_arr = dvdt = None

            for sp in spikes:
                thresh_t.append(sp["threshold_time_s"])
                thresh_v.append(sp["threshold_voltage_mV"])
                peak_t.append(sp["peak_time_s"])
                peak_v.append(sp["peak_voltage_mV"])
                trough_t.append(sp["trough_time_s"])
                trough_v.append(sp["trough_voltage_mV"])

                ahp_t = sp.get("slow_ahp_time_s")
                ahp_v = sp.get("slow_ahp_voltage_mV")
                if ahp_t is not None and not np.isnan(ahp_t):
                    slow_ahp_t.append(ahp_t)
                    slow_ahp_v.append(ahp_v)

                if dvdt is not None:
                    i0 = sp.get("threshold_index",
                                np.searchsorted(t_arr, sp["threshold_time_s"]))
                    i1 = sp.get("peak_index",
                                np.searchsorted(t_arr, sp["peak_time_s"]))
                    i2 = sp.get("trough_index",
                                np.searchsorted(t_arr, sp["trough_time_s"]))
                    i0 = max(0, min(i0, len(dvdt) - 1))
                    i1 = max(i0 + 1, min(i1, len(dvdt) - 1))
                    i2 = max(i1 + 1, min(i2, len(dvdt) - 1))

                    up_idx = i0 + int(np.argmax(dvdt[i0:i1]))
                    up_t.append(t_arr[up_idx])
                    up_dvdt.append(dvdt[up_idx])

                    dn_idx = i1 + int(np.argmin(dvdt[i1:i2]))
                    dn_t.append(t_arr[dn_idx])
                    dn_dvdt.append(dvdt[dn_idx])

        self._mk_threshold.setData(thresh_t, thresh_v)
        self._mk_peak.setData(peak_t, peak_v)
        self._mk_trough.setData(trough_t, trough_v)
        self._mk_slow_ahp.setData(slow_ahp_t, slow_ahp_v)
        self._mk_upstroke.setData(up_t, up_dvdt)
        self._mk_downstroke.setData(dn_t, dn_dvdt)

    # ------------------------------------------------------------------
    # Tau overlay
    # ------------------------------------------------------------------

    def _update_tau(self, pos: int, c) -> None:
        ref = self._ref_at(pos)
        key = (ref.filename, ref.sweep_index)
        fit = self._tau_lookup.get(key)
        if fit and fit.get("time") and fit.get("predicted"):
            c["tau"].setData(fit["time"], fit["predicted"])
        else:
            c["tau"].setData([], [])

    def _update_avg_tau(self) -> None:
        fit = self._avg_tau_fit
        if fit and fit.get("time") and fit.get("predicted"):
            self._avg_tau_curve.setData(fit["time"], fit["predicted"])
        else:
            self._avg_tau_curve.setData([], [])

    # ------------------------------------------------------------------
    # Average & Analyze Subthreshold
    # ------------------------------------------------------------------

    def _run_average_analysis(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        from wholecell.analysis.passive import (
            estimate_input_resistance,
            fit_time_constant,
            estimate_sag,
        )

        if self._current_collection is None:
            self._results_box.setPlainText("No active collection.")
            return

        checked = self._checked_sweeps()
        if len(checked) < 1:
            self._results_box.setPlainText("No sweeps checked.")
            return

        # Warn if any checked sweep contains detected spikes
        spiking = [pos for pos in checked
                   if self._spike_data.get(
                       (self._ref_at(pos).filename, self._ref_at(pos).sweep_index)
                   )]
        if not spiking:
            spiking = self._detect_spiking_sweeps(checked)
        if spiking:
            spiking_labels = [self._ref_at(p).display_label for p in spiking]
            reply = QtWidgets.QMessageBox.warning(
                self._win,
                "Spiking sweeps selected",
                f"Sweep(s) {spiking_labels} appear to contain action potentials.\n\n"
                "Subthreshold analysis on spiking data will give unreliable results.\n\n"
                "Proceed anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        try:
            step_levels, _ = self._validate_step_command(checked)
        except AssertionError as exc:
            self._results_box.setHtml(
                f'<span style="color:#f88;">Validation failed:<br>{exc}</span>'
            )
            return

        step_current_pA = step_levels[0]

        # Average voltage traces
        lowpass = self._default_lowpass_hz if self._filter_on else None
        arrays = []
        t_ref = None
        for pos in checked:
            ref = self._ref_at(pos)
            try:
                t, v, _ = self._current_collection.get_sweep_arrays(
                    ref, lowpass_hz=lowpass
                )
                arrays.append(v)
                if t_ref is None:
                    t_ref = t
            except Exception:
                continue

        if not arrays:
            self._results_box.setPlainText("Could not load sweep data.")
            return

        avg_v = np.mean(np.stack(arrays, axis=0), axis=0)

        # Show average on plot
        self._avg_curve_v.setData(t_ref, avg_v)
        if self._show_current:
            t_i, i_arr = self._get_current_trace(checked[0])
            self._avg_curve_i.setData(t_i, i_arr)
        if self._show_dvdt:
            self._avg_curve_d.setData(t_ref, np.gradient(avg_v, t_ref) / 1000.0)

        # Slice to step epoch
        ref0 = self._ref_at(checked[0])
        rec0 = self._cell.recordings[ref0.filename]
        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        epoch = rec0.get_epoch(ref0.sweep_index, epoch_idx)
        sl = slice(epoch.start_sample, epoch.end_sample)
        ep_time = t_ref[sl]
        ep_voltage = avg_v[sl]

        baseline_voltage = self._estimate_avg_baseline(t_ref, avg_v, checked[0])

        # Run analysis
        rin = estimate_input_resistance(baseline_voltage, ep_voltage, step_current_pA)
        tau_ms, fit_t, fit_v, fit_pred, tau_r2 = fit_time_constant(
            ep_time, ep_voltage, baseline_voltage
        )
        sag = estimate_sag(ep_time, ep_voltage, baseline_voltage)

        # Store tau fit for overlay
        if fit_t is not None and fit_pred is not None:
            self._avg_tau_fit = {
                "time": fit_t.tolist(),
                "predicted": fit_pred.tolist(),
            }
        else:
            self._avg_tau_fit = None
        self._update_avg_tau()

        # Store result on Cell with full provenance
        source_sweeps = [
            {"filename": self._ref_at(p).filename,
             "sweep_index": self._ref_at(p).sweep_index}
            for p in checked
        ]
        result_data = {
            "type": "averaged_passive",
            "source_sweeps": source_sweeps,
            "n_sweeps_averaged": len(checked),
            "step_current_pA": float(step_current_pA),
            "baseline_voltage_mV": float(baseline_voltage),
            "input_resistance_MOhm": float(rin),
            "time_constant_ms": float(tau_ms),
            "time_constant_r2": float(tau_r2),
            "sag_ratio": float(sag["sag_ratio"]),
            "sag_amplitude_mV": float(sag["sag_amplitude_mV"]),
            "sag_tau_ms": float(sag["sag_tau_ms"]),
            "_tau_fit": self._avg_tau_fit,
        }
        params = {
            "collection_name": self._current_collection.name,
            "epoch_index": epoch_idx,
            "lowpass_hz": self._default_lowpass_hz if self._filter_on else None,
            "n_sweeps_averaged": len(checked),
        }
        self._cell._store_result("passive_repeated_step", result_data, params)

        # Display results
        filt_str = f"LP {self._default_lowpass_hz:.0f} Hz" if self._filter_on else "raw"
        lines = [
            f"N sweeps averaged : {len(checked)}",
            f"Filter            : {filt_str}",
            f"Step current      : {step_current_pA:.1f} pA",
            f"Baseline voltage  : {baseline_voltage:.1f} mV",
            "─" * 28,
            f"Rin               : {rin:.1f} MΩ",
            f"τ_m               : {tau_ms:.2f} ms  (R²={tau_r2:.3f})",
            f"Sag ratio         : {sag['sag_ratio']:.3f}",
            f"Sag amplitude     : {sag['sag_amplitude_mV']:.2f} mV",
            f"Sag τ             : {sag['sag_tau_ms']:.1f} ms",
        ]
        self._results_box.setPlainText("\n".join(lines))
        self._refresh_analysis_checks()

    def _run_per_sweep_passive_analysis(self) -> None:
        """Analyze each checked sweep individually (no averaging).

        Unlike 'Average & Analyze Subthreshold', sweeps may have different
        step amplitudes. Each sweep is analyzed independently and all results
        are displayed in a per-sweep table.
        """
        from pyqtgraph.Qt import QtWidgets
        from wholecell.core.sweep_collection import SweepCollection, SweepRef
        from wholecell.analysis.passive import run_passive_analysis

        if self._current_collection is None:
            self._results_box.setPlainText("No active collection.")
            return

        checked = self._checked_sweeps()
        if not checked:
            self._results_box.setPlainText("No sweeps checked.")
            return

        # Warn (but don't block) if any checked sweep appears to spike
        spiking = [pos for pos in checked
                   if self._spike_data.get(
                       (self._ref_at(pos).filename, self._ref_at(pos).sweep_index)
                   )]
        if not spiking:
            spiking = self._detect_spiking_sweeps(checked)
        if spiking:
            spiking_labels = [self._ref_at(p).display_label for p in spiking]
            reply = QtWidgets.QMessageBox.warning(
                self._win,
                "Spiking sweeps selected",
                f"Sweep(s) {spiking_labels} appear to contain action potentials.\n\n"
                "Passive analysis on spiking data will give unreliable results.\n\n"
                "Proceed anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        refs = [self._ref_at(pos) for pos in checked]
        col = SweepCollection(
            name="_viewer_passive_sweep_temp",
            sweeps=refs,
            recordings=self._cell.recordings,
        )

        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        lowpass = self._default_lowpass_hz if self._filter_on else None

        try:
            result = run_passive_analysis(col, epoch_idx, lowpass_hz=lowpass)
        except Exception as exc:
            self._results_box.setPlainText(f"Per-sweep passive analysis error:\n{exc}")
            return

        params = {
            "collection_name": self._current_collection.name,
            "epoch_index": epoch_idx,
            "lowpass_hz": lowpass,
            "n_sweeps": len(checked),
            "source": "per_sweep_passive",
        }
        self._cell._store_result("passive_range", result, params)

        # Build per-sweep display table
        per_sweep = result.get("per_sweep", [])
        filt_str = f"LP {lowpass:.0f} Hz" if lowpass else "raw"

        hdr = f"{'Sw':>3}  {'I(pA)':>7}  {'Rin_SS':>7}  {'Rin_pk':>7}  {'τ(ms)':>7}  {'R²τ':>5}  {'Sag':>5}"
        sep = "─" * len(hdr)
        rows_txt = [f"N={len(per_sweep)}  Filter={filt_str}", sep, hdr, sep]

        rin_vals, tau_vals = [], []
        for r in per_sweep:
            sw = r.get("sweep_index", "?")
            i_pA = r.get("current_injection_pA", float("nan"))
            rin_ss = r.get("input_resistance_MOhm", float("nan"))
            rin_pk = r.get("input_resistance_peak_MOhm", float("nan"))
            tau = r.get("time_constant_ms", float("nan"))
            r2 = r.get("time_constant_r2", float("nan"))
            sag = r.get("sag_ratio", float("nan"))

            def _fmt(v, fmt=".1f"):
                return f"{v:{fmt}}" if v == v else "  NaN"  # NaN check

            rows_txt.append(
                f"{sw:>3}  {_fmt(i_pA):>7}  {_fmt(rin_ss):>7}  "
                f"{_fmt(rin_pk):>7}  {_fmt(tau):>7}  {_fmt(r2, '.3f'):>5}  "
                f"{_fmt(sag, '.3f'):>5}"
            )
            if rin_ss == rin_ss:
                rin_vals.append(rin_ss)
            if tau == tau:
                tau_vals.append(tau)

        rows_txt.append(sep)
        if rin_vals:
            import statistics
            rin_mean = sum(rin_vals) / len(rin_vals)
            rin_sd = statistics.stdev(rin_vals) if len(rin_vals) > 1 else 0.0
            rows_txt.append(f"Rin_SS mean ± SD : {rin_mean:.1f} ± {rin_sd:.1f} MΩ")
        if tau_vals:
            tau_mean = sum(tau_vals) / len(tau_vals)
            tau_sd = statistics.stdev(tau_vals) if len(tau_vals) > 1 else 0.0
            rows_txt.append(f"τ_m    mean ± SD : {tau_mean:.2f} ± {tau_sd:.2f} ms")

        self._results_box.setPlainText("\n".join(rows_txt))
        self._refresh_analysis_checks()

    def _refresh_analysis_checks(self) -> None:
        self._chk_fi.setChecked(bool(self._cell.results.get("fi_curve")))
        self._chk_ramp.setChecked(bool(self._cell.results.get("ramp_evoked_APs")))
        self._chk_avg.setChecked(bool(self._cell.results.get("passive_repeated_step")))
        self._chk_passive.setChecked(bool(self._cell.results.get("passive_range")))
        self._chk_vrest.setChecked(bool(self._cell.results.get("v_rest")))

    # ------------------------------------------------------------------
    # Axis label helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sweep_label(label: str) -> tuple[str, str]:
        import re
        m = re.match(r'^(.*?)\s*\(([^)]+)\)\s*$', label.strip())
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return label.strip(), ""

    def _update_axis_labels(self) -> None:
        if self._current_collection is None:
            return
        try:
            ref = self._current_collection.sweeps[0]
            rec = self._cell.recordings[ref.filename]
            y_name, y_units = self._parse_sweep_label(rec.y_label)
            c_name, c_units = self._parse_sweep_label(rec.c_label)
        except Exception:
            return

        self._plot_v.setLabel("left", y_name, units=y_units)
        self._plot_v.getAxis("left").enableAutoSIPrefix(False)
        self._plot_i.setLabel("left", c_name, units=c_units)
        self._plot_i.getAxis("left").enableAutoSIPrefix(False)

        d_name = f"d({y_name})/dt" if y_name else "dV/dt"
        d_units = f"{y_units}/ms" if y_units else "mV/ms"
        self._plot_d.setLabel("left", d_name, units=d_units)
        self._plot_d.getAxis("left").enableAutoSIPrefix(False)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _get_current_trace(self, pos: int) -> tuple[np.ndarray, np.ndarray]:
        ref = self._ref_at(pos)
        rec = self._cell.recordings[ref.filename]
        t, _, i = rec.get_sweep_arrays(ref.sweep_index)
        if not np.all(np.isnan(i)):
            return t, i
        try:
            return t, rec.get_command_waveform(ref.sweep_index)
        except Exception:
            pass
        i_synth = np.zeros(len(t))
        try:
            for ep in rec.get_epochs(ref.sweep_index):
                i_synth[ep.start_sample:ep.end_sample] = ep.level
        except Exception:
            pass
        return t, i_synth

    def _detect_spiking_sweeps(self, checked: list[int]) -> list[int]:
        spiking = []
        lowpass = self._default_lowpass_hz if self._filter_on else None
        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        for pos in checked:
            try:
                ref = self._ref_at(pos)
                rec = self._cell.recordings[ref.filename]
                ep = rec.get_epoch(ref.sweep_index, epoch_idx)
                _, v, _ = self._current_collection.get_sweep_arrays(
                    ref, lowpass_hz=lowpass
                )
                if np.any(v[ep.start_sample:ep.end_sample] > 0.0):
                    spiking.append(pos)
            except Exception:
                pass
        return spiking

    def _validate_step_command(
        self, checked: list[int]
    ) -> tuple[list[float], list[int]]:
        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        levels = []
        durations = []
        for pos in checked:
            ref = self._ref_at(pos)
            rec = self._cell.recordings[ref.filename]
            ep = rec.get_epoch(ref.sweep_index, epoch_idx)
            levels.append(ep.level)
            durations.append(ep.end_sample - ep.start_sample)

        ref_level = levels[0]
        ref_dur = durations[0]
        bad_level = [checked[i] for i, lv in enumerate(levels)
                     if abs(lv - ref_level) > 0.5]
        bad_dur = [checked[i] for i, d in enumerate(durations)
                   if d != ref_dur]

        msg_parts = []
        if bad_level:
            labels = [self._ref_at(p).display_label for p in bad_level]
            msg_parts.append(
                f"Step amplitude differs in sweeps {labels} "
                f"(expected {ref_level:.1f} pA)."
            )
        if bad_dur:
            ref0 = self._ref_at(checked[0])
            rec0 = self._cell.recordings[ref0.filename]
            ref_ms = ref_dur / rec0.sampling_rate_hz * 1000
            labels = [self._ref_at(p).display_label for p in bad_dur]
            msg_parts.append(
                f"Step duration differs in sweeps {labels} "
                f"(expected {ref_ms:.1f} ms)."
            )
        assert not msg_parts, "  ".join(msg_parts)
        return levels, durations

    def _estimate_avg_baseline(
        self, t: np.ndarray, v: np.ndarray, ref_pos: int
    ) -> float:
        epoch_idx = self._step_epoch_index if self._step_epoch_index is not None else 1
        ref = self._ref_at(ref_pos)
        rec = self._cell.recordings[ref.filename]
        if epoch_idx > 0:
            try:
                ep_pre = rec.get_epoch(ref.sweep_index, epoch_idx - 1)
                sl = slice(ep_pre.start_sample, ep_pre.end_sample)
                return float(np.mean(v[sl]))
            except (IndexError, RuntimeError):
                pass
        ep = rec.get_epoch(ref.sweep_index, epoch_idx)
        n_50ms = int(0.05 * rec.sampling_rate_hz)
        sl = slice(ep.start_sample, ep.start_sample + n_50ms)
        return float(np.mean(v[sl]))

    # ------------------------------------------------------------------
    # Export actions
    # ------------------------------------------------------------------

    def _on_open_session(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        from wholecell.core.cell import Cell

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._win, "Open Session", str(self._cell.output_dir),
            "Session files (*.json);;All files (*)"
        )
        if not path:
            return

        try:
            cell = Cell.load_session(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self._win, "Load failed", str(exc))
            return

        self._cell = cell
        self._current_collection = None
        self._cursor = 0
        self._spike_data.clear()
        self._curves.clear()

        self._cell_id_edit.setText(cell.cell_id)
        self._win.setWindowTitle(f"Trace Viewer — {cell.cell_id}")

        self._build_file_tree()
        self._update_collections_combo()

        if cell.collections:
            first = next(iter(cell.collections))
            idx = self._collections_combo.findText(first)
            if idx >= 0:
                self._collections_combo.setCurrentIndex(idx)
            self._on_collection_changed(first)

        self._results_box.setPlainText(f"Session loaded:\n{path}")
        self._refresh_analysis_checks()

    def _on_open_directory(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        from wholecell.config import get_default_data_dir

        _real_dir = self._cell.output_dir != Path(".")
        _start = str(self._cell.output_dir) if _real_dir else str(get_default_data_dir() or "")
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self._win, "Open Recording Directory", _start,
        )
        if not dir_path:
            return

        # Load all ABF files from the directory
        abf_files = sorted(Path(dir_path).glob("*.abf"))
        if not abf_files:
            QtWidgets.QMessageBox.information(
                self._win, "No ABF files", f"No .abf files found in:\n{dir_path}"
            )
            return

        loaded = []
        errors = []
        for abf_path in abf_files:
            stem = abf_path.stem
            if stem in self._cell.recordings:
                continue
            try:
                self._cell.add_recording(abf_path)
                loaded.append(stem)
            except Exception as exc:
                errors.append(f"{stem}: {exc}")

        self._cell.output_dir = Path(dir_path)

        # Auto-create a default collection for each newly loaded file
        self._auto_create_collections(loaded)

        self._build_file_tree()
        self._update_collections_combo()

        # Switch to the first newly created collection if none is active
        if loaded and self._current_collection is None:
            first = loaded[0]
            idx = self._collections_combo.findText(first)
            if idx >= 0:
                self._collections_combo.setCurrentIndex(idx)
            self._on_collection_changed(first)

        msg = f"Loaded {len(loaded)} file(s)."
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        self._results_box.setPlainText(msg)

    def _on_new_cell(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        from wholecell.core.cell import Cell

        # Settings are instrument-like — carry them forward
        saved_dvdt = self._dvdt_spin.value()
        saved_peak_window = self._peak_window_spin.value()

        from wholecell.config import get_default_data_dir
        _real_dir = self._cell.output_dir != Path(".")
        _start = str(self._cell.output_dir) if _real_dir else str(get_default_data_dir() or "")
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self._win, "Open New Cell Directory", _start,
        )
        if not dir_path:
            return

        abf_files = sorted(Path(dir_path).glob("*.abf"))
        if not abf_files:
            QtWidgets.QMessageBox.information(
                self._win, "No ABF files", f"No .abf files found in:\n{dir_path}"
            )
            return

        # Fresh cell model for the new directory
        self._cell = Cell(cell_id=Path(dir_path).name, output_dir=dir_path)

        # Reset all transient GUI state
        self._current_collection = None
        self._cursor = 0
        # _clear_curves() must come before clearing the dict — it iterates
        # self._curves to remove items from the pyqtgraph plots, then deletes
        # the keys. Clearing the dict first would orphan plot items on screen.
        self._clear_curves()
        self._clear_avg_curves()
        self._spike_data.clear()
        self._tau_lookup.clear()
        self._avg_tau_fit = None
        self._update_spike_markers()   # clears marker scatter plots
        self._list.clear()
        self._results_box.clear()
        self._cell_id_edit.setText(self._cell.cell_id)
        self._win.setWindowTitle(f"Trace Viewer — {self._cell.cell_id}")
        self._refresh_analysis_checks()

        # Restore the preserved settings
        self._dvdt_spin.setValue(saved_dvdt)
        self._peak_window_spin.setValue(saved_peak_window)

        # Load recordings from the new directory
        loaded, errors = [], []
        for abf_path in abf_files:
            try:
                self._cell.add_recording(abf_path)
                loaded.append(abf_path.stem)
            except Exception as exc:
                errors.append(f"{abf_path.stem}: {exc}")

        self._auto_create_collections(loaded)
        self._build_file_tree()
        self._update_collections_combo()

        if loaded:
            first = loaded[0]
            idx = self._collections_combo.findText(first)
            if idx >= 0:
                self._collections_combo.setCurrentIndex(idx)
            self._on_collection_changed(first)

        msg = f"New cell loaded — {len(loaded)} file(s)."
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        self._results_box.setPlainText(msg)

    def _on_save_session(self) -> None:
        try:
            path = self._cell.save_session()
            self._results_box.setPlainText(f"Session saved:\n{path}")
        except Exception as exc:
            self._results_box.setPlainText(f"Save failed:\n{exc}")

    def _on_export_spike_table(self) -> None:
        import pandas as pd
        from datetime import datetime
        from pyqtgraph.Qt import QtWidgets

        if not self._spike_data:
            QtWidgets.QMessageBox.information(
                self._win, "No spikes detected",
                "Run spike detection first (Find Spikes button)."
            )
            return

        rows = []
        for spikes in self._spike_data.values():
            rows.extend(spikes)

        if not rows:
            QtWidgets.QMessageBox.information(
                self._win, "No spikes", "No spikes detected in any sweep."
            )
            return

        df = (
            pd.DataFrame(rows)
            .drop(columns=["display_label"], errors="ignore")
            .sort_values(["filename", "sweep_index", "spike_index_in_sweep"])
            .reset_index(drop=True)
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested = str(
            self._cell.output_dir / f"{self._cell.cell_id}_spikes_{ts}.csv"
        )
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._win, "Save Spike Table", suggested, "CSV files (*.csv)"
        )
        if not filepath:
            return
        try:
            df.to_csv(filepath, index=False)
            self._results_box.setPlainText(f"Spike table saved:\n{filepath}")
        except Exception as exc:
            self._results_box.setPlainText(f"Export failed:\n{exc}")

    def _on_export_sweep_summary(self) -> None:
        from datetime import datetime
        from pyqtgraph.Qt import QtWidgets

        coll_name = self._collections_combo.currentText()
        if not coll_name:
            QtWidgets.QMessageBox.information(
                self._win, "No collection", "Select an active collection first."
            )
            return

        epoch_idx = self._step_epoch_index
        if epoch_idx is None:
            QtWidgets.QMessageBox.information(
                self._win, "No step epoch",
                "Step epoch not detected. Open a file and select sweeps first."
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested = str(
            self._cell.output_dir / f"{self._cell.cell_id}_sweep_summary_{ts}.csv"
        )
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._win, "Save Sweep Summary", suggested, "CSV files (*.csv)"
        )
        if not filepath:
            return
        try:
            path = self._cell.export_sweep_summary(coll_name, epoch_idx,
                                                    filepath=filepath)
            self._results_box.setPlainText(f"Sweep summary saved:\n{path}")
        except Exception as exc:
            self._results_box.setPlainText(f"Export failed:\n{exc}")

    def _on_export_cell_summary(self) -> None:
        from pyqtgraph.Qt import QtWidgets

        suggested = str(self._cell.output_dir / f"{self._cell.cell_id}_cell_summary.json")
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._win, "Save Cell Summary", suggested, "JSON files (*.json)"
        )
        if not filepath:
            return
        try:
            self._cell.export_cell_summary(filepath=filepath)
            self._results_box.setPlainText(f"Cell summary saved:\n{filepath}")
        except Exception as exc:
            self._results_box.setPlainText(f"Export failed:\n{exc}")

    def _on_cell_id_changed(self) -> None:
        self._cell.cell_id = self._cell_id_edit.text().strip()
        self._win.setWindowTitle(f"Trace Viewer — {self._cell.cell_id}")

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        if self._current_collection is None:
            self._status.setText(
                "No collection active. Load ABF files and add sweeps to a collection."
            )
            return
        checked = self._checked_sweeps()
        filt_str = (f"LP {self._default_lowpass_hz:.0f} Hz"
                    if self._filter_on else "raw")
        epoch_str = (str(self._step_epoch_index)
                     if self._step_epoch_index is not None else "?")
        self._status.setText(
            f"Collection: {self._current_collection.name} | "
            f"Cursor: pos {self._cursor} | Overlaid: {len(checked)} | "
            f"Step epoch: {epoch_str} | {filt_str} | "
            f"[↑↓ navigate | Space pin | A/N all/none | F filter | C current | D dV/dt | Q quit]"
        )

    # ------------------------------------------------------------------
    # Sweep list event handlers
    # ------------------------------------------------------------------

    def _on_item_changed(self, item) -> None:
        self._full_update()

    def _toggle_solo(self) -> None:
        self._solo_mode = not self._solo_mode
        self._btn_solo.setStyleSheet(
            "font-size: 11px; padding: 2px 6px;"
            + ("background: #4a3a1a; color: #fc8;" if self._solo_mode else "")
        )
        self._full_update()

    def _check_all(self) -> None:
        from pyqtgraph.Qt import QtCore
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(QtCore.Qt.CheckState.Checked)
        self._list.blockSignals(False)
        self._full_update()

    def _check_none(self) -> None:
        from pyqtgraph.Qt import QtCore
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
        self._list.blockSignals(False)
        self._full_update()

    # ------------------------------------------------------------------
    # Close event (auto-save)
    # ------------------------------------------------------------------

    def _on_close_event(self, event) -> None:
        if self._cell.recordings:
            try:
                self._cell.save_session()
            except Exception:
                pass
        event.accept()

    # ------------------------------------------------------------------
    # Keyboard handler
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        from pyqtgraph.Qt import QtCore

        if self._current_collection is None:
            return

        key = event.key()
        n = self._current_collection.n_sweeps

        if key == QtCore.Qt.Key.Key_Down:
            self._cursor = min(self._cursor + 1, n - 1)
            self._full_update()
        elif key == QtCore.Qt.Key.Key_Up:
            self._cursor = max(self._cursor - 1, 0)
            self._full_update()
        elif key in (QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Return):
            item = self._list.item(self._cursor)
            if item:
                new = (
                    QtCore.Qt.CheckState.Unchecked
                    if item.checkState() == QtCore.Qt.CheckState.Checked
                    else QtCore.Qt.CheckState.Checked
                )
                item.setCheckState(new)
        elif key == QtCore.Qt.Key.Key_A:
            self._check_all()
        elif key == QtCore.Qt.Key.Key_N:
            self._check_none()
        elif key == QtCore.Qt.Key.Key_F:
            self._filter_on = not self._filter_on
            self._full_update()
        elif key == QtCore.Qt.Key.Key_C:
            self._show_current = not self._show_current
            self._plot_i.setVisible(self._show_current)
            self._full_update()
        elif key == QtCore.Qt.Key.Key_D:
            self._show_dvdt = not self._show_dvdt
            self._plot_d.setVisible(self._show_dvdt)
            self._full_update()
        elif key == QtCore.Qt.Key.Key_S:
            self._show_spikes = not self._show_spikes
            self._full_update()
        elif key == QtCore.Qt.Key.Key_V:
            self._toggle_solo()
        elif key in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self._win.close()

    def run(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app:
            app.exec()


# ---------------------------------------------------------------------------
# Convenience launchers
# ---------------------------------------------------------------------------

def launch_viewer(
    filepath: str | Path | None = None,
    directory: str | Path | None = None,
    cell_id: str | None = None,
    lowpass_hz: float | None = None,
    epoch_index: int | None = None,
) -> None:
    """Open TraceViewer from a single ABF file or a directory.

    Parameters
    ----------
    filepath : str or Path or None
        Single ABF file to load. Creates a Cell from this file.
    directory : str or Path or None
        Directory to use as output_dir; loads all .abf files found.
        Ignored if filepath is provided.
    cell_id : str or None
        Cell identifier. Defaults to the file stem or directory name.
    lowpass_hz : float or None
    epoch_index : int or None
        Step epoch index. Auto-detected if None.
    """
    from wholecell.core.cell import Cell
    from wholecell.core.recording import Recording

    if filepath is not None:
        filepath = Path(filepath)
        cid = cell_id or filepath.stem
        cell = Cell(cell_id=cid, output_dir=filepath.parent)
        cell.add_recording(filepath)
        # Create a default collection with all sweeps
        rec = cell.recordings[filepath.stem]
        sweeps = [{"filename": filepath.stem, "sweep_index": i}
                  for i in range(rec.n_sweeps)]
        cell.create_sweep_collection(filepath.stem, sweeps)

    elif directory is not None:
        directory = Path(directory)
        cid = cell_id or directory.name
        cell = Cell(cell_id=cid, output_dir=directory)
        for abf_path in sorted(directory.glob("*.abf")):
            try:
                cell.add_recording(abf_path)
            except Exception:
                pass

    else:
        # No path given — open directory dialog
        from pyqtgraph.Qt import QtWidgets
        from wholecell.config import get_default_data_dir
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        _start = str(get_default_data_dir() or "")
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Recording Directory", _start
        )
        if not dir_path:
            return
        launch_viewer(directory=dir_path, cell_id=cell_id, lowpass_hz=lowpass_hz,
                      epoch_index=epoch_index)
        return

    viewer = TraceViewer(cell, step_epoch_index=epoch_index, lowpass_hz=lowpass_hz)
    viewer.run()


def launch_from_cell(cell, lowpass_hz: float | None = None,
                     epoch_index: int | None = None) -> None:
    """Open TraceViewer from an existing Cell object."""
    viewer = TraceViewer(cell, step_epoch_index=epoch_index, lowpass_hz=lowpass_hz)
    viewer.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Whole-cell trace viewer")
    parser.add_argument("path", nargs="?", default=None,
                        help="Path to an .abf file or a directory of .abf files. "
                             "Opens a directory picker if omitted.")
    parser.add_argument("--cell-id", default=None, help="Cell identifier")
    parser.add_argument("--lowpass", type=float, default=None,
                        help="Lowpass filter cutoff in Hz (default: from ~/.wholecell/settings.json)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Step epoch index (auto-detected if omitted)")
    args = parser.parse_args()

    if args.path is None:
        launch_viewer(lowpass_hz=args.lowpass, epoch_index=args.epoch)
    else:
        p = Path(args.path)
        if p.suffix.lower() == ".json":
            from wholecell.core.cell import Cell
            cell = Cell.load_session(p)
            launch_from_cell(cell, lowpass_hz=args.lowpass, epoch_index=args.epoch)
        elif p.is_dir():
            launch_viewer(directory=p, cell_id=args.cell_id,
                          lowpass_hz=args.lowpass, epoch_index=args.epoch)
        else:
            launch_viewer(filepath=p, cell_id=args.cell_id,
                          lowpass_hz=args.lowpass, epoch_index=args.epoch)


if __name__ == "__main__":
    main()
