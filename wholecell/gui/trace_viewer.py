"""
trace_viewer.py
---------------
Interactive trace viewer for whole-cell current-clamp recordings.

Layout
------
  Left panel  : sweep list with checkboxes, Average & Analyze button, results
  Right panel : voltage plot (+ optional dV/dt panel below)

Controls
--------
  ↑ / ↓          navigate cursor through the sweep list
  Space / Enter   toggle checkbox of the cursor sweep
  A               check all sweeps
  N               uncheck all sweeps
  F               toggle lowpass filter (2 kHz default) for all shown sweeps
  D               toggle dV/dt panel
  S               toggle spike markers
  Q / Escape      quit

The cursor sweep is always shown as a bright white trace regardless of its
checkbox state.  Checked sweeps are shown in colour (viridis-like cycle).

"Average & Analyze" averages all checked sweeps (same filter applied) and
runs passive analysis on the result.  All checked sweeps must share the same
step epoch amplitude and duration; a clear error is shown if they do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _sweep_color(sweep_index: int, n_total: int):
    import pyqtgraph as pg
    return pg.intColor(sweep_index, hues=max(n_total, 1), minValue=180)


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class TraceViewer:
    """pyqtgraph-based viewer for a single ABF Recording.

    Parameters
    ----------
    recording : Recording
    spike_result : dict or None
    passive_result : dict or None
    lowpass_hz : float or None
        Default lowpass filter cutoff toggled with F.
    epoch_index : int or None
        Zero-based index of the current step epoch used for passive analysis.
        Auto-detected via find_step_epoch if not supplied.
    """

    def __init__(
        self,
        recording,
        spike_result: dict | None = None,
        passive_result: dict | None = None,
        lowpass_hz: float | None = 2000.0,
        epoch_index: int | None = None,
    ) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._rec = recording
        self._spike_result = spike_result
        self._passive_result = passive_result
        self._default_lowpass_hz = lowpass_hz

        # Auto-detect step epoch if not provided
        if epoch_index is None:
            try:
                from wholecell.io.abf_reader import find_step_epoch
                epoch_index = find_step_epoch(recording.filepath)
            except Exception:
                epoch_index = 1
        self._step_epoch_index = epoch_index

        self._cursor = 0
        self._filter_on = False
        self._show_current = False
        self._show_dvdt = False
        self._show_spikes = False

        # spike_data: sweep_index → list of spike dicts (threshold/peak/trough times+voltages)
        self._spike_data: dict[int, list] = {}
        if spike_result:
            for sd in spike_result.get("per_sweep", []):
                self._spike_data[sd["sweep_index"]] = sd.get("spikes", [])
            self._show_spikes = True

        self._tau_lookup: dict[int, dict] = {}
        if passive_result:
            for row in passive_result.get("per_sweep", []):
                if "_tau_fit" in row:
                    self._tau_lookup[row["sweep_index"]] = row["_tau_fit"]

        # ---- Qt application ----
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        # ---- Main window ----
        self._win = QtWidgets.QWidget()
        self._win.setWindowTitle(f"Trace Viewer — {recording.filename}")
        self._win.resize(1300, 740)
        self._win.keyPressEvent = self._on_key

        main_layout = QtWidgets.QHBoxLayout(self._win)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ---- Left panel ----
        left = QtWidgets.QWidget()
        left.setFixedWidth(190)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl = QtWidgets.QLabel("Sweeps  (Space = pin)")
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        left_layout.addWidget(lbl)

        self._list = QtWidgets.QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background: #1a1a1a; color: #ddd; font-size: 12px; }"
            "QListWidget::item:selected { background: #333; }"
        )
        self._list.itemChanged.connect(self._on_item_changed)
        left_layout.addWidget(self._list, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("All")
        btn_none = QtWidgets.QPushButton("None")
        for btn in (btn_all, btn_none):
            btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        btn_all.clicked.connect(self._check_all)
        btn_none.clicked.connect(self._check_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        left_layout.addLayout(btn_row)

        btn_spikes = QtWidgets.QPushButton("Find Spikes  (S = toggle)")
        btn_spikes.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #2a2a4a; color: #aaf;"
        )
        btn_spikes.clicked.connect(self._run_spike_detection)
        left_layout.addWidget(btn_spikes)

        btn_avg = QtWidgets.QPushButton("Average && Analyze Subthreshold")
        btn_avg.setStyleSheet(
            "font-size: 11px; padding: 4px; background: #2a4a2a; color: #8f8;"
        )
        btn_avg.clicked.connect(self._run_average_analysis)
        left_layout.addWidget(btn_avg)

        # Results display
        res_lbl = QtWidgets.QLabel("Results")
        res_lbl.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        left_layout.addWidget(res_lbl)

        self._results_box = QtWidgets.QTextEdit()
        self._results_box.setReadOnly(True)
        self._results_box.setFixedHeight(180)
        self._results_box.setStyleSheet(
            "QTextEdit { background: #111; color: #cfc; font-size: 11px; "
            "font-family: monospace; border: 1px solid #333; }"
        )
        self._results_box.setPlaceholderText("No analysis yet.")
        left_layout.addWidget(self._results_box)

        main_layout.addWidget(left)

        # ---- Right: pyqtgraph plots ----
        self._pg_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self._pg_widget, stretch=1)

        self._plot_v = self._pg_widget.addPlot(row=0, col=0)
        self._plot_v.setLabel("left", "Voltage", units="mV")
        self._plot_v.setLabel("bottom", "Time", units="s")
        self._plot_v.showGrid(x=True, y=True, alpha=0.25)

        self._plot_d = self._pg_widget.addPlot(row=1, col=0)
        self._plot_d.setLabel("left", "dV/dt", units="mV/ms")
        self._plot_d.setLabel("bottom", "Time", units="s")
        self._plot_d.showGrid(x=True, y=True, alpha=0.25)
        self._plot_d.setXLink(self._plot_v)
        self._plot_d.setVisible(False)

        self._plot_i = self._pg_widget.addPlot(row=2, col=0)
        self._plot_i.setLabel("left", "Current", units="pA")
        self._plot_i.setLabel("bottom", "Time", units="s")
        self._plot_i.showGrid(x=True, y=True, alpha=0.25)
        self._plot_i.setXLink(self._plot_v)
        self._plot_i.setVisible(False)

        self._status = pg.LabelItem(justify="left")
        self._pg_widget.addItem(self._status, row=3, col=0)

        # Average trace curves (always present, hidden when empty)
        self._avg_curve_v = self._plot_v.plot(
            pen=self._make_pen("#0ff", width=3, alpha=220), name="avg"
        )
        self._avg_curve_i = self._plot_i.plot(
            pen=self._make_pen("#0ff", width=2, alpha=200)
        )
        self._avg_curve_d = self._plot_d.plot(
            pen=self._make_pen("#0ff", width=2, alpha=200)
        )

        # Global spike marker scatter plots (one set per feature, shared across sweeps)
        _sp = dict(pen=None, symbolPen=None)
        self._mk_threshold = self._plot_v.plot(
            symbol="t1", symbolSize=11, symbolBrush="#f80", **_sp)   # ▼ orange
        self._mk_peak = self._plot_v.plot(
            symbol="o",  symbolSize=10, symbolBrush="#4f4", **_sp)   # ● green
        self._mk_trough = self._plot_v.plot(
            symbol="d",  symbolSize=10, symbolBrush="#f44", **_sp)   # ◆ red
        self._mk_upstroke = self._plot_d.plot(
            symbol="t",  symbolSize=11, symbolBrush="#0ff", **_sp)   # ▲ cyan
        self._mk_downstroke = self._plot_d.plot(
            symbol="t1", symbolSize=11, symbolBrush="#f4f", **_sp)   # ▼ magenta
        for mk in (self._mk_threshold, self._mk_peak, self._mk_trough,
                   self._mk_upstroke, self._mk_downstroke):
            mk.setZValue(20)

        # Per-sweep curve sets
        self._curves: dict[int, dict] = {}

        # ---- Populate sweep list ----
        self._list.blockSignals(True)
        for i in range(recording.n_sweeps):
            self._list.addItem(self._list_item(i))
        self._list.blockSignals(False)
        self._list.setCurrentRow(0)

        self._full_update()
        self._win.show()

    # ------------------------------------------------------------------
    # List helpers
    # ------------------------------------------------------------------

    def _list_item(self, sweep_index: int):
        from pyqtgraph.Qt import QtCore, QtWidgets
        item = QtWidgets.QListWidgetItem(f"Sweep {sweep_index:3d}")
        item.setFlags(
            item.flags()
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        return item

    def _checked_sweeps(self) -> list[int]:
        from pyqtgraph.Qt import QtCore
        return [
            i for i in range(self._list.count())
            if self._list.item(i).checkState() == QtCore.Qt.CheckState.Checked
        ]

    # ------------------------------------------------------------------
    # Full refresh
    # ------------------------------------------------------------------

    def _full_update(self) -> None:
        lowpass = self._default_lowpass_hz if self._filter_on else None
        n = self._rec.n_sweeps
        to_show = set(self._checked_sweeps())

        for idx in list(self._curves.keys()):
            if idx not in to_show:
                self._remove_curves(idx)

        for idx in to_show:
            color = _sweep_color(idx, n)
            width = 1
            alpha = 200
            zval = 1

            t, v, _ = self._rec.get_sweep_arrays(idx, lowpass_hz=lowpass)

            if idx not in self._curves:
                self._curves[idx] = self._make_curves(idx)
            c = self._curves[idx]

            pen = self._make_pen(color, width, alpha)
            c["v"].setData(t, v)
            c["v"].setPen(pen)
            c["v"].setZValue(zval)

            if self._show_current:
                t_i, i_arr = self._get_current_trace(idx)
                c["i"].setData(t_i, i_arr)
                c["i"].setPen(pen)
                c["i"].setZValue(zval)
            else:
                c["i"].setData([], [])

            if self._show_dvdt:
                c["d"].setData(t, np.gradient(v, t) / 1000.0)
                c["d"].setPen(pen)
                c["d"].setZValue(zval)
            else:
                c["d"].setData([], [])

            self._update_tau(idx, c)

        self._update_spike_markers()
        self._list.setCurrentRow(self._cursor)
        self._update_status()

    def _make_curves(self, idx: int) -> dict:
        from pyqtgraph.Qt import QtCore
        c: dict = {}
        c["v"] = self._plot_v.plot(name=f"s{idx:03d}")
        c["i"] = self._plot_i.plot()
        c["d"] = self._plot_d.plot()
        c["tau"] = self._plot_v.plot(
            pen=self._make_pen("#ff0", 2, 200,
                               style=QtCore.Qt.PenStyle.DashLine)
        )
        return c

    def _remove_curves(self, idx: int) -> None:
        if idx not in self._curves:
            return
        for item in self._curves[idx].values():
            for plot in (self._plot_v, self._plot_i, self._plot_d):
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        del self._curves[idx]

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

    def _run_spike_detection(self) -> None:
        """Detect spikes in all checked sweeps and update global markers."""
        from wholecell.core.sweep_collection import SweepCollection, SweepRef
        from wholecell.analysis.spikes.finder import run_spike_detection

        checked = self._checked_sweeps()
        if not checked:
            self._results_box.setPlainText("No sweeps checked for spike detection.")
            return

        rec = self._rec
        sweeps = [SweepRef(rec.filename, i) for i in checked]
        col = SweepCollection(
            name="viewer_spikes",
            sweeps=sweeps,
            recordings={rec.filename: rec},
        )
        try:
            result = run_spike_detection(col, epoch_index=self._step_epoch_index)
        except Exception as exc:
            self._results_box.setPlainText(f"Spike detection error:\n{exc}")
            return

        for sd in result.get("per_sweep", []):
            self._spike_data[sd["sweep_index"]] = sd.get("spikes", [])

        n_spikes = sum(len(v) for v in self._spike_data.values())
        self._results_box.setPlainText(
            f"Detected {n_spikes} spike(s) across {len(checked)} sweep(s).\n"
            "Press S to toggle markers."
        )
        self._show_spikes = True
        self._update_spike_markers()

    def _update_spike_markers(self) -> None:
        """Aggregate spike feature coordinates from all checked sweeps and
        update the five global scatter plots.  dV/dt markers are computed
        from the voltage derivative within threshold→peak and peak→trough
        windows respectively."""
        visible = self._show_spikes and bool(self._spike_data)

        if not visible:
            for mk in (self._mk_threshold, self._mk_peak, self._mk_trough,
                       self._mk_upstroke, self._mk_downstroke):
                mk.setData([], [])
            return

        checked = set(self._checked_sweeps())
        thresh_t, thresh_v = [], []
        peak_t, peak_v = [], []
        trough_t, trough_v = [], []
        up_t, up_dvdt = [], []
        dn_t, dn_dvdt = [], []

        lowpass = self._default_lowpass_hz if self._filter_on else None

        for idx in checked:
            spikes = self._spike_data.get(idx)
            if not spikes:
                continue

            # Voltage + time arrays for dV/dt computation
            try:
                t_arr, v_arr, _ = self._rec.get_sweep_arrays(idx, lowpass_hz=lowpass)
                dt = t_arr[1] - t_arr[0]
                dvdt = np.gradient(v_arr, t_arr) / 1000.0  # mV/ms
            except Exception:
                t_arr = v_arr = dvdt = None

            for sp in spikes:
                thresh_t.append(sp["threshold_time_s"])
                thresh_v.append(sp["threshold_voltage_mV"])
                peak_t.append(sp["peak_time_s"])
                peak_v.append(sp["peak_voltage_mV"])
                trough_t.append(sp["trough_time_s"])
                trough_v.append(sp["trough_voltage_mV"])

                if dvdt is not None:
                    # Upstroke: max dV/dt between threshold and peak
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
        self._mk_upstroke.setData(up_t, up_dvdt)
        self._mk_downstroke.setData(dn_t, dn_dvdt)

    def _update_tau(self, idx, c) -> None:
        fit = self._tau_lookup.get(idx)
        if fit and fit.get("time") and fit.get("predicted"):
            c["tau"].setData(fit["time"], fit["predicted"])
        else:
            c["tau"].setData([], [])

    def _update_status(self) -> None:
        checked = self._checked_sweeps()
        filt_str = f"LP {self._default_lowpass_hz:.0f} Hz" if self._filter_on else "raw"
        self._status.setText(
            f"Cursor: sweep {self._cursor} | Overlaid: {len(checked)} | "
            f"Step epoch: {self._step_epoch_index} | {filt_str} | "
            f"[↑↓ navigate | Space pin | A/N all/none | F filter | C current | D dV/dt | Q quit]"
        )

    # ------------------------------------------------------------------
    # Average & Analyze
    # ------------------------------------------------------------------

    def _run_average_analysis(self) -> None:
        """Average checked sweeps and run passive analysis on the result."""
        from wholecell.analysis.passive import (
            estimate_input_resistance,
            fit_time_constant,
            estimate_sag,
        )

        checked = self._checked_sweeps()
        if len(checked) < 1:
            self._results_box.setPlainText("No sweeps checked.")
            return

        # ---- Warn if any checked sweep contains detected spikes ----
        spiking = [i for i in checked if self._spike_data.get(i)]
        if not spiking:
            # No spike result loaded — do a quick threshold scan
            spiking = self._detect_spiking_sweeps(checked)
        if spiking:
            from pyqtgraph.Qt import QtWidgets
            reply = QtWidgets.QMessageBox.warning(
                self._win,
                "Spiking sweeps selected",
                f"Sweep(s) {spiking} appear to contain action potentials.\n\n"
                "Subthreshold analysis on spiking data will give unreliable "
                "results (Rin, τ, sag are all corrupted by spikes).\n\n"
                "Proceed anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        # ---- Validate step command consistency ----
        try:
            step_levels, step_durations = self._validate_step_command(checked)
        except AssertionError as exc:
            self._results_box.setHtml(
                f'<span style="color:#f88;">Validation failed:<br>{exc}</span>'
            )
            return

        step_current_pA = step_levels[0]

        # ---- Average voltage traces ----
        lowpass = self._default_lowpass_hz if self._filter_on else None
        arrays = []
        t_ref = None
        for idx in checked:
            t, v, _ = self._rec.get_sweep_arrays(idx, lowpass_hz=lowpass)
            arrays.append(v)
            if t_ref is None:
                t_ref = t

        avg_v = np.mean(np.stack(arrays, axis=0), axis=0)

        # ---- Show average on plot ----
        self._avg_curve_v.setData(t_ref, avg_v)
        if self._show_current:
            t_i, i_arr = self._get_current_trace(checked[0])
            self._avg_curve_i.setData(t_i, i_arr)
        if self._show_dvdt:
            self._avg_curve_d.setData(t_ref, np.gradient(avg_v, t_ref) / 1000.0)

        # ---- Slice to epoch ----
        epoch = self._rec.get_epoch(checked[0], self._step_epoch_index)
        sl = slice(epoch.start_sample, epoch.end_sample)
        ep_time = t_ref[sl]
        ep_voltage = avg_v[sl]

        # ---- Baseline: epoch before step (or first 50 ms of step epoch) ----
        baseline_voltage = self._estimate_avg_baseline(
            t_ref, avg_v, checked[0]
        )

        # ---- Run analysis ----
        rin = estimate_input_resistance(baseline_voltage, ep_voltage, step_current_pA)
        tau_ms, _, _, _ = fit_time_constant(ep_time, ep_voltage, baseline_voltage)
        sag = estimate_sag(ep_time, ep_voltage, baseline_voltage)

        # ---- Display results ----
        n = len(checked)
        filt_str = f"LP {self._default_lowpass_hz:.0f} Hz" if self._filter_on else "raw"
        lines = [
            f"N sweeps averaged : {n}",
            f"Filter            : {filt_str}",
            f"Step current      : {step_current_pA:.1f} pA",
            f"Baseline voltage  : {baseline_voltage:.1f} mV",
            "─" * 28,
            f"Rin               : {rin:.1f} MΩ",
            f"τ_m               : {tau_ms:.2f} ms",
            f"Sag ratio         : {sag['sag_ratio']:.3f}",
            f"Sag amplitude     : {sag['sag_amplitude_mV']:.2f} mV",
            f"Sag τ             : {sag['sag_tau_ms']:.1f} ms",
        ]
        self._results_box.setPlainText("\n".join(lines))

    def _get_current_trace(self, sweep_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (time, current_pA) for one sweep.

        Uses the recorded current channel when available; otherwise synthesises
        a clean step waveform from the epoch command levels.
        """
        t, _, i = self._rec.get_sweep_arrays(sweep_index)
        if not np.all(np.isnan(i)):
            return t, i
        # Synthesise from epoch commands
        i_synth = np.zeros(len(t))
        try:
            for ep in self._rec.get_epochs(sweep_index):
                i_synth[ep.start_sample:ep.end_sample] = ep.level
        except Exception:
            pass
        return t, i_synth

    def _detect_spiking_sweeps(self, checked: list[int]) -> list[int]:
        """Return sweep indices that likely contain action potentials.

        Uses a simple threshold: any sample > 0 mV within the step epoch is
        treated as a spike.  Only called when no spike_result was provided.
        """
        spiking = []
        lowpass = self._default_lowpass_hz if self._filter_on else None
        for idx in checked:
            try:
                ep = self._rec.get_epoch(idx, self._step_epoch_index)
                _, v, _ = self._rec.get_sweep_arrays(idx, lowpass_hz=lowpass)
                ep_v = v[ep.start_sample:ep.end_sample]
                if np.any(ep_v > 0.0):
                    spiking.append(idx)
            except Exception:
                pass
        return spiking

    def _validate_step_command(self, checked: list[int]) -> tuple[list[float], list[int]]:
        """Assert all checked sweeps have identical step epoch command.

        Returns
        -------
        levels : list of float
        durations_samples : list of int

        Raises
        ------
        AssertionError with a descriptive message if sweeps differ.
        """
        levels = []
        durations = []
        for idx in checked:
            ep = self._rec.get_epoch(idx, self._step_epoch_index)
            levels.append(ep.level)
            durations.append(ep.end_sample - ep.start_sample)

        ref_level = levels[0]
        ref_dur = durations[0]

        bad_level = [
            checked[i] for i, lv in enumerate(levels)
            if abs(lv - ref_level) > 0.5  # 0.5 pA tolerance
        ]
        bad_dur = [
            checked[i] for i, d in enumerate(durations)
            if d != ref_dur
        ]

        msg_parts = []
        if bad_level:
            msg_parts.append(
                f"Step amplitude differs in sweeps {bad_level} "
                f"(expected {ref_level:.1f} pA)."
            )
        if bad_dur:
            ref_ms = ref_dur / self._rec.sampling_rate_hz * 1000
            msg_parts.append(
                f"Step duration differs in sweeps {bad_dur} "
                f"(expected {ref_ms:.1f} ms)."
            )
        assert not msg_parts, "  ".join(msg_parts)

        return levels, durations

    def _estimate_avg_baseline(
        self, t: np.ndarray, v: np.ndarray, ref_sweep: int
    ) -> float:
        """Return baseline voltage for the averaged trace.

        Uses the epoch before the step if available, else the first 50 ms
        of the step epoch.
        """
        if self._step_epoch_index > 0:
            try:
                ep_pre = self._rec.get_epoch(ref_sweep, self._step_epoch_index - 1)
                sl = slice(ep_pre.start_sample, ep_pre.end_sample)
                return float(np.mean(v[sl]))
            except (IndexError, RuntimeError):
                pass
        ep = self._rec.get_epoch(ref_sweep, self._step_epoch_index)
        n_50ms = int(0.05 * self._rec.sampling_rate_hz)
        sl = slice(ep.start_sample, ep.start_sample + n_50ms)
        return float(np.mean(v[sl]))

    # ------------------------------------------------------------------
    # Sweep list slots
    # ------------------------------------------------------------------

    def _on_item_changed(self, item) -> None:
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
    # Keyboard handler
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        from pyqtgraph.Qt import QtCore

        key = event.key()
        if key == QtCore.Qt.Key.Key_Down:
            self._cursor = min(self._cursor + 1, self._rec.n_sweeps - 1)
            self._full_update()
        elif key == QtCore.Qt.Key.Key_Up:
            self._cursor = max(self._cursor - 1, 0)
            self._full_update()
        elif key in (QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Return):
            item = self._list.item(self._cursor)
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
        elif key in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self._win.close()

    def run(self) -> None:
        from pyqtgraph.Qt import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app:
            app.exec()


# ---------------------------------------------------------------------------
# Convenience launcher
# ---------------------------------------------------------------------------

def launch_viewer(
    filepath: str | Path,
    sweep_index: int = 0,
    spike_result: dict | None = None,
    passive_result: dict | None = None,
    lowpass_hz: float | None = 2000.0,
    epoch_index: int | None = None,
) -> None:
    """Open TraceViewer for a single ABF file.

    Parameters
    ----------
    filepath : str or Path
    sweep_index : int
        Initial cursor position.
    spike_result : dict or None
    passive_result : dict or None
    lowpass_hz : float or None
    epoch_index : int or None
        Step epoch index for passive analysis. Auto-detected if None.
    """
    from wholecell.core.recording import Recording

    rec = Recording(filepath)
    viewer = TraceViewer(
        rec,
        spike_result=spike_result,
        passive_result=passive_result,
        lowpass_hz=lowpass_hz,
        epoch_index=epoch_index,
    )
    viewer._cursor = sweep_index
    viewer._full_update()
    viewer.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Whole-cell trace viewer")
    parser.add_argument("filepath", help="Path to .abf file")
    parser.add_argument("--sweep", type=int, default=0)
    parser.add_argument("--lowpass", type=float, default=2000.0)
    parser.add_argument("--epoch", type=int, default=None,
                        help="Step epoch index (auto-detected if omitted)")
    args = parser.parse_args()

    launch_viewer(
        args.filepath,
        sweep_index=args.sweep,
        lowpass_hz=args.lowpass,
        epoch_index=args.epoch,
    )


if __name__ == "__main__":
    main()
