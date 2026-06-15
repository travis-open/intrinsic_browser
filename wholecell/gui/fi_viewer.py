"""
fi_viewer.py
------------
Popup window for visualising F-I (frequency-current) curves.

Shows mean firing rate and peak instantaneous rate vs. injected current,
with rheobase marked and the fitted slope overlaid.

Launched from Cell.plot_fi_curve() or from the TraceViewer toolbar.
"""

from __future__ import annotations

import math


class FICurveViewer:
    """F-I curve popup window backed by pyqtgraph.

    Parameters
    ----------
    fi_result : dict
        The ``"data"`` field of a timestamped fi_curve result from Cell.
        Must have keys ``"fi_curve"`` and ``"cell_level"``.
    title : str, optional
        Window title.
    """

    def __init__(self, fi_result: dict, title: str = "F-I Curve") -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._result = fi_result
        fi_curve = fi_result.get("fi_curve", {})
        cell_level = fi_result.get("cell_level", {})

        self._currents = fi_curve.get("current_pA", [])
        self._mean_rates = fi_curve.get("mean_rate_hz", [])
        self._peak_rates = fi_curve.get("peak_rate_hz", [])
        self._rheobase = cell_level.get("rheobase_pA", float("nan"))
        self._slope = cell_level.get("fi_slope_hz_per_pA", float("nan"))
        self._intercept = cell_level.get("fi_slope_intercept_hz", float("nan"))
        self._r2 = cell_level.get("fi_slope_r2", float("nan"))
        self._slope_n = cell_level.get("fi_slope_n_points", 0)
        self._max_rate = cell_level.get("max_firing_rate_hz", float("nan"))
        self._max_peak_rate = cell_level.get("max_peak_instantaneous_rate_hz", float("nan"))

        app = QtWidgets.QApplication.instance()
        if app is None:
            import sys
            app = QtWidgets.QApplication(sys.argv)

        self._win = QtWidgets.QDialog()
        self._win.setWindowTitle(title)
        self._win.resize(700, 480)
        self._win.setStyleSheet("background: #1a1a1a; color: #ddd;")

        layout = QtWidgets.QVBoxLayout(self._win)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ---- Rate selector ----
        ctrl_row = QtWidgets.QHBoxLayout()
        rate_lbl = QtWidgets.QLabel("Y-axis:")
        rate_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self._radio_mean = QtWidgets.QRadioButton("Mean rate (Hz)")
        self._radio_peak = QtWidgets.QRadioButton("Peak instantaneous rate (Hz)")
        self._radio_both = QtWidgets.QRadioButton("Both")
        self._radio_mean.setChecked(True)
        for rb in (self._radio_mean, self._radio_peak, self._radio_both):
            rb.setStyleSheet("color: #ccc; font-size: 11px;")
            rb.toggled.connect(self._refresh_plot)

        ctrl_row.addWidget(rate_lbl)
        ctrl_row.addWidget(self._radio_mean)
        ctrl_row.addWidget(self._radio_peak)
        ctrl_row.addWidget(self._radio_both)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ---- Plot ----
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#111")
        self._plot_widget.setLabel("bottom", "Current injection (pA)")
        self._plot_widget.setLabel("left", "Firing rate (Hz)")
        self._plot_widget.getAxis("left").enableAutoSIPrefix(False)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self._plot_widget)

        # ---- Info box ----
        self._info_box = QtWidgets.QLabel()
        self._info_box.setStyleSheet(
            "color: #aaf; font-size: 11px; font-family: monospace; padding: 2px;"
        )
        layout.addWidget(self._info_box)

        self._refresh_plot()

    def _refresh_plot(self) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore

        pw = self._plot_widget
        pw.clear()

        show_mean = self._radio_mean.isChecked() or self._radio_both.isChecked()
        show_peak = self._radio_peak.isChecked() or self._radio_both.isChecked()

        currents = self._currents

        if show_mean and self._mean_rates:
            pw.plot(
                currents,
                self._mean_rates,
                pen=pg.mkPen("#4af", width=2),
                symbol="o",
                symbolBrush="#4af",
                symbolSize=7,
                name="Mean rate",
            )

        if show_peak and self._peak_rates:
            valid_pairs = [
                (c, r) for c, r in zip(currents, self._peak_rates)
                if r is not None and not (isinstance(r, float) and math.isnan(r))
            ]
            if valid_pairs:
                vc, vr = zip(*valid_pairs)
                pw.plot(
                    list(vc),
                    list(vr),
                    pen=pg.mkPen("#fa4", width=2),
                    symbol="s",
                    symbolBrush="#fa4",
                    symbolSize=7,
                    name="Peak inst. rate",
                )

        # Rheobase vertical line
        if not (self._rheobase != self._rheobase):  # not NaN
            rheo_line = pg.InfiniteLine(
                pos=self._rheobase,
                angle=90,
                pen=pg.mkPen("#8f8", width=1, style=QtCore.Qt.DashLine),
                label=f"Rheobase\n{self._rheobase:.0f} pA",
                labelOpts={"color": "#8f8", "position": 0.85},
            )
            pw.addItem(rheo_line)

        # F-I slope fit line (over the fitted range)
        if (
            show_mean
            and not math.isnan(self._slope)
            and not math.isnan(self._intercept)
            and self._slope_n >= 2
            and currents
        ):
            supra = [c for c in currents if not math.isnan(self._rheobase) and c >= self._rheobase]
            if supra:
                fit_currents = sorted(supra)[: self._slope_n]
                if fit_currents:
                    c_min, c_max = fit_currents[0], fit_currents[-1]
                    fit_x = [c_min, c_max]
                    fit_y = [self._slope * x + self._intercept for x in fit_x]
                    pw.plot(
                        fit_x,
                        fit_y,
                        pen=pg.mkPen("#f4f", width=1.5, style=QtCore.Qt.DashLine),
                    )

        # Info line
        parts = []
        if not math.isnan(self._rheobase):
            parts.append(f"Rheobase: {self._rheobase:.0f} pA")
        if not math.isnan(self._max_rate):
            parts.append(f"Max mean rate: {self._max_rate:.1f} Hz")
        if not math.isnan(self._max_peak_rate):
            parts.append(f"Max peak rate: {self._max_peak_rate:.1f} Hz")
        if not math.isnan(self._slope):
            r2_str = f"  R²={self._r2:.3f}" if not math.isnan(self._r2) else ""
            parts.append(f"F-I slope: {self._slope:.4f} Hz/pA{r2_str} (n={self._slope_n})")
        self._info_box.setText("    ".join(parts))

    def show(self) -> None:
        """Show the window (non-blocking)."""
        self._win.show()

    def exec(self) -> None:
        """Show the window and block until closed."""
        self._win.exec()
