"""
fi_viewer.py
------------
Popup window for visualising F-I (frequency-current) curves.

Shows mean firing rate and peak instantaneous rate vs. injected current,
with rheobase marked and the fitted slope overlaid.

Can draw several conditions on one axes (control vs. apamin), so a drug effect
is visible at the rig rather than only after export.

Launched from Cell.plot_fi_curve() or from the TraceViewer toolbar.
"""

from __future__ import annotations

import math

# Per-condition colours, used only when more than one series is shown. A single
# series keeps the original mean/peak colours so the familiar view is unchanged.
SERIES_COLORS = ["#4af", "#f66", "#6d6", "#fa4", "#a8f", "#4dd"]

_SINGLE_MEAN_COLOR = "#4af"
_SINGLE_PEAK_COLOR = "#fa4"


def _parse(result: dict) -> dict:
    """Flatten one fi_curve result dict into the scalars the plot needs."""
    fi_curve = result.get("fi_curve", {}) or {}
    cell_level = result.get("cell_level", {}) or {}
    nan = float("nan")
    return {
        "currents": fi_curve.get("current_pA", []),
        "mean_rates": fi_curve.get("mean_rate_hz", []),
        "peak_rates": fi_curve.get("peak_rate_hz", []),
        "rheobase": cell_level.get("rheobase_pA", nan),
        "slope": cell_level.get("fi_slope_hz_per_pA", nan),
        "intercept": cell_level.get("fi_slope_intercept_hz", nan),
        "r2": cell_level.get("fi_slope_r2", nan),
        "slope_n": cell_level.get("fi_slope_n_points", 0),
        "max_rate": cell_level.get("max_firing_rate_hz", nan),
        "max_peak_rate": cell_level.get("max_peak_instantaneous_rate_hz", nan),
        "current_at_max_rate": cell_level.get("current_at_max_firing_pA", nan),
        "dep_block_current": cell_level.get("dep_block_current_pA", nan),
    }


def _normalise(fi_result) -> list[tuple[str, dict]]:
    """Accept a single result dict or a list of ``(label, result)`` pairs."""
    if isinstance(fi_result, dict):
        return [("", _parse(fi_result))]
    return [(str(label), _parse(res)) for label, res in fi_result]


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class FICurveViewer:
    """F-I curve popup window backed by pyqtgraph.

    Parameters
    ----------
    fi_result : dict or list of (str, dict)
        Either the ``"data"`` field of a timestamped fi_curve result from Cell
        (keys ``"fi_curve"`` and ``"cell_level"``), or a list of
        ``(condition_label, result)`` pairs to overlay on one axes.
    title : str, optional
        Window title.
    """

    def __init__(self, fi_result, title: str = "F-I Curve") -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._result = fi_result
        self._series = _normalise(fi_result)

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
        self._info_box.setWordWrap(True)
        layout.addWidget(self._info_box)

        self._refresh_plot()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_series(self, pw, label: str, data: dict, index: int,
                     show_mean: bool, show_peak: bool, multi: bool) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore

        currents = data["currents"]
        if multi:
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            mean_color = peak_color = color
            mean_name = f"{label} — mean"
            peak_name = f"{label} — peak"
            peak_style = QtCore.Qt.DashLine
        else:
            mean_color, peak_color = _SINGLE_MEAN_COLOR, _SINGLE_PEAK_COLOR
            mean_name, peak_name = "Mean rate", "Peak inst. rate"
            peak_style = QtCore.Qt.SolidLine

        if show_mean and data["mean_rates"]:
            pw.plot(
                currents,
                data["mean_rates"],
                pen=pg.mkPen(mean_color, width=2),
                symbol="o",
                symbolBrush=mean_color,
                symbolSize=7,
                name=mean_name,
            )

        if show_peak and data["peak_rates"]:
            valid_pairs = [
                (c, r) for c, r in zip(currents, data["peak_rates"])
                if not _is_nan(r)
            ]
            if valid_pairs:
                vc, vr = zip(*valid_pairs)
                pw.plot(
                    list(vc),
                    list(vr),
                    pen=pg.mkPen(peak_color, width=2, style=peak_style),
                    symbol="s",
                    symbolBrush=peak_color,
                    symbolSize=7,
                    name=peak_name,
                )

        # Rheobase vertical line
        rheobase = data["rheobase"]
        if not _is_nan(rheobase):
            rheo_color = mean_color if multi else "#8f8"
            rheo_text = (
                f"{label} rheobase\n{rheobase:.0f} pA" if multi
                else f"Rheobase\n{rheobase:.0f} pA"
            )
            pw.addItem(pg.InfiniteLine(
                pos=rheobase,
                angle=90,
                pen=pg.mkPen(rheo_color, width=1, style=QtCore.Qt.DashLine),
                label=rheo_text,
                labelOpts={"color": rheo_color, "position": 0.85 - 0.08 * index},
            ))

        # F-I slope fit line (over the fitted range)
        slope, intercept, slope_n = data["slope"], data["intercept"], data["slope_n"]
        if (
            show_mean
            and not _is_nan(slope)
            and not _is_nan(intercept)
            and slope_n >= 2
            and currents
        ):
            supra = [c for c in currents if not _is_nan(rheobase) and c >= rheobase]
            if supra:
                fit_currents = sorted(supra)[:slope_n]
                if fit_currents:
                    fit_x = [fit_currents[0], fit_currents[-1]]
                    fit_y = [slope * x + intercept for x in fit_x]
                    fit_color = mean_color if multi else "#f4f"
                    pw.plot(
                        fit_x,
                        fit_y,
                        pen=pg.mkPen(fit_color, width=1.5, style=QtCore.Qt.DashLine),
                    )

    @staticmethod
    def _info_parts(data: dict) -> list[str]:
        parts = []
        if not _is_nan(data["rheobase"]):
            parts.append(f"Rheobase: {data['rheobase']:.0f} pA")
        if not _is_nan(data["max_rate"]):
            parts.append(f"Max mean rate: {data['max_rate']:.1f} Hz")
        if not _is_nan(data["max_peak_rate"]):
            parts.append(f"Max peak rate: {data['max_peak_rate']:.1f} Hz")
        if not _is_nan(data["current_at_max_rate"]):
            parts.append(f"I @ max rate: {data['current_at_max_rate']:.0f} pA")
        if not _is_nan(data["dep_block_current"]):
            parts.append(f"Dep. block: {data['dep_block_current']:.0f} pA")
        if not _is_nan(data["slope"]):
            r2_str = "" if _is_nan(data["r2"]) else f"  R²={data['r2']:.3f}"
            parts.append(
                f"F-I slope: {data['slope']:.4f} Hz/pA{r2_str} (n={data['slope_n']})"
            )
        return parts

    def _refresh_plot(self) -> None:
        pw = self._plot_widget
        pw.clear()

        multi = len(self._series) > 1
        if multi:
            # Only worth a legend when there is something to tell apart; adding
            # one unconditionally would change the familiar single-curve view.
            pw.addLegend(offset=(-10, 10), labelTextColor="#ccc")

        show_mean = self._radio_mean.isChecked() or self._radio_both.isChecked()
        show_peak = self._radio_peak.isChecked() or self._radio_both.isChecked()

        info_lines = []
        for index, (label, data) in enumerate(self._series):
            self._plot_series(pw, label, data, index, show_mean, show_peak, multi)
            parts = self._info_parts(data)
            if not parts:
                continue
            info_lines.append(
                f"{label}:  {'    '.join(parts)}" if multi else "    ".join(parts)
            )

        self._info_box.setText("\n".join(info_lines))

    def show(self) -> None:
        """Show the window (non-blocking)."""
        self._win.show()

    def exec(self) -> None:
        """Show the window and block until closed."""
        self._win.exec()
