"""
ahp_viewer.py
-------------
Popup window for visualising post-step afterhyperpolarization.

Shows medium (mAHP) and slow (sAHP) AHP against the injected step current,
either as a delta from the pre-step baseline or as absolute voltage.

Launched from Cell.plot_ahp() or from the TraceViewer sidebar.
"""

from __future__ import annotations

import math

# Series palette. The trace viewer imports these for its own AHP markers so the
# two windows cannot drift apart, and the coloured "Show:" radio labels act as
# the key for both.
MAHP_COLOR = "#4af"
SAHP_COLOR = "#fa4"

_RADIO_COLOR = "#ccc"


def _finite_pairs(xs: list, ys: list) -> tuple[list, list]:
    """Return (x, y) with any pair containing a missing value dropped."""
    pairs = [
        (x, y) for x, y in zip(xs, ys)
        if x is not None and y is not None
        and not (isinstance(x, float) and math.isnan(x))
        and not (isinstance(y, float) and math.isnan(y))
    ]
    if not pairs:
        return [], []
    x_vals, y_vals = zip(*pairs)
    return list(x_vals), list(y_vals)


def _add_radio_row(QtWidgets, layout, label_text, radios, group) -> None:
    """Append a labelled row of radio buttons bound to their own button group.

    ``radios`` is a sequence of ``(button, color)`` pairs. The colour is applied
    unconditionally, not on selection: for the measure row it is the plot key,
    so ``mAHP`` must read blue and ``sAHP`` orange even while deselected.
    """
    row = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel(label_text)
    label.setStyleSheet("color: #aaa; font-size: 11px;")
    row.addWidget(label)
    for rb, color in radios:
        rb.setStyleSheet(f"color: {color}; font-size: 11px;")
        group.addButton(rb)
        row.addWidget(rb)
    row.addStretch()
    layout.addLayout(row)


class AHPViewer:
    """AHP popup window backed by pyqtgraph.

    Parameters
    ----------
    ahp_result : dict
        The ``"data"`` field of a timestamped ahp result from Cell.
        Must have keys ``"ahp_curve"`` and ``"cell_level"``.
    title : str, optional
        Window title.
    """

    def __init__(self, ahp_result: dict, title: str = "AHP") -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets

        self._result = ahp_result
        curve = ahp_result.get("ahp_curve", {})
        self._cell_level = ahp_result.get("cell_level", {})

        self._currents = curve.get("step_current_pA", [])
        self._series = {
            "mahp": {
                "delta": curve.get("mahp_delta_mV", []),
                "absolute": curve.get("mahp_voltage_mV", []),
                "color": MAHP_COLOR,
                "symbol": "o",
                "label": "mAHP",
            },
            "sahp": {
                "delta": curve.get("sahp_delta_mV", []),
                "absolute": curve.get("sahp_voltage_mV", []),
                "color": SAHP_COLOR,
                "symbol": "s",
                "label": "sAHP",
            },
        }

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

        # ---- Selectors ----
        # Two independent radio rows. They need explicit button groups: without
        # them Qt auto-exclusivity would treat all five radios sharing the
        # dialog as one set, so picking a Y quantity would clear the measure.
        self._radio_mahp = QtWidgets.QRadioButton("mAHP")
        self._radio_sahp = QtWidgets.QRadioButton("sAHP")
        self._radio_both = QtWidgets.QRadioButton("Both")
        self._radio_delta = QtWidgets.QRadioButton("Δ from baseline (mV)")
        self._radio_absolute = QtWidgets.QRadioButton("Absolute voltage (mV)")

        self._measure_group = QtWidgets.QButtonGroup(self._win)
        self._quantity_group = QtWidgets.QButtonGroup(self._win)

        # The measure labels are tinted to their series colour — this is the
        # plot's key. "Both" stays grey: it is not a series of its own.
        _add_radio_row(
            QtWidgets, layout, "Show:",
            ((self._radio_mahp, MAHP_COLOR),
             (self._radio_sahp, SAHP_COLOR),
             (self._radio_both, _RADIO_COLOR)),
            self._measure_group,
        )
        _add_radio_row(
            QtWidgets, layout, "Y-axis:",
            ((self._radio_delta, _RADIO_COLOR),
             (self._radio_absolute, _RADIO_COLOR)),
            self._quantity_group,
        )

        self._radio_both.setChecked(True)
        self._radio_delta.setChecked(True)

        for rb in (self._radio_mahp, self._radio_sahp, self._radio_both,
                   self._radio_delta, self._radio_absolute):
            rb.toggled.connect(self._refresh_plot)

        # ---- Plot ----
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#111")
        self._plot_widget.setLabel("bottom", "Step current (pA)")
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

    def _refresh_plot(self) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore

        pw = self._plot_widget
        pw.clear()

        show_delta = self._radio_delta.isChecked()
        quantity = "delta" if show_delta else "absolute"
        pw.setLabel(
            "left",
            "Δ from baseline (mV)" if show_delta else "Voltage (mV)",
        )

        show = {
            "mahp": self._radio_mahp.isChecked() or self._radio_both.isChecked(),
            "sahp": self._radio_sahp.isChecked() or self._radio_both.isChecked(),
        }

        for key, spec in self._series.items():
            if not show[key]:
                continue
            x_vals, y_vals = _finite_pairs(self._currents, spec[quantity])
            if not x_vals:
                continue
            pw.plot(
                x_vals,
                y_vals,
                pen=pg.mkPen(spec["color"], width=2),
                symbol=spec["symbol"],
                symbolBrush=spec["color"],
                symbolSize=7,
                name=spec["label"],
            )

        if show_delta:
            pw.addItem(pg.InfiniteLine(
                pos=0.0,
                angle=0,
                pen=pg.mkPen("#666", width=1, style=QtCore.Qt.DashLine),
            ))

        self._info_box.setText("    ".join(self._info_parts()))

    def _info_parts(self) -> list[str]:
        """Cell-level summary strings for the info label."""
        cl = self._cell_level
        parts = []

        for key, label in (("mahp", "mAHP"), ("sahp", "sAHP")):
            mean = cl.get(f"mean_{key}_delta_mV", float("nan"))
            std = cl.get(f"std_{key}_delta_mV", float("nan"))
            peak = cl.get(f"max_{key}_delta_mV", float("nan"))
            peak_i = cl.get(f"current_at_max_{key}_pA", float("nan"))
            if isinstance(mean, float) and not math.isnan(mean):
                parts.append(f"{label} Δ: {mean:.2f} ± {std:.2f} mV")
            if isinstance(peak, float) and not math.isnan(peak):
                parts.append(f"max {label}: {peak:.2f} mV @ {peak_i:.0f} pA")

        n = cl.get("n_sweeps_analyzed", 0)
        skipped = cl.get("n_sweeps_skipped_nondepolarizing", 0)
        parts.append(f"n={n} depolarizing sweeps ({skipped} skipped)")

        if cl.get("any_window_truncated"):
            parts.append("⚠ search window truncated on ≥1 sweep")

        return parts

    def show(self) -> None:
        """Show the window (non-blocking)."""
        self._win.show()

    def exec(self) -> None:
        """Show the window and block until closed."""
        self._win.exec()
