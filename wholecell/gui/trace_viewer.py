"""
trace_viewer.py
---------------
Interactive trace viewer for whole-cell current-clamp recordings.

Displays voltage traces from ABF files with overlaid spike markers
(threshold, peak, trough) and passive fit overlays (tau fit, sag).

Launch
------
    python -m wholecell.gui.trace_viewer path/to/file.abf

Or from Python:
    from wholecell.gui.trace_viewer import launch_viewer
    launch_viewer("path/to/file.abf")

Keyboard shortcuts
------------------
    Left / Right arrows : previous / next sweep
    F                   : toggle lowpass filter (2 kHz default)
    D                   : toggle dV/dt panel
    S                   : toggle spike markers
    Q / Escape          : quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class TraceViewer:
    """pyqtgraph-based viewer for a single ABF Recording.

    Parameters
    ----------
    recording : Recording
        Loaded Recording object.
    spike_result : dict or None
        Output of run_spike_detection (for marker overlay). Optional.
    passive_result : dict or None
        Output of run_passive_analysis (for tau fit overlay). Optional.
    lowpass_hz : float or None
        Default lowpass filter cutoff. Toggled with F key.
    """

    def __init__(
        self,
        recording,
        spike_result: dict | None = None,
        passive_result: dict | None = None,
        lowpass_hz: float | None = 2000.0,
    ) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._rec = recording
        self._spike_result = spike_result
        self._passive_result = passive_result
        self._default_lowpass_hz = lowpass_hz

        self._sweep_index = 0
        self._filter_on = False
        self._show_dvdt = False
        self._show_spikes = spike_result is not None

        # Build spike lookup: sweep_index → list of spike dicts
        self._spike_lookup: dict[int, list] = {}
        if spike_result:
            for sweep_data in spike_result.get("data", {}).get("per_sweep", []):
                self._spike_lookup[sweep_data["sweep_index"]] = sweep_data.get("spikes", [])

        # Build tau-fit lookup: sweep_index → _tau_fit dict
        self._tau_lookup: dict[int, dict] = {}
        if passive_result:
            for row in passive_result.get("per_sweep", []):
                if "_tau_fit" in row:
                    self._tau_lookup[row["sweep_index"]] = row["_tau_fit"]

        # ---- Build Qt application and windows ----
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        self._win = pg.GraphicsLayoutWidget(
            title=f"Trace Viewer — {recording.filename}"
        )
        self._win.resize(1100, 700)

        # Voltage panel (always shown)
        self._plot_v = self._win.addPlot(row=0, col=0)
        self._plot_v.setLabel("left", "Voltage", units="mV")
        self._plot_v.setLabel("bottom", "Time", units="s")
        self._plot_v.showGrid(x=True, y=True, alpha=0.3)
        self._curve_v = self._plot_v.plot(pen=pg.mkPen("w", width=1))

        # dV/dt panel (shown when toggled)
        self._plot_d = self._win.addPlot(row=1, col=0)
        self._plot_d.setLabel("left", "dV/dt", units="mV/ms")
        self._plot_d.setLabel("bottom", "Time", units="s")
        self._plot_d.showGrid(x=True, y=True, alpha=0.3)
        self._plot_d.setXLink(self._plot_v)
        self._curve_d = self._plot_d.plot(pen=pg.mkPen("#4af", width=1))
        self._plot_d.setVisible(False)

        # Spike marker scatter plots
        self._scatter_thresh = self._plot_v.plot(
            pen=None, symbol="t", symbolSize=10,
            symbolBrush="#f70", symbolPen=None,
        )
        self._scatter_peak = self._plot_v.plot(
            pen=None, symbol="o", symbolSize=8,
            symbolBrush="#0f0", symbolPen=None,
        )
        self._scatter_trough = self._plot_v.plot(
            pen=None, symbol="d", symbolSize=8,
            symbolBrush="#f44", symbolPen=None,
        )

        # Tau fit overlay
        self._curve_tau = self._plot_v.plot(
            pen=pg.mkPen("#ff0", width=2, style=2)  # dashed yellow
        )

        # Status label
        self._label = pg.LabelItem(justify="left")
        self._win.addItem(self._label, row=2, col=0)

        # Keyboard handler
        self._win.keyPressEvent = self._on_key

        self._update()
        self._win.show()

    # ------------------------------------------------------------------
    # Update display
    # ------------------------------------------------------------------

    def _update(self) -> None:
        lowpass = self._default_lowpass_hz if self._filter_on else None
        t, v, i = self._rec.get_sweep_arrays(self._sweep_index, lowpass_hz=lowpass)

        self._curve_v.setData(t, v)

        if self._show_dvdt:
            dvdt = np.gradient(v, t) / 1000.0
            self._curve_d.setData(t, dvdt)

        self._update_spike_markers(t, v)
        self._update_tau_overlay()
        self._update_label()

    def _update_spike_markers(self, t: np.ndarray, v: np.ndarray) -> None:
        if not self._show_spikes:
            self._scatter_thresh.setData([], [])
            self._scatter_peak.setData([], [])
            self._scatter_trough.setData([], [])
            return

        spikes = self._spike_lookup.get(self._sweep_index, [])
        if not spikes:
            self._scatter_thresh.setData([], [])
            self._scatter_peak.setData([], [])
            self._scatter_trough.setData([], [])
            return

        thresh_t = [s["threshold_time_s"] for s in spikes]
        thresh_v = [s["threshold_voltage_mV"] for s in spikes]
        peak_t = [s["peak_time_s"] for s in spikes]
        peak_v = [s["peak_voltage_mV"] for s in spikes]
        trough_t = [s["trough_time_s"] for s in spikes]
        trough_v = [s["trough_voltage_mV"] for s in spikes]

        self._scatter_thresh.setData(thresh_t, thresh_v)
        self._scatter_peak.setData(peak_t, peak_v)
        self._scatter_trough.setData(trough_t, trough_v)

    def _update_tau_overlay(self) -> None:
        fit = self._tau_lookup.get(self._sweep_index)
        if fit and fit.get("time") and fit.get("predicted"):
            self._curve_tau.setData(fit["time"], fit["predicted"])
        else:
            self._curve_tau.setData([], [])

    def _update_label(self) -> None:
        filt_str = f"LP {self._default_lowpass_hz:.0f} Hz" if self._filter_on else "raw"
        dvdt_str = "dV/dt ON" if self._show_dvdt else ""
        spike_str = "spikes ON" if self._show_spikes else ""
        parts = [p for p in [filt_str, dvdt_str, spike_str] if p]
        status = " | ".join(parts)
        self._label.setText(
            f"Sweep {self._sweep_index + 1}/{self._rec.n_sweeps}    {status}    "
            f"[←→ navigate | F filter | D dV/dt | S spikes | Q quit]"
        )

    # ------------------------------------------------------------------
    # Keyboard handler
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        from pyqtgraph.Qt import QtCore

        key = event.key()
        if key == QtCore.Qt.Key.Key_Right:
            self._sweep_index = min(self._sweep_index + 1, self._rec.n_sweeps - 1)
            self._update()
        elif key == QtCore.Qt.Key.Key_Left:
            self._sweep_index = max(self._sweep_index - 1, 0)
            self._update()
        elif key == QtCore.Qt.Key.Key_F:
            self._filter_on = not self._filter_on
            self._update()
        elif key == QtCore.Qt.Key.Key_D:
            self._show_dvdt = not self._show_dvdt
            self._plot_d.setVisible(self._show_dvdt)
            self._update()
        elif key == QtCore.Qt.Key.Key_S:
            self._show_spikes = not self._show_spikes
            self._update()
        elif key in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self._win.close()

    def run(self) -> None:
        """Enter the Qt event loop (blocking)."""
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
) -> None:
    """Open TraceViewer for a single ABF file.

    Parameters
    ----------
    filepath : str or Path
    sweep_index : int
        Initial sweep to display.
    spike_result : dict or None
        Detection result from run_spike_detection for marker overlay.
    passive_result : dict or None
        Result from run_passive_analysis for tau fit overlay.
    lowpass_hz : float or None
        Default lowpass filter cutoff toggled with F key. Default 2 kHz.
    """
    from wholecell.core.recording import Recording

    rec = Recording(filepath)
    viewer = TraceViewer(
        rec,
        spike_result=spike_result,
        passive_result=passive_result,
        lowpass_hz=lowpass_hz,
    )
    viewer._sweep_index = sweep_index
    viewer._update()
    viewer.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Whole-cell trace viewer")
    parser.add_argument("filepath", help="Path to .abf file")
    parser.add_argument("--sweep", type=int, default=0, help="Initial sweep index")
    parser.add_argument("--lowpass", type=float, default=2000.0,
                        help="Default lowpass filter Hz (toggled with F)")
    args = parser.parse_args()

    launch_viewer(args.filepath, sweep_index=args.sweep, lowpass_hz=args.lowpass)


if __name__ == "__main__":
    main()
