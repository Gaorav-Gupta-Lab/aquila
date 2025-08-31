from dataclasses import dataclass
from PySide6 import QtCore, QtWidgets, QtGui
from pathlib import Path
import pandas as pd

from aquila_utils import process_image_file, plot_results

@dataclass
class AquilaParams:
    input_dir: str
    output_dir: str
    sample_groups: str
    channels: list[str] | None
    sigmaA: float
    sigmaB: float
    prominence: float
    min_nucleus_area: int
    max_nucleus_area: int
    dapi_foreground: str
    blur_sigma: float
    seed_radius: int
    maxima_smooth_sigma: float
    maxima_min_distance: int
    extensions: str
    copy_original: bool

class Worker(QtCore.QObject):
    finished = QtCore.Signal()
    log = QtCore.Signal(str)

    def __init__(self, params: AquilaParams):
        super().__init__()
        self.params = params
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    @QtCore.Slot()
    def run(self):
        all_results = []
        in_dir = Path(self.params.input_dir)
        exts = {e.strip().lower() for e in self.params.extensions.split(",")}
        files = [p for p in sorted(in_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in exts]
        if not files:
            self.log.emit("[WARN] No matching files found.")
        for i, fp in enumerate(files, 1):
            if self._stop_flag:
                self.log.emit("[INFO] Stopped by user.")
                break
            self.log.emit(f"[{i}/{len(files)}] Processing {fp.name} …")
            try:
                info = process_image_file(
                    fp, self.params,
                    output_dir=Path(self.params.output_dir),
                    log=self.log.emit
                )
                all_results.append(info["results_df"])
            except Exception as e:
                self.log.emit(f"[ERR] {fp.name}: {e}")
        if not self._stop_flag and all_results:
            combined = pd.concat(all_results, ignore_index=True)
            summary_dir = Path(self.params.output_dir) / "Summary"
            summary_dir.mkdir(exist_ok=True)
            combined.to_csv(summary_dir / "all_results_summary.csv", index=False)
            plot_path = plot_results(combined, summary_dir, order=self.params.sample_groups)
            if plot_path:
                self.log.emit(f"[OK] Wrote violin summary → {plot_path.name}")
        self.finished.emit()

class AquilaWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AQUILA — Analyze Foci")
        self.setMinimumSize(820, 600)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # -----------------
        # Directories Group
        # -----------------
        dir_group = QtWidgets.QGroupBox("Directories")
        dir_layout = QtWidgets.QFormLayout(dir_group)

        self.in_edit = QtWidgets.QLineEdit()
        self.in_edit.setPlaceholderText("Enter directory where images are stored")
        btn_in = QtWidgets.QPushButton("Browse…")
        btn_in.clicked.connect(self._browse_input)
        in_row = QtWidgets.QHBoxLayout()
        in_row.addWidget(self.in_edit)
        in_row.addWidget(btn_in)
        dir_layout.addRow("Input directory:", in_row)

        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText("Enter results directory (leaving this blank will save results to input directory)")
        btn_out = QtWidgets.QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_output)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_row.addWidget(btn_out)
        dir_layout.addRow("Output directory:", out_row)

        main_layout.addWidget(dir_group)

        # --------------------------
        # Detection Parameters Group
        # --------------------------
        detect_group = QtWidgets.QGroupBox("Focus Detection Parameters")
        detect_layout = QtWidgets.QGridLayout(detect_group)

        self.sigmaA = QtWidgets.QDoubleSpinBox(); self.sigmaA.setRange(0.1, 50.0); self.sigmaA.setValue(1.0)
        self.sigmaB = QtWidgets.QDoubleSpinBox(); self.sigmaB.setRange(0.1, 50.0); self.sigmaB.setValue(8.0)
        self.prominence = QtWidgets.QDoubleSpinBox(); self.prominence.setRange(0.0, 5000.0); self.prominence.setValue(35.0)

        label_sigmaA = QtWidgets.QLabel("Sigma A (DoG):")
        label_sigmaA.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        detect_layout.addWidget(label_sigmaA, 0, 0); detect_layout.addWidget(self.sigmaA, 0, 1)
        label_sigmaB = QtWidgets.QLabel("Sigma B (DoG):")
        label_sigmaB.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        detect_layout.addWidget(label_sigmaB, 0, 2); detect_layout.addWidget(self.sigmaB, 0, 3)
        label_prominence = QtWidgets.QLabel("Prominence:")
        label_prominence.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        detect_layout.addWidget(label_prominence, 0, 4); detect_layout.addWidget(self.prominence, 0, 5)

        main_layout.addWidget(detect_group)


        # ------------------
        # Sample Groups & Channels
        # ------------------
        groups_box = QtWidgets.QGroupBox("Sample Groups and Channels")
        groups_grid = QtWidgets.QGridLayout(groups_box)   # parented; no need to call setLayout()

        # Sample groups row
        lbl_groups = QtWidgets.QLabel("Sample group names:")
        self.sample_groups = QtWidgets.QLineEdit()
        lbl_groups.setBuddy(self.sample_groups)
        self.sample_groups.setPlaceholderText("Enter group names separated by commas")
        self.sample_groups.setClearButtonEnabled(True)

        # Put label in col 0, line edit spans all cols
        groups_grid.addWidget(lbl_groups, 0, 0)
        groups_grid.addWidget(self.sample_groups, 0, 1, 1, -1)

        # Channels (built via loop)
        self.channels = []
        channel_items = ["None", "DAPI", "TexasRed", "Cy5", "GFP"]

        for i in range(5):
            lbl = QtWidgets.QLabel(f"Channel {i+1}:")
            cmb = QtWidgets.QComboBox()
            cmb.addItems(channel_items)

            # Default channels for testing, change upon release
            if i == 0:
                cmb.setCurrentText("DAPI")
            if i == 1:
                cmb.setCurrentText("TexasRed")

            cmb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)  # keep compact
            # Labels on row 1 (cols 2..4), combos on row 2 (cols 2..4)
            col = 2 + i
            groups_grid.addWidget(lbl, 1, col)
            groups_grid.addWidget(cmb, 2, col)
            self.channels.append(cmb)

        # Layout behavior & spacing
        groups_grid.setHorizontalSpacing(12)
        groups_grid.setVerticalSpacing(6)
        groups_grid.setContentsMargins(10, 10, 10, 10)

        # # Make the groups QLineEdit expand; keep channel columns snug
        groups_grid.setColumnStretch(0, 0)   # label
        groups_grid.setColumnStretch(1, 1)   # expanding field
        groups_grid.setColumnStretch(2, 0)
        groups_grid.setColumnStretch(3, 0)
        groups_grid.setColumnStretch(4, 0)

        # Add the group to your window's main layout
        main_layout.addWidget(groups_box)

        # --------------------------
        # Nucleus Segmentation Group
        # --------------------------
        seg_group = QtWidgets.QGroupBox("Nucleus Segmentation")
        seg_layout = QtWidgets.QGridLayout(seg_group)

        self.min_area = QtWidgets.QSpinBox(); self.min_area.setRange(0, 100000); self.min_area.setValue(200)
        self.max_area = QtWidgets.QSpinBox(); self.max_area.setRange(0, 100000); self.max_area.setValue(5000)
        self.foreground = QtWidgets.QComboBox(); self.foreground.addItems(["bright", "dark"])
        self.blur_sigma = QtWidgets.QDoubleSpinBox(); self.blur_sigma.setRange(0.0, 10.0); self.blur_sigma.setValue(1.8)
        self.seed_radius = QtWidgets.QSpinBox(); self.seed_radius.setRange(1, 50); self.seed_radius.setValue(3)

        label_min_area = QtWidgets.QLabel("Min area (px):")
        label_min_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        seg_layout.addWidget(label_min_area, 0, 0); seg_layout.addWidget(self.min_area, 0, 1)
        label_max_area = QtWidgets.QLabel("Max area (px):")
        label_max_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        seg_layout.addWidget(label_max_area, 0, 2); seg_layout.addWidget(self.max_area, 0, 3)

        label_foreground = QtWidgets.QLabel("DAPI foreground:")
        label_foreground.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        seg_layout.addWidget(label_foreground, 1, 0); seg_layout.addWidget(self.foreground, 1, 1)
        label_blur_sigma = QtWidgets.QLabel("Blur σ:")
        label_blur_sigma.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        seg_layout.addWidget(label_blur_sigma, 1, 2); seg_layout.addWidget(self.blur_sigma, 1, 3)
        label_seed_radius = QtWidgets.QLabel("Seed radius:")
        label_seed_radius.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        seg_layout.addWidget(label_seed_radius, 1, 4); seg_layout.addWidget(self.seed_radius, 1, 5)

        main_layout.addWidget(seg_group)

        # -------------------------
        # Advanced Parameters Group
        # -------------------------
        adv_group = QtWidgets.QGroupBox("Advanced")
        adv_layout = QtWidgets.QGridLayout(adv_group)

        self.maxima_sigma = QtWidgets.QDoubleSpinBox(); self.maxima_sigma.setRange(0.0, 10.0); self.maxima_sigma.setValue(1.0)
        self.min_dist = QtWidgets.QSpinBox(); self.min_dist.setRange(1, 50); self.min_dist.setValue(1)
        self.exts = QtWidgets.QLineEdit(".tif,.tiff,.png,.jpg,.jpeg")
        self.exts.setClearButtonEnabled(True)
        self.copy_original = QtWidgets.QCheckBox("Copy original images into output folder"); self.copy_original.setChecked(True)

        adv_layout.addWidget(QtWidgets.QLabel("Maxima smooth σ:"), 0, 0); adv_layout.addWidget(self.maxima_sigma, 0, 1)
        adv_layout.addWidget(QtWidgets.QLabel("Minimum distance:"), 0, 2); adv_layout.addWidget(self.min_dist, 0, 3)
        adv_layout.addWidget(QtWidgets.QLabel("Extensions:"), 1, 0); adv_layout.addWidget(self.exts, 1, 1, 1, 3)
        adv_layout.addWidget(self.copy_original, 2, 0, 1, 4)

        main_layout.addWidget(adv_group)

        # ------------------
        # Run / Stop Buttons
        # ------------------
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("▶ Run Analysis")
        self.btn_run.setStyleSheet("""
            font-size: 16pt;
            border: 1px solid white;
        """)
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setMinimumWidth(180)

        self.btn_stop = QtWidgets.QPushButton("■ Stop")
        self.btn_stop.setStyleSheet("""
            font-size: 16pt;
            border: 1px solid white;
        """)
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setMinimumWidth(180)
        self.btn_stop.setEnabled(False)

        self.btn_run.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)

        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        main_layout.addLayout(btn_layout)

        # ----------
        # Log Output
        # ----------
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QtGui.QFont("Consolas", 10))
        log_layout.addWidget(self.log_box)
        main_layout.addWidget(log_group)

        # ----------
        # Status Bar
        # ----------
        self.status = QtWidgets.QLabel("Idle.")
        main_layout.addWidget(self.status)

        # -------------------------
        # Stylesheet for Clean Look
        # -------------------------
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #aaa;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QLabel {
                font-size: 11pt;
            }
            QPushButton {
                padding: 5px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QPlainTextEdit {
                background-color: #111;
                color: #eee;
                border: 1px solid #444;
            }
        """)

    # -----------------
    # Directory Pickers
    # -----------------
    def _browse_input(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose input directory")
        if d:
            self.in_edit.setText(d)

    def _browse_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if d:
            self.out_edit.setText(d)

    # ------------------------------------------------
    # Stored parameter list to access outside of class
    # ------------------------------------------------
    def _params(self) -> AquilaParams:
        input_dir=self.in_edit.text().strip()
        output_dir=self.out_edit.text().strip()

        if not output_dir:
            output_dir = input_dir

        return AquilaParams(
            input_dir=input_dir,
            output_dir=output_dir,
            sample_groups=[g.strip() for g in self.sample_groups.text().strip().split(",")],
            channels = [
                c.currentText() if c.currentText() != 'None' else None 
                for c in self.channels
            ] if self.channels else None,
            sigmaA=self.sigmaA.value(),
            sigmaB=self.sigmaB.value(),
            prominence=self.prominence.value(),
            min_nucleus_area=self.min_area.value(),
            max_nucleus_area=self.max_area.value(),
            dapi_foreground=self.foreground.currentText(),
            blur_sigma=self.blur_sigma.value(),
            seed_radius=self.seed_radius.value(),
            maxima_smooth_sigma=self.maxima_sigma.value(),
            maxima_min_distance=self.min_dist.value(),
            extensions=self.exts.text().strip(),
            copy_original=self.copy_original.isChecked()
        )


    # -----------------------
    # Upon "RUN" Button press
    # -----------------------
    def _start(self):
        params = self._params()
        if not Path(params.input_dir).is_dir():
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid input directory")
            return

        self.log_box.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Running…")

        print(params.channels)
        print(params.sample_groups)

        self.thread = QtCore.QThread()
        self.worker = Worker(params)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Reset UI when done
        self.worker.finished.connect(self._finish_run)

        self.thread.start()

    # ------------------------
    # Upon "STOP" Button press
    # ------------------------
    def _stop(self):
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
            self.status.setText("Stopping…")

    # -----------------------------
    # Executes after successful run
    # -----------------------------
    def _finish_run(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("Idle.")

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)