import shutil
import warnings
from pathlib import Path
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
from PIL import Image
import re

from PySide6 import QtCore, QtWidgets, QtGui

from skimage import filters, morphology, feature, measure, segmentation, util, exposure
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------
# Core image helpers
# ------------------
def robust_read_image(img_path: Path):
    """Return HxWxC uint8 (RGB/RGBA->RGB). Avoid OME; fallback to Pillow."""
    try:
        arr = tiff.imread(str(img_path), is_ome=False, key=0)
    except Exception:
        with Image.open(img_path) as im:
            im.seek(0)
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            arr = np.array(im)
    # Normalize HxWxC
    if arr.ndim == 2:
        raise ValueError(f"Expected color image (>=3 channels), got single-channel: {img_path}")
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.shape[-1] == 4:  # drop alpha
            arr = arr[..., :3]
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):  # CxHxW
        arr = np.moveaxis(arr, 0, -1)
        return arr[..., :3].astype(np.uint8, copy=False)
    if arr.ndim == 4 and arr.shape[0] <= 4:  # take first frame if small
        arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return arr[..., :3].astype(np.uint8, copy=False)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
            return arr[..., :3].astype(np.uint8, copy=False)
    raise ValueError(f"Unhandled TIFF layout {arr.shape} for {img_path}")

def truncate_name(name: str, token: str = "Top") -> str:
    idx = name.find(token)
    return name[:idx] if idx >= 0 else name

def difference_of_gaussians_uint8_like(img_uint8: np.ndarray, sigmaA: float, sigmaB: float):
    """Return DoG as float (≈0..255 after shift), WITHOUT rescaling to uint16 for computation."""
    gA = filters.gaussian(img_uint8, sigma=sigmaA, preserve_range=True)
    gB = filters.gaussian(img_uint8, sigma=sigmaB, preserve_range=True)
    dog = gA - gB
    dog_shift = dog - dog.min()
    return dog_shift.astype(np.float32, copy=False)

def segment_nuclei_from_dapi(dapi_u8: np.ndarray,
                             min_area_px: int,
                             max_area_px: int,
                             foreground: str,
                             blur_sigma: float,
                             seed_radius: int):
    
    """Robust nuclei segmentation with erosion-based markers to avoid midline splits."""
    dapi_blur = filters.gaussian(dapi_u8, sigma=blur_sigma, preserve_range=True)

    thr = filters.threshold_otsu(dapi_blur)
    mask = (dapi_blur > thr) if foreground == "bright" else (dapi_blur < thr)

    # Clean small objects and clean up nuclei
    mask = morphology.remove_small_objects(mask, min_area_px)
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = morphology.remove_small_holes(mask, area_threshold=128)

    dist = ndi.distance_transform_edt(mask)

    # Erosion-based markers (1 seed per nucleus)
    eroded = morphology.binary_erosion(mask, morphology.disk(seed_radius))
    markers = measure.label(eroded)

    if markers.max() == 0:
        valid_d = dist[mask]
        h = max(1.0, float(np.median(valid_d) * 0.4)) if valid_d.size else 1.0
        hmax = morphology.h_maxima(dist, h)
        markers = measure.label(hmax)

    labels = segmentation.watershed(-dist, markers=markers, mask=mask)

    # Remove edge-touching/small without merging neighbors
    H, W = labels.shape
    drop = set()
    for p in measure.regionprops(labels):
        # small
        if p.area < min_area_px:
            drop.add(p.label); continue
        # too big (likely merged clumps); only if enabled (default 5000px)
        if max_area_px and p.area > max_area_px:
            drop.add(p.label); continue
        # touches border
        minr, minc, maxr, maxc = p.bbox
        if minr == 0 or minc == 0 or maxr == H or maxc == W:
            drop.add(p.label)

    if drop:
        labels = labels.copy()
        labels[np.isin(labels, list(drop))] = 0

    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels

def find_maxima_hprominence(dog_float: np.ndarray, h_prom: float, smooth_sigma: float = 1.0, min_distance: int = 1):
    if smooth_sigma and smooth_sigma > 0:
        dogf = filters.gaussian(dog_float, sigma=smooth_sigma, preserve_range=True)
    else:
        dogf = dog_float
    dogf = dogf - dogf.min()
    hmax_mask = morphology.h_maxima(dogf, h_prom)
    coords = feature.peak_local_max(dogf, labels=hmax_mask, min_distance=min_distance)
    return coords

def measure_foci(labels: np.ndarray, foci_rc: np.ndarray, intensity_img: np.ndarray):
    if len(foci_rc):
        lab_at = labels[foci_rc[:, 0], foci_rc[:, 1]]
        keep = lab_at > 0 
        foci_rc = foci_rc[keep]
        lab_at = lab_at[keep]
        inten = intensity_img[foci_rc[:, 0], foci_rc[:, 1]]
        df_points = pd.DataFrame({"nucleus_label": lab_at.astype(int),
                                  "y": foci_rc[:, 0], "x": foci_rc[:, 1],
                                  "focus_intensity": inten.astype(float)})
    else:
        df_points = pd.DataFrame(columns=["nucleus_label", "y", "x", "focus_intensity"])

    areas = {p.label: p.area for p in measure.regionprops(labels)}
    if not df_points.empty:
        agg = df_points.groupby("nucleus_label")["focus_intensity"].agg(
            foci_count="count", mean_focus_intensity="mean",
            median_focus_intensity="median", max_focus_intensity="max"
        ).reset_index().fillna(0)
    else:
        agg = pd.DataFrame(columns=["nucleus_label", "foci_count",
                                    "mean_focus_intensity", "median_focus_intensity", "max_focus_intensity"])

    
    df_nuclei = (pd.DataFrame({"nucleus_label": list(areas.keys()), "area_px": list(areas.values())})
                 .merge(agg, on="nucleus_label", how="left")
                 .fillna({"foci_count": 0}))
    df_nuclei["foci_count"] = df_nuclei["foci_count"].astype(int)
    return df_nuclei, df_points

def save_overlay_png(dest_png: Path, base_img: np.ndarray, labels: np.ndarray, foci_rc: np.ndarray):
    base = base_img - base_img.min()
    base = (255.0 * (base / (base.max() if base.max() > 0 else 1))).astype(np.uint8)
    if base.ndim == 2:
        rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        rgb = base.copy()

    contours = measure.find_contours(labels > 0, 0.5)
    for c in contours:
        pts = np.flip(c.astype(np.int32), axis=1).reshape(-1, 1, 2)
        cv2.polylines(rgb, [pts], True, (0, 255, 0), 1)

    for (r, c) in foci_rc:
        cv2.circle(rgb, (int(c), int(r)), 2, (0, 0, 255), -1)

    import cv2 as _cv2
    _cv2.imwrite(str(dest_png), rgb)

def process_image_file(img_path: Path, params, output_dir=None, log=lambda *_: None):
    """
    params: AquilaParams
    """
    file_name = img_path.name
    sample_name = truncate_name(file_name, "Top")

    sample_group = re.search(r"\(([^)]+)\)", sample_name)
    sample_group = sample_group.group(1) if sample_group else None

    if output_dir:
        dest_dir = output_dir / sample_name
    else:
        dest_dir = img_path.parent / sample_name
        log(f"[WARN] No Valid output directory specified, saving to: {dest_dir}")
    dest_dir.mkdir(exist_ok=True)

    # Optional: copy original file
    if params.copy_original:
        try:
            shutil.copy2(img_path, dest_dir / file_name)
        except Exception as e:
            log(f"[WARN] Could not copy original image: {e}")

    arr = robust_read_image(img_path)
    pla_u8 = arr[..., 0]  # Red (PLA)
    dapi_u8 = arr[..., 2] # Blue (DAPI)

    # DoG
    dog = difference_of_gaussians_uint8_like(pla_u8, params.sigmaA, params.sigmaB)
    dog_out = dest_dir / f"{sample_name}-PLA_halo_post_threshold.tif"
    dog_save = exposure.rescale_intensity(dog, out_range=(0, 65535)).astype(np.uint16)
    tiff.imwrite(str(dog_out), dog_save)

    # Nuclei
    labels = segment_nuclei_from_dapi(dapi_u8,
                                      min_area_px=params.min_nucleus_area,
                                      max_area_px=params.max_nucleus_area,
                                      foreground=params.dapi_foreground,
                                      blur_sigma=params.blur_sigma,
                                      seed_radius=params.seed_radius)
    nuclei_out = dest_dir / f"{sample_name}_nucleusROIs.tif"
    tiff.imwrite(str(nuclei_out), labels.astype(np.uint16))

    # Maxima
    coords = find_maxima_hprominence(dog, h_prom=params.prominence,
                                     smooth_sigma=params.maxima_smooth_sigma,
                                     min_distance=params.maxima_min_distance)
    H, W = dog.shape
    if len(coords):
        inside = (coords[:, 0] > 0) & (coords[:, 0] < H - 1) & (coords[:, 1] > 0) & (coords[:, 1] < W - 1)
        coords = coords[inside]

    # Measure
    df_nuclei, df_points = measure_foci(labels, coords, dog)
    results_csv = dest_dir / f"{sample_name}_foci_results.csv"
    points_csv = dest_dir / f"{sample_name}_foci_points.csv"

    df_nuclei.insert(0, "sample_name", sample_name)
    df_points.insert(0, "sample_name", sample_name)
    df_nuclei.insert(1, "sample_group", sample_group)
    df_points.insert(1, "sample_group", sample_group)

    df_nuclei.to_csv(results_csv, index=False)
    df_points.to_csv(points_csv, index=False)

    # Overlay
    overlay_png_path = dest_dir / f"{sample_name}_overlay.png"
    save_overlay_png(overlay_png_path, dog, labels, coords)

    log(f"[OK] {file_name} -> {results_csv.name}  (nuclei={len(df_nuclei)}, foci={len(df_points)})")

    return {
    "sample_name": sample_name,
    "results_csv": str(results_csv),
    "results_df": df_nuclei,
    "points_df": df_points
    }

def plot_results(input_df: pd.DataFrame, out_dir: Path, order=None):
    """
    Make a violinplot of foci per nucleus by sample_group.
    Assumes results_df has columns: ['sample_name','nucleus_label','area_px','foci_count',...]
    """
    if input_df.empty or "foci_count" not in input_df.columns:
        return None

    df = input_df.copy()
    counts = (df.groupby(['sample_group', 'foci_count'], observed=True)
                 .size()
                 .reset_index(name='count'))
    binned_df = counts.copy()
    binned_df['foci_count'] = pd.cut(binned_df['foci_count'], bins=[0, 2, 4, 6, 8, np.inf], right=False)
    binned_df = binned_df.groupby(['foci_count', 'sample_group']).sum().reset_index()

    palette_violin = ["lightgray", "#5680C4", "#96BBCF"]
    palette_bars   = ['#181E3B', '#487394', '#C2C7BA', '#EBCC8E', '#D66955']
    base = 'v3KO'  # baseline used for delta_no_base


    # Stacked bar proportions from binned_df (one row per sample_group; columns: sample_group, foci_count, count)
    pivot_df = (binned_df.pivot(index='sample_group', columns='foci_count', values='count')
                        .fillna(0)
                        .reindex(order))
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # ---- layout: violin (top-left), bar (top-right), bar (bottom, span 2 cols) ----
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.1, 1.0], width_ratios=[1.05, 1])

    ax_violin_count = fig.add_subplot(gs[0, :])
    ax_violin_area = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

# ---- (1) Violin with avg. foci counts per nucleus (top row) ----
    sns.violinplot(
        data=df,
        x="sample_group",
        y="foci_count",
        inner=None,
        linewidth=2,
        hue="sample_group",
        linecolor="black",
        order=order,
        palette=palette_violin if (order is None or len(set(df["sample_group"])) <= 3) else None,
        fill=True,
        density_norm="count",
        bw_adjust=0.8,
        inner_kws={"color": "black", "box_width": 10, "whis_width": 2, "linestyle": '-'},
        ax=ax_violin_count
    )
    sns.boxplot(
        data=df, 
        x="sample_group", 
        y="foci_count",
        order=order, fliersize=0, width=0.035, linecolor="k", color="white",
        boxprops=dict(alpha=1), whiskerprops=dict(alpha=0.9),
        ax=ax_violin_count
    )

    ax_violin_count.set_title("Foci per Nucleus")
    ax_violin_count.set_xlabel("Sample Group")
    ax_violin_count.set_ylabel("Foci per nucleus")
    
# ---- (2) Violin with avg. foci counts per 1000um (bottom left) ----
    # Violin metric (density normalized by area):
    df = df[df["area_px"] > 0].copy()
    df["foci_per_1000px2"] = df["foci_count"] / (df["area_px"] / 1000.0)
    df["foci_per_1000px2_log2"] = np.log2(df["foci_per_1000px2"])
    sns.violinplot(
        data=df,
        x="sample_group",
        y="foci_per_1000px2_log2",
        inner=None,
        linewidth=2,
        hue="sample_group",
        linecolor="black",
        order=order,
        palette=palette_violin if (order is None or len(set(df["sample_group"])) <= 3) else None,
        fill=True,
        density_norm="count",
        bw_adjust=0.8,
        inner_kws={"color": "black", "box_width": 10, "whis_width": 2, "linestyle": '-'},
        ax=ax_violin_area
    )
    sns.boxplot(
        data=df,
        x="sample_group",
        y="foci_per_1000px2_log2",
        order=order, fliersize=0, width=0.035, linecolor="k", color="white",
        boxprops=dict(alpha=1), whiskerprops=dict(alpha=0.9),
        ax=ax_violin_area
    )

    ax_violin_area.set_title("Foci per 1000px2")
    ax_violin_area.set_xlabel("Sample Group")
    ax_violin_area.set_ylabel("Log2(Foci per 1000px2)")

# ---- (3) Stacked bar with foci bins (bottom right) ----
    bottom = np.zeros(len(pivot_df), dtype=float)
    for color, bin_label in zip(palette_bars, pivot_df.columns):
        vals = pivot_df[bin_label].values
        bars = ax_bar.bar(pivot_df.index, vals, bottom=bottom, color=color, label=str(bin_label))
        # Label segments (optional; hide tiny ones)
        for v, b, bt in zip(vals, bars, bottom):
            if v >= 0.05:
                ax_bar.text(b.get_x()+b.get_width()/2, bt+v/2, f"{v*100:.0f}%",
                            ha="center", va="center", color="white", fontsize=9, weight="bold")
        bottom += vals

    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Proportion")
    ax_bar.set_xlabel("")
    ax_bar.set_title("Composition of foci bins", pad=6)
    ax_bar.legend(title="Num Foci Bin", bbox_to_anchor=(1.02, 1), loc="upper left",
                borderaxespad=0., frameon=False)
    
    sns.despine(ax=ax_bar)

    out_path = Path(out_dir) / "segmentation_results.png"
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

@dataclass
class AquilaParams:
    input_dir: str
    output_dir: str
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
            plot_path = plot_results(combined, summary_dir, order=['v3KO', 'HPQ-', 'HPQ+'])
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
        btn_in = QtWidgets.QPushButton("Browse…")
        btn_in.clicked.connect(self._browse_input)
        in_row = QtWidgets.QHBoxLayout()
        in_row.addWidget(self.in_edit)
        in_row.addWidget(btn_in)
        dir_layout.addRow("Input directory:", in_row)

        self.out_edit = QtWidgets.QLineEdit()
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

        detect_layout.addWidget(QtWidgets.QLabel("Sigma A (DoG):"), 0, 0); detect_layout.addWidget(self.sigmaA, 0, 1)
        detect_layout.addWidget(QtWidgets.QLabel("Sigma B (DoG):"), 0, 2); detect_layout.addWidget(self.sigmaB, 0, 3)
        detect_layout.addWidget(QtWidgets.QLabel("Prominence:"), 0, 4); detect_layout.addWidget(self.prominence, 0, 5)

        main_layout.addWidget(detect_group)

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

        seg_layout.addWidget(QtWidgets.QLabel("Min area (px):"), 0, 0); seg_layout.addWidget(self.min_area, 0, 1)
        seg_layout.addWidget(QtWidgets.QLabel("Max area (px):"), 0, 2); seg_layout.addWidget(self.max_area, 0, 3)
        seg_layout.addWidget(QtWidgets.QLabel("DAPI foreground:"), 1, 0); seg_layout.addWidget(self.foreground, 1, 1)
        seg_layout.addWidget(QtWidgets.QLabel("Blur σ:"), 1, 2); seg_layout.addWidget(self.blur_sigma, 1, 3)
        seg_layout.addWidget(QtWidgets.QLabel("Seed radius:"), 1, 4); seg_layout.addWidget(self.seed_radius, 1, 5)

        main_layout.addWidget(seg_group)

        # -------------------------
        # Advanced Parameters Group
        # -------------------------
        adv_group = QtWidgets.QGroupBox("Advanced")
        adv_layout = QtWidgets.QGridLayout(adv_group)

        self.maxima_sigma = QtWidgets.QDoubleSpinBox(); self.maxima_sigma.setRange(0.0, 10.0); self.maxima_sigma.setValue(1.0)
        self.min_dist = QtWidgets.QSpinBox(); self.min_dist.setRange(1, 50); self.min_dist.setValue(1)
        self.exts = QtWidgets.QLineEdit(".tif,.tiff,.png,.jpg,.jpeg")
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

    def _params(self) -> AquilaParams:
        input_dir=self.in_edit.text().strip()
        output_dir=self.out_edit.text().strip()

        if not output_dir:
            output_dir = input_dir

        return AquilaParams(
            input_dir=input_dir,
            output_dir=output_dir,
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


    def _start(self):
        params = self._params()
        if not Path(params.input_dir).is_dir():
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid input directory")
            return

        self.log_box.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Running…")

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

    def _stop(self):
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
            self.status.setText("Stopping…")

    def _finish_run(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("Idle.")

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)