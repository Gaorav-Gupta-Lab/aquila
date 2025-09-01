import shutil
import warnings
from pathlib import Path
import re

warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
from PIL import Image
import re

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

def find_group(sample_name: str, group_names: list[str]) -> str | None:
    # Normalize sample name (lowercase for case-insensitivity)
    name = sample_name.lower()
    groups = [g.lower() for g in group_names]

    # 1. First try to match by exact token (split sample_name into words/numbers)
    # tokens = re.findall(r"[A-Za-z0-9\+\-_]+", name)
    # for token in tokens:
    #     if token in groups:
    #         return group_names[groups.index(token)]

    # 2. If no token match, try embedded search
    for g in groups:
        if g in name:
            return group_names[groups.index(g)]
    #     else:
    #         print("not here either")

    return "Default"

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

    group_name = find_group(sample_name, params.sample_groups)

    # group_name = re.search(r"\(([^)]+)\)", sample_name)
    # group_name = group_name.group(1) if group_name else None

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
    red_u8 = arr[..., 0]  # Red (PLA)
    green_u8 = arr[..., 1]  # Green (GFP)
    blue_u8 = arr[..., 2] # Blue (DAPI)

    # DoG
    dog = difference_of_gaussians_uint8_like(red_u8, params.sigmaA, params.sigmaB)
    dog_out = dest_dir / f"{sample_name}-PLA_halo_post_threshold.tif"
    dog_save = exposure.rescale_intensity(dog, out_range=(0, 65535)).astype(np.uint16)
    tiff.imwrite(str(dog_out), dog_save)

    # Nuclei
    labels = segment_nuclei_from_dapi(blue_u8,
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
    df_nuclei.insert(1, "sample_group", group_name)
    df_points.insert(1, "sample_group", group_name)

    df_nuclei.to_csv(results_csv, index=False)
    df_points.to_csv(points_csv, index=False)

    # Overlay
    overlay_png_path = dest_dir / f"{sample_name}_overlay.png"
    save_overlay_png(overlay_png_path, dog, labels, coords)

    log(f"[OK] Assigned to Group {group_name} -- (Nuclei: {len(df_nuclei)}, Foci: {len(df_points)})")
    # log(f"[OK] {file_name} -> {results_csv.name}  (nuclei={len(df_nuclei)}, foci={len(df_points)})")

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
    palette_bars   = ['#F2C447', '#F76218', '#FF1D68', '#B10065', '#740580']
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
        palette="Blues",
        # palette=palette_violin if (order is None or len(set(df["sample_group"])) <= 3) else None,
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
        palette="Reds",
        # palette=palette_violin if (order is None or len(set(df["sample_group"])) <= 3) else None,
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