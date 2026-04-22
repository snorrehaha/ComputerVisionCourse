import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these before running
# ═══════════════════════════════════════════════════════════════════════════════
base_dirs = [
    "Potato_Early_blight",
]
# Paths to test images  (add as many as you like)
IMAGE_PATHS = []

for d in base_dirs:
    files = sorted([
        f for f in os.listdir(d)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    IMAGE_PATHS.extend([os.path.join(d, f) for f in files[:5]])

print(IMAGE_PATHS)





# Resize all images to this before processing
IMAGE_SIZE = (256, 256)

# ── Segmentation ───────────────────────────────────────────────────────────────
SEG_SAT_THRESHOLD   = 50    # min saturation to be considered leaf
SEG_V_SHADOW_THRESH = 40    # max V to be considered shadow (excluded)
SEG_KERNEL_SIZE     = 5     # morphology kernel size
SEG_CLOSE_ITER      = 3     # morphological closing iterations
SEG_OPEN_ITER       = 2     # morphological opening iterations

# ── Disease spot detection ─────────────────────────────────────────────────────
SPOT_DIST_THRESH    = 0.3   # watershed distance threshold (lower → finer split)
SPOT_MIN_AREA       = 30    # minimum spot size in pixels
HEALTHY_THRESH      = 0.02  # spot coverage fraction below which leaf is Healthy

# ── Color bin quantization ─────────────────────────────────────────────────────
# Manual bin centroids — H is OpenCV scale (0–179), S is 0–255.
# To calibrate: run sample_hsv_at_click() on your images and average the clicks.
COLOR_BIN_CENTROIDS = {
    "green":  (50, 170),    # healthy leaf vivid green   — high S
    "yellow": (27, 195),    # late blight yellow halo    — high S warm hue
    "brown":  (13, 110),    # necrotic tissue            — lower S cool hue
}

# Weight on the saturation dimension relative to hue.
# Raise to make saturation differences matter more (healthy vs pale blight).
S_WEIGHT = 1.5

# Display colours for the flat color-bin image (RGB)
BIN_DISPLAY_COLORS_RGB = {
    "green":  ( 34, 139,  34),
    "yellow": (210, 180,  30),
    "brown":  (101,  55,   0),
}

# ── HSV disease colour ranges (OpenCV: H 0-179, S/V 0-255) ────────────────────
#
# Early Blight browns (#513e30, #9f8f80, #382514)
#   Sampled HSV: H 13-15, S 50-164, V 56-159
EARLY_BLIGHT_LOWER       = np.array([ 9,  20,  26], dtype=np.uint8)
EARLY_BLIGHT_UPPER       = np.array([19, 195, 189], dtype=np.uint8)

# Late Blight yellow (#c9c561, #ced16a, #b9bb56)
#   Sampled HSV: H 29-31, S 126-138, V 187-209
LATE_BLIGHT_YELLOW_LOWER = np.array([25,  96, 157], dtype=np.uint8)
LATE_BLIGHT_YELLOW_UPPER = np.array([35, 168, 239], dtype=np.uint8)

# Late Blight brown (#23120a, #876450, #9e8d73)
#   Sampled HSV: H 10-18, S 69-182, V 35-158
LATE_BLIGHT_BROWN_LOWER  = np.array([ 6,  39,  15], dtype=np.uint8)
LATE_BLIGHT_BROWN_UPPER  = np.array([22, 212, 188], dtype=np.uint8)

# Fraction of spot pixels that must be yellow to call Late Blight
LATE_BLIGHT_YELLOW_RATIO_THRESH = 0.10

# Overlay colours per disease class (RGB)
SPOT_COLOURS = {
    "Healthy":      (  0, 200,  50),
    "Early Blight": (255, 200,   0),
    "Late Blight":  (180,  40,   0),
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(path: str, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Load image, resize, and convert to HSV."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.resize(img, size)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ═══════════════════════════════════════════════════════════════════════════════
# LEAF SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _split_hsv(img: np.ndarray):
    return cv2.split(img)


def _threshold_saturation(s: np.ndarray, thresh: int = SEG_SAT_THRESHOLD):
    _, mask = cv2.threshold(s, thresh, 255, cv2.THRESH_BINARY)
    return mask


def _exclude_dark_shadows(mask: np.ndarray, v: np.ndarray,
                           v_thresh: int = SEG_V_SHADOW_THRESH):
    bright = (v > v_thresh).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, bright)


def _morph_cleanup(mask: np.ndarray, kernel: np.ndarray,
                   close_iter: int = SEG_CLOSE_ITER,
                   open_iter:  int = SEG_OPEN_ITER):
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iter)


def _largest_contour_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def segment_plant_hsv(img_hsv: np.ndarray) -> np.ndarray:
    """
    Full leaf segmentation pipeline:
      HSV → S-threshold → shadow exclusion → morphology → largest contour.

    Returns binary leaf mask (uint8, 0/255).
    """
    _, s, v = _split_hsv(img_hsv)
    kernel = np.ones((SEG_KERNEL_SIZE, SEG_KERNEL_SIZE), np.uint8)
    mask = _threshold_saturation(s)
    mask = _exclude_dark_shadows(mask, v)
    mask = _morph_cleanup(mask, kernel)
    return _largest_contour_mask(mask).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# BROWN SPOT DETECTION  (watershed)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_spot_submasks(img_hsv: np.ndarray, leaf_mask: np.ndarray):
    early       = cv2.inRange(img_hsv, EARLY_BLIGHT_LOWER,       EARLY_BLIGHT_UPPER)
    late_yellow = cv2.inRange(img_hsv, LATE_BLIGHT_YELLOW_LOWER,  LATE_BLIGHT_YELLOW_UPPER)
    late_brown  = cv2.inRange(img_hsv, LATE_BLIGHT_BROWN_LOWER,   LATE_BLIGHT_BROWN_UPPER)
    for m in (early, late_yellow, late_brown):
        cv2.bitwise_and(m, m, dst=m, mask=leaf_mask)
    return early, late_yellow, late_brown


def detect_brown_spots_watershed(img_hsv: np.ndarray, leaf_mask: np.ndarray,
                                  dist_thresh: float = SPOT_DIST_THRESH,
                                  min_spot_area: int  = SPOT_MIN_AREA):
    """
    Detect and label individual disease spots using watershed.

    Returns
    -------
    labeled_spots : int32 array — each unique value > 0 is one spot
    spot_mask     : binary mask of all retained spot pixels (uint8, 0/255)
    n_spots       : number of spots after size filtering
    submasks      : dict with keys 'early', 'late_yellow', 'late_brown'
    """
    early, late_yellow, late_brown = _build_spot_submasks(img_hsv, leaf_mask)
    submasks = {"early": early, "late_yellow": late_yellow, "late_brown": late_brown}

    spot_mask = cv2.bitwise_or(early, late_yellow)
    spot_mask = cv2.bitwise_or(spot_mask, late_brown)

    kernel = np.ones((3, 3), np.uint8)
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    sure_bg = cv2.dilate(spot_mask, kernel, iterations=3)
    dist    = cv2.distanceTransform(spot_mask, cv2.DIST_L2, 5)

    if dist.max() == 0:
        empty = np.zeros_like(spot_mask, dtype=np.int32)
        return empty, spot_mask, 0, submasks

    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg    = sure_fg.astype(np.uint8)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0

    bgr     = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    markers = cv2.watershed(bgr, markers)

    labeled_spots = np.where(markers > 1, markers, 0).astype(np.int32)

    for lbl in np.unique(labeled_spots):
        if lbl == 0:
            continue
        if np.sum(labeled_spots == lbl) < min_spot_area:
            labeled_spots[labeled_spots == lbl] = 0

    spot_mask_clean = (labeled_spots > 0).astype(np.uint8) * 255
    n_spots         = len(np.unique(labeled_spots)) - 1   # exclude background 0

    return labeled_spots, spot_mask_clean, n_spots, submasks


# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_leaf(leaf_mask: np.ndarray, spot_mask: np.ndarray,
                  img_hsv: np.ndarray, n_spots: int, submasks: dict,
                  healthy_thresh: float = HEALTHY_THRESH,
                  yellow_ratio_thresh: float = LATE_BLIGHT_YELLOW_RATIO_THRESH):
    """
    Classify leaf as Healthy / Early Blight / Late Blight.

    Decision logic
    --------------
    Healthy      — spot coverage < healthy_thresh  OR  n_spots == 0
    Late Blight  — olive/yellow pixels ≥ yellow_ratio_thresh of spot area
    Early Blight — brown spots present but no significant yellow component
    """
    leaf_area  = int(np.sum(leaf_mask  > 0))
    spot_area  = int(np.sum(spot_mask  > 0))
    spot_ratio = spot_area / leaf_area if leaf_area > 0 else 0.0

    yellow_area  = int(np.sum(submasks["late_yellow"] > 0))
    yellow_ratio = yellow_area / spot_area if spot_area > 0 else 0.0

    h, _, v = _split_hsv(img_hsv)
    if spot_area > 0:
        mean_spot_h = float(np.mean(h[spot_mask > 0].astype(float)))
        mean_spot_v = float(np.mean(v[spot_mask > 0].astype(float)))
    else:
        mean_spot_h = mean_spot_v = None

    info = {
        "leaf_area_px":  leaf_area,
        "spot_area_px":  spot_area,
        "spot_coverage": spot_ratio,
        "n_spots":       n_spots,
        "yellow_area_px":yellow_area,
        "yellow_ratio":  yellow_ratio,
        "mean_spot_h":   mean_spot_h,
        "mean_spot_v":   mean_spot_v,
    }

    if spot_ratio < healthy_thresh or n_spots == 0:
        return "Healthy", info
    if yellow_ratio >= yellow_ratio_thresh:
        return "Late Blight", info
    return "Early Blight", info


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR BIN QUANTIZATION  (lighting-independent)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Feature space: (sin H, cos H, S_norm)
#   • V is completely dropped    → luminosity independent
#   • H encoded as unit-circle   → no 0/179 wraparound discontinuity
#   • S retained with extra weight so healthy (vivid green) ≠ pale blight green
#
# Assignment: nearest-centroid in that 3D space (= K-means with fixed centers)


def _leaf_hs_features(img_hsv: np.ndarray, leaf_mask: np.ndarray):
    """
    Extract (sin H, cos H, S_norm) for every leaf pixel.

    Returns
    -------
    features : (N, 3) float32  — [sin_h, cos_h, s_norm]
    yx       : (N, 2) int32    — [row, col] of each pixel
    """
    ys, xs = np.where(leaf_mask > 0)
    if len(ys) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.int32)

    h = img_hsv[ys, xs, 0].astype(np.float32)
    s = img_hsv[ys, xs, 1].astype(np.float32)

    h_rad    = h / 179.0 * 2.0 * np.pi   # OpenCV H (0-179) → radians
    features = np.stack([np.sin(h_rad), np.cos(h_rad), s / 255.0], axis=1)
    return features.astype(np.float32), np.stack([ys, xs], axis=1).astype(np.int32)


def _centroids_to_features(centroids: dict):
    """
    Convert {name: (H_ocv, S)} centroid dict to the same (sin H, cos H, S_norm)
    feature space used by _leaf_hs_features.

    Returns
    -------
    names  : list of bin name strings (in matrix row order)
    matrix : (K, 3) float32 centroid feature vectors
    """
    names, vecs = [], []
    for name, (h, s) in centroids.items():
        h_rad = h / 179.0 * 2.0 * np.pi
        vecs.append([np.sin(h_rad), np.cos(h_rad), s / 255.0])
        names.append(name)
    return names, np.array(vecs, dtype=np.float32)


def quantize_leaf_colors(img_hsv: np.ndarray, leaf_mask: np.ndarray,
                          centroids: dict = None,
                          s_weight: float = S_WEIGHT):
    """
    Assign every leaf pixel to the nearest manually-defined color bin,
    using lighting-independent (sin H, cos H, S) feature space.

    V channel is completely ignored → output is luminosity independent.
    S is retained with extra weight so healthy vivid green is distinguished
    from pale/blighted tissue even when hue is similar.

    Parameters
    ----------
    img_hsv   : HSV image (uint8, H×W×3)
    leaf_mask : binary leaf mask (uint8, 0/255)
    centroids : dict {name: (H_ocv, S)} — defaults to COLOR_BIN_CENTROIDS
    s_weight  : multiplier on S dimension (>1 → saturation matters more)

    Returns
    -------
    label_map   : (H, W) int32; -1 = background, 0/1/… = bin index
    bin_names   : list of bin name strings in index order
    percentages : dict {bin_name: float} — % of leaf pixels, sums to 100
    """
    if centroids is None:
        centroids = COLOR_BIN_CENTROIDS

    features, yx = _leaf_hs_features(img_hsv, leaf_mask)
    label_map    = np.full(leaf_mask.shape, -1, dtype=np.int32)
    percentages  = {n: 0.0 for n in centroids}

    if len(features) == 0:
        return label_map, list(centroids.keys()), percentages

    # Apply saturation weight to both pixel features and centroids
    fw = features.copy();   fw[:, 2] *= s_weight
    bin_names, cmat = _centroids_to_features(centroids)
    cw = cmat.copy();       cw[:, 2] *= s_weight

    # Vectorised nearest-centroid: (N, K, 3) → distances (N, K) → argmin (N,)
    diffs       = fw[:, np.newaxis, :] - cw[np.newaxis, :, :]
    assignments = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)

    label_map[yx[:, 0], yx[:, 1]] = assignments

    total = len(assignments)
    for idx, name in enumerate(bin_names):
        count = int(np.sum(assignments == idx))
        percentages[name] = round(100.0 * count / total, 2) if total > 0 else 0.0

    return label_map, bin_names, percentages


def render_color_bins(shape: tuple, label_map: np.ndarray, bin_names: list,
                       display_colors: dict = None,
                       bg_color: tuple = (25, 25, 25)) -> np.ndarray:
    """
    Paint each leaf pixel with its flat bin color.
    All texture and lighting information is stripped — output is purely chromatic.
    """
    if display_colors is None:
        display_colors = BIN_DISPLAY_COLORS_RGB
    out = np.full((*shape, 3), bg_color, dtype=np.uint8)
    for idx, name in enumerate(bin_names):
        out[label_map == idx] = display_colors.get(name, (128, 128, 128))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CENTROID CALIBRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def sample_hsv_at_click(img_hsv: np.ndarray,
                         title: str = "Click representative pixels, then close"):
    """
    Interactive calibration tool.  Click pixels of a single class; the terminal
    prints each pixel's H and S values.  Average 10+ clicks per class and use
    those means as your COLOR_BIN_CENTROIDS values.

    Usage
    -----
        img = load_image("some_leaf.jpg")
        sample_hsv_at_click(img, "Click all the brown necrotic spots")
        sample_hsv_at_click(img, "Click healthy green tissue")
    """
    rgb     = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    samples = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= y < img_hsv.shape[0] and 0 <= x < img_hsv.shape[1]:
            h, s, v = img_hsv[y, x]
            samples.append((int(h), int(s)))
            print(f"  ({x:3d},{y:3d})  H={h:3d}  S={s:3d}  V={v:3d}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()

    if samples:
        mean_h = round(float(np.mean([s[0] for s in samples])), 1)
        mean_s = round(float(np.mean([s[1] for s in samples])), 1)
        print(f"\n  → Centroid for this class:  H={mean_h}  S={mean_s}")
        print(f'     Paste into COLOR_BIN_CENTROIDS:  "class_name": ({int(mean_h)}, {int(mean_s)})')

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSION & OVERLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))


def to_rgb(img_hsv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def overlay_spots(img_rgb: np.ndarray, spot_mask: np.ndarray,
                  labeled_spots: np.ndarray, label: str,
                  alpha: float = 0.45,
                  outline_colour: tuple = (255, 255, 255),
                  outline_thick: int = 1) -> np.ndarray:
    """Draw a transparent coloured overlay of detected spots on an RGB image."""
    fill_colour = SPOT_COLOURS.get(label, (255, 100, 0))
    overlay     = img_rgb.copy().astype(np.float32)
    fill        = np.array(fill_colour, dtype=np.float32)
    spot_bool   = spot_mask > 0
    overlay[spot_bool] = (1 - alpha) * overlay[spot_bool] + alpha * fill
    result = np.clip(overlay, 0, 255).astype(np.uint8)
    for lbl in np.unique(labeled_spots):
        if lbl == 0:
            continue
        binary = (labeled_spots == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, outline_colour, outline_thick)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  ①  — Full disease detection pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def show_full_pipeline(images_hsv: list, titles: list):
    """
    4-row grid per image:
        Row 0 — Original RGB
        Row 1 — Leaf mask (shadows excluded)
        Row 2 — Segmented leaf
        Row 3 — Disease spot overlay + classification label
    """
    n = len(images_hsv)
    fig, axes = plt.subplots(4, n, figsize=(5 * n, 16))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i, (img_hsv, title) in enumerate(zip(images_hsv, titles)):
        rgb          = to_rgb(img_hsv)
        leaf_mask    = segment_plant_hsv(img_hsv)
        labeled_spots, spot_mask, n_spots, submasks = detect_brown_spots_watershed(
            img_hsv, leaf_mask)
        label, info  = classify_leaf(leaf_mask, spot_mask, img_hsv, n_spots, submasks)
        seg_rgb      = apply_mask(rgb, leaf_mask)
        spot_overlay = overlay_spots(seg_rgb, spot_mask, labeled_spots, label)
        cov_pct      = info["spot_coverage"] * 100

        axes[0, i].imshow(rgb);                  axes[0, i].set_title(f"{title}\nOriginal RGB",              fontsize=9); axes[0, i].axis("off")
        axes[1, i].imshow(leaf_mask, cmap="gray");axes[1, i].set_title(f"{title}\nLeaf Mask",                fontsize=9); axes[1, i].axis("off")
        axes[2, i].imshow(seg_rgb);               axes[2, i].set_title(f"{title}\nSegmented Leaf",           fontsize=9); axes[2, i].axis("off")
        axes[3, i].imshow(spot_overlay)
        colour_rgb = tuple(c / 255 for c in SPOT_COLOURS.get(label, (200, 200, 200)))
        axes[3, i].legend(handles=[mpatches.Patch(color=colour_rgb, label=label)],
                          loc="lower right", fontsize=7)
        axes[3, i].set_title(
            f"{title}\n{label}  |  {n_spots} spots  |  {cov_pct:.1f}% coverage",
            fontsize=9, color="darkred" if "Blight" in label else "darkgreen")
        axes[3, i].axis("off")

    plt.suptitle("Disease Detection Pipeline", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  ②  — Color bin analysis
# ═══════════════════════════════════════════════════════════════════════════════

def show_color_bin_analysis(images_hsv: list, titles: list,
                             centroids: dict = None,
                             s_weight: float = S_WEIGHT):
    """
    3-panel row per image:
        Col 0 — Original RGB
        Col 1 — Flat color bins (lighting stripped)
        Col 2 — Percentage bar chart

    The bar chart is the main deliverable: a 3-number feature vector
    (green %, yellow %, brown %) that is luminosity-independent and
    directly comparable across images captured under different lighting.
    """
    if centroids is None:
        centroids = COLOR_BIN_CENTROIDS

    n = len(images_hsv)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (img_hsv, title) in enumerate(zip(images_hsv, titles)):
        leaf_mask = segment_plant_hsv(img_hsv)
        label_map, bin_names, pct = quantize_leaf_colors(
            img_hsv, leaf_mask, centroids=centroids, s_weight=s_weight)
        flat_rgb = render_color_bins(leaf_mask.shape, label_map, bin_names)
        orig_rgb = to_rgb(img_hsv)

        # Original
        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f"{title}\nOriginal", fontsize=9)
        axes[i, 0].axis("off")

        # Flat bins
        axes[i, 1].imshow(flat_rgb)
        axes[i, 1].set_title(f"{title}\nColor Bins (V stripped)", fontsize=9)
        patches = [mpatches.Patch(
            color=[c / 255 for c in BIN_DISPLAY_COLORS_RGB.get(name, (128, 128, 128))],
            label=name) for name in bin_names]
        axes[i, 1].legend(handles=patches, loc="lower right", fontsize=7)
        axes[i, 1].axis("off")

        # Bar chart
        ax = axes[i, 2]
        bar_colors = [tuple(c / 255 for c in BIN_DISPLAY_COLORS_RGB.get(name, (128, 128, 128)))
                      for name in bin_names]
        values = [pct[name] for name in bin_names]
        bars   = ax.bar(bin_names, values, color=bar_colors, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.set_ylabel("% of leaf pixels")
        ax.set_title(f"{title}\nColor Distribution", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle("Color Bin Analysis (Lighting Independent)", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  ③  — Combined view
# ═══════════════════════════════════════════════════════════════════════════════

def show_combined_analysis(images_hsv: list, titles: list,
                            centroids: dict = None,
                            s_weight: float = S_WEIGHT):
    """
    Full 4-row, 2-column-per-image grid:

        Row 0 — Original RGB  |  Leaf mask
        Row 1 — Segmented     |  Flat color bins
        Row 2 — Spot overlay  |  Color distribution bar chart
        Row 3 — Summary text panel (monospaced, spans both columns)
    """
    if centroids is None:
        centroids = COLOR_BIN_CENTROIDS

    n   = len(images_hsv)
    fig, axes = plt.subplots(4, n * 2, figsize=(7 * n, 18))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i, (img_hsv, title) in enumerate(zip(images_hsv, titles)):
        cp = i * 2      # pipeline column index
        cb = i * 2 + 1  # bin / chart column index

        rgb          = to_rgb(img_hsv)
        leaf_mask    = segment_plant_hsv(img_hsv)
        labeled_spots, spot_mask, n_spots, submasks = detect_brown_spots_watershed(
            img_hsv, leaf_mask)
        label, info  = classify_leaf(leaf_mask, spot_mask, img_hsv, n_spots, submasks)
        label_map, bin_names, pct = quantize_leaf_colors(
            img_hsv, leaf_mask, centroids=centroids, s_weight=s_weight)
        flat_rgb     = render_color_bins(leaf_mask.shape, label_map, bin_names)
        seg_rgb      = apply_mask(rgb, leaf_mask)
        spot_overlay = overlay_spots(seg_rgb, spot_mask, labeled_spots, label)
        cov_pct      = info["spot_coverage"] * 100

        # Row 0
        axes[0, cp].imshow(rgb);                   axes[0, cp].set_title(f"{title}\nOriginal RGB", fontsize=9);  axes[0, cp].axis("off")
        axes[0, cb].imshow(leaf_mask, cmap="gray"); axes[0, cb].set_title(f"{title}\nLeaf Mask",   fontsize=9);  axes[0, cb].axis("off")

        # Row 1
        axes[1, cp].imshow(seg_rgb);  axes[1, cp].set_title("Segmented Leaf",   fontsize=9); axes[1, cp].axis("off")
        axes[1, cb].imshow(flat_rgb); axes[1, cb].set_title("Flat Color Bins",  fontsize=9)
        patches = [mpatches.Patch(
            color=[c / 255 for c in BIN_DISPLAY_COLORS_RGB.get(name, (128, 128, 128))],
            label=name) for name in bin_names]
        axes[1, cb].legend(handles=patches, loc="lower right", fontsize=7)
        axes[1, cb].axis("off")

        # Row 2
        colour_rgb = tuple(c / 255 for c in SPOT_COLOURS.get(label, (200, 200, 200)))
        axes[2, cp].imshow(spot_overlay)
        axes[2, cp].legend(handles=[mpatches.Patch(color=colour_rgb, label=label)],
                            loc="lower right", fontsize=7)
        axes[2, cp].set_title(
            f"{label}  |  {n_spots} spots  |  {cov_pct:.1f}% coverage",
            fontsize=9, color="darkred" if "Blight" in label else "darkgreen")
        axes[2, cp].axis("off")

        ax = axes[2, cb]
        bar_colors = [tuple(c / 255 for c in BIN_DISPLAY_COLORS_RGB.get(name, (128, 128, 128)))
                      for name in bin_names]
        values = [pct[name] for name in bin_names]
        bars   = ax.bar(bin_names, values, color=bar_colors, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.set_ylabel("% of leaf pixels")
        ax.set_title("Color Distribution (lighting independent)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Row 3 — summary text
        for col in (cp, cb):
            axes[3, col].axis("off")
        summary = (
            f"Diagnosis    : {label}\n"
            f"Spots found  : {n_spots}\n"
            f"Coverage     : {cov_pct:.2f}%\n"
            f"Leaf area    : {info['leaf_area_px']:,} px\n"
            f"Spot area    : {info['spot_area_px']:,} px\n"
            f"Yellow ratio : {info['yellow_ratio']:.3f}\n"
            f"\nColor bins:\n"
            + "\n".join(f"  {name:8s}  {pct[name]:6.1f}%" for name in bin_names)
        )
        axes[3, cp].text(
            0.05, 0.95, summary,
            transform=axes[3, cp].transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

    plt.suptitle("Leaf Disease Analysis — Combined View", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run(paths: list):
    if not paths:
        print("No images provided.")
        print("  Option 1 — add paths to IMAGE_PATHS at the top of this file.")
        print("  Option 2 — pass paths as CLI args:  python leaf_analysis.py img1.jpg img2.jpg")
        return

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print("ERROR — file(s) not found:")
        for p in missing:
            print(f"  {p}")
        return

    titles = [os.path.basename(p) for p in paths]
    images = [load_image(p) for p in paths]

    print(f"Loaded {len(images)} image(s).\n")

    # ── View 1: disease detection pipeline ──────────────────────────────────
    #show_full_pipeline(images, titles)

    # ── View 2: color bin analysis ──────────────────────────────────────────
    #show_color_bin_analysis(images, titles)

    # ── View 3: combined (both + summary panel) ─────────────────────────────
    show_combined_analysis(images, titles)

    # ── Terminal output ──────────────────────────────────────────────────────
    print("═" * 60)
    print("NUMERIC RESULTS")
    print("═" * 60)
    for img_hsv, title in zip(images, titles):
        leaf_mask = segment_plant_hsv(img_hsv)
        labeled_spots, spot_mask, n_spots, submasks = detect_brown_spots_watershed(
            img_hsv, leaf_mask)
        label, info = classify_leaf(leaf_mask, spot_mask, img_hsv, n_spots, submasks)
        _, bin_names, pct = quantize_leaf_colors(img_hsv, leaf_mask)

        print(f"\n{title}")
        print(f"  Diagnosis    : {label}")
        print(f"  Spots        : {n_spots}")
        print(f"  Coverage     : {info['spot_coverage'] * 100:.2f}%")
        print(f"  Yellow ratio : {info['yellow_ratio']:.3f}")
        print("  Color bins   :")
        for name in bin_names:
            print(f"    {name:8s}  {pct[name]:6.1f}%")
    print()


if __name__ == "__main__":
    cli_paths = sys.argv[1:]
    run(cli_paths if cli_paths else IMAGE_PATHS)