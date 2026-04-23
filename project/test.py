import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np

# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────

def LoadImage(path, size=(256, 256)):
    """Load image, resize, and convert to HSV."""
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


# ─────────────────────────────────────────────
# LEAF SEGMENTATION  (with shadow exclusion)
# ─────────────────────────────────────────────

def split_hsv(img):
    """Split HSV image into H, S, V channels."""
    return cv2.split(img)


def threshold_saturation(s_channel, thresh=50):
    """Binary threshold on S channel — main cue for separating leaf from grey bg."""
    _, mask = cv2.threshold(s_channel, thresh, 255, cv2.THRESH_BINARY)
    return mask


def exclude_dark_shadows(mask, v_channel, v_thresh=40):
    """
    Remove near-black shadow pixels from the leaf mask.
    Pixels with V < v_thresh are considered shadow and are excluded.
    """
    bright_enough = (v_channel > v_thresh).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, bright_enough)


def morph_cleanup(mask, kernel, close_iter=3, open_iter=2):
    """Apply morphological closing then opening to clean noise."""
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return opened


def largest_contour_mask(mask):
    """Keep only the largest connected component."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return clean_mask


def segment_plant_hsv(img_hsv, sat_threshold=50, v_shadow_thresh=40,
                      kernel_size=5, close_iter=3, open_iter=2):
    """
    Full leaf segmentation pipeline:
      HSV → S-threshold → shadow exclusion → morphology → largest contour

    Args:
        img_hsv:          HSV image (uint8)
        sat_threshold:    Min saturation to be considered leaf (default 50)
        v_shadow_thresh:  Max V to be considered shadow and excluded (default 40)
        kernel_size:      Morphology kernel size
        close_iter:       Closing iterations
        open_iter:        Opening iterations

    Returns:
        Binary leaf mask (uint8, 0/255)
    """
    _, s, v = split_hsv(img_hsv)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = threshold_saturation(s, sat_threshold)
    mask = exclude_dark_shadows(mask, v, v_shadow_thresh)   # ← NEW: drop shadows
    mask = morph_cleanup(mask, kernel, close_iter, open_iter)
    mask = largest_contour_mask(mask)

    return mask.astype(np.uint8)


# ─────────────────────────────────────────────
# BROWN SPOT DETECTION  (watershed)
# ─────────────────────────────────────────────

# ── HSV colour ranges derived from sampled hex values ──────────────────────────
#
# All ranges use OpenCV conventions: H 0-179, S 0-255, V 0-255
# Margins of ±4 H / ±30 S / ±30 V added around the sampled cluster.
#
#  Early Blight browns (#513e30, #9f8f80, #382514)
#    Sampled HSV: H 13-15, S  50-164, V  56-159
EARLY_BLIGHT_LOWER = np.array([ 9,  20,  26], dtype=np.uint8)
EARLY_BLIGHT_UPPER = np.array([19, 195, 189], dtype=np.uint8)

#  Late Blight yellow (#c9c561, #ced16a, #b9bb56)
#    Sampled HSV: H 29-31, S 126-138, V 187-209
LATE_BLIGHT_YELLOW_LOWER = np.array([25,  96, 157], dtype=np.uint8)
LATE_BLIGHT_YELLOW_UPPER = np.array([35, 168, 239], dtype=np.uint8)

#  Late Blight brown (#23120a, #876450, #9e8d73)
#    Sampled HSV: H 10-18, S  69-182, V  35-158
LATE_BLIGHT_BROWN_LOWER = np.array([ 6,  39,  15], dtype=np.uint8)
LATE_BLIGHT_BROWN_UPPER = np.array([22, 212, 188], dtype=np.uint8)

# Ratio of yellow pixels inside the combined spot mask above which we call Late Blight
LATE_BLIGHT_YELLOW_RATIO_THRESH = 0.10


def _build_spot_submasks(img_hsv, leaf_mask):
    """
    Return three binary sub-masks (all restricted to leaf area):
        early_mask        — early-blight brown tones
        late_yellow_mask  — late-blight olive/yellow tones
        late_brown_mask   — late-blight dark brown tones

    Having separate masks lets classify_leaf() check the yellow
    component independently, which is the clearest Late Blight cue.
    """
    early_mask       = cv2.inRange(img_hsv, EARLY_BLIGHT_LOWER,      EARLY_BLIGHT_UPPER)
    late_yellow_mask = cv2.inRange(img_hsv, LATE_BLIGHT_YELLOW_LOWER, LATE_BLIGHT_YELLOW_UPPER)
    late_brown_mask  = cv2.inRange(img_hsv, LATE_BLIGHT_BROWN_LOWER,  LATE_BLIGHT_BROWN_UPPER)

    for m in (early_mask, late_yellow_mask, late_brown_mask):
        cv2.bitwise_and(m, m, dst=m, mask=leaf_mask)

    return early_mask, late_yellow_mask, late_brown_mask


def detect_brown_spots_watershed(img_hsv, leaf_mask,
                                  dist_thresh=0.3,
                                  min_spot_area=30):
    """
    Detect and label individual disease spots using watershed.

    Pipeline:
      1. Three named HSV masks (early brown, late yellow, late brown)
         → combined spot mask restricted to leaf area
      2. Morphological cleanup  (open → close)
      3. Distance transform → sure-foreground markers
      4. Watershed to separate touching spots
      5. Remove blobs smaller than min_spot_area

    Args:
        img_hsv:        HSV image (uint8)
        leaf_mask:      Binary leaf mask (uint8, 0/255)
        dist_thresh:    Fraction of max distance used as sure-foreground
                        threshold. Lower → more markers → finer split.
        min_spot_area:  Minimum pixel area to keep a labelled spot.

    Returns:
        labeled_spots:   int32 array — each unique value > 0 is one spot
        spot_mask:       Binary mask of all retained spot pixels (uint8, 0/255)
        n_spots:         Number of detected spots after size filtering
        submasks:        dict with keys 'early', 'late_yellow', 'late_brown'
                         (raw per-class masks before morphology, for classification)
    """
    # 1. Per-class masks
    early_mask, late_yellow_mask, late_brown_mask = _build_spot_submasks(
        img_hsv, leaf_mask
    )
    submasks = {
        "early":       early_mask,
        "late_yellow": late_yellow_mask,
        "late_brown":  late_brown_mask,
    }

    # Union of all disease pixels
    spot_mask = cv2.bitwise_or(early_mask, late_yellow_mask)
    spot_mask = cv2.bitwise_or(spot_mask,  late_brown_mask)

    # 2. Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Watershed markers via distance transform
    sure_bg = cv2.dilate(spot_mask, kernel, iterations=3)

    dist = cv2.distanceTransform(spot_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        empty = np.zeros_like(spot_mask, dtype=np.int32)
        return empty, spot_mask, 0, submasks

    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1       # background → 1, foreground → 2+
    markers[unknown == 255] = 0 # unknown → 0 (watershed will decide)

    # 4. Watershed (requires BGR)
    bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    markers = cv2.watershed(bgr, markers)

    # markers == -1 → boundary; markers == 1 → background
    labeled_spots = np.where(markers > 1, markers, 0).astype(np.int32)

    # 5. Remove tiny blobs
    for lbl in np.unique(labeled_spots):
        if lbl == 0:
            continue
        if np.sum(labeled_spots == lbl) < min_spot_area:
            labeled_spots[labeled_spots == lbl] = 0

    spot_mask_clean = (labeled_spots > 0).astype(np.uint8) * 255
    n_spots = len(np.unique(labeled_spots)) - 1   # exclude 0

    return labeled_spots, spot_mask_clean, n_spots, submasks


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

def classify_leaf(leaf_mask, spot_mask, img_hsv, n_spots, submasks,
                  healthy_thresh=0.02,
                  yellow_ratio_thresh=LATE_BLIGHT_YELLOW_RATIO_THRESH):
    """
    Classify a leaf into one of three categories:

    Healthy      — spot coverage < healthy_thresh  OR  n_spots == 0
    Late Blight  — olive/yellow pixels make up > yellow_ratio_thresh of
                   the total spot area (the yellow component is the
                   clearest Late Blight marker)
    Early Blight — brown spots present but no significant yellow component

    Decision is based on the raw sub-masks from _build_spot_submasks so
    that morphological cleanup doesn't affect the colour vote.

    Args:
        leaf_mask:           Binary leaf mask (uint8)
        spot_mask:           Combined binary spot mask (uint8, 0/255)
        img_hsv:             HSV image
        n_spots:             Number of watershed-labelled spots
        submasks:            Dict from detect_brown_spots_watershed:
                               'early', 'late_yellow', 'late_brown'
        healthy_thresh:      Coverage fraction below which leaf is Healthy
        yellow_ratio_thresh: Fraction of spot pixels that must be yellow
                             to call Late Blight (default 0.10)

    Returns:
        label (str), info dict
    """
    leaf_area = int(np.sum(leaf_mask > 0))
    spot_area = int(np.sum(spot_mask > 0))
    spot_ratio = spot_area / leaf_area if leaf_area > 0 else 0.0

    yellow_area = int(np.sum(submasks["late_yellow"] > 0))
    yellow_ratio = yellow_area / spot_area if spot_area > 0 else 0.0

    h, _, v = split_hsv(img_hsv)
    if spot_area > 0:
        mean_spot_h = float(np.mean(h[spot_mask > 0].astype(float)))
        mean_spot_v = float(np.mean(v[spot_mask > 0].astype(float)))
    else:
        mean_spot_h = mean_spot_v = None

    info = {
        "leaf_area_px":   leaf_area,
        "spot_area_px":   spot_area,
        "spot_coverage":  spot_ratio,
        "n_spots":        n_spots,
        "yellow_area_px": yellow_area,
        "yellow_ratio":   yellow_ratio,
        "mean_spot_h":    mean_spot_h,
        "mean_spot_v":    mean_spot_v,
    }

    if spot_ratio < healthy_thresh or n_spots == 0:
        return "Healthy", info

    # Late Blight has a visible olive/yellow lesion ring — Early Blight does not
    if yellow_ratio >= yellow_ratio_thresh:
        label = "Late Blight"
    else:
        label = "Early Blight"

    return label, info


# ─────────────────────────────────────────────
# OVERLAY
# ─────────────────────────────────────────────

# Overlay colour per class (RGB)
SPOT_COLOURS = {
    "Healthy":      (0,   200,  50),   # green  — shouldn't appear, just in case
    "Early Blight": (255, 200,   0),   # yellow-orange
    "Late Blight":  (180,  40,   0),   # dark red-brown
}


def overlay_spots(img_rgb, spot_mask, labeled_spots, label,
                  alpha=0.45, outline_colour=(255, 255, 255), outline_thick=1):
    """
    Draw a transparent coloured overlay of brown spots on an RGB image.

    Each detected spot region is filled with a semi-transparent colour
    (keyed to the disease label) and outlined for clarity.

    Args:
        img_rgb:        RGB image (uint8)
        spot_mask:      Binary spot mask (uint8, 0/255)
        labeled_spots:  Watershed-labelled spots (int32)
        label:          Classification label string
        alpha:          Spot fill opacity (0 = invisible, 1 = opaque)
        outline_colour: RGB colour for spot outlines
        outline_thick:  Contour thickness in pixels

    Returns:
        result (uint8 RGB image with overlay applied)
    """
    fill_colour = SPOT_COLOURS.get(label, (255, 100, 0))

    overlay = img_rgb.copy().astype(np.float32)
    fill = np.array(fill_colour, dtype=np.float32)

    # Fill spot pixels
    spot_bool = spot_mask > 0
    overlay[spot_bool] = (
        (1 - alpha) * overlay[spot_bool] + alpha * fill
    )

    result = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw per-spot contours
    for spot_label in np.unique(labeled_spots):
        if spot_label == 0:
            continue
        binary = (labeled_spots == spot_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, outline_colour, outline_thick)

    return result


# ─────────────────────────────────────────────
# CONVERSION HELPERS
# ─────────────────────────────────────────────

def apply_mask(img, mask):
    """Apply binary mask to image."""
    return cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))


def to_rgb(img_hsv):
    """Convert HSV image to RGB for display."""
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def show_full_pipeline(images_hsv, titles):
    """
    Runs the full pipeline on each image and displays a 4-row grid:

    Row 1 — Original RGB
    Row 2 — Leaf mask (with shadows excluded)
    Row 3 — Segmented leaf (masked)
    Row 4 — Brown-spot overlay + classification label
    """
    n = len(images_hsv)
    fig, axes = plt.subplots(4, n, figsize=(5 * n, 16))

    for i, (img_hsv, title) in enumerate(zip(images_hsv, titles)):

        rgb = to_rgb(img_hsv)

        # --- Leaf segmentation ---
        leaf_mask = segment_plant_hsv(img_hsv)

        # --- Brown spot detection ---
        labeled_spots, spot_mask, n_spots, submasks = detect_brown_spots_watershed(
            img_hsv, leaf_mask
        )

        # --- Classification ---
        label, info = classify_leaf(leaf_mask, spot_mask, img_hsv, n_spots, submasks)

        # --- Overlay ---
        segmented_rgb = apply_mask(rgb, leaf_mask)
        spot_overlay  = overlay_spots(segmented_rgb, spot_mask, labeled_spots, label)

        cov_pct = info["spot_coverage"] * 100

        # Row 0: Original
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"{title}\nOriginal", fontsize=9)
        axes[0, i].axis("off")

        # Row 1: Leaf mask
        axes[1, i].imshow(leaf_mask, cmap="gray")
        axes[1, i].set_title(f"{title}\nLeaf Mask (shadows excluded)", fontsize=9)
        axes[1, i].axis("off")

        # Row 2: Segmented leaf
        axes[2, i].imshow(segmented_rgb)
        axes[2, i].set_title(f"{title}\nSegmented Leaf", fontsize=9)
        axes[2, i].axis("off")

        # Row 3: Spot overlay + label
        axes[3, i].imshow(spot_overlay)
        colour_rgb = tuple(c / 255 for c in SPOT_COLOURS.get(label, (200, 200, 200)))
        patch = mpatches.Patch(color=colour_rgb, label=label)
        axes[3, i].legend(handles=[patch], loc="lower right", fontsize=7)
        axes[3, i].set_title(
            f"{title}\n{label}  |  {n_spots} spots  |  {cov_pct:.1f}% coverage",
            fontsize=9,
            color="darkred" if "Blight" in label else "darkgreen"
        )
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.show()


def show_segmentation_grid(images, masks, seg_masks, titles):
    """
    Legacy display helper (original 3-row grid).
    Kept for backwards compatibility.
    """
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))

    for i in range(n):
        axes[0, i].imshow(to_rgb(images[i]))
        axes[0, i].set_title(f"{titles[i]} - Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(masks[i], cmap="gray")
        axes[1, i].set_title(f"{titles[i]} - Mask")
        axes[1, i].axis("off")

        rgb  = to_rgb(images[i])
        mask = seg_masks[i].astype(np.uint8)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
        segmented = cv2.bitwise_and(rgb, rgb, mask=mask)

        axes[2, i].imshow(segmented)
        axes[2, i].set_title(f"{titles[i]} - Segmented")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# HISTOGRAM HELPERS  (unchanged)
# ─────────────────────────────────────────────

def plot_hsv_histograms(segmented_images, titles):
    fig, axes = plt.subplots(3, len(segmented_images), figsize=(15, 10))
    for i, img in enumerate(segmented_images):
        h, s, v = cv2.split(img)
        axes[0, i].hist(h.ravel(), bins=50, color='orange')
        axes[0, i].set_title(f"{titles[i]} - Hue");  axes[0, i].set_xlim([0, 180])
        axes[1, i].hist(s.ravel(), bins=50, color='green')
        axes[1, i].set_title(f"{titles[i]} - Saturation"); axes[1, i].set_xlim([0, 255])
        axes[2, i].hist(v.ravel(), bins=50, color='gray')
        axes[2, i].set_title(f"{titles[i]} - Value"); axes[2, i].set_xlim([0, 255])
    plt.tight_layout(); plt.show()


def plot_rgb_histograms(segmented_images, titles):
    fig, axes = plt.subplots(3, len(segmented_images), figsize=(15, 10))
    for i, img in enumerate(segmented_images):
        rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        r, g, b = cv2.split(rgb)
        axes[0, i].hist(r.ravel(), bins=50, color='red')
        axes[0, i].set_title(f"{titles[i]} - Red"); axes[0, i].set_xlim([0, 255])
        axes[1, i].hist(g.ravel(), bins=50, color='green')
        axes[1, i].set_title(f"{titles[i]} - Green"); axes[1, i].set_xlim([0, 255])
        axes[2, i].hist(b.ravel(), bins=50, color='blue')
        axes[2, i].set_title(f"{titles[i]} - Blue"); axes[2, i].set_xlim([0, 255])
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────
# QUICK TEST  (run as script)
# ─────────────────────────────────────────────


if __name__ == "__main__":

    img_healthy = LoadImage('Potato-Plants.jpg')
    img_early = LoadImage('Potato_Early_blight/000_early_blight.JPG')
    img_late = LoadImage('Potato_Late_blight/000_late_blight.JPG')
    images = [img_healthy, img_early, img_late]

    titles = ["healthy", "early", "late"]



    show_full_pipeline(images, titles)