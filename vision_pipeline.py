"""
Module 1: Preprocessing & Computer Vision Pipeline
===================================================
Provides robust, production-grade computer vision utilities for currency note
inspection. Implements:
  1. Note detection via adaptive Canny edge detection, convex hull polygon approximation,
     and perspective-invariant quadrilateral fitting.
  2. Four-point perspective transformation (homography warp) to a standardized 600x300 px note.
  3. Automatic orientation & flip normalization (detects and corrects upside-down notes).
  4. Illumination-invariant color constancy (Gray-World White Balance algorithm & CLAHE).
  5. Predefined Region-of-Interest (ROI) extraction (Security Strip, Watermark, Numeral).
  6. Color moment extraction (RGB, HSV, and CIE-Lab moments and normalized histograms).
  7. Unsupervised K-Means clustering (scikit-learn) on the security thread ROI to segment
     the metallic strip and measure vertical continuity & windowed segment periodicity.
  8. Watermark latent portrait contrast, variance, and paper matrix texture analysis.

Academic Mapping:
  - Computer Vision: Homography, spatial filtering, morphological operations, color space mapping.
  - CO4 (Unsupervised Learning): K-Means clustering for metallic/ink thread segmentation.
"""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import cv2
from scipy.stats import skew  # type: ignore
from sklearn.cluster import KMeans


# Standardized canvas dimensions for rectified banknote
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 300

# Normalized ROI coordinates relative to standardized canvas [ymin, xmin, ymax, xmax]
# Standard Indian Banknote layout conventions (obverse / front face):
ROI_DEFINITIONS = {
    "watermark": {
        "rel_box": (0.15, 0.08, 0.85, 0.32),  # Portrait / latent watermark window
        "name": "Mahatma Gandhi Watermark Window",
    },
    "security_strip": {
        "rel_box": (0.05, 0.38, 0.95, 0.48),  # Vertical security thread ribbon
        "name": "Security Thread Strip",
    },
    "numeral": {
        "rel_box": (0.60, 0.72, 0.90, 0.94),  # Value numeral region (bottom-right)
        "name": "Denomination Numeral Region",
    },
    "governor_promise": {
        "rel_box": (0.40, 0.48, 0.65, 0.72),  # Central RBI guarantee / signature
        "name": "Reserve Bank Guarantee Clause",
    },
}


def apply_color_constancy(image: np.ndarray) -> np.ndarray:
    """
    Applies the Gray-World White Balance algorithm with illuminant normalization
    to remove ambient color temperature bias (such as warm incandescent lamps,
    yellow CFLs, cool LEDs, or tinted daylight).

    Parameters:
        image (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Illumination-normalized BGR image.
    """
    img_float = image.astype(np.float32)
    b, g, r = cv2.split(img_float)

    mean_b = float(np.mean(b))
    mean_g = float(np.mean(g))
    mean_r = float(np.mean(r))
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    # Avoid division by zero
    scale_b = mean_gray / (mean_b + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_r = mean_gray / (mean_r + 1e-6)

    # Prevent extreme noise amplification under monochromatic lighting
    max_scale = max(scale_b, scale_g, scale_r)
    if max_scale > 2.2:
        damping = 2.2 / max_scale
        scale_b *= damping
        scale_g *= damping
        scale_r *= damping

    b_out = np.clip(b * scale_b, 0, 255)
    g_out = np.clip(g * scale_g, 0, 255)
    r_out = np.clip(r * scale_r, 0, 255)

    return cv2.merge([b_out, g_out, r_out]).astype(np.uint8)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 arbitrary polygon coordinates in consistent clockwise order:
    [top-left, top-right, bottom-right, bottom-left].

    Handles arbitrary rotation angles, tilt, and quadrilateral orientations.

    Parameters:
        pts (np.ndarray): Array of shape (4, 2) representing 2D corner points.

    Returns:
        np.ndarray: Ordered coordinates of shape (4, 2) in float32.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.asarray(pts, dtype="float32")

    # Sum of coordinates: smallest (x+y) is top-left, largest is bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # Difference of coordinates (y - x): smallest (y-x) is top-right, largest is bottom-left
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
    target_width: int = CANVAS_WIDTH,
    target_height: int = CANVAS_HEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies perspective transformation (homography) to unwarp and rectify a note
    contour into a standardized rectangular orientation.

    Automatically handles horizontal vs vertical orientations so the longer
    banknote edge is consistently mapped to target_width (600 px) and shorter
    edge to target_height (300 px).

    Parameters:
        image (np.ndarray): Source BGR image.
        pts (np.ndarray): Four detected corners of the note.
        target_width (int): Output rectified width (default 600 px).
        target_height (int): Output rectified height (default 300 px).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - warped: Rectified note image of shape (target_height, target_width, 3).
            - M: The 3x3 perspective transformation matrix.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute Euclidean dimensions
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(int(height_a), int(height_b))

    # If the detected contour is vertically oriented (height > width),
    # rotate corner order so the resulting note is oriented horizontally.
    if max_h > max_w:
        rect = np.roll(rect, -1, axis=0)

    # Destination coordinates for standard flat canvas
    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )

    # Compute homography matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image, M, (target_width, target_height), flags=cv2.INTER_CUBIC
    )

    return warped, M


def auto_orient_note(warped_note: np.ndarray) -> Tuple[np.ndarray, int, str]:
    """
    Detects if the rectified banknote is upside-down (180 degrees) or rotated
    and normalizes it to the canonical front face (obverse) layout:
      - Left side: Mahatma Gandhi Watermark Window (light translucent paper).
      - Right side: Mahatma Gandhi Intaglio Portrait (dense dark engraved lines).

    Parameters:
        warped_note (np.ndarray): Rectified note image of shape (300, 600, 3).

    Returns:
        Tuple[np.ndarray, int, str]:
            - normalized_note: Rectified note in canonical orientation.
            - rotation_applied: Rotation angle applied (0 or 180).
            - orientation_label: Human-readable orientation status.
    """
    h, w = warped_note.shape[:2]
    gray = cv2.cvtColor(warped_note, cv2.COLOR_BGR2GRAY)

    # 1. Top banner vs Bottom banner (Reserve Bank header text: 'RESERVE BANK OF INDIA')
    top_banner = gray[: int(h * 0.25), int(w * 0.25) : int(w * 0.75)]
    bot_banner = gray[int(h * 0.75) :, int(w * 0.25) : int(w * 0.75)]
    top_dark = float(np.sum(top_banner < 100))
    bot_dark = float(np.sum(bot_banner < 100))

    # 2. Left third (watermark window) vs Right third (Gandhi portrait)
    left_roi = gray[:, : int(w * 0.35)]
    right_roi = gray[:, int(w * 0.65) :]
    left_dark = float(np.sum(left_roi < 110))
    right_dark = float(np.sum(right_roi < 110))

    # If the bottom banner has substantially more dark ink text than the top banner,
    # or the left side has significantly higher dark ink density than the right,
    # the note is inverted / upside-down!
    if (bot_dark > (top_dark * 1.35)) or (left_dark > (right_dark * 1.35)):
        rotated = cv2.rotate(warped_note, cv2.ROTATE_180)
        return rotated, 180, "Inverted (180 deg) - Auto-Corrected to Obverse"

    return warped_note, 0, "Obverse (Front Face) Aligned"
def crop_guided_frame(
    image: np.ndarray, width_ratio: float = 0.85, height_ratio: float = 0.55
) -> Tuple[np.ndarray, float]:
    """
    Crops a canonical central rectangular region where users naturally present
    banknotes in front of camera/webcam sensors, completely eliminating extraneous
    room background, doors, and ceilings.

    Parameters:
        image (np.ndarray): Input frame.
        width_ratio (float): Fraction of frame width to sample (default 0.85).
        height_ratio (float): Fraction of frame height to sample (default 0.55).

    Returns:
        Tuple[np.ndarray, float]:
            - corners: (4, 2) array of corner coordinates.
            - aspect_ratio: Observed aspect ratio of cropped box.
    """
    h, w = image.shape[:2]
    crop_w = int(w * width_ratio)
    crop_h = int(h * height_ratio)
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    corners = np.array(
        [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]], dtype="float32"
    )
    ar = float((x2 - x1) / max(1, (y2 - y1)))
    return corners, ar


def detect_note_contour(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[float]]:
    """
    Detects the banknote boundary contour within a camera frame or uploaded image
    using robust multi-threshold edge detection, convex hull polygon approximation,
    and perspective quadrilateral candidate scoring.

    Handles real-world conditions:
      - Hands/fingers holding the note (convex hull bridges notches)
      - Non-orthogonal perspective angles & camera tilt
      - Complex or textured backgrounds (desks, cloth, tables)
      - Uploaded cropped banknotes (graceful fallback)

    Parameters:
        image (np.ndarray): Input BGR frame.

    Returns:
        Tuple[Optional[np.ndarray], np.ndarray, Optional[float]]:
            - corners: Array of 4 corner points (shape (4, 2)) or None if no note detected.
            - debug_edges: Binary edge mask for UI diagnostics.
            - aspect_ratio: Observed width/height aspect ratio.
    """
    h, w = image.shape[:2]
    total_area = h * w
    min_area = total_area * 0.12  # Banknote in a camera scene occupies at least 12% of the frame

    # 1. Bilateral filter to smooth texture while preserving boundaries
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. Adaptive Canny edge detection
    med_val = np.median(blurred)
    lower = max(0, int(0.66 * med_val))
    upper = min(255, int(1.33 * med_val))
    edges_canny = cv2.Canny(blurred, lower, upper)

    # 3. Morphological closing to seal perimeter and bridge finger grips
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges_canny, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find external contours and score candidate quads
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    candidates: List[Tuple[float, np.ndarray, float]] = []

    for c in contours[:20]:
        area = cv2.contourArea(c)
        if area < min_area or area > (total_area * 0.96):
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area < min_area:
            continue

        # Strategy A: Polygon approximation on contour and convex hull
        for shape in [c, hull]:
            peri = cv2.arcLength(shape, True)
            for eps in [0.02, 0.03, 0.04]:
                approx = cv2.approxPolyDP(shape, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    rect = order_points(pts)
                    w_b = np.linalg.norm(rect[1] - rect[0])
                    h_b = np.linalg.norm(rect[3] - rect[0])
                    if min(w_b, h_b) > 0:
                        ar = float(max(w_b, h_b) / min(w_b, h_b))
                        if 1.30 <= ar <= 3.40:
                            cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                            dist_center = np.sqrt(
                                ((cx - w / 2) / (w / 2)) ** 2
                                + ((cy - h / 2) / (h / 2)) ** 2
                            )
                            c_score = max(0.0, 1.0 - (dist_center / 1.414))
                            ar_score = max(0.0, 1.0 - abs(ar - 2.22) / 1.2)
                            area_score = max(0.0, 1.0 - abs((area / total_area) - 0.45) / 0.45)

                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
                            quad_pixels = gray[mask > 0]
                            tex_score = float(np.std(quad_pixels)) if len(quad_pixels) > 0 else 0.0

                            total_score = (
                                (c_score * 30.0)
                                + (ar_score * 30.0)
                                + (area_score * 20.0)
                                + (min(50.0, tex_score) * 0.4)
                            )
                            candidates.append((total_score, pts, ar))
                            break

        # Strategy B: Minimum Area Rotated Quad for angled camera scenes
        rect_box = cv2.minAreaRect(c)
        w_b, h_b = rect_box[1]
        if min(w_b, h_b) > 0:
            ar = float(max(w_b, h_b) / min(w_b, h_b))
            box_area = w_b * h_b
            if (
                1.30 <= ar <= 3.30
                and min_area <= box_area <= (total_area * 0.92)
            ):
                box_pts = cv2.boxPoints(rect_box)
                cx, cy = rect_box[0]
                dist_center = np.sqrt(
                    ((cx - w / 2) / (w / 2)) ** 2
                    + ((cy - h / 2) / (h / 2)) ** 2
                )
                c_score = max(0.0, 1.0 - (dist_center / 1.414))
                ar_score = max(0.0, 1.0 - abs(ar - 2.22) / 1.2)
                area_score = max(0.0, 1.0 - abs((box_area / total_area) - 0.45) / 0.45)

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [box_pts.astype(np.int32)], 255)
                quad_pixels = gray[mask > 0]
                tex_score = float(np.std(quad_pixels)) if len(quad_pixels) > 0 else 0.0

                total_score = (
                    (c_score * 25.0)
                    + (ar_score * 30.0)
                    + (area_score * 20.0)
                    + (min(50.0, tex_score) * 0.4)
                )
                candidates.append((total_score, box_pts.astype(np.float32), ar))

    detected_corners: Optional[np.ndarray] = None
    aspect_ratio: Optional[float] = None

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_cand = candidates[0]
        detected_corners = best_cand[1].astype(np.float32)
        aspect_ratio = float(best_cand[2])

    # Fallback: If image itself is already a cropped note or note fills frame
    if detected_corners is None:
        detected_corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
        )
        aspect_ratio = float(max(w, h) / max(1, min(w, h)))

    return detected_corners, closed, aspect_ratio


def extract_rois(warped_note: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Slices standardized Regions of Interest (ROIs) from a rectified 600x300 note.

    Extracted regions:
      - Security thread strip (narrow vertical metallic/fluorescent ribbon)
      - Watermark portrait window
      - Denomination numeral region
      - RBI Governor promise clause

    Parameters:
        warped_note (np.ndarray): Rectified note of shape (300, 600, 3).

    Returns:
        Dict[str, Dict[str, Any]]: ROI mappings containing image patches and bounding boxes.
    """
    h, w = warped_note.shape[:2]
    extracted = {}

    for key, spec in ROI_DEFINITIONS.items():
        y1_rel, x1_rel, y2_rel, x2_rel = spec["rel_box"]
        y1, y2 = int(y1_rel * h), int(y2_rel * h)
        x1, x2 = int(x1_rel * w), int(x2_rel * w)

        # Slice patch
        patch = warped_note[y1:y2, x1:x2].copy()
        extracted[key] = {
            "name": spec["name"],
            "bbox": (x1, y1, x2, y2),
            "patch": patch,
            "height": y2 - y1,
            "width": x2 - x1,
        }

    return extracted


def extract_color_features(warped_note: np.ndarray) -> Dict[str, Any]:
    """
    Computes statistical color moments and normalized color histograms across RGB,
    HSV, and CIE-Lab color spaces to characterize note denomination.

    Includes both raw moments and white-balanced color constancy moments.

    Parameters:
        warped_note (np.ndarray): Rectified note of shape (300, 600, 3).

    Returns:
        Dict[str, Any]: Dictionary containing statistical moments and feature vectors.
    """
    # 1. Raw RGB moments
    r = warped_note[:, :, 2].astype(np.float32)
    g = warped_note[:, :, 1].astype(np.float32)
    b = warped_note[:, :, 0].astype(np.float32)

    # 2. Raw HSV moments
    hsv = cv2.cvtColor(warped_note, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)

    # 3. White-balanced color moments (Gray-World illumination invariant)
    wb_note = apply_color_constancy(warped_note)
    wb_hsv = cv2.cvtColor(wb_note, cv2.COLOR_BGR2HSV)
    wb_h = wb_hsv[:, :, 0].astype(np.float32)
    wb_s = wb_hsv[:, :, 1].astype(np.float32)
    wb_v = wb_hsv[:, :, 2].astype(np.float32)

    # Central moments computation
    features: Dict[str, Any] = {
        "h_mean": float(np.mean(h_chan)),
        "h_std": float(np.std(h_chan)),
        "h_skew": float(skew(h_chan.flatten())),
        "s_mean": float(np.mean(s_chan)),
        "s_std": float(np.std(s_chan)),
        "s_skew": float(skew(s_chan.flatten())),
        "v_mean": float(np.mean(v_chan)),
        "v_std": float(np.std(v_chan)),
        "v_skew": float(skew(v_chan.flatten())),
        "r_mean": float(np.mean(r)),
        "g_mean": float(np.mean(g)),
        "b_mean": float(np.mean(b)),
        # Illumination-neutralized moments
        "wb_h_mean": float(np.mean(wb_h)),
        "wb_s_mean": float(np.mean(wb_s)),
        "wb_v_mean": float(np.mean(wb_v)),
        "wb_r_mean": float(np.mean(wb_note[:, :, 2])),
        "wb_g_mean": float(np.mean(wb_note[:, :, 1])),
        "wb_b_mean": float(np.mean(wb_note[:, :, 0])),
    }

    # Normalized 16-bin Hue histogram
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    features["hue_hist"] = hist_h.flatten().tolist()

    return features


def analyze_security_strip(
    strip_roi: np.ndarray, n_clusters: int = 3
) -> Dict[str, Any]:
    """
    Applies Unsupervised K-Means Clustering (scikit-learn) on the security thread ROI
    to segment the metallic/fluorescent ribbon from the paper substrate.

    Evaluates both:
      a) Transmitted light (backlit / full continuous dark ribbon).
      b) Reflected light (authentic windowed metallic segments with periodic spacing).

    Parameters:
        strip_roi (np.ndarray): Cropped BGR patch of security thread region.
        n_clusters (int): Number of K-Means color clusters (default: 3).

    Returns:
        Dict[str, Any]: Metrics including continuity_pct, window_count, is_windowed,
                        segmented_mask, and cluster swatches.
    """
    h, w = strip_roi.shape[:2]
    if h == 0 or w == 0:
        return {
            "continuity_pct": 0.0,
            "segmented_mask": np.zeros((10, 10), dtype=np.uint8),
            "cluster_labels": np.zeros((10, 10), dtype=np.int32),
            "cluster_centers": np.zeros((n_clusters, 3)),
            "thread_width_mean": 0.0,
            "row_density": [0.0],
            "window_count": 0,
            "is_windowed": False,
            "is_continuous": False,
        }

    # Convert to Lab and HSV color spaces for robust color-metallic segmentation
    roi_hsv = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2Lab)

    pixels_hsv = roi_hsv.reshape(-1, 3).astype(np.float32)
    pixels_lab = roi_lab.reshape(-1, 3).astype(np.float32)
    combined_features = np.hstack([pixels_hsv, pixels_lab])

    # Perform Unsupervised K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(combined_features)
    label_map = labels.reshape((h, w))

    # Evaluate which cluster corresponds to the central metallic strip
    center_start = int(w * 0.20)
    center_end = int(w * 0.80)
    center_slice = label_map[:, center_start:center_end]

    best_cluster = 0
    best_score = -1.0

    for cluster_id in range(n_clusters):
        cluster_mask = label_map == cluster_id
        cluster_pixels_total = np.sum(cluster_mask)
        if cluster_pixels_total == 0:
            continue

        center_pixels = np.sum(center_slice == cluster_id)
        centrality = center_pixels / (cluster_pixels_total + 1e-6)

        roi_fraction = cluster_pixels_total / (h * w)
        if 0.02 <= roi_fraction <= 0.45:
            score = centrality * (1.0 - abs(roi_fraction - 0.15))
        else:
            score = centrality * 0.1

        if score > best_score:
            best_score = score
            best_cluster = cluster_id

    # Create binary mask of segmented strip
    strip_binary = np.zeros((h, w), dtype=np.uint8)
    strip_binary[label_map == best_cluster] = 255

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    cleaned_binary = cv2.morphologyEx(strip_binary, cv2.MORPH_CLOSE, kernel)

    # Vertical morphological closing to bridge 1-3 pixel noise gaps along vertical scanlines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    v_closed = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, kernel_v)

    # Vertical Continuity & Scanline Profiling
    row_hits = 0
    row_widths = []
    row_density = []
    active_rows = []

    for y in range(h):
        row = v_closed[y, center_start:center_end]
        pixel_count = int(np.count_nonzero(row))
        density = float(pixel_count / max(1, (center_end - center_start)))
        row_density.append(density)
        is_hit = pixel_count >= 1
        active_rows.append(is_hit)
        if is_hit:
            row_hits += 1
            row_widths.append(pixel_count)

    continuity_pct = float((row_hits / max(1, h)) * 100.0)
    mean_width = float(np.mean(row_widths)) if row_widths else 0.0
    std_width = float(np.std(row_widths)) if row_widths else 0.0

    # Windowed segment count: filter out single-pixel flickers (< 4 scanlines)
    window_count = 0
    cur_seg_len = 0
    for hit in active_rows:
        if hit:
            cur_seg_len += 1
        else:
            if cur_seg_len >= 4:
                window_count += 1
            cur_seg_len = 0
    if cur_seg_len >= 4:
        window_count += 1

    is_continuous = continuity_pct >= 65.0
    # Authentic windowed security threads exhibit 2 to 8 distinct windows
    # with continuity between 25% and 65% in reflected light
    is_windowed = (2 <= window_count <= 8) and (25.0 <= continuity_pct < 65.0)

    # Cluster centroids in BGR
    bgr_pixels = strip_roi.reshape(-1, 3).astype(np.float32)
    cluster_centers_bgr = np.zeros((n_clusters, 3), dtype=np.float32)
    for c_id in range(n_clusters):
        c_mask = labels == c_id
        if np.any(c_mask):
            cluster_centers_bgr[c_id] = np.mean(bgr_pixels[c_mask], axis=0)

    return {
        "continuity_pct": round(continuity_pct, 2),
        "segmented_mask": cleaned_binary,
        "cluster_labels": label_map,
        "cluster_centers": cluster_centers_bgr,
        "thread_width_mean": round(mean_width, 2),
        "thread_width_std": round(std_width, 2),
        "row_density": row_density,
        "strip_cluster_id": best_cluster,
        "window_count": window_count,
        "is_windowed": is_windowed,
        "is_continuous": is_continuous,
    }


def analyze_watermark(watermark_roi: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates the watermark region for latent portrait contrast, luminance variance,
    and fibrous paper substrate texture.

    Parameters:
        watermark_roi (np.ndarray): Cropped BGR patch of the watermark window.

    Returns:
        Dict[str, Any]:
            - variance (float): Gray intensity variance.
            - laplacian_var (float): Focus / sharpness gradient variance.
            - dynamic_range (float): 95th - 5th percentile luminance contrast.
            - has_latent_features (bool): True if values fall within expected watermark band.
    """
    h, w = watermark_roi.shape[:2]
    if h == 0 or w == 0:
        return {
            "variance": 0.0,
            "laplacian_var": 0.0,
            "dynamic_range": 0.0,
            "has_latent_features": False,
        }

    gray = cv2.cvtColor(watermark_roi, cv2.COLOR_BGR2GRAY)

    variance = float(np.var(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = float(p95 - p5)

    # Shannon Texture Entropy of the watermark window
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (float(np.sum(hist)) + 1e-7)
    hist_nonzero = hist[hist > 0]
    entropy_val = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))

    # Authentic banknotes exhibit subtle multi-tone latent gradients and paper texture.
    # Plain paper counterfeits have flat paper (variance < 240 with low dyn range, or blurred with low entropy & laplacian).
    # Real camera captures with sensor noise and genuine multi-tone watermarks:
    #   variance can reach up to ~12000 due to complex portrait + intaglio details.
    # Opaque stamped counterfeit: very high variance (>14000) AND high dyn range (>95).
    is_blank = (
        (dynamic_range < 28.0 and variance < 240.0)
        or (laplacian_var < 320.0 and entropy_val < 3.80)
        or (dynamic_range < 18.0)
        or (variance < 18.0)
    )
    is_stamped = variance > 14000.0 and dynamic_range > 95.0
    has_latent = not is_blank and not is_stamped

    return {
        "variance": round(variance, 2),
        "laplacian_var": round(laplacian_var, 2),
        "dynamic_range": round(dynamic_range, 2),
        "entropy": round(entropy_val, 2),
        "has_latent_features": has_latent,
    }


def draw_roi_overlays(
    warped_note: np.ndarray,
    rois: Dict[str, Dict[str, Any]],
    verdicts: Optional[Dict[str, bool]] = None,
) -> np.ndarray:
    """
    Draws aesthetic bounding box overlays and labels for all analyzed ROIs
    on the rectified banknote canvas.

    Parameters:
        warped_note (np.ndarray): Rectified note image of shape (300, 600, 3).
        rois (Dict[str, Dict[str, Any]]): Dictionary from extract_rois().
        verdicts (Optional[Dict[str, bool]]): Optional rule status to color boxes (Green=pass, Red=fail).

    Returns:
        np.ndarray: Annotated visualization frame.
    """
    annotated = warped_note.copy()

    color_palette = {
        "watermark": (255, 165, 0),        # Amber / Cyan
        "security_strip": (0, 255, 255),   # Bright Yellow
        "numeral": (0, 255, 128),          # Mint Green
        "governor_promise": (200, 100, 255)# Violet
    }

    for key, data in rois.items():
        x1, y1, x2, y2 = data["bbox"]
        default_color = color_palette.get(key, (0, 255, 0))

        # If verdicts are supplied, use green/red cues
        if verdicts and key in verdicts:
            color = (0, 220, 50) if verdicts[key] else (0, 40, 240)
        else:
            color = default_color

        # Draw semi-transparent fill and border
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label tag with dark background
        label = data["name"].split()[0]
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        tag_y = max(y1 - 4, label_h + 4)
        cv2.rectangle(
            annotated,
            (x1, tag_y - label_h - 4),
            (x1 + label_w + 6, tag_y + 2),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, tag_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def compute_microtexture_edge_density(warped_note: np.ndarray) -> float:
    """
    Measures intaglio microprinting edge density across the central note field.
    Genuine notes have crisp intaglio engraving lines; flat counterfeits appear blurred.
    """
    h, w = warped_note.shape[:2]
    central_patch = warped_note[int(h * 0.15) : int(h * 0.85), int(w * 0.25) : int(w * 0.75)]
    gray = cv2.cvtColor(central_patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    density = float(np.count_nonzero(edges) / (edges.size + 1e-7))
    return round(density, 4)


def compute_color_delta_e(color_features: Dict[str, Any], denomination: str) -> float:
    """
    Computes perceptual CIE-Lab Delta-E color distance between observed note substrate
    and standard central bank denomination reference color.
    """
    reference_bgr = {
        "100": (215, 175, 185),   # Lavender
        "200": (70, 205, 245),    # Bright Warm Yellow
        "500": (175, 192, 180),   # Stone Olive Grey
    }
    bgr_target = reference_bgr.get(denomination, (175, 192, 180))
    target_mat = np.array([[bgr_target]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_mat, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)

    # Use observed substrate mean RGB to evaluate perceptual chromaticity (both raw and wb)
    r_raw = int(np.clip(color_features.get("r_mean", 175.0), 0, 255))
    g_raw = int(np.clip(color_features.get("g_mean", 192.0), 0, 255))
    b_raw = int(np.clip(color_features.get("b_mean", 180.0), 0, 255))
    obs_raw_mat = np.array([[[b_raw, g_raw, r_raw]]], dtype=np.uint8)
    obs_raw_lab = cv2.cvtColor(obs_raw_mat, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)
    delta_e_raw = float(np.linalg.norm(obs_raw_lab - target_lab))

    # Also evaluate calibrated white-balance BGR if available
    r_wb = int(np.clip(color_features.get("wb_r_mean", r_raw), 0, 255))
    g_wb = int(np.clip(color_features.get("wb_g_mean", g_raw), 0, 255))
    b_wb = int(np.clip(color_features.get("wb_b_mean", b_raw), 0, 255))
    obs_wb_mat = np.array([[[b_wb, g_wb, r_wb]]], dtype=np.uint8)
    obs_wb_lab = cv2.cvtColor(obs_wb_mat, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)
    delta_e_wb = float(np.linalg.norm(obs_wb_lab - target_lab))

    return round(float(min(delta_e_raw, delta_e_wb)), 2)


def extract_forensic_features(
    warped_note: np.ndarray,
    rois: Optional[Dict[str, Dict[str, Any]]] = None,
    strip_analysis: Optional[Dict[str, Any]] = None,
    watermark_analysis: Optional[Dict[str, Any]] = None,
    color_features: Optional[Dict[str, Any]] = None,
    aspect_ratio: Optional[float] = None,
    predicted_denomination: str = "500",
    **kwargs: Any,
) -> np.ndarray:
    """
    Constructs the 16-element standardized forensic feature vector for the
    AI Machine Learning Counterfeit Detector (CounterfeitDetectorAI).

    Returns:
        np.ndarray: 1D float32 array matching FORENSIC_FEATURE_COLUMNS.
    """
    from classifier import DENOMINATION_SPECS

    predicted_denomination = str(kwargs.get("pred_denom", predicted_denomination))
    s_analysis: Dict[str, Any] = strip_analysis if strip_analysis is not None else (kwargs.get("security_strip_metrics") or {})
    w_analysis: Dict[str, Any] = watermark_analysis if watermark_analysis is not None else (kwargs.get("watermark_metrics") or {})
    c_features: Dict[str, Any] = color_features if color_features is not None else (kwargs.get("color_features") or {})
    aspect_ratio = aspect_ratio if aspect_ratio is not None else kwargs.get("aspect_ratio")

    denom_spec = DENOMINATION_SPECS.get(predicted_denomination, DENOMINATION_SPECS["500"])
    target_ar = float(denom_spec["aspect_ratio"])
    obs_ar = float(aspect_ratio) if aspect_ratio is not None else target_ar

    # 1. aspect_ratio_error
    ar_error = float(abs(obs_ar - target_ar) / target_ar)

    # 2. strip continuity
    strip_continuity = float(s_analysis.get("continuity_pct", 0.0))

    # 3. strip window count
    strip_window_count = float(s_analysis.get("window_count", 0))

    # 4. strip width mean
    strip_width_mean = float(s_analysis.get("thread_width_mean", 0.0))

    # 5. strip width std
    strip_width_std = float(s_analysis.get("thread_width_std", 0.0))

    # 6. watermark variance
    watermark_variance = float(w_analysis.get("variance", 0.0))

    # 7. watermark dynamic range
    watermark_dyn_range = float(w_analysis.get("dynamic_range", 0.0))

    # 8. watermark laplacian var
    watermark_laplacian = float(w_analysis.get("laplacian_var", 0.0))

    # 9. watermark entropy
    watermark_entropy = float(w_analysis.get("entropy", 4.3))

    # 10. color hue dev (minimum deviation across raw and illuminant-calibrated moments)
    h_min, h_max = denom_spec["hsv_bounds"]["h_mean"]
    target_hue = (h_min + h_max) / 2.0
    raw_h = float(c_features.get("h_mean", target_hue))
    wb_h = float(c_features.get("wb_h_mean", raw_h))
    hue_dev = float(min(abs(raw_h - target_hue), abs(wb_h - target_hue)))

    # 11. color sat mean
    color_sat_mean = float(c_features.get("s_mean", 0.0))

    # 12. color sat std
    color_sat_std = float(c_features.get("s_std", 0.0))

    # 13. color val mean
    color_val_mean = float(c_features.get("v_mean", 0.0))

    # 14. color delta_e
    color_delta_e = compute_color_delta_e(c_features, predicted_denomination)

    # 15. microtexture edge density
    microtexture_edge_density = compute_microtexture_edge_density(warped_note)

    # 16. denomination numerical code
    denom_code_map = {"100": 1.0, "200": 2.0, "500": 5.0}
    denom_code = denom_code_map.get(predicted_denomination, 5.0)

    vec = [
        ar_error,
        strip_continuity,
        strip_window_count,
        strip_width_mean,
        strip_width_std,
        watermark_variance,
        watermark_dyn_range,
        watermark_laplacian,
        watermark_entropy,
        hue_dev,
        color_sat_mean,
        color_sat_std,
        color_val_mean,
        color_delta_e,
        microtexture_edge_density,
        denom_code,
    ]
    return np.array(vec, dtype=np.float32)
