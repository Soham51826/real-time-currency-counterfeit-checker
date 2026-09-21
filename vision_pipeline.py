"""
Module 1: Preprocessing & Computer Vision Pipeline
===================================================
Provides robust, production-grade computer vision utilities for currency note
inspection. Implements:
  1. Note detection via adaptive Canny edge detection & polygonal contour approximation.
  2. Four-point perspective transformation (homography warp) to a standardized 600x300 px note.
  3. Predefined Region-of-Interest (ROI) extraction (Security Strip, Watermark, Numeral).
  4. Color moment extraction (RGB & HSV mean, std dev, skewness) and normalized histograms.
  5. Unsupervised K-Means clustering (scikit-learn) on the security thread ROI to segment
     the metallic strip and measure vertical continuity percentage.
  6. Watermark latent portrait contrast and variance analysis.

Academic Mapping:
  - Computer Vision: Homography, spatial filtering, morphological operations, color space mapping.
  - CO4 (Unsupervised Learning): K-Means clustering for metallic/ink thread segmentation.
"""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import cv2
from scipy.stats import skew
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


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 arbitrary polygon coordinates in consistent clockwise order:
    [top-left, top-right, bottom-right, bottom-left].

    Mathematical formulation:
      - Top-left point has the smallest (x + y) sum.
      - Bottom-right point has the largest (x + y) sum.
      - Top-right point has the smallest difference (y - x).
      - Bottom-left point has the largest difference (y - x).

    Parameters:
        pts (np.ndarray): Array of shape (4, 2) representing 2D corner points.

    Returns:
        np.ndarray: Ordered coordinates of shape (4, 2) in float32.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.asarray(pts, dtype="float32")

    # Sum of coordinates
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # Difference of coordinates (y - x)
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
        # Shift corners cyclically
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


def detect_note_contour(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[float]]:
    """
    Detects the banknote boundary contour within a camera frame or uploaded image
    using adaptive edge detection and polygonal convex approximation.

    Processing Stages:
      1. Grayscale conversion and bilateral filtering (preserves edges while removing paper grain).
      2. Adaptive Canny edge detection based on Otsu median threshold.
      3. Morphological dilation to bridge broken boundary edges.
      4. Contour retrieval & sorting by area.
      5. Douglas-Peucker polygon approximation (cv2.approxPolyDP) to identify 4-point quadrilateral.

    Parameters:
        image (np.ndarray): Input BGR frame.

    Returns:
        Tuple[Optional[np.ndarray], np.ndarray, Optional[float]]:
            - corners: Array of 4 corner points (shape (4, 2)) or None if no note detected.
            - debug_edges: Binary edge mask for UI diagnostics.
            - aspect_ratio: Observed width/height aspect ratio.
    """
    h, w = image.shape[:2]
    min_area = (h * w) * 0.08  # Note must occupy at least 8% of the frame

    # 1. Bilateral filter to reduce noise while keeping edge sharpness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. Adaptive Canny thresholds
    med_val = np.median(blurred)
    lower = int(max(0, 0.66 * med_val))
    upper = int(min(255, 1.33 * med_val))
    edges = cv2.Canny(blurred, lower, upper)

    # 3. Morphological closing to seal perimeter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detected_corners: Optional[np.ndarray] = None
    aspect_ratio: Optional[float] = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            break

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Look for 4-point convex quadrilateral
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            detected_corners = pts
            rect = order_points(pts)
            w_box = np.linalg.norm(rect[1] - rect[0])
            h_box = np.linalg.norm(rect[3] - rect[0])
            if h_box > 0:
                aspect_ratio = float(max(w_box, h_box) / min(w_box, h_box))
            break

    # Fallback: If no strict 4-sided polygon is found but a large contour exists,
    # use minimum area bounding box
    if detected_corners is None and len(contours) > 0:
        largest_c = contours[0]
        if cv2.contourArea(largest_c) >= min_area:
            rect_box = cv2.minAreaRect(largest_c)
            box_pts = cv2.boxPoints(rect_box)
            detected_corners = np.int32(box_pts)
            w_b, h_b = rect_box[1]
            if min(w_b, h_b) > 0:
                aspect_ratio = float(max(w_b, h_b) / min(w_b, h_b))

    # Second Fallback: If entire image is already a cropped banknote
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
    Computes statistical color moments and normalized color histograms across RGB
    and HSV color spaces to characterize note denomination.

    Features extracted:
      - Mean, standard deviation, and skewness for Hue, Saturation, Value.
      - Mean, standard deviation, and skewness for Red, Green, Blue.
      - Dominant HSV color vector.

    Parameters:
        warped_note (np.ndarray): Rectified note of shape (300, 600, 3).

    Returns:
        Dict[str, Any]: Dictionary containing statistical moments and feature vector.
    """
    # RGB moments
    r = warped_note[:, :, 2].astype(np.float32)
    g = warped_note[:, :, 1].astype(np.float32)
    b = warped_note[:, :, 0].astype(np.float32)

    # HSV moments (OpenCV HSV: H in [0, 179], S in [0, 255], V in [0, 255])
    hsv = cv2.cvtColor(warped_note, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)

    # Central moments computation
    features = {
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
    to segment the metallic/fluorescent ribbon from the paper substrate and
    computes vertical continuity percentage.

    Academic Concept (CO4: Unsupervised Learning):
      Genuine banknotes feature an embedded metallic ribbon that spans the entire
      vertical span of the note with continuous metallic/luster response. Counterfeits
      frequently exhibit discontinuous breaks, simulated pencil/ink lines, or missing strips.

    Parameters:
        strip_roi (np.ndarray): Cropped BGR patch of security thread region.
        n_clusters (int): Number of K-Means color clusters (default 3: strip, substrate, print).

    Returns:
        Dict[str, Any]:
            - continuity_pct (float): Percentage of vertical rows containing the strip (0 to 100%).
            - segmented_mask (np.ndarray): Binary segmentation mask (255 = strip, 0 = substrate).
            - cluster_labels (np.ndarray): 2D cluster label map.
            - cluster_centers (np.ndarray): RGB cluster centroid values.
            - thread_width_mean (float): Average horizontal width of detected strip in pixels.
            - row_density (List[float]): Vertical line-by-line thread detection profile.
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
        }

    # Convert to Lab and HSV color spaces for robust color-metallic segmentation
    roi_hsv = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2Lab)

    # Feature space for clustering: [H, S, V, L, a, b]
    pixels_hsv = roi_hsv.reshape(-1, 3).astype(np.float32)
    pixels_lab = roi_lab.reshape(-1, 3).astype(np.float32)
    # Combine normalized features: V (brightness), S (saturation), L (lightness)
    combined_features = np.hstack([pixels_hsv, pixels_lab])

    # Perform Unsupervised K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    labels = kmeans.fit_predict(combined_features)
    label_map = labels.reshape((h, w))

    # Identify the cluster corresponding to the metallic/security strip:
    # A genuine security strip is vertically aligned near the horizontal center of the ROI.
    # We evaluate which cluster has the highest concentration within the central 60% of ROI width.
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
        # Ratio of cluster's presence inside the central strip column vs total ROI
        centrality = center_pixels / (cluster_pixels_total + 1e-6)

        # Strip should be relatively compact (between 3% and 40% of the ROI area)
        roi_fraction = cluster_pixels_total / (h * w)
        if 0.02 <= roi_fraction <= 0.45:
            score = centrality * (1.0 - abs(roi_fraction - 0.15))
        else:
            score = centrality * 0.1

        if score > best_score:
            best_score = score
            best_cluster = cluster_id

    # Create binary mask of the segmented security strip
    strip_binary = np.zeros((h, w), dtype=np.uint8)
    strip_binary[label_map == best_cluster] = 255

    # Morphological opening/closing to filter minor speckle noise while preserving line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    cleaned_binary = cv2.morphologyEx(strip_binary, cv2.MORPH_CLOSE, kernel)

    # Vertical Continuity Measurement:
    # For every horizontal scanline (row y), check if strip pixels exist near central zone
    row_hits = 0
    row_widths = []
    row_density = []

    for y in range(h):
        row = cleaned_binary[y, center_start:center_end]
        pixel_count = np.count_nonzero(row)
        row_density.append(float(pixel_count / max(1, (center_end - center_start))))
        if pixel_count >= 1:
            row_hits += 1
            row_widths.append(pixel_count)

    continuity_pct = float((row_hits / max(1, h)) * 100.0)
    mean_width = float(np.mean(row_widths)) if row_widths else 0.0

    # Cluster centroids in BGR for visual swatch display
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
        "row_density": row_density,
        "strip_cluster_id": int(best_cluster),
    }


def analyze_watermark(watermark_roi: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates the watermark region for latent portrait contrast, luminance variance,
    and gradient distribution.

    Academic Concept:
      Authentic watermarks are embedded within the paper matrix during manufacturing,
      creating subtle multi-tone gradations. Counterfeits commonly have either:
        a) Completely blank paper (variance near 0).
        b) Harsh opaque printed lines or stamps (excessive gradient spikes / harsh edges).

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

    # 1. Variance of pixel intensity
    variance = float(np.var(gray))

    # 2. Laplacian operator for high-frequency structural transitions
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. Dynamic contrast range (5th to 95th percentile)
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = float(p95 - p5)

    # Variance floor lowered from 80 to 40 and dynamic range set to 27
    # to match Rule 4 in expert_system.py. Camera-captured images produce
    # softer watermark regions due to JPEG compression and autofocus blur.
    # Genuine samples: dyn_range 31-35. Blank counterfeit: 25 -> threshold at 27 is safe.
    # Legitimate watermarks: variance ~40-1400, dynamic range ~27-180
    has_latent = bool(40.0 <= variance <= 1400.0 and dynamic_range >= 27.0)

    return {
        "variance": round(variance, 2),
        "laplacian_var": round(laplacian_var, 2),
        "dynamic_range": round(dynamic_range, 2),
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
