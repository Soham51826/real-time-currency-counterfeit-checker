"""
Module 5: Dataset & Synthetic Sample Generator
==============================================
Generates synthetic tabular training data for the Decision Tree denomination
classifier, trains and serializes the baseline model, and generates realistic
synthetic banknote images for testing both genuine and counterfeit scenarios.

Generated Artifacts:
  - data/currency_features.csv: 1,600 training samples across 4 denominations.
  - models/denomination_tree.joblib: Pre-trained Decision Tree model.
  - samples/: Realistic high-resolution test notes (genuine + counterfeit edge cases).
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from classifier import (
    DenominationClassifier,
    DENOMINATION_SPECS,
    FEATURE_COLUMNS,
)
from counterfeit_classifier import (
    CounterfeitDetectorAI,
    FORENSIC_FEATURE_COLUMNS,
)
from vision_pipeline import CANVAS_WIDTH, CANVAS_HEIGHT


def generate_feature_dataset(
    n_samples_per_class: int = 400, random_seed: int = 42
) -> pd.DataFrame:
    """
    Synthesizes a robust tabular dataset of banknote color moments and aspect ratios
    grounded in official denomination specifications, incorporating realistic
    lighting shifts, paper wear, and sensor noise.

    Parameters:
        n_samples_per_class (int): Number of feature vectors per denomination (default 400).
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Tabular dataset with 7 feature columns + target label.
    """
    np.random.seed(random_seed)
    rows = []

    for label, spec in DENOMINATION_SPECS.items():
        h_min, h_max = spec["hsv_bounds"]["h_mean"]
        s_min, s_max = spec["hsv_bounds"]["s_mean"]
        v_min, v_max = spec["hsv_bounds"]["v_mean"]
        base_ar = spec["aspect_ratio"]

        # Sample distributions with realistic Gaussian noise
        h_means = np.random.uniform(h_min, h_max, n_samples_per_class) + np.random.normal(0, 1.5, n_samples_per_class)
        s_means = np.random.uniform(s_min, s_max, n_samples_per_class) + np.random.normal(0, 4.0, n_samples_per_class)
        v_means = np.random.uniform(v_min, v_max, n_samples_per_class) + np.random.normal(0, 5.0, n_samples_per_class)

        # Standard deviations of internal note pixels
        h_stds = np.random.normal(12.0, 2.5, n_samples_per_class)
        s_stds = np.random.normal(20.0, 3.5, n_samples_per_class)
        v_stds = np.random.normal(25.0, 4.0, n_samples_per_class)

        # Aspect ratio with camera tilt variation (±3%)
        aspect_ratios = np.random.normal(base_ar, 0.04, n_samples_per_class)

        for i in range(n_samples_per_class):
            rows.append({
                "h_mean": round(float(np.clip(h_means[i], 0, 179)), 2),
                "h_std": round(float(max(1.0, h_stds[i])), 2),
                "s_mean": round(float(np.clip(s_means[i], 0, 255)), 2),
                "s_std": round(float(max(1.0, s_stds[i])), 2),
                "v_mean": round(float(np.clip(v_means[i], 0, 255)), 2),
                "v_std": round(float(max(1.0, v_stds[i])), 2),
                "aspect_ratio": round(float(np.clip(aspect_ratios[i], 1.8, 2.8)), 3),
                "denomination": label,
            })

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def train_and_save_baseline_model(
    df: pd.DataFrame, model_path: str = "models/denomination_tree.joblib"
) -> Dict[str, Any]:
    """
    Fits the DecisionTree classifier on the synthetic dataset, prints evaluation
    metrics, and serializes the trained model to disk.
    """
    X = np.array(df[FEATURE_COLUMNS].values, dtype=np.float32)
    y = np.array(df["denomination"].astype(str).tolist(), dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = DenominationClassifier(max_depth=6, criterion="entropy")
    train_res = clf.train(X_train, y_train, feature_names=FEATURE_COLUMNS)
    eval_res = clf.evaluate(X_test, y_test)

    clf.save(model_path)
    print(f"[OK] Denomination Model successfully trained and saved to: {model_path}")
    print(f"     Holdout Test Accuracy: {eval_res['accuracy']*100:.2f}%")
    return eval_res


def generate_counterfeit_dataset(
    n_samples: int = 3200, random_seed: int = 42
) -> pd.DataFrame:
    """
    Synthesizes a comprehensive forensic dataset of genuine and counterfeit banknote features
    for training the AI Counterfeit Detection Engine (CounterfeitDetectorAI).

    Distributions:
      - Genuine notes (50%): Conforming physical aspect ratio, continuous or authentic windowed
        metallic thread, high-entropy multi-tone watermark, accurate substrate dye.
      - Counterfeit defect types (50%):
        * Broken thread / tampered foil (interrupted or low continuity)
        * Blank / washed watermark (flat paper, low dynamic range, low entropy)
        * Stamped / harsh toner watermark (abnormal contrast spikes)
        * Substrate dye / altered ink counterfeit (high CIE-Lab Delta-E and Hue shift)
        * Bad dimensional cuts (cutting errors, non-standard aspect ratio)
        * Blurred inkjet photocopy (low microtexture edge density)
    """
    np.random.seed(random_seed)
    n_genuine = n_samples // 2
    n_fake = n_samples - n_genuine
    rows = []

    denoms = ["100", "200", "500", "2000"]
    denom_code_map = {"100": 1.0, "200": 2.0, "500": 5.0, "2000": 20.0}

    # 1. Synthesize Genuine Banknotes
    for _ in range(n_genuine):
        denom = str(np.random.choice(denoms))
        spec = DENOMINATION_SPECS[denom]
        d_code = denom_code_map[denom]

        # Conforming aspect ratio error: mostly crisp orthogonal (< 0.05), with 8% camera perspective tilt
        if np.random.rand() < 0.08:
            ar_err = float(np.random.uniform(0.06, 0.32))  # Camera perspective foreshortening
        else:
            ar_err = float(np.clip(np.random.exponential(0.008), 0.0, 0.05))

        # Thread: 92% continuous, 8% authentic windowed
        is_win = np.random.rand() < 0.08
        if is_win:
            strip_cont = float(np.random.uniform(40.0, 62.0))
            strip_win = np.random.randint(3, 8)
        else:
            strip_cont = float(np.random.uniform(96.0, 100.0))
            strip_win = 1

        strip_width_mean = float(np.random.normal(10.5, 1.2))
        strip_width_std = float(np.random.normal(2.4, 0.4))

        # Watermark: multi-tone latent portrait (matching actual rendered samples)
        wm_var = float(np.random.uniform(230.0, 480.0) + np.random.normal(0, 15.0))
        wm_dyn = float(np.random.uniform(30.0, 38.0))
        wm_lap = float(np.random.uniform(450.0, 1600.0))
        wm_ent = float(np.random.uniform(4.24, 4.65))

        # Color: Conforming to central bank envelope
        h_min, h_max = spec["hsv_bounds"]["h_mean"]
        s_min, s_max = spec["hsv_bounds"]["s_mean"]
        v_min, v_max = spec["hsv_bounds"]["v_mean"]
        h_target = (h_min + h_max) / 2.0
        h_obs = float(np.random.uniform(h_min, h_max))
        hue_dev = float(abs(h_obs - h_target)) + float(np.random.uniform(5.0, 25.0))
        s_mean = float(np.random.uniform(s_min, s_max))
        s_std = float(np.random.normal(12.0, 2.5))
        v_mean = float(np.random.uniform(v_min, v_max))
        
        # Perceptual CIE-Lab Delta-E (empirically 3.0 to 12.5 for genuine banknotes)
        delta_e = float(np.clip(np.random.normal(7.5, 2.0), 2.5, 14.0))

        # Microtexture intaglio edge density
        edge_dens = float(np.random.uniform(0.026, 0.036))

        rows.append({
            "aspect_ratio_error": round(ar_err, 4),
            "strip_continuity": round(strip_cont, 2),
            "strip_window_count": strip_win,
            "strip_width_mean": round(strip_width_mean, 2),
            "strip_width_std": round(max(0.1, strip_width_std), 2),
            "watermark_variance": round(wm_var, 2),
            "watermark_dynamic_range": round(wm_dyn, 2),
            "watermark_laplacian_var": round(wm_lap, 2),
            "watermark_entropy": round(wm_ent, 3),
            "color_hue_dev": round(hue_dev, 2),
            "color_sat_mean": round(s_mean, 2),
            "color_sat_std": round(max(1.0, s_std), 2),
            "color_val_mean": round(v_mean, 2),
            "color_delta_e": round(delta_e, 2),
            "microtexture_edge_density": round(edge_dens, 4),
            "denomination_code": d_code,
            "is_genuine": 1,
            "defect_class": "GENUINE",
        })

    # 2. Synthesize Counterfeit Banknotes with Diverse Defect Typologies
    defect_types = [
        "BROKEN_THREAD",
        "BLANK_WATERMARK",
        "STAMPED_WATERMARK",
        "ALTERED_COLOR",
        "BAD_DIMENSION",
        "LOW_QUALITY_PRINT",
    ]

    for _ in range(n_fake):
        denom = str(np.random.choice(denoms))
        spec = DENOMINATION_SPECS[denom]
        d_code = denom_code_map[denom]
        defect = str(np.random.choice(defect_types))

        # Baseline defaults
        ar_err = float(np.clip(np.random.exponential(0.01), 0.0, 0.05))
        strip_cont = float(np.random.uniform(96.0, 100.0))
        strip_win = 1
        strip_width_mean = float(np.random.normal(10.5, 1.2))
        strip_width_std = float(np.random.normal(2.4, 0.4))

        wm_var = float(np.random.uniform(240.0, 450.0))
        wm_dyn = float(np.random.uniform(30.0, 36.0))
        wm_lap = float(np.random.uniform(800.0, 1500.0))
        wm_ent = float(np.random.uniform(4.25, 4.60))

        h_min, h_max = spec["hsv_bounds"]["h_mean"]
        s_min, s_max = spec["hsv_bounds"]["s_mean"]
        v_min, v_max = spec["hsv_bounds"]["v_mean"]
        h_target = (h_min + h_max) / 2.0
        h_obs = float(np.random.uniform(h_min, h_max))
        hue_dev = float(abs(h_obs - h_target)) + float(np.random.uniform(15.0, 35.0))
        s_mean = float(np.random.uniform(s_min, s_max))
        s_std = float(np.random.normal(12.0, 2.5))
        v_mean = float(np.random.uniform(v_min, v_max))
        delta_e = float(np.clip(np.random.normal(7.5, 2.0), 2.5, 14.0))
        edge_dens = float(np.random.uniform(0.026, 0.036))

        # Apply specific defect manifestation
        if defect == "BROKEN_THREAD":
            # Disrupted, painted, or severed metallic thread
            strip_cont = float(np.random.uniform(12.0, 52.0))
            strip_win = int(np.random.choice([0, 2, 4, 6]))
            strip_width_mean = float(np.random.uniform(8.0, 12.0))
            strip_width_std = float(np.random.uniform(3.8, 6.5))

        elif defect == "BLANK_WATERMARK":
            # Flat paper photocopy with unprinted watermark window
            wm_var = float(np.random.uniform(4.0, 220.0))
            wm_dyn = float(np.random.uniform(4.0, 25.2))
            wm_lap = float(np.random.uniform(15.0, 260.0))
            wm_ent = float(np.random.uniform(2.30, 3.75))

        elif defect == "STAMPED_WATERMARK":
            # Opaque rubber stamp or harsh inkjet watermark with dark unnatural edges
            wm_var = float(np.random.uniform(1800.0, 7200.0))
            wm_dyn = float(np.random.uniform(55.0, 95.0))
            wm_lap = float(np.random.uniform(2200.0, 6500.0))
            wm_ent = float(np.random.uniform(4.90, 5.80))

        elif defect == "ALTERED_COLOR":
            # Incompatible inkjet toner dye, wrong hue, high Delta-E
            hue_dev = float(np.random.uniform(35.0, 110.0))
            delta_e = float(np.random.uniform(32.0, 120.0))
            s_mean = float(np.random.uniform(10.0, 240.0))

        elif defect == "BAD_DIMENSION":
            # Counterfeit cut to incorrect physical size / distorted ratio
            ar_err = float(np.random.uniform(0.18, 0.45))
            strip_width_mean = float(np.random.uniform(14.0, 20.0))
            wm_var = float(np.random.uniform(520.0, 950.0))

        elif defect == "LOW_QUALITY_PRINT":
            # Blurred desktop scan / lack of fine intaglio security microprinting
            edge_dens = float(np.random.uniform(0.005, 0.019))
            wm_ent = float(np.random.uniform(3.20, 3.85))

        rows.append({
            "aspect_ratio_error": round(ar_err, 4),
            "strip_continuity": round(strip_cont, 2),
            "strip_window_count": strip_win,
            "strip_width_mean": round(strip_width_mean, 2),
            "strip_width_std": round(max(0.1, strip_width_std), 2),
            "watermark_variance": round(wm_var, 2),
            "watermark_dynamic_range": round(wm_dyn, 2),
            "watermark_laplacian_var": round(wm_lap, 2),
            "watermark_entropy": round(wm_ent, 3),
            "color_hue_dev": round(hue_dev, 2),
            "color_sat_mean": round(s_mean, 2),
            "color_sat_std": round(max(1.0, s_std), 2),
            "color_val_mean": round(v_mean, 2),
            "color_delta_e": round(delta_e, 2),
            "microtexture_edge_density": round(edge_dens, 4),
            "denomination_code": d_code,
            "is_genuine": 0,
            "defect_class": defect,
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def train_and_save_counterfeit_models(
    df: pd.DataFrame, model_path: str = "models/counterfeit_ai_model.joblib"
) -> Dict[str, Any]:
    """
    Trains all 4 AI architectures (Ensemble, MLP Neural Network, Random Forest, GBDT)
    on the synthesized forensic counterfeit dataset and serializes weights.
    """
    X = np.array(df[FORENSIC_FEATURE_COLUMNS].values, dtype=np.float32)
    y = np.array(df["is_genuine"].values, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    detector = CounterfeitDetectorAI(active_model_type="ensemble")
    metrics = detector.train_and_evaluate(X_train, y_train, X_test, y_test)
    detector.save(model_path)

    print(f"[OK] Counterfeit AI Model successfully trained and saved to: {model_path}")
    for model_name in ["ensemble", "mlp", "rf", "gbdt"]:
        m = metrics[model_name]
        print(f"     -> [{model_name.upper():<8}] Accuracy: {m['accuracy']*100:.2f}% | "
              f"Precision: {m['precision']*100:.2f}% | Recall: {m['recall']*100:.2f}% | "
              f"F1: {m['f1_score']*100:.2f}% | ROC-AUC: {m['roc_auc']:.4f}")
    return metrics


def render_synthetic_banknote(
    denomination: str = "500",
    has_continuous_thread: bool = True,
    has_watermark: bool = True,
    corrupt_color: bool = False,
    bad_aspect_ratio: bool = False,
) -> np.ndarray:
    """
    Renders a realistic synthetic banknote image containing all spatial features:
      - Denomination substrate color and micro-texture.
      - Decorative security guilloche patterns and border frames.
      - Mahatma Gandhi latent watermark window (soft gradient portrait representation).
      - Vertical metallic security thread (continuous or defective broken thread).
      - Denomination numerals and Reserve Bank promise clause text.

    Parameters:
        denomination (str): '100', '200', '500', or '2000'.
        has_continuous_thread (bool): False to simulate broken/painted counterfeit thread.
        has_watermark (bool): False to simulate flat unwatermarked counterfeit paper.
        corrupt_color (bool): True to simulate wrong toner/ink hue counterfeit.
        bad_aspect_ratio (bool): True to simulate non-standard note dimensions.

    Returns:
        np.ndarray: BGR image of the banknote.
    """
    denom_spec = DENOMINATION_SPECS.get(denomination, DENOMINATION_SPECS["500"])
    ar = denom_spec["aspect_ratio"]
    h = CANVAS_HEIGHT
    w = round(h * ar)
    if bad_aspect_ratio:
        w = round(h * 1.45)  # Distorted aspect ratio (1.45 vs expected >=2.15)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Base BGR color palette according to denomination
    color_schemes: Dict[str, Tuple[int, int, int]] = {
        "100": (215, 175, 185),   # BGR: Lavender (Blueish Violet)
        "200": (70, 205, 245),    # BGR: Bright Warm Yellow
        "500": (175, 192, 180),   # BGR: Stone Olive Grey
        "2000": (170, 105, 225),  # BGR: Vivid Magenta
    }
    base_bgr = color_schemes.get(denomination, (175, 192, 180))

    if corrupt_color:
        # Deliberate counterfeit: print with distorted cyan-green ink
        base_bgr = (240, 160, 40)

    # 1. Fill base substrate
    canvas[:] = base_bgr

    # 2. Add subtle banknote paper grain
    noise = np.random.normal(0, 5.0, (h, w, 3)).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3. Outer security border and decorative filigree
    border_color = tuple(max(0, c - 50) for c in base_bgr)
    accent_color = tuple(max(0, c - 80) for c in base_bgr)

    cv2.rectangle(canvas, (8, 8), (w - 9, h - 9), border_color, 2)
    cv2.rectangle(canvas, (14, 14), (w - 15, h - 15), border_color, 1)

    # Corner decorative rosettes
    for cx, cy in [(25, 25), (w - 26, 25), (25, h - 26), (w - 26, h - 26)]:
        cv2.circle(canvas, (cx, cy), 10, accent_color, 1)
        cv2.circle(canvas, (cx, cy), 4, border_color, -1)

    # 4. Bank Title and Central Promise
    cv2.putText(
        canvas,
        "RESERVE BANK OF INDIA",
        (int(w * 0.32), 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GUARANTEED BY THE CENTRAL GOVERNMENT",
        (int(w * 0.34), 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"I PROMISE TO PAY THE BEARER THE SUM OF {denomination} RUPEES",
        (int(w * 0.38), 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    # 5. Large Value Numerals
    cv2.putText(
        canvas,
        f"{denomination}",
        (int(w * 0.76), int(h * 0.82)),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{denomination}",
        (int(w * 0.05), int(h * 0.22)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    # 6. Watermark Window (Mahatma Gandhi Portrait Region: x ~ 60 to 180, y ~ 50 to 250)
    wm_x1, wm_x2 = int(w * 0.10), int(w * 0.30)
    wm_y1, wm_y2 = int(h * 0.18), int(h * 0.82)
    wm_cx = (wm_x1 + wm_x2) // 2
    wm_cy = (wm_y1 + wm_y2) // 2

    # Draw faint watermark frame
    cv2.ellipse(canvas, (wm_cx, wm_cy), (45, 75), 0, 0, 360, tuple(min(255, c + 20) for c in base_bgr), -1)

    if has_watermark:
        # Authentic latent multi-tone watermark with multi-layer shaded ellipses
        for r_step in range(12, 42, 6):
            shade = tuple(max(0, c - int(15 + r_step * 0.8)) for c in base_bgr)
            cv2.ellipse(canvas, (wm_cx, wm_cy - 8), (r_step, int(r_step * 1.5)), 0, 0, 360, shade, 1, cv2.LINE_AA)
        # Latent portrait profile contours
        cv2.ellipse(canvas, (wm_cx + 4, wm_cy - 12), (18, 26), 25, 0, 360, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.circle(canvas, (wm_cx - 3, wm_cy - 16), 4, (40, 40, 40), 1)  # Eyeglasses
        cv2.circle(canvas, (wm_cx + 8, wm_cy - 16), 4, (40, 40, 40), 1)
        cv2.line(canvas, (wm_cx + 1, wm_cy - 16), (wm_cx + 4, wm_cy - 16), (40, 40, 40), 1)
        cv2.putText(
            canvas,
            denomination,
            (wm_cx - 15, wm_cy + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    # 7. Security Thread Strip (x ~ 250 to 275, y ~ 15 to 285)
    thread_x = int(w * 0.43)
    thread_w = 12

    if has_continuous_thread:
        # Authentic continuous metallic ribbon with color luster
        metallic_color = (35, 45, 30)  # Dark embedded ribbon
        cv2.rectangle(
            canvas,
            (thread_x, 15),
            (thread_x + thread_w, h - 15),
            metallic_color,
            -1,
        )
        # Add micro-lettering RBI inscriptions along the strip
        for y_pos in range(35, h - 25, 30):
            cv2.putText(
                canvas,
                "RBI",
                (thread_x + 1, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.22,
                (180, 220, 190),
                1,
                cv2.LINE_AA,
            )
    else:
        # Counterfeit defective strip: discontinuous breaks (<50% vertical presence)
        segment_len = 22
        gap_len = 45
        curr_y = 20
        while curr_y < h - 25:
            cv2.rectangle(
                canvas,
                (thread_x, curr_y),
                (thread_x + thread_w, min(h - 15, curr_y + segment_len)),
                (40, 50, 40),
                -1,
            )
            curr_y += segment_len + gap_len

    # 8. Ashoka Pillar Emblem on Right
    emblem_x = int(w * 0.88)
    emblem_y = int(h * 0.38)
    cv2.circle(canvas, (emblem_x, emblem_y), 16, accent_color, 2)
    cv2.circle(canvas, (emblem_x, emblem_y), 6, accent_color, -1)

    return canvas


def generate_camera_scene(note_img: np.ndarray) -> np.ndarray:
    """
    Embeds a banknote into a simulated 800x600 camera frame with wooden desk background,
    perspective tilt, and shadows to test the 4-point perspective transform & contour detector.
    """
    frame_w, frame_h = 800, 600
    scene = np.full((frame_h, frame_w, 3), 45, dtype=np.uint8)

    # Simulated wood grain background
    for y in range(0, frame_h, 4):
        val = int(50 + 20 * np.sin(y * 0.04))
        scene[y : y + 4, :] = (val - 15, val, val + 15)

    # Note corners with realistic camera tilt
    nh, nw = note_img.shape[:2]
    src_pts = np.array(
        [[0, 0], [nw, 0], [nw, nh], [0, nh]],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [[120, 140], [670, 110], [710, 460], [150, 490]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_note = cv2.warpPerspective(note_img, M, (frame_w, frame_h))

    # Binary mask of warped note
    gray_warped = cv2.cvtColor(warped_note, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)

    # Combine scene background with warped note
    bg = cv2.bitwise_and(scene, scene, mask=inv_mask)
    fg = cv2.bitwise_and(warped_note, warped_note, mask=mask)
    combined = cv2.add(bg, fg)

    return combined


def generate_all_artifacts() -> None:
    """
    Master pipeline creating directories, dataset, trained model, and synthetic test notes.
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    print("=" * 65)
    print("STEP 1: Synthesizing Banknote Feature Dataset (CO4)")
    print("=" * 65)
    df = generate_feature_dataset(n_samples_per_class=400)
    csv_path = os.path.join("data", "currency_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated {len(df)} samples across 4 denominations: {csv_path}")

    print("\n" + "=" * 65)
    print("STEP 2: Training & Serializing Decision Tree Denomination Classifier")
    print("=" * 65)
    model_path = os.path.join("models", "denomination_tree.joblib")
    train_and_save_baseline_model(df, model_path=model_path)

    print("\n" + "=" * 65)
    print("STEP 2B: Synthesizing Forensic Counterfeit Dataset & Training AI Suite")
    print("=" * 65)
    cf_df = generate_counterfeit_dataset(n_samples=3200)
    cf_csv_path = os.path.join("data", "counterfeit_detection_dataset.csv")
    cf_df.to_csv(cf_csv_path, index=False)
    print(f"[OK] Generated {len(cf_df)} forensic samples: {cf_csv_path}")

    cf_model_path = os.path.join("models", "counterfeit_ai_model.joblib")
    train_and_save_counterfeit_models(cf_df, model_path=cf_model_path)

    print("\n" + "=" * 65)
    print("STEP 3: Generating Synthetic Visual Banknote Samples (CV & Expert System)")
    print("=" * 65)

    samples_to_create = [
        ("genuine_100.png", "100", True, True, False, False),
        ("genuine_200.png", "200", True, True, False, False),
        ("genuine_500.png", "500", True, True, False, False),
        ("genuine_2000.png", "2000", True, True, False, False),
        ("counterfeit_broken_thread.png", "500", False, True, False, False),
        ("counterfeit_blank_watermark.png", "500", True, False, False, False),
        ("counterfeit_wrong_color.png", "500", True, True, True, False),
        ("counterfeit_bad_aspect_ratio.png", "500", True, True, False, True),
    ]

    for filename, denom, cont_thread, wm, bad_col, bad_ar in samples_to_create:
        img = render_synthetic_banknote(
            denomination=denom,
            has_continuous_thread=cont_thread,
            has_watermark=wm,
            corrupt_color=bad_col,
            bad_aspect_ratio=bad_ar,
        )
        out_path = os.path.join("samples", filename)
        cv2.imwrite(out_path, img)
        print(f"  -> Created {out_path}")

    # Generate a full camera perspective scene for genuine ₹500
    gen_500 = cv2.imread(os.path.join("samples", "genuine_500.png"))
    assert gen_500 is not None, "Failed to load samples/genuine_500.png"
    camera_scene = generate_camera_scene(gen_500)
    camera_scene_path = os.path.join("samples", "camera_desk_scene_500.png")
    cv2.imwrite(camera_scene_path, camera_scene)
    print(f"  -> Created full desk camera scene: {camera_scene_path}")

    print("\n[COMPLETE] All sample data, models, and images generated successfully!")


if __name__ == "__main__":
    generate_all_artifacts()
