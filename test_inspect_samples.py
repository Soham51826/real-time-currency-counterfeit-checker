"""
Script: test_inspect_samples.py
Interactive CLI utility to inspect currency banknote samples and test individual features.
Runs full computer vision rectification, K-Means security thread segmentation,
Decision Tree classification, and forward-chaining expert system rule verification.
"""

import os
import sys
import cv2
import numpy as np

from vision_pipeline import (
    detect_note_contour,
    four_point_transform,
    auto_orient_note,
    apply_color_constancy,
    extract_rois,
    extract_color_features,
    analyze_security_strip,
    analyze_watermark,
)
from classifier import DenominationClassifier
from expert_system import VerificationExpertSystem, Verdict

def inspect_all_samples(samples_dir: str = "samples"):
    print("=" * 85)
    print("CURRENCYGUARD AI: SAMPLE INSPECTION UTILITY")
    print("=" * 85)
    
    if not os.path.exists(samples_dir):
        print(f"[ERROR] Samples directory not found at: {samples_dir}")
        return
        
    model_path = os.path.join("models", "denomination_tree.joblib")
    classifier = DenominationClassifier()
    if os.path.exists(model_path):
        classifier.load(model_path)
    else:
        print(f"[WARNING] Model weights not found at {model_path}.")

    expert_system = VerificationExpertSystem(tolerance_pct=0.08)

    sample_files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"Discovered {len(sample_files)} sample files in '{samples_dir}/':\n")

    for idx, fname in enumerate(sample_files, start=1):
        fpath = os.path.join(samples_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"[{idx:02d}] Could not read: {fname}")
            continue

        corners, _, aspect_ratio = detect_note_contour(img)
        is_perspective = "desk_scene" in fname

        if corners is not None:
            warped, _ = four_point_transform(img, corners)
            warped, rot_deg, rot_reason = auto_orient_note(warped)
        else:
            warped = cv2.resize(img, (600, 300))
            is_perspective = False

        warped_wb = apply_color_constancy(warped)
        color_feats = extract_color_features(warped)
        color_feats_wb = extract_color_features(warped_wb)
        
        color_feats["wb_h_mean"] = color_feats_wb["h_mean"]
        color_feats["wb_s_mean"] = color_feats_wb["s_mean"]
        color_feats["wb_v_mean"] = color_feats_wb["v_mean"]
        pred_denom, pred_conf, proba_dict = classifier.predict_robust(color_feats, aspect_ratio=aspect_ratio)
        rois = extract_rois(warped)

        scan_mode = "perspective" if is_perspective else "normal"
        strip_metrics = analyze_security_strip(rois["security_strip"]["patch"], n_clusters=3)
        wm_metrics = analyze_watermark(rois["watermark"]["patch"])

        report = expert_system.evaluate(
            predicted_denomination=pred_denom,
            classifier_confidence=pred_conf,
            aspect_ratio=aspect_ratio,
            security_strip_metrics=strip_metrics,
            watermark_metrics=wm_metrics,
            color_features=color_feats,
            scan_mode=scan_mode,
        )

        status_tag = f"[{report.verdict.value}]"
        failed_count = report.rules_total_count - report.rules_passed_count
        print(f"[{idx:02d}] {fname:<32} -> INR {pred_denom:<4} ({pred_conf*100:5.1f}%) | "
              f"Strip: {strip_metrics['continuity_pct']:5.1f}% | "
              f"WM Var: {wm_metrics['variance']:6.1f} | "
              f"{status_tag:<12} (Failed Rules: {failed_count})")

    print("\n" + "=" * 85)
    print("INSPECTION COMPLETE")
    print("=" * 85)

if __name__ == "__main__":
    inspect_all_samples()
