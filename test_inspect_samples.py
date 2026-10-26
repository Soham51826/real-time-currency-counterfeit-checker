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
    extract_forensic_features,
)
from classifier import DenominationClassifier
from counterfeit_classifier import CounterfeitDetectorAI
from expert_system import VerificationExpertSystem, Verdict

def inspect_all_samples(samples_dir: str = "samples"):
    print("=" * 100)
    print("CURRENCYGUARD AI: ADVANCED NEURO-SYMBOLIC SAMPLE INSPECTION UTILITY")
    print("=" * 100)
    
    if not os.path.exists(samples_dir):
        print(f"[ERROR] Samples directory not found at: {samples_dir}")
        return
        
    denom_model_path = os.path.join("models", "denomination_tree.joblib")
    classifier = DenominationClassifier()
    if os.path.exists(denom_model_path):
        classifier.load(denom_model_path)
    else:
        print(f"[WARNING] Denomination model weights not found at {denom_model_path}.")

    counterfeit_model_path = os.path.join("models", "counterfeit_ai_model.joblib")
    counterfeit_ai = CounterfeitDetectorAI(model_type="ensemble")
    if os.path.exists(counterfeit_model_path):
        counterfeit_ai.load(counterfeit_model_path)
        print("[PASS] Counterfeit AI Suite loaded successfully.")
    else:
        print(f"[WARNING] Counterfeit AI model weights not found at {counterfeit_model_path}.")
        counterfeit_ai = None

    expert_system = VerificationExpertSystem(tolerance_pct=0.08, counterfeit_ai=counterfeit_ai)

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

        forensic_features = extract_forensic_features(
            warped_note=warped,
            pred_denom=pred_denom,
            aspect_ratio=aspect_ratio,
            security_strip_metrics=strip_metrics,
            watermark_metrics=wm_metrics,
            color_features=color_feats,
            rois=rois,
        )

        report = expert_system.evaluate(
            predicted_denomination=pred_denom,
            classifier_confidence=pred_conf,
            aspect_ratio=aspect_ratio,
            security_strip_metrics=strip_metrics,
            watermark_metrics=wm_metrics,
            color_features=color_feats,
            forensic_features=forensic_features,
            scan_mode=scan_mode,
        )

        ai_prob = report.ai_authenticity_prob * 100.0 if report.ai_authenticity_prob is not None else 0.0
        ai_verdict = report.ai_verdict.value if isinstance(report.ai_verdict, Verdict) else (str(report.ai_verdict) if report.ai_verdict else "N/A")
        status_tag = f"[{report.verdict.value}]"
        
        print(f"[{idx:02d}] {fname:<30} | INR {pred_denom:<4} ({pred_conf*100:5.1f}%) | "
              f"AI: {ai_verdict:<11} ({ai_prob:5.1f}%) | "
              f"Rules: {report.rules_passed_count}/{report.rules_total_count} | "
              f"Consensus: {status_tag}")
        if report.verdict == Verdict.COUNTERFEIT:
            print(f"     -> Failed Rules: {[r.rule_name for r in report.rules if not r.passed]}")

    print("\n" + "=" * 100)
    print("INSPECTION COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    inspect_all_samples()
