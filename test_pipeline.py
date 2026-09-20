"""
Verification Test Script: End-to-End Currency Inspection & Authenticity
========================================================================
Validates all 4 modules:
  1. Vision Pipeline: Contour detection, 4-point perspective warp, ROI extraction.
  2. K-Means Clustering: Security strip segmentation & continuity percentage calculation.
  3. Decision Tree: Denomination classification & posterior probabilities.
  4. Expert System: Forward-chaining inference engine on genuine and counterfeit edge cases.
"""

import os
import sys
import cv2
import numpy as np

from vision_pipeline import (
    detect_note_contour,
    four_point_transform,
    extract_rois,
    extract_color_features,
    analyze_security_strip,
    analyze_watermark,
    draw_roi_overlays,
)
from classifier import DenominationClassifier, FEATURE_COLUMNS
from expert_system import VerificationExpertSystem, Verdict


def test_full_pipeline() -> bool:
    print("=" * 70)
    print("STARTING END-TO-END SYSTEM VERIFICATION")
    print("=" * 70)

    # 1. Load trained classifier
    model_path = os.path.join("models", "denomination_tree.joblib")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    classifier = DenominationClassifier()
    assert classifier.load(model_path), "Failed to load model weights."
    print("[PASS] Model loaded successfully.")
    print("Decision Tree Root Logic:\n" + classifier.explain_decision_rules()[:300] + "...\n")

    # 2. Initialize Expert System
    expert_system = VerificationExpertSystem(tolerance_pct=0.05)
    print(f"[PASS] Expert System initialized. PEAS Agent: {expert_system.peas_model.agent_name}")

    # 3. Test genuine notes
    genuine_notes = [
        ("genuine_100.png", "100"),
        ("genuine_200.png", "200"),
        ("genuine_500.png", "500"),
        ("genuine_2000.png", "2000"),
    ]

    for filename, expected_denom in genuine_notes:
        img_path = os.path.join("samples", filename)
        img = cv2.imread(img_path)
        assert img is not None, f"Could not read {img_path}"

        # Detect contour & warp
        corners, _, aspect_ratio = detect_note_contour(img)
        warped, _ = four_point_transform(img, corners)

        # Extract features
        color_feats = extract_color_features(warped)
        feat_vec = classifier.build_feature_vector(color_feats, aspect_ratio)
        pred_denom, confidence, proba_dict = classifier.predict(feat_vec)

        # ROIs & Deep Analysis
        rois = extract_rois(warped)
        strip_metrics = analyze_security_strip(rois["security_strip"]["patch"], n_clusters=3)
        wm_metrics = analyze_watermark(rois["watermark"]["patch"])

        # Expert System
        report = expert_system.evaluate(
            predicted_denomination=pred_denom,
            classifier_confidence=confidence,
            aspect_ratio=aspect_ratio,
            security_strip_metrics=strip_metrics,
            watermark_metrics=wm_metrics,
            color_features=color_feats,
        )

        print(f"Testing {filename}:")
        print(f"  -> Predicted Denomination: INR {pred_denom} (Confidence: {confidence*100:.1f}%)")
        print(f"  -> Security Strip Continuity: {strip_metrics['continuity_pct']}%")
        print(f"  -> Watermark Variance: {wm_metrics['variance']} (Latent: {wm_metrics['has_latent_features']})")
        print(f"  -> Expert System Verdict: {report.verdict.value} (Passed {report.rules_passed_count}/{report.rules_total_count} rules)")

        assert pred_denom == expected_denom, f"Expected {expected_denom}, got {pred_denom}"
        assert report.verdict == Verdict.GENUINE, f"Expected GENUINE for {filename}, got {report.verdict.value}"

    print("\n[PASS] All 4 Genuine Banknotes Verified with 100% Accuracy and GENUINE Verdict!\n")

    # 4. Test Counterfeit Scenarios
    counterfeit_cases = [
        ("counterfeit_broken_thread.png", "RULE_2_SECURITY_THREAD"),
        ("counterfeit_blank_watermark.png", "RULE_4_WATERMARK_CONTRAST"),
        ("counterfeit_wrong_color.png", "RULE_3_COLOR_CONSISTENCY"),
        ("counterfeit_bad_aspect_ratio.png", "RULE_1_DIMENSION"),
    ]

    for filename, expected_failed_rule in counterfeit_cases:
        img_path = os.path.join("samples", filename)
        img = cv2.imread(img_path)
        assert img is not None

        corners, _, aspect_ratio = detect_note_contour(img)
        warped, _ = four_point_transform(img, corners)

        color_feats = extract_color_features(warped)
        feat_vec = classifier.build_feature_vector(color_feats, aspect_ratio)
        pred_denom, confidence, _ = classifier.predict(feat_vec)

        rois = extract_rois(warped)
        strip_metrics = analyze_security_strip(rois["security_strip"]["patch"], n_clusters=3)
        wm_metrics = analyze_watermark(rois["watermark"]["patch"])

        report = expert_system.evaluate(
            predicted_denomination=pred_denom,
            classifier_confidence=confidence,
            aspect_ratio=aspect_ratio,
            security_strip_metrics=strip_metrics,
            watermark_metrics=wm_metrics,
            color_features=color_feats,
        )

        print(f"Testing Counterfeit Case ({filename}):")
        print(f"  -> Verdict: {report.verdict.value}")
        failed_rule_ids = [r.rule_id for r in report.diagnostic_checklist if not r.passed]
        print(f"  -> Failed Rules: {failed_rule_ids}")

        assert report.verdict in (Verdict.COUNTERFEIT, Verdict.SUSPICIOUS), f"Expected Counterfeit/Suspicious, got {report.verdict}"
        assert expected_failed_rule in failed_rule_ids, f"Expected {expected_failed_rule} to fail, but failed: {failed_rule_ids}"

    print("\n[PASS] All Counterfeit Scenarios Successfully Flagged by Forward-Chaining Engine!\n")

    # 5. Test Full Perspective Camera Scene
    desk_scene_path = os.path.join("samples", "camera_desk_scene_500.png")
    scene_img = cv2.imread(desk_scene_path)
    corners, edges, aspect_ratio = detect_note_contour(scene_img)
    assert corners is not None, "Failed to detect note contour in camera scene."
    warped, _ = four_point_transform(scene_img, corners)
    assert warped.shape == (300, 600, 3), f"Unexpected warped shape {warped.shape}"
    print(f"[PASS] Successfully detected banknote contour in tilted camera desk scene!")
    print(f"       Detected Corners:\n{corners}")

    print("\n" + "=" * 70)
    print("ALL MODULE VERIFICATION TESTS PASSED (100% SUCCESS)!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
