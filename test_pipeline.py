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
    extract_forensic_features,
)
from classifier import DenominationClassifier, FEATURE_COLUMNS
from counterfeit_classifier import CounterfeitDetectorAI
from expert_system import VerificationExpertSystem, Verdict


def test_full_pipeline() -> bool:
    print("=" * 70)
    print("STARTING END-TO-END SYSTEM VERIFICATION WITH COUNTERFEIT AI SUITE")
    print("=" * 70)

    # 1. Load trained denomination classifier
    model_path = os.path.join("models", "denomination_tree.joblib")
    assert os.path.exists(model_path), f"Denomination model not found at {model_path}"
    classifier = DenominationClassifier()
    assert classifier.load(model_path), "Failed to load denomination model weights."
    print("[PASS] Denomination Decision Tree loaded successfully.")

    # 2. Load trained Counterfeit AI Detector
    cf_model_path = os.path.join("models", "counterfeit_ai_model.joblib")
    assert os.path.exists(cf_model_path), f"Counterfeit model not found at {cf_model_path}"
    cf_detector = CounterfeitDetectorAI(active_model_type="ensemble")
    assert cf_detector.load(cf_model_path), "Failed to load Counterfeit AI model weights."
    print("[PASS] Counterfeit AI Suite (Ensemble/MLP/RF/GBDT) loaded successfully.")

    # 3. Initialize Neuro-Symbolic Expert System
    expert_system = VerificationExpertSystem(tolerance_pct=0.08, counterfeit_detector=cf_detector)
    print(f"[PASS] Neuro-Symbolic Expert System initialized. PEAS Agent: {expert_system.peas_model.agent_name}")

    # 4. Test genuine notes
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
        assert corners is not None and aspect_ratio is not None, f"Contour detection failed for {filename}"
        warped, _ = four_point_transform(img, corners)

        # Extract features
        color_feats = extract_color_features(warped)
        feat_vec = classifier.build_feature_vector(color_feats, aspect_ratio)
        pred_denom, confidence, proba_dict = classifier.predict(feat_vec)

        # ROIs & Deep Analysis
        rois = extract_rois(warped)
        strip_metrics = analyze_security_strip(rois["security_strip"]["patch"], n_clusters=3)
        wm_metrics = analyze_watermark(rois["watermark"]["patch"])

        forensic_features = extract_forensic_features(
            warped_note=warped,
            rois=rois,
            strip_analysis=strip_metrics,
            watermark_analysis=wm_metrics,
            color_features=color_feats,
            aspect_ratio=aspect_ratio,
            predicted_denomination=pred_denom,
        )

        # Neuro-Symbolic Expert System Evaluation
        report = expert_system.evaluate(
            predicted_denomination=pred_denom,
            classifier_confidence=confidence,
            aspect_ratio=aspect_ratio,
            security_strip_metrics=strip_metrics,
            watermark_metrics=wm_metrics,
            color_features=color_feats,
            forensic_features=forensic_features,
            ai_model_type="ensemble",
        )

        assert report.ai_verdict is not None
        print(f"Testing {filename}:")
        print(f"  -> Predicted Denomination: INR {pred_denom} (Confidence: {confidence*100:.1f}%)")
        print(f"  -> AI Counterfeit Model: {report.ai_verdict.value} (Authenticity Prob: {report.ai_genuine_prob*100:.1f}%)")
        print(f"  -> Security Strip Continuity: {strip_metrics['continuity_pct']}%")
        print(f"  -> Watermark Entropy: {wm_metrics.get('entropy', 0.0):.2f} bits (Variance: {wm_metrics['variance']:.1f})")
        print(f"  -> Consensus Verdict: {report.verdict.value} (Passed {report.rules_passed_count}/{report.rules_total_count} rules)")
        print(f"  -> Status: {report.neuro_symbolic_status}")

        assert pred_denom == expected_denom, f"Expected {expected_denom}, got {pred_denom}"
        assert report.verdict == Verdict.GENUINE, f"Expected GENUINE for {filename}, got {report.verdict.value}"
        assert report.ai_verdict == Verdict.GENUINE, f"Expected AI GENUINE for {filename}, got {report.ai_verdict}"

    print("\n[PASS] All 4 Genuine Banknotes Verified with 100% Accuracy and GENUINE Verdict!\n")

    # 5. Test Counterfeit Scenarios
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
        assert corners is not None and aspect_ratio is not None, f"Contour detection failed for {filename}"
        warped, _ = four_point_transform(img, corners)

        color_feats = extract_color_features(warped)
        feat_vec = classifier.build_feature_vector(color_feats, aspect_ratio)
        pred_denom, confidence, _ = classifier.predict(feat_vec)

        rois = extract_rois(warped)
        strip_metrics = analyze_security_strip(rois["security_strip"]["patch"], n_clusters=3)
        wm_metrics = analyze_watermark(rois["watermark"]["patch"])

        forensic_features = extract_forensic_features(
            warped_note=warped,
            rois=rois,
            strip_analysis=strip_metrics,
            watermark_analysis=wm_metrics,
            color_features=color_feats,
            aspect_ratio=aspect_ratio,
            predicted_denomination=pred_denom,
        )

        report = expert_system.evaluate(
            predicted_denomination=pred_denom,
            classifier_confidence=confidence,
            aspect_ratio=aspect_ratio,
            security_strip_metrics=strip_metrics,
            watermark_metrics=wm_metrics,
            color_features=color_feats,
            forensic_features=forensic_features,
            ai_model_type="ensemble",
        )

        assert report.ai_verdict is not None
        print(f"Testing Counterfeit Case ({filename}):")
        print(f"  -> AI Counterfeit Model: {report.ai_verdict.value} (Authenticity Prob: {report.ai_genuine_prob*100:.1f}%)")
        print(f"  -> Consensus Verdict: {report.verdict.value}")
        failed_rule_ids = [r.rule_id for r in report.diagnostic_checklist if not r.passed]
        print(f"  -> Failed Rules: {failed_rule_ids}")

        assert report.verdict in (Verdict.COUNTERFEIT, Verdict.SUSPICIOUS), f"Expected Counterfeit/Suspicious, got {report.verdict}"
        assert report.ai_verdict == Verdict.COUNTERFEIT, f"Expected AI COUNTERFEIT for {filename}, got {report.ai_verdict}"
        assert expected_failed_rule in failed_rule_ids, f"Expected {expected_failed_rule} to fail, but failed: {failed_rule_ids}"

    print("\n[PASS] All Counterfeit Scenarios Flagged by BOTH AI Model and Forward-Chaining Engine!\n")

    # 5. Test Full Perspective Camera Scene
    desk_scene_path = os.path.join("samples", "camera_desk_scene_500.png")
    scene_img = cv2.imread(desk_scene_path)
    assert scene_img is not None, f"Could not read {desk_scene_path}"
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
