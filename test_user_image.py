import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import cv2
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
from expert_system import VerificationExpertSystem

# Portable image path resolution: CLI arg -> user image -> default sample
default_sample = os.path.join("samples", "genuine_100.png")
captured_img = r"C:\Users\sahil\.gemini\antigravity-ide\brain\627b35ba-383d-4eb2-8ff4-4c86ccae4533\.user_uploaded\media_1792937667466.png"

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    img_path = sys.argv[1]
elif os.path.exists(captured_img):
    img_path = captured_img
else:
    img_path = default_sample

print(f"Loading image for verification: {img_path}")
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")
print("Image shape:", img.shape)

corners, edges, aspect_ratio = detect_note_contour(img)
print(f"Corners detected: {corners is not None}, AR: {aspect_ratio}")

if corners is not None:
    warped, _ = four_point_transform(img, corners)
    warped, rot_deg, reason = auto_orient_note(warped)
    print(f"Warped shape: {warped.shape}, Rotated: {rot_deg} deg ({reason})")
else:
    warped = cv2.resize(img, (600, 300))
    aspect_ratio = img.shape[1] / img.shape[0]

warped_wb = apply_color_constancy(warped)
color_feats = extract_color_features(warped)
color_feats_wb = extract_color_features(warped_wb)
color_feats["wb_h_mean"] = color_feats_wb["h_mean"]
color_feats["wb_s_mean"] = color_feats_wb["s_mean"]
color_feats["wb_v_mean"] = color_feats_wb["v_mean"]

clf = DenominationClassifier()
clf.load("models/denomination_tree.joblib")
denom, conf, probas = clf.predict_robust(color_feats, aspect_ratio=aspect_ratio)
print(f"\n[CLASSIFICATION] Predicted Denomination: INR {denom} (Confidence: {conf*100:.1f}%)")
print("Class probabilities:", probas)

rois = extract_rois(warped)
strip = analyze_security_strip(rois["security_strip"]["patch"], n_clusters=3)
wm = analyze_watermark(rois["watermark"]["patch"])
print(f"\n[SECURITY THREAD] Continuity: {strip['continuity_pct']:.1f}%, Windows: {strip.get('num_windows', 0)}")
print(f"[WATERMARK] Variance: {wm['variance']:.1f}, Has Latent: {wm['has_latent_features']}, Dynamic Range: {wm.get('dynamic_range', 0)}")

exp = VerificationExpertSystem(tolerance_pct=0.10)
report = exp.evaluate(
    predicted_denomination=denom,
    classifier_confidence=conf,
    aspect_ratio=aspect_ratio,
    security_strip_metrics=strip,
    watermark_metrics=wm,
    color_features=color_feats,
    scan_mode="perspective"
)
print(f"\n[EXPERT SYSTEM VERDICT] {report.verdict.value} (Confidence: {report.overall_confidence*100:.1f}%)")
for r in report.diagnostic_checklist:
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {r.rule_name} (Severity: {r.severity.value}, Penalty: -{r.score_penalty}): {r.observed_value} vs expected {r.expected_threshold}\n       Explanation: {r.explanation}")
