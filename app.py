"""
Module 4: User Interface & Forensic Dashboard
==============================================
Production-grade, modern Streamlit dashboard for real-time banknote denomination
recognition and counterfeit feature verification.

Academic Curriculum Standards Mapped:
  - CO1: Agent Modeling (Formal PEAS setup viewer).
  - CO3: Rule-Based Expert System (Forward-chaining inference engine diagnostic checklist).
  - CO4: Machine Learning (Supervised Decision Tree Classifier & Unsupervised K-Means Clustering).
  - CV: 4-Point Perspective Transform, Adaptive Canny, Color Moments & Histogram Extraction.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple, List, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from vision_pipeline import (
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    detect_note_contour,
    four_point_transform,
    extract_rois,
    extract_color_features,
    analyze_security_strip,
    analyze_watermark,
    draw_roi_overlays,
)
from classifier import (
    DenominationClassifier,
    DENOMINATION_SPECS,
    FEATURE_COLUMNS,
)
from expert_system import (
    VerificationExpertSystem,
    Verdict,
    RuleSeverity,
    AgentPEAS,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CurrencyGuard AI - Banknote & Counterfeit Checker",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Futuristic Cyber-Forensic CSS Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Base Theme Adjustments */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(13, 20, 36, 0.95) 0%, rgba(7, 10, 19, 0.98) 90%);
}

/* Main Header Card */
.header-container {
    background: linear-gradient(135deg, rgba(25, 36, 64, 0.6) 0%, rgba(14, 20, 38, 0.8) 100%);
    border: 1px solid rgba(0, 210, 255, 0.2);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(12px);
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 50%, #00F59B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}

.header-badges {
    display: flex;
    gap: 10px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.curriculum-tag {
    background: rgba(0, 210, 255, 0.12);
    border: 1px solid rgba(0, 210, 255, 0.3);
    color: #64D2FF;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

/* Glassmorphic Cards */
.glass-card {
    background: rgba(18, 26, 47, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(10px);
}

/* Verdict Banners */
.verdict-banner-genuine {
    background: linear-gradient(135deg, rgba(0, 245, 155, 0.15) 0%, rgba(16, 185, 129, 0.3) 100%);
    border: 2px solid #00F59B;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 0 35px rgba(0, 245, 155, 0.25);
    animation: pulseGreen 3s infinite alternate;
}

.verdict-banner-counterfeit {
    background: linear-gradient(135deg, rgba(255, 51, 102, 0.2) 0%, rgba(239, 68, 68, 0.35) 100%);
    border: 2px solid #FF3366;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255, 51, 102, 0.35);
    animation: pulseRed 2s infinite alternate;
}

.verdict-banner-suspicious {
    background: linear-gradient(135deg, rgba(255, 184, 0, 0.15) 0%, rgba(245, 158, 11, 0.3) 100%);
    border: 2px solid #FFB800;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 0 35px rgba(255, 184, 0, 0.25);
}

.verdict-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

/* Rule Diagnostic Cards */
.rule-card-pass {
    background: rgba(16, 185, 129, 0.08);
    border-left: 4px solid #00F59B;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.rule-card-fail {
    background: rgba(239, 68, 68, 0.1);
    border-left: 4px solid #FF3366;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.rule-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    font-size: 0.95rem;
    color: #E2E8F0;
}

.rule-badge-pass {
    background: #00F59B;
    color: #052E16;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 2px 10px;
    border-radius: 12px;
}

.rule-badge-fail {
    background: #FF3366;
    color: #FFFFFF;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 2px 10px;
    border-radius: 12px;
}

.rule-details {
    font-size: 0.86rem;
    color: #94A3B8;
    margin-top: 6px;
    line-height: 1.4;
}

.rule-metric-row {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #CBD5E1;
}

/* Quick stat chips */
.stat-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 8px 14px;
    border-radius: 10px;
    text-align: center;
}
.stat-pill-num {
    font-size: 1.3rem;
    font-weight: 700;
    color: #00D2FF;
}
.stat-pill-lbl {
    font-size: 0.7rem;
    color: #94A3B8;
    text-transform: uppercase;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Application Resources (Model & Expert System)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system_resources() -> Tuple[DenominationClassifier, VerificationExpertSystem]:
    """Loads the trained Decision Tree model and initializes the Expert System."""
    model_path = os.path.join("models", "denomination_tree.joblib")
    classifier = DenominationClassifier(max_depth=6, criterion="entropy")

    # If model doesn't exist, generate training data & train on the fly
    if not os.path.exists(model_path):
        from generate_sample_data import generate_all_artifacts
        generate_all_artifacts()

    classifier.load(model_path)
    expert_system = VerificationExpertSystem(tolerance_pct=0.05)
    return classifier, expert_system


classifier, expert_system = load_system_resources()


# -----------------------------------------------------------------------------
# Header Presentation
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <div class="header-title">CURRENCYGUARD AI</div>
        <div style="color: #94A3B8; font-size: 1.05rem; margin-top: 6px;">
            Real-Time Banknote Recognition & Counterfeit Feature Verification Engine
        </div>
        <div class="header-badges">
            <span class="curriculum-tag">CO1: PEAS Agent Modeling</span>
            <span class="curriculum-tag">CO3: Forward-Chaining Expert System</span>
            <span class="curriculum-tag">CO4: Decision Tree & K-Means Clustering</span>
            <span class="curriculum-tag">Computer Vision: Homography & Edge Contours</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar Configuration & Input Selector
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Input Source Selection")
    input_mode = st.radio(
        "Choose Inspection Mode:",
        ["📁 Preset Test Samples", "📤 Upload Image", "📸 Webcam Capture"],
        index=0,
    )

    selected_image: Optional[np.ndarray] = None
    input_source_name: str = ""

    if input_mode == "📁 Preset Test Samples":
        preset_options = {
            "Genuine ₹100 (Lavender)": "samples/genuine_100.png",
            "Genuine ₹200 (Bright Yellow)": "samples/genuine_200.png",
            "Genuine ₹500 (Stone Grey)": "samples/genuine_500.png",
            "Genuine ₹2000 (Magenta)": "samples/genuine_2000.png",
            "Counterfeit: Broken Security Thread": "samples/counterfeit_broken_thread.png",
            "Counterfeit: Blank Watermark Window": "samples/counterfeit_blank_watermark.png",
            "Counterfeit: Altered Substrate Color": "samples/counterfeit_wrong_color.png",
            "Counterfeit: Distorted Aspect Ratio": "samples/counterfeit_bad_aspect_ratio.png",
            "Tilted Camera Scene (₹500 Desk View)": "samples/camera_desk_scene_500.png",
        }
        choice = st.selectbox("Select Sample Note:", list(preset_options.keys()))
        sample_path = preset_options[choice]
        if os.path.exists(sample_path):
            selected_image = cv2.imread(sample_path)
            input_source_name = choice

    elif input_mode == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Banknote Image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            selected_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            input_source_name = uploaded_file.name

    elif input_mode == "📸 Webcam Capture":
        cam_photo = st.camera_input("Position note horizontally in frame:")
        if cam_photo is not None:
            file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
            selected_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            input_source_name = "Live Camera Capture"

    st.markdown("---")
    st.markdown("### ⚙️ Inspection Thresholds")
    tolerance_slider = st.slider(
        "Aspect Ratio Tolerance (Rule 1)",
        min_value=0.03,
        max_value=0.15,
        value=0.05,
        step=0.01,
        format="±%0.2f",
    )
    expert_system.tolerance_pct = tolerance_slider

    strip_clusters = st.slider(
        "K-Means Clusters (Security Strip)",
        min_value=2,
        max_value=4,
        value=3,
        step=1,
    )

    st.markdown("---")
    # PEAS Model Popover / Expander in Sidebar
    with st.expander("📘 Formal PEAS Agent Architecture (CO1)", expanded=False):
        peas = expert_system.peas_model
        st.markdown(f"**Agent**: `{peas.agent_name}`")
        st.markdown("**Performance Measures (P):**")
        for p in peas.performance_measures:
            st.markdown(f"- *{p}*")
        st.markdown("**Environment (E):**")
        for e in peas.environment:
            st.markdown(f"- *{e}*")
        st.markdown("**Actuators (A):**")
        for a in peas.actuators:
            st.markdown(f"- *{a}*")
        st.markdown("**Sensors (S):**")
        for s in peas.sensors:
            st.markdown(f"- *{s}*")


# -----------------------------------------------------------------------------
# Main Analysis Pipeline Execution
# -----------------------------------------------------------------------------
if selected_image is None:
    st.info("👆 Please select a preset test sample, upload a banknote image, or take a webcam photo from the sidebar.")
    st.stop()

# Execution timer for real-time latency measurement
start_time = time.perf_counter()

# Step 1: Detect note contour & compute aspect ratio
detected_corners, edges_map, aspect_ratio = detect_note_contour(selected_image)

# Step 2: 4-point perspective warp to standard 600x300 canvas
if detected_corners is not None:
    warped_note, homography_matrix = four_point_transform(
        selected_image, detected_corners, CANVAS_WIDTH, CANVAS_HEIGHT
    )
else:
    warped_note = cv2.resize(selected_image, (CANVAS_WIDTH, CANVAS_HEIGHT))

# Step 3: Feature Extraction (Color moments & HSV distributions)
color_features = extract_color_features(warped_note)
feature_vec = classifier.build_feature_vector(color_features, aspect_ratio)

# Step 4: Supervised ML Denomination Prediction (CO4)
pred_denom, confidence, proba_dict = classifier.predict(feature_vec)

# Step 5: Region of Interest (ROI) extraction
rois = extract_rois(warped_note)

# Step 6: Unsupervised K-Means Security Strip Continuity (CO4)
strip_roi = rois["security_strip"]["patch"]
strip_analysis = analyze_security_strip(strip_roi, n_clusters=strip_clusters)

# Step 7: Watermark Latent Portrait Analysis
watermark_roi = rois["watermark"]["patch"]
watermark_analysis = analyze_watermark(watermark_roi)

# Step 8: Forward-Chaining Rule-Based Expert System (CO3)
report = expert_system.evaluate(
    predicted_denomination=pred_denom,
    classifier_confidence=confidence,
    aspect_ratio=aspect_ratio,
    security_strip_metrics=strip_analysis,
    watermark_metrics=watermark_analysis,
    color_features=color_features,
)

elapsed_ms = (time.perf_counter() - start_time) * 1000.0


# -----------------------------------------------------------------------------
# Top Row: Authenticity Verdict Banner & High-Level KPIs
# -----------------------------------------------------------------------------
denom_info = DENOMINATION_SPECS.get(pred_denom, DENOMINATION_SPECS["500"])

col_verdict, col_metrics = st.columns([1.6, 1.0])

with col_verdict:
    if report.verdict == Verdict.GENUINE:
        st.markdown(
            f"""
            <div class="verdict-banner-genuine">
                <div class="verdict-title" style="color: #00F59B;">✔ GENUINE BANKNOTE</div>
                <div style="font-size: 1.15rem; color: #E2E8F0; font-weight: 600;">
                    {denom_info['display_name']} • Authenticity Confirmed
                </div>
                <div style="font-size: 0.9rem; color: #A7F3D0; margin-top: 8px;">
                    {report.summary_explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif report.verdict == Verdict.COUNTERFEIT:
        st.markdown(
            f"""
            <div class="verdict-banner-counterfeit">
                <div class="verdict-title" style="color: #FF3366;">✖ COUNTERFEIT DETECTED</div>
                <div style="font-size: 1.15rem; color: #FEE2E2; font-weight: 600;">
                    Defective {denom_info['display_name']} • Security Standard Violation
                </div>
                <div style="font-size: 0.9rem; color: #FECACA; margin-top: 8px;">
                    {report.summary_explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-banner-suspicious">
                <div class="verdict-title" style="color: #FFB800;">⚠ SUSPICIOUS CURRENCY</div>
                <div style="font-size: 1.15rem; color: #FEF3C7; font-weight: 600;">
                    {denom_info['display_name']} • Manual Secondary Inspection Recommended
                </div>
                <div style="font-size: 0.9rem; color: #FDE68A; margin-top: 8px;">
                    {report.summary_explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_metrics:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div class="stat-pill">
                <div class="stat-pill-num">₹{pred_denom}</div>
                <div class="stat-pill-lbl">Recognized Value</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-num">{confidence*100:.1f}%</div>
                <div class="stat-pill-lbl">Tree Confidence</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-num" style="color: {'#00F59B' if report.rules_passed_count == report.rules_total_count else '#FF3366'};">
                    {report.rules_passed_count}/{report.rules_total_count}
                </div>
                <div class="stat-pill-lbl">Rules Satisfied</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-num" style="color: #64D2FF;">{elapsed_ms:.1f}ms</div>
                <div class="stat-pill-lbl">Pipeline Latency</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 1: Visual Computer Vision Studio (Contour, Warp & Overlays)
# -----------------------------------------------------------------------------
st.markdown("### 🔬 Computer Vision Spatial Inspection")

cv_tab1, cv_tab2 = st.tabs(["🖼️ Rectified Note & ROI Overlays", "📐 Contour Detection & Edge Map"])

with cv_tab1:
    col_vis1, col_vis2 = st.columns([1.5, 1.0])
    with col_vis1:
        # Generate color-coded overlays based on rule verdicts
        rule_dict = {r.rule_id: r.passed for r in report.diagnostic_checklist}
        verdict_flags = {
            "security_strip": rule_dict.get("RULE_2_SECURITY_THREAD", True),
            "watermark": rule_dict.get("RULE_4_WATERMARK_CONTRAST", True),
            "numeral": True,
            "governor_promise": True,
        }
        annotated_note = draw_roi_overlays(warped_note, rois, verdict_flags)
        st.image(
            cv2.cvtColor(annotated_note, cv2.COLOR_BGR2RGB),
            caption="Homography Rectified Banknote (600x300 px) with Predefined Inspection ROIs",
            use_container_width=True,
        )

    with col_vis2:
        st.markdown(
            f"""
            <div class="glass-card" style="font-size: 0.88rem;">
                <div style="font-weight: 700; color: #00D2FF; margin-bottom: 8px;">ROI Inspection Legend</div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div><span style="color: #00F59B; font-weight: bold;">■</span> <b>Green Box</b>: Security feature passed central bank rules.</div>
                    <div><span style="color: #FF3366; font-weight: bold;">■</span> <b>Red Box</b>: Counterfeit violation or defective region.</div>
                    <div><span style="color: #FFB800; font-weight: bold;">■</span> <b>Yellow Ribbon</b>: Vertical Metallic Security Thread.</div>
                    <div><span style="color: #38BDF8; font-weight: bold;">■</span> <b>Cyan Oval</b>: Mahatma Gandhi Latent Watermark.</div>
                </div>
                <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); color: #94A3B8;">
                    <b>Contour Aspect Ratio:</b> <code>{f"{aspect_ratio:.2f}" if aspect_ratio else 'N/A'}</code> (Spec: <code>{denom_info['aspect_ratio']}</code>)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with cv_tab2:
    col_edge1, col_edge2 = st.columns(2)
    with col_edge1:
        # Original frame with drawn polygon
        disp_orig = selected_image.copy()
        if detected_corners is not None:
            cv2.polylines(
                disp_orig,
                [np.int32(detected_corners)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3,
            )
            for pt in detected_corners:
                cv2.circle(disp_orig, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
        st.image(
            cv2.cvtColor(disp_orig, cv2.COLOR_BGR2RGB),
            caption=f"Input Source: {input_source_name} (Detected Note Boundary in Green)",
            use_container_width=True,
        )
    with col_edge2:
        st.image(
            edges_map,
            caption="Adaptive Canny Edge Detection & Bilateral Spatial Filtering",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Section 2: Forward-Chaining Expert System Rule Checklist (CO3)
# -----------------------------------------------------------------------------
st.markdown("### 📋 Expert System Forward-Chaining Diagnostic Trace (CO3)")

for res in report.diagnostic_checklist:
    card_cls = "rule-card-pass" if res.passed else "rule-card-fail"
    badge_cls = "rule-badge-pass" if res.passed else "rule-badge-fail"
    badge_txt = "PASS" if res.passed else "FAIL"

    st.markdown(
        f"""
        <div class="{card_cls}">
            <div class="rule-header">
                <span>{res.rule_name} <span style="font-size: 0.75rem; color: #94A3B8;">[{res.rule_id}]</span></span>
                <span class="{badge_cls}">{badge_txt}</span>
            </div>
            <div class="rule-details">
                {res.explanation}
            </div>
            <div class="rule-metric-row">
                <span><b>Observed:</b> {res.observed_value}</span>
                <span>•</span>
                <span><b>Threshold:</b> {res.expected_threshold}</span>
                <span>•</span>
                <span><b>Severity:</b> {res.severity.value}</span>
                <span>•</span>
                <span><b>Penalty:</b> -{res.score_penalty:.0f} pts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Section 3: Machine Learning & Forensic Features (CO4)
# -----------------------------------------------------------------------------
st.markdown("### 🧠 Forensic Machine Learning & Feature Analysis (CO4)")

ml_tab1, ml_tab2, ml_tab3 = st.tabs(
    [
        "🧵 K-Means Security Thread Segmentation",
        "🌳 Decision Tree Denomination Classifier",
        "💧 Watermark & Color Moment Profiles",
    ]
)

# Tab 1: K-Means Thread Segmentation
with ml_tab1:
    col_km1, col_km2, col_km3 = st.columns([1.0, 1.2, 1.2])

    with col_km1:
        st.markdown("**Security Strip Patch (Raw)**")
        st.image(
            cv2.cvtColor(strip_roi, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=f"Size: {strip_roi.shape[1]}x{strip_roi.shape[0]} px",
        )

    with col_km2:
        st.markdown("**K-Means Segmented Metallic Ribbon**")
        st.image(
            strip_analysis["segmented_mask"],
            use_container_width=True,
            caption="Binary Mask of Metallic Ribbon Cluster",
        )

    with col_km3:
        st.markdown("**Continuity & Clustering Metrics**")
        cont_val = strip_analysis["continuity_pct"]
        st.metric("Vertical Continuity", f"{cont_val}%", delta=f"{cont_val - 80.0:.1f}% vs Threshold")
        st.metric("Mean Thread Thickness", f"{strip_analysis['thread_width_mean']} px")

        # Visual swatch of cluster centroids
        st.markdown("<div style='font-size: 0.8rem; color: #94A3B8;'>Cluster Centroid Colors (K-Means):</div>", unsafe_allow_html=True)
        centers_bgr = strip_analysis["cluster_centers"]
        swatch_cols = st.columns(len(centers_bgr))
        for idx, col in enumerate(swatch_cols):
            b, g, r = centers_bgr[idx]
            hex_c = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            with col:
                st.markdown(
                    f"""<div style="background-color: {hex_c}; height: 32px; border-radius: 6px; border: 1px solid #555;" title="{hex_c}"></div>
                    <div style="font-size: 0.65rem; text-align: center; color: #888; font-family: monospace;">K{idx}</div>""",
                    unsafe_allow_html=True,
                )

# Tab 2: Decision Tree Class Probabilities
with ml_tab2:
    col_tree1, col_tree2 = st.columns([1.2, 1.5])
    with col_tree1:
        st.markdown("**Denomination Class Posterior Probabilities**")
        prob_df = pd.DataFrame(
            {
                "Denomination": [f"₹{k}" for k in proba_dict.keys()],
                "Probability": list(proba_dict.values()),
            }
        )
        st.bar_chart(prob_df.set_index("Denomination"))

    with col_tree2:
        st.markdown("**Supervised Decision Tree Logic Path**")
        st.text_area(
            "Induction Rule Splitting Trace (sklearn.tree.export_text):",
            classifier.explain_decision_rules(),
            height=200,
        )

# Tab 3: Watermark & Moments
with ml_tab3:
    col_wm1, col_wm2, col_wm3 = st.columns([1.0, 1.2, 1.2])

    with col_wm1:
        st.markdown("**Watermark Window Patch**")
        st.image(
            cv2.cvtColor(watermark_roi, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption="Latent Portrait Region",
        )

    with col_wm2:
        st.markdown("**Watermark Latent Metrics**")
        st.metric("Gray Intensity Variance", f"{watermark_analysis['variance']}", "Normal: 80 - 1400")
        st.metric("Dynamic Contrast Range", f"{watermark_analysis['dynamic_range']}", "Normal: >= 28")
        st.metric("Laplacian Focus Gradient", f"{watermark_analysis['laplacian_var']}")

    with col_wm3:
        st.markdown("**Substrate Statistical Moments**")
        moments_df = pd.DataFrame(
            {
                "Channel": ["Hue (H)", "Saturation (S)", "Value (V)"],
                "Mean": [color_features["h_mean"], color_features["s_mean"], color_features["v_mean"]],
                "Std Dev": [color_features["h_std"], color_features["s_std"], color_features["v_std"]],
                "Skewness": [color_features["h_skew"], color_features["s_skew"], color_features["v_skew"]],
            }
        )
        st.dataframe(moments_df, hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# Footer & Audit Download
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Automated Banknote Inspection & Anti-Counterfeit System • Production Reference Implementation "
    "compliant with AI Curriculum Outcomes (CO1, CO3, CO4)."
)
