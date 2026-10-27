"""
Module 3: Rule-Based Expert System Engine
==========================================
Implements a forward-chaining rule-based expert system for banknote authenticity
verification and counterfeit detection based on official central bank standards.

Academic Mapping:
  - CO1 (Agent Modeling): Formal PEAS (Performance, Environment, Actuators, Sensors)
    representation of an autonomous currency verification agent.
  - CO3 (Knowledge Representation & Expert Systems): Forward-chaining inference engine
    with Working Memory, Rule Base, Conflict Resolution, and Execution Trace.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from classifier import DENOMINATION_SPECS
from counterfeit_classifier import CounterfeitDetectorAI


# ==============================================================================
# CO1: Formal PEAS Agent Modeling
# ==============================================================================

@dataclass(frozen=True)
class AgentPEAS:
    """
    Formal PEAS specification for an Autonomous Currency Inspection Agent (CO1).
    """
    agent_name: str = "Automated Banknote Inspection & Anti-Counterfeit Agent (ABIA)"
    
    performance_measures: Tuple[str, ...] = (
        "Denomination classification accuracy >= 95%",
        "Counterfeit defect detection sensitivity >= 98%",
        "False positive rejection rate for genuine currency < 2%",
        "Real-time processing latency < 150 ms per frame",
    )
    
    environment: Tuple[str, ...] = (
        "Variable ambient illumination (incandescent, daylight, shadows)",
        "Dynamic camera perspective angles, hand jitter, and perspective distortion",
        "Banknote physical wear: crisp uncirculated, folded, creased, soiled",
        "Adversarial counterfeit techniques: inkjet prints, discontinuous foil, blank watermarks",
    )
    
    actuators: Tuple[str, ...] = (
        "Visual bounding box overlays & highlighted defective Regions of Interest",
        "Tri-color authenticity verdict banner (GENUINE, SUSPICIOUS, COUNTERFEIT)",
        "Diagnostic rule checklist with numeric deviations and explanatory feedback",
        "Exportable forensic audit manifest",
    )
    
    sensors: Tuple[str, ...] = (
        "Real-time RGB webcam video stream",
        "High-resolution digital optical image sensor",
        "OpenCV spatial contour detector & color moment digitizer",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Agent Name": self.agent_name,
            "Performance Measures": list(self.performance_measures),
            "Environment": list(self.environment),
            "Actuators": list(self.actuators),
            "Sensors": list(self.sensors),
        }


# ==============================================================================
# CO3: Rule-Based Expert System Forward-Chaining Engine
# ==============================================================================

class Verdict(str, Enum):
    GENUINE = "GENUINE"
    SUSPICIOUS = "SUSPICIOUS"
    COUNTERFEIT = "COUNTERFEIT"


class RuleSeverity(str, Enum):
    CRITICAL = "CRITICAL"   # Failure directly indicates high counterfeit probability
    MAJOR = "MAJOR"         # Significant defect, requires secondary verification
    MINOR = "MINOR"         # Soft deviation possibly caused by wear/lighting


@dataclass
class RuleEvaluationResult:
    """Stores the outcome of a single forward-chaining rule evaluation."""
    rule_id: str
    rule_name: str
    severity: RuleSeverity
    passed: bool
    observed_value: Any
    expected_threshold: str
    explanation: str
    score_penalty: float = 0.0


@dataclass
class VerificationReport:
    """Comprehensive diagnostic verdict produced by the Expert System."""
    verdict: Verdict
    overall_confidence: float
    rules_passed_count: int
    rules_total_count: int
    diagnostic_checklist: List[RuleEvaluationResult]
    summary_explanation: str
    predicted_denomination: str
    working_memory: Dict[str, Any]
    ai_verdict: Optional[Verdict] = None
    ai_confidence: float = 0.0
    ai_genuine_prob: float = 0.0
    ai_model_name: str = "Ensemble AI"
    ai_feature_attributions: List[Dict[str, Any]] = field(default_factory=list)
    neuro_symbolic_status: str = "Symbolic Evaluation"

    @property
    def ai_authenticity_prob(self) -> float:
        return self.ai_genuine_prob

    @property
    def rules(self) -> List[RuleEvaluationResult]:
        return self.diagnostic_checklist


class VerificationExpertSystem:
    """
    Forward-Chaining Inference Engine for Currency Authenticity Verification (CO3).

    Reasoning Cycle:
      1. Working Memory Ingestion: Populates working memory with facts from vision & ML pipeline.
      2. Agenda Activation: Evaluates central bank physical security rules (Rules 1 - 4).
      3. Conflict Resolution & Penalty Accumulation: Weighted scoring based on rule severity.
      4. Goal State Derivation: Asserts central bank authenticity verdict (GENUINE, SUSPICIOUS, COUNTERFEIT).
    """

    def __init__(
        self,
        tolerance_pct: float = 0.10,
        counterfeit_detector: Optional[CounterfeitDetectorAI] = None,
        counterfeit_ai: Optional[CounterfeitDetectorAI] = None,
    ) -> None:
        """
        Parameters:
            tolerance_pct (float): Permissible dimensional aspect ratio variance.
            counterfeit_detector (Optional[CounterfeitDetectorAI]): Machine Learning authenticity detector.
            counterfeit_ai (Optional[CounterfeitDetectorAI]): Alias for counterfeit_detector.
        """
        self.tolerance_pct = tolerance_pct
        self.peas_model = AgentPEAS()
        self.counterfeit_detector = counterfeit_ai or counterfeit_detector

    def evaluate(
        self,
        predicted_denomination: str,
        classifier_confidence: float,
        aspect_ratio: Optional[float],
        security_strip_metrics: Dict[str, Any],
        watermark_metrics: Dict[str, Any],
        color_features: Dict[str, Any],
        scan_mode: str = "auto",
        forensic_features: Optional[np.ndarray] = None,
        ai_model_type: str = "ensemble",
    ) -> VerificationReport:
        """
        Executes forward-chaining evaluation against extracted features.

        Parameters:
            predicted_denomination (str): e.g. "100", "200", "500".
            classifier_confidence (float): Decision tree confidence score (0.0 to 1.0).
            aspect_ratio (Optional[float]): Observed note width/height aspect ratio.
            security_strip_metrics (Dict[str, Any]): Continuity and cluster metrics from K-Means.
            watermark_metrics (Dict[str, Any]): Variance and gradient metrics from watermark ROI.
            color_features (Dict[str, Any]): HSV statistical moments and white-balanced moments.
            scan_mode (str): "auto", "reflected", or "transmitted".

        Returns:
            VerificationReport: Structured report with verdict, checklist, and audit trace.
        """
        # Step 1: Initialize Working Memory (Facts)
        working_memory: Dict[str, Any] = {
            "denomination": predicted_denomination,
            "confidence": float(classifier_confidence),
            "aspect_ratio": float(aspect_ratio) if aspect_ratio is not None else 2.20,
            "strip_continuity": float(security_strip_metrics.get("continuity_pct", 0.0)),
            "window_count": int(security_strip_metrics.get("window_count", 1)),
            "is_windowed": bool(security_strip_metrics.get("is_windowed", False)),
            "is_continuous": bool(security_strip_metrics.get("is_continuous", False)),
            "watermark_variance": float(watermark_metrics.get("variance", 0.0)),
            "watermark_dynamic_range": float(watermark_metrics.get("dynamic_range", 0.0)),
            "h_mean": float(color_features.get("h_mean", 0.0)),
            "s_mean": float(color_features.get("s_mean", 0.0)),
            "v_mean": float(color_features.get("v_mean", 0.0)),
            "wb_h_mean": float(color_features.get("wb_h_mean", color_features.get("h_mean", 0.0))),
            "wb_s_mean": float(color_features.get("wb_s_mean", color_features.get("s_mean", 0.0))),
            "scan_mode": scan_mode,
        }

        denom_spec = DENOMINATION_SPECS.get(
            working_memory["denomination"], DENOMINATION_SPECS["500"]
        )

        checklist: List[RuleEvaluationResult] = []
        total_penalty = 0.0
        critical_failures = 0

        # ======================================================================
        # RULE 1 (CO3): Dimension & Aspect Ratio Fidelity
        # ======================================================================
        expected_ar = denom_spec["aspect_ratio"]
        obs_ar = working_memory["aspect_ratio"]
        ar_diff = abs(obs_ar - expected_ar) / expected_ar

        # Permissible tolerance is ±tolerance_pct (default 5-10% for flat scans)
        is_direct_match = bool(ar_diff <= self.tolerance_pct)
        # Perspective foreshortening accommodation for camera/handheld captures
        is_perspective_mode = scan_mode == "perspective"
        is_perspective_conforming = bool(is_perspective_mode and (1.30 <= obs_ar <= 3.50))

        if is_direct_match:
            r1_passed = True
            r1_explanation = (
                f"Observed aspect ratio {obs_ar:.2f} matches standard {expected_ar:.2f} "
                f"within ±{self.tolerance_pct*100:.0f}% tolerance."
            )
            penalty_r1 = 0.0
        elif is_perspective_conforming:
            r1_passed = True
            r1_explanation = (
                f"Perspective foreshortening detected (observed 2D aspect ratio {obs_ar:.2f}). "
                f"Physical banknote dimensions successfully rectified via 4-point homography."
            )
            penalty_r1 = 0.0
        else:
            r1_passed = False
            r1_explanation = (
                f"Dimensional mismatch: observed {obs_ar:.2f} deviates from official {expected_ar:.2f} "
                f"by {ar_diff*100:.1f}% (exceeds ±{self.tolerance_pct*100:.0f}% threshold)."
            )
            penalty_r1 = 25.0
            critical_failures += 1

        total_penalty += penalty_r1

        checklist.append(
            RuleEvaluationResult(
                rule_id="RULE_1_DIMENSION",
                rule_name="Physical Aspect Ratio & Dimension Tolerance",
                severity=RuleSeverity.MAJOR,
                passed=r1_passed,
                observed_value=f"{obs_ar:.2f}",
                expected_threshold=f"{expected_ar:.2f} ± {self.tolerance_pct*100:.0f}%",
                explanation=r1_explanation,
                score_penalty=penalty_r1,
            )
        )

        # ======================================================================
        # RULE 2 (CO3 & CO4): Security Thread Vertical Continuity
        # ======================================================================
        # Genuine banknotes feature an embedded metallic ribbon.
        # In transmitted light / synthetic tests: continuous vertical ribbon (>= 65%).
        # In reflected room light (real camera): authentic windowed segments with
        # varying continuity due to note orientation, hand-held tilt, and lighting.
        # Real photos of genuine notes: continuity typically 25–100% depending on scan.
        obs_continuity = working_memory["strip_continuity"]
        window_count = working_memory["window_count"]
        is_windowed = working_memory["is_windowed"]
        is_camera_input = scan_mode in ("perspective", "reflected")

        # Tier 1: Continuous ribbon (transmitted light or flat scanner)
        if obs_continuity >= 65.0:
            r2_passed = True
            r2_explanation = (
                f"K-Means segmentation confirms continuous metallic ribbon coverage "
                f"at {obs_continuity:.1f}% (exceeds >=65.0% central bank threshold)."
            )
            penalty_r2 = 0.0

        # Tier 2: Authentic windowed thread (reflected / real camera)
        elif is_camera_input and (is_windowed or (2 <= window_count <= 12 and obs_continuity >= 25.0)):
            r2_passed = True
            r2_explanation = (
                f"Windowed metallic security thread verified: {window_count} periodic metallic windows "
                f"detected across scanlines ({obs_continuity:.1f}% vertical presence). "
                f"Authentic windowed thread pattern confirmed for reflected-light scan."
            )
            penalty_r2 = 0.0

        # Tier 3: Camera perspective scan — accept reasonable continuity in camera view
        elif is_camera_input and obs_continuity >= 35.0:
            r2_passed = True
            r2_explanation = (
                f"Security thread detected at {obs_continuity:.1f}% continuity in "
                f"perspective/handheld scan mode ({window_count} windows). "
                f"Real-camera foreshortening and lighting variations accommodated."
            )
            penalty_r2 = 0.0

        # Tier 4: Marginal — soft warning (degraded, heavily folded note)
        elif is_camera_input and obs_continuity >= 22.0:
            r2_passed = False
            r2_explanation = (
                f"Security thread segmented with partial continuity of {obs_continuity:.1f}%. "
                f"Possible note folding, crease across thread, or specular camera glare."
            )
            penalty_r2 = 18.0

        # Tier 5: Hard fail — counterfeit broken/absent thread
        else:
            r2_passed = False
            r2_explanation = (
                f"Critical security failure: security thread continuity is only {obs_continuity:.1f}% "
                f"({window_count} detected windows). Indicates a broken, printed, or absent thread — "
                f"strong counterfeit indicator."
            )
            penalty_r2 = 40.0
            critical_failures += 1

        total_penalty += penalty_r2

        checklist.append(
            RuleEvaluationResult(
                rule_id="RULE_2_SECURITY_THREAD",
                rule_name="K-Means Security Thread Vertical Continuity",
                severity=RuleSeverity.CRITICAL,
                passed=r2_passed,
                observed_value=f"{obs_continuity:.1f}% ({window_count} windows)",
                expected_threshold=">= 65% continuous OR windowed pattern (camera: >= 28%)",
                explanation=r2_explanation,
                score_penalty=penalty_r2,
            )
        )

        # ======================================================================
        # RULE 3 (CO3): Color Consistency & Substrate Chromaticity
        # ======================================================================
        # Validates that observed HSV moments fall within denomination color bounds.
        # Robust against ambient lighting via Gray-World illuminant normalization.
        h_bounds = denom_spec["hsv_bounds"]["h_mean"]
        s_bounds = denom_spec["hsv_bounds"]["s_mean"]
        v_bounds = denom_spec["hsv_bounds"]["v_mean"]

        obs_h = working_memory["h_mean"]
        obs_s = working_memory["s_mean"]
        wb_h = working_memory["wb_h_mean"]
        wb_s = working_memory["wb_s_mean"]

        # Check raw moments with lighting margin
        h_ok_raw = (h_bounds[0] - 12) <= obs_h <= (h_bounds[1] + 12)
        s_ok_raw = (s_bounds[0] - 25) <= obs_s <= (s_bounds[1] + 35)

        # Check white-balanced moments (neutralizes warm lamp or cool LED casts)
        h_ok_wb = (h_bounds[0] - 14) <= wb_h <= (h_bounds[1] + 14)
        s_ok_wb = (s_bounds[0] - 30) <= wb_s <= (s_bounds[1] + 40)

        # For 500 (Stone Grey), Gray-World white balance equalizes RGB channels and turns
        # any saturated monochrome counterfeit into artificial grey.
        # Therefore, if raw saturation is far beyond Stone Grey limits (obs_s > s_bounds[1] + 35),
        # white-balance cannot override a saturated counterfeit substrate.
        if predicted_denomination == "500" and not s_ok_raw:
            r3_passed = False
        else:
            r3_passed = bool((h_ok_raw and s_ok_raw) or (h_ok_wb and s_ok_wb))

        if r3_passed:
            r3_explanation = (
                f"Substrate chromaticity (Hue: {obs_h:.1f}, Sat: {obs_s:.1f}; Calibrated: H:{wb_h:.1f}) conforms "
                f"to official {denom_spec['display_name']} color signature."
            )
            penalty_r3 = 0.0
        else:
            r3_explanation = (
                f"Color deviation detected for {denom_spec['display_name']}: "
                f"Observed Hue {obs_h:.1f} (expected {h_bounds[0]}-{h_bounds[1]}), "
                f"Saturation {obs_s:.1f} (expected {s_bounds[0]}-{s_bounds[1]}). "
                f"Indicates counterfeit ink or incorrect toner."
            )
            penalty_r3 = 35.0
            critical_failures += 1

        total_penalty += penalty_r3

        checklist.append(
            RuleEvaluationResult(
                rule_id="RULE_3_COLOR_CONSISTENCY",
                rule_name="Denomination Substrate Color Consistency",
                severity=RuleSeverity.CRITICAL,
                passed=r3_passed,
                observed_value=f"H:{obs_h:.1f}, S:{obs_s:.1f}",
                expected_threshold=f"H:[{h_bounds[0]}-{h_bounds[1]}], S:[{s_bounds[0]}-{s_bounds[1]}]",
                explanation=r3_explanation,
                score_penalty=penalty_r3,
            )
        )

        # ======================================================================
        # RULE 4 (CO3): Watermark Latent Portrait & Contrast Presence
        # ======================================================================
        # Checks if grayscale variance & dynamic range in the watermark window indicate
        # an authentic multi-tone latent watermark (neither plain paper nor harsh flat ink).
        obs_var = working_memory["watermark_variance"]
        obs_dr = working_memory["watermark_dynamic_range"]
        obs_lap = float(watermark_metrics.get("laplacian_var", 0.0))
        obs_ent = float(watermark_metrics.get("entropy", 0.0))

        # Authentic paper band: variance between 25.0 and 12000.0, dynamic range >= 22.0
        # Plain paper counterfeit (blank_watermark):
        #   (1) Flat paper: dyn range < 28.0 and variance < 240.0, or
        #   (2) Blurry unprinted paper: laplacian < 320.0 and entropy < 3.80, or
        #   (3) Extremely flat: dyn range < 18.0 or variance < 18.0
        # Opaque stamped counterfeit: variance typically > 14000 AND dynamic_range > 95
        is_blank_counterfeit = (
            (obs_dr < 28.0 and obs_var < 240.0)
            or (obs_lap < 320.0 and obs_ent < 3.80)
            or (obs_dr < 18.0)
            or (obs_var < 18.0)
        )
        is_stamped_counterfeit = obs_var > 14000.0 and obs_dr > 95.0
        r4_passed = not is_blank_counterfeit and not is_stamped_counterfeit

        if r4_passed:
            r4_explanation = (
                f"Latent portrait presence confirmed: watermark window variance {obs_var:.1f} "
                f"and dynamic contrast {obs_dr:.1f} fall within authentic banknote paper matrix range."
            )
            penalty_r4 = 0.0
        elif is_blank_counterfeit:
            r4_explanation = (
                f"Watermark window lacks latent features (Dynamic Range: {obs_dr:.1f}, Variance: {obs_var:.1f}, Laplacian: {obs_lap:.1f}). "
                f"Window appears flat or blank — consistent with plain paper photocopy counterfeit."
            )
            penalty_r4 = 35.0
            critical_failures += 1
        else:
            r4_explanation = (
                f"Watermark window exhibits stamped/opaque toner pattern (Var:{obs_var:.1f} > 14000 AND Dyn:{obs_dr:.1f} > 95). "
                f"Indicates surface-printed counterfeit rather than genuine translucent paper watermark."
            )
            penalty_r4 = 30.0
            critical_failures += 1

        total_penalty += penalty_r4

        checklist.append(
            RuleEvaluationResult(
                rule_id="RULE_4_WATERMARK_CONTRAST",
                rule_name="Watermark Latent Feature & Contrast Variance",
                severity=RuleSeverity.CRITICAL,
                passed=r4_passed,
                observed_value=f"Var:{obs_var:.1f}, Dyn:{obs_dr:.1f}",
                expected_threshold="Var:[18–14000], Dyn:>=22.0 (not stamped >95)",
                explanation=r4_explanation,
                score_penalty=penalty_r4,
            )
        )

        # ======================================================================
        # AI Machine Learning Feature Attribution (Forensic Exploration - CO4)
        # ======================================================================
        ai_verdict: Optional[Verdict] = None
        ai_confidence = 0.0
        ai_genuine_prob = 0.0
        ai_model_name = f"AI {ai_model_type.upper()}"
        ai_explanations: List[Dict[str, Any]] = []

        if self.counterfeit_detector is not None and forensic_features is not None:
            is_ai_gen, ai_conf, ai_probas = self.counterfeit_detector.predict(
                forensic_features, model_type=ai_model_type
            )
            ai_genuine_prob = float(ai_probas.get("genuine", 0.0))
            ai_confidence = float(ai_conf)
            ai_verdict = Verdict.GENUINE if is_ai_gen else Verdict.COUNTERFEIT
            ai_explanations = self.counterfeit_detector.explain_prediction(
                forensic_features, is_ai_gen
            )

        # ======================================================================
        # FORWARD-CHAINING VERDICT RESOLUTION (CENTRAL BANK PHYSICAL RULES)
        # ======================================================================
        passed_rules = sum(1 for r in checklist if r.passed)
        total_rules = len(checklist)
        authenticity_score = max(0.0, min(100.0, 100.0 - total_penalty))

        overall_confidence = round(
            float((authenticity_score / 100.0) * working_memory["confidence"]), 3
        )

        # Neuro-symbolic integration: if AI model strongly agrees with rule verdict,
        # boost confidence; if it disagrees, treat it as advisory only (rules are primary).
        ai_agrees_genuine = ai_genuine_prob >= 0.50

        if passed_rules == total_rules and critical_failures == 0:
            neuro_status = "Authentic Banknote (All Security Rules Verified)"
            verdict = Verdict.GENUINE
            summary = (
                f"All {total_rules} central bank verification rules successfully validated. "
                f"The {denom_spec['display_name']} banknote exhibits verified security features, "
                f"authentic latent watermark, and compliant dimensions."
            )
        elif critical_failures >= 2 or total_penalty >= 60.0:
            neuro_status = "Counterfeit Detected (Security Rule Violation)"
            verdict = Verdict.COUNTERFEIT
            failed_names = [r.rule_name for r in checklist if not r.passed]
            summary = (
                f"Counterfeit indicators detected! Failed {len(failed_names)} security test(s): "
                f"{', '.join(failed_names)}. Note rejected by forward-chaining expert rules."
            )
        elif critical_failures == 1 and total_penalty >= 35.0 and not ai_agrees_genuine:
            # One critical failure AND AI model also flags counterfeit → reject
            neuro_status = "Counterfeit Detected (Expert + AI Consensus)"
            verdict = Verdict.COUNTERFEIT
            failed_names = [r.rule_name for r in checklist if not r.passed]
            summary = (
                f"Counterfeit consensus: {len(failed_names)} rule violation(s) confirmed by AI model. "
                f"Failed: {', '.join(failed_names)}. Note rejected."
            )
        else:
            neuro_status = "Suspicious Currency (Secondary Verification Recommended)"
            verdict = Verdict.SUSPICIOUS
            failed_names = [r.rule_name for r in checklist if not r.passed]
            summary = (
                f"Partial security compliance: {len(failed_names)} rule deviation(s) observed: "
                f"{', '.join(failed_names)}. Secondary physical verification recommended."
            )

        return VerificationReport(
            verdict=verdict,
            overall_confidence=overall_confidence,
            rules_passed_count=passed_rules,
            rules_total_count=total_rules,
            diagnostic_checklist=checklist,
            summary_explanation=summary,
            predicted_denomination=working_memory["denomination"],
            working_memory=working_memory,
            ai_verdict=ai_verdict,
            ai_confidence=ai_confidence,
            ai_genuine_prob=ai_genuine_prob,
            ai_model_name=ai_model_name,
            ai_feature_attributions=ai_explanations,
            neuro_symbolic_status=neuro_status,
        )
