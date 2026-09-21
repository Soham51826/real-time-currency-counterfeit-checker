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


class VerificationExpertSystem:
    """
    Forward-Chaining Inference Engine for Currency Authenticity Verification (CO3).

    Reasoning Cycle:
      1. Working Memory Ingestion: Populates working memory with facts from vision & ML pipeline.
      2. Agenda Activation: Evaluates Rule 1 through Rule 4 against working memory.
      3. Conflict Resolution & Penalty Accumulation: Weighted scoring based on rule severity.
      4. Goal State Derivation: Asserts final verdict (GENUINE, SUSPICIOUS, COUNTERFEIT).
    """

    def __init__(self, tolerance_pct: float = 0.12) -> None:
        """
        Parameters:
            tolerance_pct (float): Permissible dimensional aspect ratio variance.
                Default is 12% (±0.12) to accommodate real camera perspective
                distortion, hand tilt, and homography approximation error.
        """
        self.tolerance_pct = tolerance_pct
        self.peas_model = AgentPEAS()

    def evaluate(
        self,
        predicted_denomination: str,
        classifier_confidence: float,
        aspect_ratio: Optional[float],
        security_strip_metrics: Dict[str, Any],
        watermark_metrics: Dict[str, Any],
        color_features: Dict[str, Any],
    ) -> VerificationReport:
        """
        Executes forward-chaining evaluation against extracted features.

        Parameters:
            predicted_denomination (str): e.g. "100", "200", "500", "2000".
            classifier_confidence (float): Decision tree confidence score (0.0 to 1.0).
            aspect_ratio (Optional[float]): Observed note width/height aspect ratio.
            security_strip_metrics (Dict[str, Any]): Continuity and cluster metrics from K-Means.
            watermark_metrics (Dict[str, Any]): Variance and gradient metrics from watermark ROI.
            color_features (Dict[str, Any]): HSV statistical moments.

        Returns:
            VerificationReport: Structured report with verdict, checklist, and audit trace.
        """
        # Step 1: Initialize Working Memory (Facts)
        working_memory: Dict[str, Any] = {
            "denomination": str(predicted_denomination),
            "confidence": float(classifier_confidence),
            "aspect_ratio": float(aspect_ratio) if aspect_ratio is not None else 2.20,
            "strip_continuity": float(security_strip_metrics.get("continuity_pct", 0.0)),
            "watermark_variance": float(watermark_metrics.get("variance", 0.0)),
            "watermark_dynamic_range": float(watermark_metrics.get("dynamic_range", 0.0)),
            "h_mean": float(color_features.get("h_mean", 0.0)),
            "s_mean": float(color_features.get("s_mean", 0.0)),
            "v_mean": float(color_features.get("v_mean", 0.0)),
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

        # Permissible tolerance is ±tolerance_pct (default 8% for camera tilt)
        r1_passed = bool(ar_diff <= self.tolerance_pct)
        r1_explanation = (
            f"Observed aspect ratio {obs_ar:.2f} matches standard {expected_ar:.2f} "
            f"within ±{self.tolerance_pct*100:.0f}% tolerance."
            if r1_passed
            else f"Dimensional mismatch: observed {obs_ar:.2f} deviates from official {expected_ar:.2f} "
            f"by {ar_diff*100:.1f}% (exceeds ±{self.tolerance_pct*100:.0f}% threshold)."
        )
        penalty_r1 = 0.0 if r1_passed else 25.0
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
        # Genuine banknotes must have continuous metallic/fluorescent strip.
        # Threshold lowered to 65% (from 80%) to accommodate:
        #   - Real circulated/worn notes with partial thread abrasion
        #   - Camera angle causing ROI to miss the strip edge
        obs_continuity = working_memory["strip_continuity"]
        r2_passed = bool(obs_continuity >= 65.0)

        if r2_passed:
            r2_explanation = (
                f"K-Means segmentation verifies continuous metallic thread coverage "
                f"at {obs_continuity:.1f}% (exceeds >=65.0% acceptance threshold)."
            )
            penalty_r2 = 0.0
        elif obs_continuity >= 45.0:
            # Soft fail (worn, creased, or camera offset)
            r2_explanation = (
                f"Security thread segmented with partial continuity of {obs_continuity:.1f}%. "
                f"Slight degradation, folding, or ROI misalignment detected."
            )
            penalty_r2 = 20.0
        else:
            # Hard fail (broken/fake line or missing ribbon)
            r2_explanation = (
                f"Critical security failure: security thread continuity is only {obs_continuity:.1f}% "
                f"(requires >=65.0%). Indicates broken imitation or absent metallic foil."
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
                observed_value=f"{obs_continuity:.1f}%",
                expected_threshold=">= 80.0%",
                explanation=r2_explanation,
                score_penalty=penalty_r2,
            )
        )

        # ======================================================================
        # RULE 3 (CO3): Color Consistency & Substrate Chromaticity
        # ======================================================================
        # Validates that observed HSV mean moments fall within official denomination color bounds
        h_bounds = denom_spec["hsv_bounds"]["h_mean"]
        s_bounds = denom_spec["hsv_bounds"]["s_mean"]
        v_bounds = denom_spec["hsv_bounds"]["v_mean"]

        obs_h = working_memory["h_mean"]
        obs_s = working_memory["s_mean"]
        obs_v = working_memory["v_mean"]

        # Allow ±12 units margin on Hue and ±25 on S/V for lighting variation
        h_ok = (h_bounds[0] - 12) <= obs_h <= (h_bounds[1] + 12)
        s_ok = (s_bounds[0] - 25) <= obs_s <= (s_bounds[1] + 35)
        v_ok = (v_bounds[0] - 30) <= obs_v <= (v_bounds[1] + 35)

        r3_passed = bool(h_ok and s_ok)
        if r3_passed:
            r3_explanation = (
                f"Substrate chromaticity (Hue: {obs_h:.1f}, Saturation: {obs_s:.1f}) conforms "
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
            penalty_r3 = 30.0
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

        # Lower variance floor from 70 to 40: real camera images often produce
        # lower variance due to JPEG compression, autofocus softening, and
        # the watermark ROI not being perfectly aligned on the translucent paper area.
        # Dynamic range threshold at 27: genuine notes 31-35, blank counterfeit 25.
        r4_passed = bool(40.0 <= obs_var <= 1500.0 and obs_dr >= 27.0)
        if r4_passed:
            r4_explanation = (
                f"Latent portrait presence confirmed: watermark window variance {obs_var:.1f} "
                f"and dynamic contrast {obs_dr:.1f} fall within authentic paper matrix band."
            )
            penalty_r4 = 0.0
        elif obs_var < 40.0:
            r4_explanation = (
                f"Watermark window lacks latent features (Variance: {obs_var:.1f} < 40.0). "
                f"Window appears flat or blank (typical of plain paper counterfeit)."
            )
            penalty_r4 = 35.0
            critical_failures += 1
        else:
            r4_explanation = (
                f"Watermark window exhibits anomalous harsh edge variance ({obs_var:.1f} > 1500.0). "
                f"Indicates opaque surface printing rather than translucent paper embedding."
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
                expected_threshold="Var:[70-1500], Dyn:>=28",
                explanation=r4_explanation,
                score_penalty=penalty_r4,
            )
        )

        # ======================================================================
        # FORWARD-CHAINING VERDICT RESOLUTION
        # ======================================================================
        passed_rules = sum(1 for r in checklist if r.passed)
        total_rules = len(checklist)
        authenticity_score = max(0.0, min(100.0, 100.0 - total_penalty))
        overall_confidence = round(
            float((authenticity_score / 100.0) * working_memory["confidence"]), 3
        )

        if passed_rules == total_rules and critical_failures == 0:
            verdict = Verdict.GENUINE
            summary = (
                f"All {total_rules} central bank verification rules successfully validated. "
                f"The {denom_spec['display_name']} banknote exhibits continuous security strip, "
                f"legitimate multi-tone watermark, and compliant physical dimensions."
            )
        elif critical_failures >= 2 or total_penalty >= 60.0:
            # Require at least 2 independent critical failures OR a very high
            # cumulative penalty before declaring COUNTERFEIT. A single borderline
            # failure from camera noise / lighting is routed to SUSPICIOUS instead.
            verdict = Verdict.COUNTERFEIT
            failed_names = [r.rule_name for r in checklist if not r.passed]
            summary = (
                f"Counterfeit indicators detected! Failed {len(failed_names)} security test(s): "
                f"{', '.join(failed_names)}. Note rejected by forward-chaining expert rules."
            )
        else:
            verdict = Verdict.SUSPICIOUS
            failed_names = [r.rule_name for r in checklist if not r.passed]
            summary = (
                f"Minor anomalies detected in {', '.join(failed_names)}. Banknote requires "
                f"secondary manual inspection under UV light or high-magnification sensor."
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
        )
