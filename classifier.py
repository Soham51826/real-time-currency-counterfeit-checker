"""
Module 2: Machine Learning Denomination Classifier
===================================================
Implements a supervised Decision Tree Classifier (sklearn.tree.DecisionTreeClassifier)
for recognizing currency note denominations (e.g. ₹100, ₹200, ₹500, ₹2000) based on
statistical color moments (HSV) and physical aspect ratios.

Academic Mapping:
  - CO4 (Supervised Learning): Decision Tree induction, information gain / entropy,
    probabilistic confidence scoring, and multi-class confusion matrix evaluation.
"""

from typing import Dict, List, Tuple, Optional, Any, Literal
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score


# Standard denomination specifications (Mahatma Gandhi New Series)
# Includes official dimensions (mm), aspect ratios, and characteristic HSV envelopes.
DENOMINATION_SPECS: Dict[str, Dict[str, Any]] = {
    "100": {
        "label": "100",
        "display_name": "₹100 (Lavender)",
        "color_name": "Lavender",
        "dimensions_mm": (142, 66),
        "aspect_ratio": round(142 / 66, 2),  # ~2.15
        "hsv_bounds": {
            # Accommodates lavender under warm incandescent/CFL to cool LED camera lighting
            "h_mean": (95, 160),
            # Widened from (35,115) → (18,140): camera compression lowers saturation
            "s_mean": (18, 140),
            "v_mean": (110, 235),
        },
    },
    "200": {
        "label": "200",
        "display_name": "₹200 (Bright Yellow)",
        "color_name": "Bright Yellow",
        "dimensions_mm": (146, 66),
        "aspect_ratio": round(146 / 66, 2),  # ~2.21
        "hsv_bounds": {
            # Widened from (16,34) → (8,48): golden-yellow can shift warm/cool under camera
            "h_mean": (8, 48),
            # Widened from (120,230) → (70,245): saturation varies with camera exposure
            "s_mean": (70, 245),
            "v_mean": (145, 255),
        },
    },
    "500": {
        "label": "500",
        "display_name": "₹500 (Stone Grey)",
        "color_name": "Stone Grey",
        "dimensions_mm": (150, 66),
        "aspect_ratio": round(150 / 66, 2),  # ~2.27
        "hsv_bounds": {
            # Widened from (35,75) → (20,95): olive-grey note has broad hue spread
            "h_mean": (20, 95),
            # Widened from (15,75) → (5,105): very low-saturation note, easily affected by WB
            "s_mean": (5, 105),
            "v_mean": (80, 210),
        },
    },
    "2000": {
        "label": "2000",
        "display_name": "₹2000 (Magenta)",
        "color_name": "Magenta",
        "dimensions_mm": (166, 66),
        "aspect_ratio": round(166 / 66, 2),  # ~2.51
        "hsv_bounds": {
            # Widened from (145,172) → (132,178): deep pink/magenta can wrap in HSV space
            "h_mean": (132, 178),
            # Widened from (110,225) → (65,240): saturation spread due to multi-tone print
            "s_mean": (65, 240),
            "v_mean": (115, 240),
        },
    },
}

FEATURE_COLUMNS: List[str] = [
    "h_mean",
    "h_std",
    "s_mean",
    "s_std",
    "v_mean",
    "v_std",
    "aspect_ratio",
]


class DenominationClassifier:
    """
    Supervised Decision Tree Classifier for Banknote Denomination Recognition.

    Attributes:
        model (DecisionTreeClassifier): Scikit-learn tree estimator.
        classes_ (List[str]): List of denomination target labels.
        is_trained (bool): Status flag indicating if model weights are fitted.
    """

    def __init__(
        self, max_depth: int = 6, criterion: Literal["entropy", "gini", "log_loss"] = "entropy"
    ) -> None:
        """
        Initializes the Decision Tree classifier with academic parameters.

        Parameters:
            max_depth (int): Maximum tree depth to prevent overfitting (default: 6).
            criterion (str): Splitting measure ('entropy' for Information Gain or 'gini').
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.model: DecisionTreeClassifier = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=42,
            min_samples_split=4,
            min_samples_leaf=2,
        )
        self.classes_: List[str] = sorted(list(DENOMINATION_SPECS.keys()))
        self.is_trained: bool = False

    @staticmethod
    def build_feature_vector(
        color_features: Dict[str, Any], aspect_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Extracts and aligns a standard numerical feature vector from color moments.

        Parameters:
            color_features (Dict[str, Any]): Dictionary of moments from vision_pipeline.
            aspect_ratio (Optional[float]): Aspect ratio (width / height); defaults to 2.2 if None.

        Returns:
            np.ndarray: 1D numerical array matching FEATURE_COLUMNS.
        """
        ratio_val = float(aspect_ratio) if aspect_ratio is not None else 2.20
        # Clip ratio to reasonable bounds to guard against detection artifacts
        ratio_val = float(np.clip(ratio_val, 1.2, 3.5))

        vec = [
            float(color_features.get("h_mean", 0.0)),
            float(color_features.get("h_std", 0.0)),
            float(color_features.get("s_mean", 0.0)),
            float(color_features.get("s_std", 0.0)),
            float(color_features.get("v_mean", 0.0)),
            float(color_features.get("v_std", 0.0)),
            ratio_val,
        ]
        return np.array(vec, dtype=np.float32)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Trains the Decision Tree on historical or synthetic banknote feature distributions.

        Parameters:
            X (np.ndarray): Training feature matrix of shape (N, 7).
            y (np.ndarray): Target denomination labels of shape (N,).
            feature_names (Optional[List[str]]): Column labels.

        Returns:
            Dict[str, Any]: Training accuracy and feature importance scores.
        """
        self.model.fit(X, y)
        self.classes_ = [str(c) for c in self.model.classes_]
        self.is_trained = True

        train_preds = self.model.predict(X)
        acc = accuracy_score(y, train_preds)

        importances = dict(
            zip(
                feature_names or FEATURE_COLUMNS,
                [round(float(v), 4) for v in self.model.feature_importances_],
            )
        )

        return {
            "training_accuracy": round(float(acc), 4),
            "feature_importances": importances,
            "classes": self.classes_,
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Calculates academic performance metrics on a holdout test dataset.

        Returns:
            Dict[str, Any]:
                - accuracy: Overall accuracy score.
                - confusion_matrix: 2D integer matrix of actual vs predicted counts.
                - precision_recall_f1: Per-class metrics dictionary.
                - classification_report: Formatted text report.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=self.classes_)
        zero_div: Any = 0
        prec, rec, f1, support = precision_recall_fscore_support(
            y_test, preds, labels=self.classes_, zero_division=zero_div
        )

        prec_arr = np.asarray(prec)
        rec_arr = np.asarray(rec)
        f1_arr = np.asarray(f1)
        supp_arr = np.asarray(support) if support is not None else np.zeros(len(self.classes_))

        metrics_per_class = {}
        for i, c in enumerate(self.classes_):
            metrics_per_class[c] = {
                "precision": round(float(prec_arr[i]), 4),
                "recall": round(float(rec_arr[i]), 4),
                "f1_score": round(float(f1_arr[i]), 4),
                "support": int(supp_arr[i]),
            }

        report_str = classification_report(
            y_test, preds, labels=self.classes_, zero_division=zero_div
        )

        return {
            "accuracy": round(float(acc), 4),
            "confusion_matrix": cm.tolist(),
            "metrics_per_class": metrics_per_class,
            "classification_report": report_str,
        }

    def predict(
        self, feature_vector: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts banknote denomination along with posterior confidence scores.

        Parameters:
            feature_vector (np.ndarray): 1D array of shape (7,) or 2D array of shape (1, 7).

        Returns:
            Tuple[str, float, Dict[str, float]]:
                - predicted_denomination (str): e.g. "500"
                - confidence (float): Highest class probability (0.0 to 1.0)
                - class_probabilities (Dict[str, float]): Probability for each denomination.
        """
        if not self.is_trained:
            # Fallback heuristic using closest HSV mean bounds
            return self._heuristic_predict(feature_vector)

        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector

        pred_class = str(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0]

        proba_dict = {
            str(cls_name): round(float(prob), 4)
            for cls_name, prob in zip(self.model.classes_, probabilities)
        }
        confidence = float(np.max(probabilities))

        return pred_class, round(confidence, 4), proba_dict

    def predict_robust(
        self,
        color_features: Dict[str, Any],
        aspect_ratio: Optional[float] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Illumination-robust denomination prediction.
        Evaluates the Decision Tree on both raw and Gray-World white-balanced color moments.
        If ambient room light (e.g., warm incandescent or cool LED) shifted the raw Hue/Sat,
        the white-balanced vector neutralizes the ambient lighting cast and restores
        canonical central bank denomination recognition.

        Parameters:
            color_features (Dict[str, Any]): Color moments dictionary from vision_pipeline.
            aspect_ratio (Optional[float]): Observed or rectified aspect ratio.

        Returns:
            Tuple[str, float, Dict[str, float]]:
                - predicted_denomination (str)
                - confidence (float)
                - class_probabilities (Dict[str, float])
        """
        raw_vec = self.build_feature_vector(color_features, aspect_ratio)
        pred_raw, conf_raw, proba_raw = self.predict(raw_vec)

        # If raw prediction is high confidence (>= 0.70), trust raw color.
        # Gray-World white balance subtracts the dominant color of monochrome/dominant banknotes,
        # which can artificially turn Lavender (₹100) or Magenta (₹2000) into Stone Grey (₹500).
        if conf_raw >= 0.70:
            return pred_raw, conf_raw, proba_raw

        # Evaluate white-balanced moments only if raw prediction has lower confidence
        if "wb_h_mean" in color_features:
            wb_features = dict(color_features)
            wb_features["h_mean"] = color_features["wb_h_mean"]
            wb_features["s_mean"] = color_features["wb_s_mean"]
            wb_features["v_mean"] = color_features["wb_v_mean"]
            wb_vec = self.build_feature_vector(wb_features, aspect_ratio)
            pred_wb, conf_wb, proba_wb = self.predict(wb_vec)

            # Choose white-balanced if it yields higher confidence than raw
            if conf_wb > conf_raw:
                return pred_wb, conf_wb, proba_wb

        return pred_raw, conf_raw, proba_raw

    def _heuristic_predict(
        self, feature_vector: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """Fallback prediction rule if model weights are not yet loaded."""
        h_mean = float(feature_vector[0])
        s_mean = float(feature_vector[2])

        # Lavender ₹100: Hue 120-145
        if 115 <= h_mean <= 145:
            return "100", 0.90, {"100": 0.90, "200": 0.03, "500": 0.05, "2000": 0.02}
        # Bright Yellow ₹200: Hue 15-35
        elif 10 <= h_mean <= 35:
            return "200", 0.92, {"100": 0.02, "200": 0.92, "500": 0.04, "2000": 0.02}
        # Magenta ₹2000: Hue 145-175
        elif 145 < h_mean <= 178:
            return "2000", 0.88, {"100": 0.05, "200": 0.02, "500": 0.05, "2000": 0.88}
        # Stone Grey ₹500: Low saturation or Hue 35-85
        else:
            return "500", 0.85, {"100": 0.05, "200": 0.05, "500": 0.85, "2000": 0.05}

    def explain_decision_rules(self) -> str:
        """
        Exports human-readable text representation of the Decision Tree logic rules
        for academic explainability and inspection.
        """
        if not self.is_trained:
            return "Model is not trained yet."
        return export_text(self.model, feature_names=FEATURE_COLUMNS)

    def save(self, filepath: str) -> None:
        """Serializes model weights and metadata to disk via joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model": self.model,
            "classes": self.classes_,
            "is_trained": self.is_trained,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
        }
        joblib.dump(payload, filepath)

    def load(self, filepath: str) -> bool:
        """Loads serialized model weights from disk via joblib."""
        if not os.path.exists(filepath):
            return False
        payload = joblib.load(filepath)
        self.model = payload["model"]
        self.classes_ = payload.get("classes", list(DENOMINATION_SPECS.keys()))
        self.is_trained = payload.get("is_trained", True)
        self.max_depth = payload.get("max_depth", self.max_depth)
        self.criterion = payload.get("criterion", self.criterion)
        return True
