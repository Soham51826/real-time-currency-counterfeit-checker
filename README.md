# CurrencyGuard AI: Real-Time Currency Note Recognition & Counterfeit Feature Checker

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-red.svg)](https://streamlit.io)

A production-grade, modular, and fully runnable Python application engineered to detect currency denominations and rigorously inspect security features for counterfeit detection in real time.

Designed and structured in strict compliance with Academic AI Curriculum Standards:
- **CO1 (Agent Modeling)**: Formal PEAS (Performance, Environment, Actuators, Sensors) framework.
- **CO3 (Knowledge Representation & Expert Systems)**: Forward-chaining rule engine evaluating formal central bank specifications.
- **CO4 (Supervised & Unsupervised Machine Learning)**:
  - **Supervised**: Decision Tree Classifier for denomination recognition with confidence metrics and explainability rules.
  - **Unsupervised**: K-Means Clustering on the security thread Region of Interest (ROI) to segment the metallic/fluorescent ribbon and calculate vertical continuity.
- **Computer Vision Pipeline**: Adaptive Canny edge detection, Douglas-Peucker convex polygon contour approximation, four-point perspective homography warp to a standard $600 \times 300\text{ px}$ canvas, and color moment analysis.

---

## 🏛️ System Architecture

```
                                [ CAMERA / IMAGE INPUT ]
                                           │
                           ┌───────────────▼────────────────┐
                           │   Module 1: Vision Pipeline    │
                           │  - Adaptive Canny & Contours   │
                           │  - 4-Point Perspective Warp    │
                           │  - ROI Extraction (Strip, WM)  │
                           │  - K-Means Color Segmentation  │
                           └───────┬──────────────┬─────────┘
                                   │              │
             Extracted HSV Moments │              │ ROI Masks & Continuity
                                   ▼              ▼
                 ┌──────────────────────┐  ┌───────────────────────┐
                 │ Module 2: Classifier │  │ Module 3: Expert Sys  │
                 │ Decision Tree (CO4)  │  │ Forward-Chaining (CO3)│
                 │ Denomination Predict │  │ Rules 1-4 Verification│
                 └──────────┬───────────┘  └──────────┬────────────┘
                            │                         │
                            └──────────────┬──────────┘
                                           │
                           ┌───────────────▼────────────────┐
                           │   Module 4: Streamlit UI       │
                           │  - Real-time Visual Overlays   │
                           │  - Diagnostic Rule Checklist   │
                           │  - Authenticity Verdict Banner │
                           └────────────────────────────────┘
```

---

## 📘 Academic Curriculum Mapping

### 1. Agent Modeling (CO1): Formal PEAS Setup
The system models an autonomous rational agent encapsulated as `AgentPEAS` in `expert_system.py`:

| PEAS Element | Specification |
| :--- | :--- |
| **Performance Measures (P)** | - Denomination recognition accuracy $\ge 95\%$<br>- Counterfeit detection sensitivity $\ge 98\%$<br>- False positive rate for genuine currency $< 2\%$<br>- Frame processing latency $< 150\text{ ms}$ |
| **Environment (E)** | - Variable ambient lighting (incandescent, daylight, shadows)<br>- Hand jitter, rotation, tilt, and perspective distortion<br>- Banknote physical condition (crisp uncirculated, creased, soiled)<br>- Adversarial counterfeits (color photocopies, broken thread imitations, blank watermark paper) |
| **Actuators (A)** | - Dynamic bounding box overlays highlighting inspected ROIs<br>- High-contrast Tri-Color Verdict Banner (`GENUINE`, `SUSPICIOUS`, `COUNTERFEIT`)<br>- Forward-chaining diagnostic checklist with measured values vs thresholds<br>- Real-time metric scorecards |
| **Sensors (S)** | - Live RGB optical webcam video stream (`st.camera_input`)<br>- High-resolution image upload sensor (`st.file_uploader`)<br>- OpenCV edge digitizer & color moment sensor |

---

### 2. Forward-Chaining Expert System (CO3)
Located in `expert_system.py`, the `VerificationExpertSystem` implements an AI forward-chaining cycle:
1. **Working Memory Ingestion**: Populates facts (aspect ratio, K-Means continuity percentage, watermark intensity variance, mean HSV color values).
2. **Rule Base Evaluation**:
   - **Rule 1 (Physical Dimensions)**: Verifies note width-to-height aspect ratio against official bank specifications within $\pm 5\%$ tolerance.
   - **Rule 2 (Security Thread Continuity)**: Verifies segmented metallic ribbon maintains $\ge 80\%$ vertical continuity from top to bottom.
   - **Rule 3 (Substrate Color Consistency)**: Validates that mean Hue and Saturation fall within the chromaticity envelope of the predicted denomination.
   - **Rule 4 (Watermark Latent Contrast)**: Checks pixel intensity variance ($80 \le \sigma^2 \le 1400$) and dynamic range ($\ge 28$) in the watermark window to confirm multi-tone latent portrait presence (flags flat paper or harsh toner stamps).
3. **Conflict Resolution & Penalty Scoring**: Calculates weighted penalty points based on rule severity (`CRITICAL`, `MAJOR`, `MINOR`).
4. **Goal State Derivation**: Emits structured diagnosis:
   - `GENUINE` (Emerald Green): All rules pass; full security compliance.
   - `SUSPICIOUS` (Amber Yellow): Minor soft deviation (e.g. slight folding or camera tilt); manual inspection recommended.
   - `COUNTERFEIT` (Crimson Red): Critical failure in security thread, watermark, or substrate color.

---

### 3. Machine Learning Classifiers (CO4)

#### Supervised Learning: Decision Tree (`classifier.py`)
- Fits an `sklearn.tree.DecisionTreeClassifier(max_depth=6, criterion='entropy')` using Information Gain on statistical HSV moments and aspect ratio.
- Achieves **100% holdout test accuracy** on distinct denomination classes (₹100, ₹200, ₹500, ₹2000).
- Provides model persistence (`models/denomination_tree.joblib`) via `joblib`.
- Features human-interpretable rule exports via `export_text()`.

#### Unsupervised Learning: K-Means Clustering (`vision_pipeline.py`)
- Applies `sklearn.cluster.KMeans(n_clusters=3)` on combined HSV and CIE-Lab color spaces within the security thread ROI.
- Segments the embedded metallic thread from the paper substrate and background printing without requiring prior manual labeling.
- Measures vertical continuity percentage across horizontal scanlines.

---

## 📂 Project Structure

```
Real-Time Currency and Counterfeit checker/
├── app.py                     # Module 4: Modern Streamlit Forensic Dashboard & UI
├── vision_pipeline.py         # Module 1: OpenCV Preprocessing, Homography & K-Means
├── classifier.py              # Module 2: Supervised Decision Tree Denomination Classifier
├── expert_system.py           # Module 3: Forward-Chaining Authenticity Inference Engine
├── generate_sample_data.py    # Module 5: Dataset generation, Model training & Synthetic notes
├── test_pipeline.py           # Automated end-to-end verification test suite
├── requirements.txt           # Pinned production dependencies
├── data/
│   └── currency_features.csv  # 1,600 training samples across 4 denominations
├── models/
│   └── denomination_tree.joblib # Serialized Decision Tree weights
└── samples/                   # Pre-rendered test banknotes (Genuine + Counterfeit test cases)
    ├── genuine_100.png
    ├── genuine_200.png
    ├── genuine_500.png
    ├── genuine_2000.png
    ├── counterfeit_broken_thread.png
    ├── counterfeit_blank_watermark.png
    ├── counterfeit_wrong_color.png
    ├── counterfeit_bad_aspect_ratio.png
    └── camera_desk_scene_500.png
```

---

## 🚀 Quickstart & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/Soham51826/real-time-currency-counterfeit-checker.git
cd real-time-currency-counterfeit-checker
```

### 2. Create and Activate a Virtual Environment
**On Windows (Command Prompt / PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Verification Tests
Executes full regression tests across all genuine and counterfeit samples:
```bash
python test_pipeline.py
```

### 5. Launch the Forensic Dashboard
- **Windows One-Click**: Simply double-click `run_app.bat` (automatically detects `.venv`, system Python, or PATH Streamlit).
- **Cross-Platform Terminal**:
  ```bash
  streamlit run app.py
  ```
Access the application in your browser at `http://localhost:8501`.

---

## 🧪 Built-in Test Cases
You can instantly evaluate the system using the preset dropdown in the Streamlit sidebar:
1. **Genuine ₹100, ₹200, ₹500, ₹2000**: Evaluates valid notes; all 4 forward-chaining rules pass with `GENUINE` banner.
2. **Counterfeit: Broken Security Thread**: Simulates broken or painted metallic ribbon; fails Rule 2 (`continuity < 50%`) with `COUNTERFEIT` banner.
3. **Counterfeit: Blank Watermark Window**: Simulates flat unwatermarked paper; fails Rule 4 (`variance < 70`) with `COUNTERFEIT` banner.
4. **Counterfeit: Altered Substrate Color**: Simulates wrong printer toner; fails Rule 3 (`color consistency violation`) with `COUNTERFEIT` banner.
5. **Counterfeit: Distorted Aspect Ratio**: Simulates non-standard paper size ($1.45$ vs $>2.15$ standard); fails Rule 1 with `COUNTERFEIT` banner.
6. **Tilted Camera Scene (₹500 Desk View)**: Real-world simulated desk scene with perspective skew; tests 4-point homography warp and contour detection.
