# CurrencyGuard AI: Real-Time Currency Note Recognition & Counterfeit Feature Checker

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-red.svg)](https://streamlit.io)

A production-grade, modular, and fully runnable Python application engineered to detect currency denominations and rigorously inspect security features for counterfeit detection in real time using **Hybrid Symbolic Expert Rules, Supervised Machine Learning, and Computer Vision**.

Designed and structured in strict compliance with Academic AI Curriculum Standards:
- **CO1 (Agent Modeling)**: Formal PEAS (Performance, Environment, Actuators, Sensors) framework for the Automated Banknote Inspection Agent (ABIA).
- **CO3 (Knowledge Representation & Expert Systems)**: Forward-chaining inference engine evaluating formal central bank specifications across physical security rules.
- **CO4 (Supervised & Unsupervised Machine Learning)**:
  - **Supervised Denomination AI**: Decision Tree Classifier with entropy information gain on statistical color moments and aspect ratio (97.5% accuracy).
  - **Supervised Counterfeit AI Suite**: 4-model architecture suite (Soft-Voting Calibrated Ensemble, Deep Multi-Layer Perceptron Neural Net, Random Forest, and Gradient Boosting) trained on 3,200 forensic samples across 16 standardized physical features (100% holdout accuracy, 99.92% 5-fold CV).
  - **Explainable AI (XAI)**: Feature attribution ranking identifying the exact physical cues causing verdict outcomes.
  - **Unsupervised Learning**: K-Means Clustering on the security thread Region of Interest (ROI) to segment the metallic ribbon and compute vertical continuity.
- **Computer Vision Pipeline**: Adaptive Otsu-Canny edge detection, convex polygon contour approximation, four-point perspective homography warp to a standard $600 \times 300\text{ px}$ canvas, Gray-World illuminant constancy, and auto-orientation.

---

## 💡 Pure On-Device Architecture: Why is there an "AI / ML Verdict" without any Gemini API?

> [!IMPORTANT]
> **Understanding Artificial Intelligence in this Project (Local Machine Learning vs. Cloud LLMs):**
> In popular conversation, the term **"AI"** is frequently confused with **Cloud Generative Large Language Models (LLMs)** like Google Gemini, ChatGPT, or Claude.
> 
> In Computer Science and the university AI curriculum, **Artificial Intelligence is a broad academic discipline**:
> 1. **Zero Cloud Calls / Zero Gemini API**:
>    - This project contains **ZERO external API calls, ZERO cloud servers, and ZERO API keys**.
>    - You do not need any Gemini API key, billing account, or internet connection to run this system.
> 2. **Where the "AI Model Verdict" and "AI Authenticity" Actually Come From**:
>    - The verdict and authenticity percentage displayed on your dashboard are computed by **Classical Scikit-Learn Supervised Machine Learning models** running **100% locally on your computer's CPU in $<1\text{ ms}$**.
>    - Located in [`counterfeit_classifier.py`](counterfeit_classifier.py), this local AI engine ingests **16 mathematical computer vision forensic features** (Shannon texture entropy of paper fibers, watermark luminance variance, intaglio microtexture edge density, security thread vertical continuity, and CIE-Lab perceptual color distance $\Delta E$).
>    - It evaluates these features using a **Soft-Voting Calibrated Ensemble** (combining a 120-tree Random Forest, a Deep Multi-Layer Perceptron neural network with 64-32-16 ReLU nodes, and Gradient Boosted Decision Trees).
> 3. **The Neuro-Symbolic Consensus (CO3 + CO4)**:
>    - The final **Tri-Color Verdict Banner** (`GENUINE`, `SUSPICIOUS`, `COUNTERFEIT`) is computed by a **Neuro-Symbolic Expert System** ([`expert_system.py`](expert_system.py)):
>      - **Symbolic AI (CO3)**: Evaluates deterministic production rules based on official Reserve Bank of India (RBI) physical security specifications (thread continuity, watermark presence, dimension bounds, and substrate color).
>      - **Sub-Symbolic Machine Learning AI (CO4)**: Provides a probabilistic authenticity score based on statistical pattern recognition of forensic paper features.
>    - When **both** the physical rules pass and the local ML model indicates authentic banknote substrate, the system issues the **GENUINE** consensus verdict!

---

## 🏛️ System Architecture

```
                                [ CAMERA / IMAGE INPUT ]
                                           │
                           ┌───────────────▼────────────────┐
                           │   Module 1: Vision Pipeline    │
                           │  - Adaptive Canny & Contours   │
                           │  - 4-Point Homography Warp     │
                           │  - 4-Way Auto-Orientation      │
                           │  - Dual Color Constancy        │
                           │  - ROI Extraction (Strip, WM)  │
                           │  - 16-D Forensic Feature Vec   │
                           └───────┬──────────────┬─────────┘
                                   │              │
             Extracted HSV Moments │              │ Standardized 16-D Vector
             & Warped Canvas Image │              │ & Cropped Forensic ROIs
                                   ▼              ▼
                     ┌────────────────────────┐  ┌───────────────────────┐
                     │  Decision Tree (CO4)   │  │ Counterfeit AI Suite  │
                     │  - Entropy Splitting   │  │ - Ensemble, MLP, RF,  │
                     │  - Denomination Model  │  │   GBDT & XAI Rankings │
                     └──────────┬─────────────┘  └──────────┬────────────┘
                                │ Denomination              │ Authenticity Prob & XAI
                                └──────────────┬────────────┘
                                               ▼
                               ┌────────────────────────────────┐
                               │ Module 3: Expert System Engine │
                               │ - Forward-Chaining (CO3)       │
                               │ - Rules 1-4 Verification       │
                               │ - Neuro-Symbolic Consensus     │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │ Module 4: Streamlit UI         │
                               │ - Real-time Visual Overlays    │
                               │ - Tri-Color Authenticity Banner│
                               │ - Local AI Forensic Tab Panels │
                               │ - Forward-Chaining Audit HUD   │
                               └────────────────────────────────┘
```

---

## 📘 Academic Curriculum Mapping

### 1. Agent Modeling (CO1): Formal PEAS Setup
The system models an autonomous rational agent encapsulated as `AgentPEAS` in `expert_system.py`:

| PEAS Element | Specification |
| :--- | :--- |
| **Performance Measures (P)** | - Denomination recognition accuracy $\ge 95\%$ (Achieved: $97.5\%$)<br>- Counterfeit detection accuracy: $100.0\%$ (Holdout), $99.92\%$ ($5$-Fold CV)<br>- False negative rate for counterfeits: $0.0\%$<br>- Real-time frame processing latency $< 150\text{ ms}$ |
| **Environment (E)** | - Variable ambient lighting (incandescent 2700K, cool LED, ambient daylight)<br>- Hand jitter, rotation, tilt, and perspective distortion<br>- Banknote physical condition (crisp uncirculated, creased, soiled)<br>- Adversarial counterfeits (color photocopies, broken thread imitations, blank watermark paper, altered dyes) |
| **Actuators (A)** | - Dynamic bounding box overlays highlighting inspected ROIs<br>- High-contrast Tri-Color Verdict Banner (`GENUINE`, `SUSPICIOUS`, `COUNTERFEIT`)<br>- Forward-chaining diagnostic checklist with measured values vs thresholds<br>- Real-time AI authenticity confidence gauges and XAI attribution rankings |
| **Sensors (S)** | - Live RGB optical webcam video stream (`st.camera_input`)<br>- High-resolution image upload sensor (`st.file_uploader`)<br>- OpenCV edge digitizer, color moment sensor & CIE-Lab Delta-E calculator |

---

### 2. Forward-Chaining Expert System & Central Bank Rule Verification (CO3)
Located in `expert_system.py`, the `VerificationExpertSystem` implements an AI forward-chaining cycle over 4 formal physical security rules:
1. **Working Memory Ingestion**: Populates facts (aspect ratio, K-Means continuity percentage, watermark intensity variance, entropy, CIE-Lab Delta-E, AI authenticity probability).
2. **Rule Base Evaluation**:
   - **Rule 1 (Physical Dimensions)**: Verifies note width-to-height aspect ratio against official specifications within $\pm 10\%$ perspective tolerance.
   - **Rule 2 (Security Thread Continuity)**: Verifies segmented metallic ribbon maintains $\ge 65\%$ continuity (transmitted) or 3-8 periodic windows with $\ge 45\%$ continuity (reflected).
   - **Rule 3 (Substrate Color Consistency)**: Validates that mean Hue and Saturation fall within the chromaticity envelope of the predicted denomination.
   - **Rule 4 (Watermark Latent Contrast)**: Checks pixel intensity variance ($25 \le \sigma^2 \le 1400$) and dynamic range ($\ge 26.5$) in the watermark window to confirm multi-tone latent portrait presence.
3. **Neuro-Symbolic Consensus**:
   - `GENUINE` (Emerald Green): Both symbolic rules and Machine Learning model concur (Score $\ge 85\%$ & AI Authenticity $\ge 50\%$).
   - `SUSPICIOUS` (Amber Yellow): Divergent conclusion between rules and AI model, or soft wear degradation; prompts manual cashier review.
   - `COUNTERFEIT` (Crimson Red): Critical failure in thread/watermark or low AI authenticity score ($< 35\%$).

---

### 3. Machine Learning Classifiers (CO4)

#### Supervised Denomination Classifier (`classifier.py`)
- Fits an `sklearn.tree.DecisionTreeClassifier(max_depth=6, criterion='entropy')` using Information Gain on statistical HSV moments and aspect ratio.
- Achieves **95.3% holdout test accuracy** on active circulating denomination classes (₹100 Lavender, ₹200 Bright Yellow, ₹500 Stone Grey) with sub-millisecond execution ($<0.8\text{ ms}$).

#### Supervised Counterfeit AI Suite (`counterfeit_classifier.py`)
- Operates on a **16-dimensional standardized forensic feature vector**:
  1. Aspect ratio deviation
  2. Security strip vertical continuity (%)
  3. Security thread window segment count
  4. Security thread mean ribbon thickness
  5. Security thread width standard deviation
  6. Watermark grayscale intensity variance
  7. Watermark dynamic luminance contrast range
  8. Watermark high-frequency Laplacian sharpness
  9. Watermark Shannon texture entropy (paper fiber randomness)
  10. Substrate hue deviation from canonical currency standard
  11. Substrate saturation mean
  12. Substrate saturation standard deviation
  13. Substrate value (brightness) mean
  14. CIE-Lab perceptual color Delta-E
  15. Intaglio microprint edge density
  16. Denomination numerical identifier
- Evaluates 4 distinct AI architectures:
  1. **Soft-Voting Ensemble AI**: 100% Test Accuracy, 99.92% 5-Fold Cross-Validation.
  2. **Deep Multi-Layer Perceptron (MLP)**: 3-layer neural network (64-32-16 ReLU nodes) with Adam optimizer and early stopping (99.88% Accuracy).
  3. **Random Forest Classifier**: 120 orthogonal decision trees with Gini impurity splitting (100% Accuracy).
  4. **Gradient Boosted Decision Trees (GBDT)**: Sequential boosting for high-margin separation (100% Accuracy).
- **Explainable AI (XAI)**: Computes local feature attribution rankings to explain why a note is flagged.

#### Unsupervised Learning: K-Means Clustering (`vision_pipeline.py`)
- Applies `sklearn.cluster.KMeans(n_clusters=3)` on combined HSV and CIE-Lab color spaces within the security thread ROI.
- Segments the embedded metallic thread from paper substrate and background print without requiring manual ground truth annotations.

---


## 📂 Project Structure

```
Real-Time Currency and Counterfeit checker/
├── app.py                         # Module 4: Modern Streamlit Forensic Dashboard & Dual UI
├── vision_pipeline.py             # Module 1: OpenCV Preprocessing, Homography, K-Means & 16-D Features
├── classifier.py                  # Module 2: Supervised Decision Tree Denomination Classifier
├── counterfeit_classifier.py      # Module 2B: Supervised Counterfeit AI Suite (Ensemble, MLP, RF, GBDT)
├── expert_system.py               # Module 3: Forward-Chaining Inference Engine & Neuro-Symbolic Consensus
├── generate_sample_data.py        # Module 5: Dataset synthesis & Model training pipeline
├── test_pipeline.py               # Automated end-to-end verification test suite
├── test_inspect_samples.py        # Advanced interactive sample inspection utility
├── generate_presentation_pptx.py  # Presentation generator (12-slide dark-mode technical defense)
├── generate_report_docx.py        # Academic project report generator (.docx)
├── requirements.txt               # Pinned local production dependencies
├── data/
│   ├── currency_features.csv      # Denomination training data (1,200 samples)
│   └── counterfeit_detection_dataset.csv # 16-D Counterfeit forensic dataset (3,200 samples)
├── models/
│   ├── denomination_tree.joblib   # Serialized Denomination Tree model
│   ├── counterfeit_ai_model.joblib# Serialized Counterfeit AI models (Ensemble, MLP, RF, GBDT)
│   └── counterfeit_model_metrics.json # Performance metrics, CV scores, confusion matrices
└── samples/                       # Pre-rendered test banknotes (Genuine + Defective test cases)
    ├── genuine_100.png
    ├── genuine_200.png
    ├── genuine_500.png
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
Executes full regression tests across all genuine and counterfeit test cases:
```bash
python test_pipeline.py
```

### 6. Launch the Forensic Dashboard
- **Windows One-Click**: Simply double-click `run_app.bat` (automatically detects `.venv`, system Python, or PATH Streamlit).
- **Cross-Platform Terminal**:
  ```bash
  streamlit run app.py
  ```
Access the application in your browser at `http://localhost:8501`.

---

## 🧪 Built-in Test Cases
You can instantly evaluate the system using the preset dropdown in the Streamlit sidebar:
1. **Genuine ₹100, ₹200, ₹500**: Evaluates active circulating banknotes (reflecting the official demonetization/withdrawal of the ₹2000 note); all 4 forward-chaining rules pass with `GENUINE` banner.
2. **Counterfeit: Broken Security Thread**: Simulates broken or painted metallic ribbon; fails Rule 2 (`continuity < 50%`) with `COUNTERFEIT` banner.
3. **Counterfeit: Blank Watermark Window**: Simulates flat unwatermarked paper; fails Rule 4 (`variance < 70`) with `COUNTERFEIT` banner.
4. **Counterfeit: Altered Substrate Color**: Simulates wrong printer toner; fails Rule 3 (`color consistency violation`) with `COUNTERFEIT` banner.
5. **Counterfeit: Distorted Aspect Ratio**: Simulates non-standard paper size ($1.45$ vs $>2.15$ standard); fails Rule 1 with `COUNTERFEIT` banner.
6. **Tilted Camera Scene (₹500 Desk View)**: Real-world simulated desk scene with perspective skew; tests 4-point homography warp and contour detection.
