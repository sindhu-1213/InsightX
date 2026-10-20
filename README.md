# 🩻 INSIGHT-X: AI-Powered Multimodal Clinical Decision Support System

[![CI](https://github.com/your-username/insightx/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/insightx/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel%20%2F%20PWA-00d4ff?style=flat&logo=vercel)](https://insightx.web.app)
[![API Status](https://img.shields.io/badge/Hugging%20Face-Docker%20API-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!CAUTION]
> **Research Demo Only — Not a Certified Medical Device**  
> InsightX is an academic research demonstration and is not certified as a medical device by the FDA, CE, or any regulatory health authority. Model outputs represent algorithmic predictions and visual attention heatmaps for investigational and research purposes only, not clinical diagnoses. Model outputs must never be used to guide diagnostic decisions, patient management, or clinical interventions. Always consult a qualified radiologist or physician.

---

## 🎬 Product Demo

<!-- DEMO GIF PLACEHOLDER -->
```markdown
![INSIGHT-X Interactive Demo](docs/assets/demo.gif)
```
*(Demo GIF placeholder: Drag-and-drop radiograph upload, iPhone HEIC in-browser conversion, low-latency ONNX classification, and real-time Grad-CAM opacity adjustments)*

---

## 📌 Executive Summary

INSIGHT-X is an academic research prototype for automated chest radiograph finding classification.  
The vision model fine-tunes a MobileNetV2 backbone on the Indiana University Chest X-Ray collection using patient-level group splits.  
It deploys ONNX models for low-latency inference (`insightx.onnx`, 8.51 MB) and zero-TensorFlow serving-time Grad-CAM explainability in pure NumPy (`insightx_gradcam.onnx`, 8.50 MB).  
Platt probability scaling fitted on validation data improves probability calibration on the test split, reducing Brier score from 0.2167 to 0.1947.  
The client delivers a responsive React Progressive Web App with Firebase Authentication and Firestore history logging.  

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client["Frontend Client (React PWA on Vercel)"]
        UI["React 19 + Vite Application"]
        PWA["PWA Service Worker & Manifest<br/>(Offline Precaching & Mobile Install)"]
        PRE["Client-Side Preprocessor<br/>(HEIC/HEIF to JPEG & 1024px Resize)"]
        
        UI --> PWA
        UI --> PRE
    end

    subgraph Firebase["Cloud Services (Firebase)"]
        AUTH["Firebase Authentication<br/>(Email/Password & Google Sign-In)"]
        DB[("Cloud Firestore<br/>users/{uid}/analyses Records")]
    end

    subgraph Backend["Serving Backend (FastAPI on Hugging Face Spaces)"]
        API["FastAPI Application (Port 7860)"]
        VERIFY["Token Verifier & Rate Limiter<br/>(firebase-admin SDK)"]
        ONNX["ONNX Runtime Engine<br/>(insightx_gradcam.onnx opset 13)"]
        CALIB["Platt Scaling Calibrator<br/>(models/calibration.json)"]
        CAM["NumPy Grad-CAM Engine<br/>(Zero-TensorFlow Serving Heatmap)"]

        API --> VERIFY
        API --> ONNX
        ONNX --> CALIB
        ONNX --> CAM
    end

    UI <-->|"Authentication Tokens & State"| AUTH
    UI -->|"Store Structured Findings (No Raw Images)"| DB
    PRE -->|"POST /predict (Multipart Image + Bearer Token)"| API
    API -->|"Calibrated Probability & Base64 Grad-CAM"| UI
```

---

## 📊 Empirical Evaluation & Metrics

All metrics reported below were computed on the held-out patient-level test split ($N = 1,095$ images across $564$ unique patient UIDs) as recorded in `reports/metrics.json` and `reports/comparison.md`:

### Model Performance (Test Split)

| Metric | Score | Specification |
|---|---|---|
| **ROC-AUC** | **0.7563** | **95% Bootstrap CI: [0.7180, 0.7921]** (1,000 resamples by patient `uid`) |
| **Accuracy** | **66.48%** (0.6648) | Validation-chosen Youden threshold ($0.6522$) |
| **F1-Score** | **0.6697** | Harmonic mean of precision & recall (Abnormal positive) |
| **Sensitivity** (@ Youden Threshold) | **56.45%** (0.5645) | 372 / 659 abnormal images flagged |
| **Specificity** (@ Youden Threshold) | **81.65%** (0.8165) | 356 / 436 normal images identified |
| **Sensitivity @ 90% Specificity** | **45.37%** (0.4537) | Operating threshold: $0.7291$ (299 / 659 abnormal flagged) |
| **Specificity Achieved @ 90% Spec Threshold** | **87.84%** (0.8784) | 383 / 436 normal images correctly identified |
| **Brier Score (Uncalibrated)** | **0.2167** | Raw neural network sigmoid output |
| **Brier Score (Calibrated)** | **0.1947** | Post-Platt scaling fitted on validation split |

### Confusion Matrix (Test Split, Threshold = 0.6522)

| Ground Truth \ Predicted | Predicted Normal (`0`) | Predicted Abnormal (`1`) |
|---|---|---|
| **Actual Normal (`0`)** | **356** (True Negative) | **80** (False Positive) |
| **Actual Abnormal (`1`)** | **287** (False Negative) | **372** (True Positive) |

---

## 🔬 Modality Comparison: Text vs Image vs Fused

Test split baseline comparison as documented in `reports/comparison.md` and `reports/baselines.json`:

| Modality / Baseline Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Majority Class Baseline** | 60.18% | 0.7514 | 0.5000 |
| **Frozen MobileNetV2 GAP** (C=0.01) | 64.84% | 0.6980 | 0.7182 |
| **Text-Only (Findings TF-IDF, 1,000 features)** | 90.68% | 0.9257 | 0.9676 |
| **Fused (Image + Text Concatenation)** | 89.50% | 0.9119 | 0.9629 |
| **Fine-Tuned MobileNetV2 (Served Model)** | **66.48%** | **0.6697** | **0.7563** |

### Why the Text Baseline Scores High (Label Leakage)
The Indiana University disease annotations (`Problems` / MeSH terms) were extracted directly from the written radiology reports by clinical catalogers. As a result, tokens in the `findings` narrative share direct lexical and semantic overlap with the assigned abnormality categories. Evaluating a text model against labels extracted from that exact narrative yields artificially high performance due to label circularity.

### Why the Served System is Image-Only
In actual hospital intake workflows, an official radiology report **does not exist at scan acquisition time**:
1. An automated decision support tool must analyze raw radiograph pixel data immediately after acquisition.
2. Requiring an authored radiologist report at inference time would render the AI redundant, as a human clinician would already have completed the diagnostic reading.
3. Therefore, InsightX serving is strictly **image-only**, taking chest radiographs as input to produce classifications and Grad-CAM attention overlays.

---

## ⚡ Run Locally in Three Commands

You can run both the FastAPI ONNX backend and the Vite React frontend locally using three commands:

```bash
# 1. Install Python and Web dependencies
pip install -r requirements.txt && npm --prefix web install

# 2. Launch the FastAPI ONNX serving API (starts on http://localhost:7860)
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload

# 3. Launch the React PWA frontend (starts on http://localhost:5173)
npm --prefix web run dev
```

---

## 🚀 Hugging Face Spaces Deployment (Docker SDK)

The API is containerized using `api/Dockerfile` with a minimal runtime environment (no TensorFlow dependency, pure ONNX Runtime on CPU) and exposes port `7860`:

```bash
# Build Docker image
docker build -f api/Dockerfile -t insightx-api .

# Run locally or in cloud container runtime
docker run -p 7860:7860 \
  -e FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account","project_id":"..."}' \
  -e ALLOWED_ORIGINS="*" \
  insightx-api
```

### Endpoints
- `GET /health`: Health status and active model version.
- `GET /samples`: List of available demo sample IDs.
- `GET /samples/{id}/predict`: Guest trial prediction without authentication (rate limited to 10 req/min per IP).
- `POST /predict`: Authenticated prediction for user-uploaded radiographs (`PNG`, `JPEG`, max 8 MB). Requires a valid Firebase Bearer token with a verified email address (rate limited to 10 req/min per UID).

---

## ⚠️ System Limitations & Failure Modes

1. **Single-Institution Cohort**: All training and testing cases originate from a single health system (Indiana University). The model has not been evaluated across multi-institutional datasets (e.g. CheXpert, MIMIC-CXR) and may experience domain shift under different radiography machines, exposure settings, or patient demographics.
2. **Label Noise from Report-Derived Annotations**: Ground truth labels were generated from automated NLP and MeSH cataloging of unstructured clinical reports rather than multi-reader panel adjudication or histopathological confirmation.
3. **Mixed Frontal and Lateral Views**: The collection mixes posteroanterior (PA), anteroposterior (AP), and lateral projections under a unified binary target without projection-specific conditioning.
4. **Subtle & Micro-Lesions**: Global Average Pooling across the $7 \times 7$ feature grid can dilute localized focal abnormalities (such as solitary pulmonary nodules $< 1$ cm or apical pneumothoraces).
5. **Medical Artifacts**: Foreign bodies, pacemakers, surgical clips, monitoring leads, and anatomical rotation may generate spurious Grad-CAM activations.
6. **No Sub-Pathology Discrimination**: The binary classification targets broad abnormality, without distinguishing between benign chronic changes (e.g., granulomas) and acute emergencies (e.g., tension pneumothorax).
7. **No Clinical Validation**: The system has not undergone prospective clinical trials, reader study validation, or safety audits. It must not be deployed for clinical care.

---

## 📄 License & Attribution

This project is open-source under the [MIT License](LICENSE).

### Dataset Attribution
Chest radiograph images and radiology reports are sourced from the **Indiana University Chest X-ray Collection** (Demner-Fushman et al., 2016, accessed via the NLM Open-i service).
- **Citation**: Demner-Fushman D, Kohli MD, Rosenman MB, Shooshan SE, Rodriguez L, Antani S, Thoma GR, McDonald CJ. *Preparing a collection of radiology examinations for distribution and retrieval*. Journal of the American Medical Informatics Association (JAMIA), 2016 Mar;23(2):304-310. doi:10.1093/jamia/ocv080.
- **Provider**: U.S. National Library of Medicine (NLM), National Institutes of Health, Open-i service.
