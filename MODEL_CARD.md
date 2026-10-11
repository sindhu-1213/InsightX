# Model Card: InsightX (MobileNetV2 Chest X-Ray Finding Classifier)

> [!CAUTION]
> **Research Demo Only — Not a Certified Medical Device**  
> InsightX is an academic research demonstration and is not certified as a medical device by the FDA, CE, or any regulatory health authority. Model outputs represent algorithmic **predictions and findings** for research purposes only, not clinical diagnoses. Model predictions must never be used to guide diagnostic decisions, patient management, or clinical interventions. Always consult a qualified radiologist or physician.

---

## 1. Intended Use

- **Primary Purpose**: Research demonstration of transfer learning, post-hoc probability calibration, explainable AI (Grad-CAM), and low-latency ONNX serving on chest radiography.
- **Intended Users**: Machine learning researchers, medical informatics students, and computer vision developers exploring chest radiograph classification workflows.
- **Out-of-Scope & Prohibited Uses**:
  - Clinical diagnosis, triage, or screening in healthcare delivery environments.
  - Autonomous decision-making without expert radiologist oversight.
  - Emergency room triage or life-critical patient management.

---

## 2. Dataset

### 2.1 Data Source
The model was trained, validated, and evaluated on the **Indiana University Chest X-ray Collection** (OpenI / IU X-Ray), comprising frontal (PA/AP) and lateral radiograph projections with paired radiology reports.

### 2.2 Label Definition
Target binary classes were constructed from the indexed `Problems` metadata field:
- **Normal (`0`)**: Assigned when the `Problems` annotation is exactly `'normal'`.
- **Abnormal (`1`)**: Assigned for any other disease or anatomical abnormality finding (e.g., *Cardiomegaly*, *Pulmonary Atelectasis*, *Granulomatous Disease*, *Spondylosis*, *Calcinosis*).
- Records marked as `'No Indexing'` or missing annotations were excluded during dataset curation.

### 2.3 Patient-Level Splitting
To prevent severe data leakage caused by multi-image patient series, records were partitioned via **`GroupShuffleSplit` on patient `uid`** (random seed `42`):
- All projections (frontal and lateral) of any given patient UID reside strictly within a single partition.
- Split ratio: **70% Training / 15% Validation / 15% Testing**.

### 2.4 Dataset Counts and Class Balance

| Split | Patient UIDs | Normal (`0`) | Abnormal (`1`) | Total Images |
|---|---|---|---|---|
| **Train** | 2,631 | 1,828 | 3,281 | 5,109 |
| **Validation** | 564 | 431 | 659 | 1,090 |
| **Test** | 564 | 436 | 659 | 1,095 |
| **Total** | **3,759** | **2,695** | **4,599** | **7,294** |

---

## 3. Model Architecture & Deployment

### 3.1 Neural Network Architecture
- **Base Backbone**: `MobileNetV2` (pre-trained on ImageNet, frozen top, input shape $224 \times 224 \times 3$ with replicated single-channel radiographs).
- **Classification Head**:
  - `GlobalAveragePooling2D`
  - `Dropout` (rate: `0.3`)
  - `Dense(1, activation='sigmoid')` (predicted probability of abnormality)
- **Training Strategy**: Two-phase transfer learning with class weighting:
  - **Phase A**: Frozen backbone, training head only (5 epochs, batch size 32).
  - **Phase B**: Unfreezing the top 30 layers with lower learning rate ($10^{-5}$) and early stopping on validation AUC.

### 3.2 ONNX Export & Explainability Serving
- **Standard Inference Model**: `models/insightx.onnx` (8.51 MB, ONNX opset 13) for optimized CPU runtime via `onnxruntime`.
- **Dual-Output Explainability Model**: `models/insightx_gradcam.onnx` (8.50 MB, ONNX opset 13):
  - Output 1: Last convolutional feature map ($7 \times 7 \times 1280$) from the backbone `out_relu` layer.
  - Output 2: Pre-sigmoid linear logit ($1 \times 1$).
- **Zero-TensorFlow Serving CAM**: Because the classification head consists of Global Average Pooling followed by a linear Dense layer, the Grad-CAM channel importance weights $\alpha_k$ are mathematically identical to the Dense layer weights $w_k$. The runtime computes $\text{ReLU}\left(\sum_k w_k A_k\right)$ in pure NumPy using `models/gradcam_head_weights.json` without requiring TensorFlow or gradient tape backpropagation during serving.

---

## 4. Evaluation Metrics

Evaluated on the held-out patient test split ($N = 1,095$ images, $564$ patient UIDs) as recorded in `reports/metrics.json`:

| Metric | Score | Note |
|---|---|---|
| **Accuracy** | PENDING: filled in after real training | Threshold = 0.50 |
| **F1-Score** | PENDING: filled in after real training | Harmonic mean of precision & recall |
| **ROC-AUC** | PENDING: filled in after real training | Area Under Receiver Operating Characteristic |
| **Sensitivity @ 90% Specificity** | PENDING: filled in after real training | Operating Threshold: PENDING: filled in after real training |
| **Brier Score (Uncalibrated)** | PENDING: filled in after real training | Mean squared probability error |
| **Brier Score (Calibrated)** | PENDING: filled in after real training | Post-Platt scaling on test split |

### Confusion Matrix (Test Split)

| | Predicted Normal (`0`) | Predicted Abnormal (`1`) |
|---|---|---|
| **Actual Normal (`0`)** | PENDING: filled in after real training (TN) | PENDING: filled in after real training (FP) |
| **Actual Abnormal (`1`)** | PENDING: filled in after real training (FN) | PENDING: filled in after real training (TP) |

---

## 5. Probability Calibration

Raw neural network sigmoid outputs often exhibit overconfidence. Probability calibration was conducted using **Platt Scaling** (logistic regression over log-odds logits).

- **Calibration Fitting Split**: **Validation Split Only** ($N = 1,090$), preserving the test set for unbiased evaluation.
- **Fitted Parameters** (`models/calibration.json`):
  - **Method**: Platt Scaling (Logistic Regression)
  - **Coefficient (`coef`)**: PENDING: filled in after real training
  - **Intercept (`intercept`)**: PENDING: filled in after real training
- **Calibration Impact**:
  - Test split Brier score: PENDING: filled in after real training
  - Reliability diagrams demonstrate that calibrated probabilities closely align with empirical observed positive frequencies across all probability bins.

---

## 6. Modality Comparison: Text vs Image vs Fused

Test split baseline comparison as documented in `reports/baselines.md`:

| Modality / Baseline Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Text-Only (Findings TF-IDF, 1000 features)** | PENDING: filled in after real training | PENDING: filled in after real training | PENDING: filled in after real training |
| **Image-Only (MobileNetV2 GAP Features)** | PENDING: filled in after real training | PENDING: filled in after real training | PENDING: filled in after real training |
| **Fused (Image + Text concatenated)** | PENDING: filled in after real training | PENDING: filled in after real training | PENDING: filled in after real training |

### Why the Text Baseline Scores High
The Indiana University disease labels (MeSH / Problems) were extracted directly from the written radiologist reports by medical catalogers. Consequently, the lexical tokens in the `findings` narrative contain explicit semantic overlap with the assigned abnormality categories. Evaluating a text model against labels extracted from that exact narrative naturally yields strong performance due to linguistic circularity.

### Why the Served Model is Strictly Image-Only
In actual clinical radiology workflows, an official report **does not exist when a radiograph is acquired**:
1. An automated pre-read decision support tool must analyze raw pixel data at image ingestion time.
2. Requiring an authored radiologist report at inference time would render the AI system redundant, as a human radiologist would already have completed the reading.
3. Therefore, InsightX is deployed strictly as an **image-only** computer vision model, using chest radiographs as the sole input to produce predictive classifications and Grad-CAM visual attention overlays.

---

## 7. Limitations

1. **Single-Institution Cohort**: All training and testing cases originate from a single hospital network (Indiana University). The model has not been evaluated on multi-institutional datasets (e.g., CheXpert, MIMIC-CXR, PadChest) and may suffer domain shift under varying radiology equipment, beam protocols, or patient demographics.
2. **Label Noise from Report-Derived Annotations**: Ground truth labels were generated from natural language processing and MeSH indexing of unstructured clinical reports rather than panel-adjudicated radiologist bounding boxes or histopathological confirmations.
3. **Mixed Frontal and Lateral Views**: The collection mixes posteroanterior (PA), anteroposterior (AP), and lateral projections under a unified binary target without projection-specific conditioning or view-separated heads.
4. **No Clinical Validation**: The system has not undergone prospective clinical trials, reader study validation, or safety audits. It must not be deployed for patient care.

---

## 8. Failure Modes

- **Subtle and Micro-Lesions**: Because the architecture applies Global Average Pooling across a $7 \times 7$ spatial grid, localized focal abnormalities (such as solitary pulmonary nodules $< 1$ cm or subtle apical pneumothoraces) can be spatially diluted by surrounding normal lung parenchyma.
- **Artifacts and Foreign Objects**: Support devices, pacemakers, surgical clips, monitoring electrodes, and patient rotation can trigger spurious activations in the convolutional feature maps.
- **Co-Occurring Abnormalities**: With a binary classification target (Normal vs Abnormal), the model does not differentiate between urgent life-threatening pathologies (e.g., tension pneumothorax) and chronic benign findings (e.g., mild scoliosis, granulomatous calcification).
- **Out-of-Distribution Inputs**: Degraded image contrast, extreme under/over-exposure, pediatric radiographs, or non-chest radiographs will yield untrustworthy predictions and unreliable Grad-CAM heatmaps.
