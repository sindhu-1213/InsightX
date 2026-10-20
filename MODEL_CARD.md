# Model Card: InsightX (MobileNetV2 Chest X-Ray Finding Classifier)

> [!CAUTION]
> **Research Demonstration Only — Not a Certified Medical Device**  
> InsightX is an academic research demonstration and is not certified as a medical device by the FDA, CE, or any health regulatory authority. Model outputs represent algorithmic predictions and activations for investigational research purposes only. **This system must not be used for medical screening, clinical diagnosis, patient management, or clinical decision support.** Always consult a qualified radiologist or physician.

---

## 1. Intended Use

- **Primary Purpose**: Academic research investigation into transfer learning, probability calibration (Platt scaling), explainable AI (Grad-CAM), and low-latency ONNX serving on chest radiography.
- **Intended Users**: Machine learning researchers, medical informatics students, and computer vision developers exploring chest radiograph classification workflows.
- **Prohibited & Out-of-Scope Uses**:
  - Any clinical diagnostic decision, screening, or patient triage in healthcare delivery settings.
  - Autonomous evaluation without qualified radiologist review.
  - Emergency room triage or life-critical patient management.

---

## 2. Dataset & Cohort Accounting

### 2.1 Data Source
The model was trained, validated, and evaluated on the **Indiana University Chest X-ray Collection** (OpenI / IU X-Ray), comprising frontal (PA/AP) and lateral radiograph projections with paired radiology reports.

### 2.2 Data Cleaning & Exclusions Accounting
- **Input Projections Rows**: 7,466 (`data/raw/indiana_projections.csv`)
- **Input Reports Rows**: 3,851 (`data/raw/indiana_reports.csv`)
- **Rows Dropped (Total: 173)**:
  - **Unmatched UID**: 0 rows dropped.
  - **Missing Image File on Disk**: 1 row dropped (`1000_IM-0003-1001.dcm.png`).
  - **`'No Indexing'` Annotation**: 172 rows dropped (unindexed or missing report annotations).
  - **Empty / NaN Problem String**: 0 rows dropped.
- **Retained Clean Dataset**: **7,293 images** across **3,759 unique patients**.
  - **Normal (`0`)**: 2,695 images (37.0%)
  - **Abnormal (`1`)**: 4,598 images (63.0%)

### 2.3 Label Definition & Known Limits
- **Definition**: Target binary labels were assigned based on the report-indexed `Problems` metadata field:
  - **Normal (`0`)**: Assigned when `Problems` is strictly `'normal'`.
  - **Abnormal (`1`)**: Assigned for any other clinical finding or pathology term (e.g., *Cardiomegaly*, *Pulmonary Atelectasis*, *Calcinosis*, *Pleural Effusion*).
- **Label Limits**:
  - Labels were derived by catalogers using MeSH terminology extracted from written radiologist reports, **not from independent radiologist review of the images**.
  - The same study-level label is assigned to both views (frontal and lateral) of a study, even if a focal finding is only visible on one projection.
  - Studies marked `'No Indexing'` were excluded.

### 2.4 Patient-Level Partitioning
To prevent data leakage across multi-image studies, records were split using **`GroupShuffleSplit` on patient `uid`** (random seed `42`):
- All projections of any patient reside strictly within a single split.

| Split | Patient UIDs | Normal (`0`) | Abnormal (`1`) | Total Images | Class Balance (% Abnormal) |
|---|---|---|---|---|---|
| **Train** | 2,631 | 1,828 | 3,280 | 5,108 | 64.21% |
| **Validation** | 564 | 431 | 659 | 1,090 | 60.46% |
| **Test** | 564 | 436 | 659 | 1,095 | 60.18% |
| **Total** | **3,759** | **2,695** | **4,598** | **7,293** | **63.05%** |

---

## 3. Model Architecture & Training

### 3.1 Architecture
- **Backbone**: `MobileNetV2` (pre-trained ImageNet weights, `include_top=False`, input shape $224 \times 224 \times 3$, single-channel grayscale replicated across 3 RGB channels, float32 scaled to $[-1, 1]$ via `preprocess_input`).
- **Backbone Feature Layer**: Named last convolutional layer `out_relu` (output shape $7 \times 7 \times 1280$).
- **Head**:
  - `GlobalAveragePooling2D(name='gap')`
  - `Dropout(0.3, name='head_dropout')`
  - `Dense(1, activation='sigmoid', name='prediction')`
- **Batch Normalization**: Kept frozen (`training=False`) throughout both training phases to preserve pre-trained population statistics.
- **Random Seed**: 42 across Python, NumPy, and TensorFlow.
- **Class Weights**: Balanced weights computed from train split: class `0` (Normal) = `1.3972`, class `1` (Abnormal) = `0.7787`.

### 3.2 Two-Phase Training Strategy
- **Phase A**: Base frozen, head trained with Adam learning rate $10^{-3}$ for 5 epochs.
- **Phase B**: Top 30 layers unfrozen, Adam learning rate $10^{-5}$ for 14 epochs (early stopping on validation AUC, patience 5).

### 3.3 Training Compute & Outlier Epoch
Recorded in `reports/train_log.json`:
- **Phase A (5 epochs)**: [74.00, 66.96, 68.01, 69.64, 69.33] seconds (mean: 69.59 s/epoch, total: 347.94 seconds).
- **Phase B (14 epochs)**: [100.23, 97.40, 99.17, 33616.63, 82.61, 106.59, 105.33, 130.17, 104.02, 98.45, 101.89, 92.81, 94.39, 140.17] seconds.
- **Note on Compute Outlier**: Epoch 4 of Phase B recorded an execution time of **33,616.63 seconds** because the host machine entered sleep state during overnight execution.
- **Active Compute**: Excluding this sleep-state outlier epoch, the remaining 13 Phase B epochs averaged **104.09 seconds/epoch** (total active Phase B compute: 1,353.23 seconds). Total active training compute across both phases was **1,701.17 seconds** (~28.35 minutes). Logged runtime with sleep was 35,320.88 seconds.

---

## 4. Test Split Evaluation Metrics

Evaluated exactly once on the held-out patient test split ($N = 1{,}095$ images, 564 patients) as recorded in `reports/metrics.json`:

| Metric | Measured Score | Specification / Operating Threshold |
|---|---|---|
| **ROC-AUC** | **0.7563** | **95% Bootstrap CI: [0.7180, 0.7921]** (1,000 resamples by patient `uid`) |
| **Accuracy** | **0.6648** (66.48%) | At validation-chosen Youden threshold `0.6522` |
| **F1-Score** | **0.6697** | Harmonic mean of precision & recall (Abnormal class) |
| **Sensitivity** (@ Youden Threshold) | **0.5645** (56.45%) | 372 / 659 abnormal images flagged |
| **Specificity** (@ Youden Threshold) | **0.8165** (81.65%) | 356 / 436 normal images flagged |
| **Sensitivity @ 90% Spec Threshold** | **0.4537** (45.37%) | At validation 90% spec threshold `0.7291` (299 / 659 abnormal flagged) |
| **Specificity Achieved @ 90% Spec** | **0.8784** (87.84%) | 383 / 436 normal images correctly identified |
| **Brier Score (Uncalibrated)** | **0.2167** | Mean squared probability error before Platt scaling |
| **Brier Score (Calibrated)** | **0.1947** | Mean squared probability error after Platt scaling |

### Test Confusion Matrix (@ Youden Threshold `0.6522`)

| Ground Truth \ Predicted | Predicted Normal (`0`) | Predicted Abnormal (`1`) | Total Actual |
|---|---|---|---|
| **Actual Normal (`0`)** | **356** (True Negative) | **80** (False Positive) | 436 |
| **Actual Abnormal (`1`)** | **287** (False Negative) | **372** (True Positive) | 659 |

---

## 5. Operating Points: Stated Plainly

Two operating thresholds were selected using the **validation split only** and evaluated on the test split:

1. **Default Operating Point (Youden's $J$ Threshold = `0.6522`)**:
   - **Test Sensitivity**: **56.45%** (about 56%)
   - **Test Specificity**: **81.65%** (about 82%)
   - *Plain Summary*: At this threshold, the model catches about 56% of abnormal radiographs, meaning **approximately 43.55% of abnormal studies are missed (false negatives)**.
2. **High-Specificity Operating Point (90% Specificity Threshold = `0.7291`)**:
   - **Test Sensitivity**: **45.37%** (about 45%)
   - **Test Specificity**: **87.84%** (about 88%)
   - *Plain Summary*: Restricting false alarms pushes sensitivity down to about 45%, meaning **more than half (54.63%) of abnormal radiographs are missed**.

---

## 6. Probability Calibration

Probability calibration was conducted using **Platt Scaling** (logistic regression over logit log-odds) fitted **strictly on the Validation Split ($N = 1{,}090$)** to avoid test set contamination:
- **Fitted Parameters** (`models/calibration.json`):
  - Coefficient ($\beta$): `0.8151`
  - Intercept ($\alpha$): `0.6796`
- **Impact**:
  - Test Brier score dropped from **0.2167** (uncalibrated) to **0.1947** (calibrated).
  - Reliability diagrams ([reports/calibration_curve.png](file:///c:/dev/insightx/reports/calibration_curve.png)) confirm improved alignment between mean predicted probability and observed fraction of positives.

---

## 7. Results by Projection (View)

Evaluated on the held-out test split, grouped by projection:

| Projection | Image Count | Normal Count | Abnormal Count | ROC-AUC | 95% Bootstrap CI (by `uid`) |
|---|---|---|---|---|---|
| **Frontal (PA / AP)** | 563 | 228 | 335 | **0.7587** | [0.7187, 0.8011] |
| **Lateral** | 532 | 208 | 324 | **0.7558** | [0.7096, 0.7961] |

Performance is consistent across projections with slightly higher discrimination on frontal views.

---

## 8. Results by Finding (Abnormal Token vs Normal)

Evaluated on the held-out test split for the top 10 most common abnormal finding tokens, using all test Normal cases ($n = 436$) as negatives:

| Finding Token | Abnormal Test Images | ROC-AUC vs Normal | Reliability Flag |
|---|---|---|---|
| **Pleural Effusion** | 47 | **0.9530** | Sufficient ($\ge 30$ images) — Strongest |
| **Cardiomegaly** | 79 | **0.9130** | Sufficient ($\ge 30$ images) — Strongest |
| **Pulmonary Atelectasis** | 86 | **0.8546** | Sufficient ($\ge 30$ images) |
| **Lung** | 136 | **0.8494** | Sufficient ($\ge 30$ images) |
| **Cicatrix** | 49 | **0.8377** | Sufficient ($\ge 30$ images) |
| **Opacity** | 111 | **0.8215** | Sufficient ($\ge 30$ images) |
| **Calcinosis** | 78 | **0.7974** | Sufficient ($\ge 30$ images) |
| **Thoracic Vertebrae** | 64 | **0.7940** | Sufficient ($\ge 30$ images) |
| **Spine** | 63 | **0.7899** | Sufficient ($\ge 30$ images) |
| **Calcified Granuloma** | 67 | **0.6550** | Sufficient ($\ge 30$ images) — Weakest |

- **Key Takeaway**: Gross anatomical and fluid-accumulation findings (**Pleural Effusion** AUC 0.9530, **Cardiomegaly** AUC 0.9130) achieve strong discrimination. In contrast, small focal abnormalities (**Calcified Granuloma** AUC 0.6550) exhibit the weakest discrimination, as small calcified nodules are diluted by global pooling.

---

## 9. Baseline Comparison & Technical Discussion

From `reports/comparison.md` and `reports/baselines.json`:

| Model Architecture | Modality | Accuracy | F1 (Abnormal) | ROC-AUC | 95% Bootstrap CI | Brier Score |
|---|---|---|---|---|---|---|
| **Majority Class** | None | 0.6018 | **0.7514** | 0.5000 | N/A | N/A |
| **Frozen MobileNetV2 GAP** (C=0.01) | Image Only | 0.6484 | 0.6980 | 0.7182 | N/A | N/A |
| **TF-IDF Findings** (C=0.01) | Text Only | 0.9068 | 0.9257 | 0.9676 | N/A | N/A |
| **Fused Multimodal** (C=0.01) | Image + Text | 0.8950 | 0.9119 | 0.9629 | N/A | N/A |
| **Fine-Tuned MobileNetV2** (Ours) | Image Only | **0.6648** | 0.6697 | **0.7563** | [**0.7180**, **0.7921**] | **0.1947** |

### Critical Technical Notes:
1. **Why the Text Baseline Reaches High AUC (0.9676) — Label Leakage**: The dataset target labels (`Problems` / MeSH terms) were extracted by catalogers directly from the text radiology reports. A text model reading the findings narrative experiences direct lexical circularity (label leakage).
2. **Why the Served Model is Strictly Image-Only**: In actual healthcare operations, a radiologist report **does not exist when a scan is performed**. Requiring an authored report at inference time would render the AI redundant. InsightX is therefore deployed strictly as an image-only model.
3. **Why ROC-AUC is the Fair Metric over F1**: The majority-class baseline (always predicting Abnormal) achieves an F1 of **0.7514**, higher than the fine-tuned model's **0.6697**. This occurs because 60.18% test class prevalence artificially inflates F1 for majority predictors. ROC-AUC evaluates true ranking ability across all decision thresholds (0.7563 vs 0.5000).

---

## 10. Failure Modes Analysis

From validation diagnostics in `reports/diagnostics_val.md`:

### Confident False Positives (Ground Truth: Normal, Highest Predicted Probability)
1. `3252_IM-1542-1001.dcm.png` (Frontal, Problems: normal, predicted prob: **0.9479**)
2. `1318_IM-0205-1001.dcm.png` (Frontal, Problems: normal, predicted prob: **0.9078**)
3. `3252_IM-1542-2001.dcm.png` (Lateral, Problems: normal, predicted prob: **0.9067**)
4. `3453_IM-1676-0001-0001.dcm.png` (Frontal, Problems: normal, predicted prob: **0.9025**)

### Confident False Negatives (Ground Truth: Abnormal, Lowest Predicted Probability)
1. `456_IM-2087-2001.dcm.png` (Lateral, Problems: Granulomatous Disease, predicted prob: **0.0262**)
2. `600_IM-2192-2001.dcm.png` (Lateral, Problems: Calcinosis; Calcified Granuloma; Fractures, Bone, predicted prob: **0.0450**)
3. `167_IM-0441-1002.dcm.png` (Lateral, Problems: Calcified Granuloma, predicted prob: **0.0567**)
4. `395_IM-2019-1001.dcm.png` (Frontal, Problems: Calcinosis, predicted prob: **0.0596**)

### Failure Root Causes:
- **Spatial Dilution of Focal Findings**: Global Average Pooling over a $7 \times 7$ grid dilutes small calcified granulomas and micro-nodules against normal lung parenchyma.
- **Anatomical Rotation & Contrast Non-Uniformity**: Patient body habitus and beam exposure variations can trigger spurious activations in normal studies.

---

## 11. Limitations

1. **Single-Institution Cohort**: All training and testing cases originate from a single healthcare system (Indiana University). The model has not been evaluated across multi-institutional datasets (e.g., CheXpert, MIMIC-CXR) and may experience domain shift under different scanners, beam energies, or patient demographics.
2. **Label Noise from Report-Derived Annotations**: Ground truth labels were generated from automated NLP and MeSH cataloging of unstructured clinical reports rather than multi-reader panel adjudication or biopsy confirmation.
3. **Mixed Frontal and Lateral Views**: The collection mixes posteroanterior (PA), anteroposterior (AP), and lateral projections under a unified binary target without projection-specific conditioning.
4. **Validation-Selected Checkpoint**: Operating thresholds and Platt calibration were selected using the validation split.
5. **No External Validation**: Performance on independent cohorts outside IU Health has not been established.
6. **No Clinical Validation**: The system has not undergone prospective clinical trials, reader study validation, or safety audits. It must not be deployed for clinical care.

---

## 12. Heatmap Explainability Method

- **Method**: Class Activation Mapping (CAM) computed directly from the final convolutional feature map $A$ ($7 \times 7 \times 1280$ from `out_relu`) and Dense classification head weights $W$ ($1280 \times 1$):
  $$\text{CAM} = \text{ReLU}\left(\sum_k w_k A_k\right)$$
- **Equivalence to Grad-CAM**: For an architecture consisting of Global Average Pooling followed by a single Dense layer, the channel gradients $\alpha_k = \frac{1}{H \cdot W} \sum_{i,j} \frac{\partial y}{\partial A_{i,j}^k} = \frac{w_k}{H \cdot W}$. Because heatmaps are min-max normalized, the constant factor $\frac{1}{H \cdot W}$ cancels out entirely, making this pure NumPy computation mathematically identical to Grad-CAM.
- **Serving Runtime**: Executed in pure NumPy via `models/gradcam_head_weights.json` without requiring TensorFlow or automatic differentiation tapes at serving time.
- **Clinical Notice**: **Heatmaps represent unverified neural activation patterns and must NOT be interpreted as clinically validated or diagnostic lesion localizations.**

---

## 13. Reproducibility Commands

```bash
# 1. Run baseline comparison
python insightx/baselines.py

# 2. Re-extract weights and verify model loading
python -c "from insightx.model_io import load_trained_model; load_trained_model()"

# 3. Run validation diagnostics
python scripts/diagnostics_val.py

# 4. Run validation calibration and one-shot test evaluation
python -m insightx.evaluate

# 5. Generate Grad-CAM test overlays
python -m insightx.gradcam

# 6. Export ONNX models
python -m insightx.export

# 7. Run test verification suite
pytest
```
