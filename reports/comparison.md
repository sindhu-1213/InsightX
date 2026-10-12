# InsightX: Model Comparison & Test Performance

> **Notice**: Research Demo Only - This tool is not a certified medical device.
> Outputs are AI predictions for research investigation only and must not guide
> clinical decisions.

This report summarizes the held-out **Test Split (n = 1095)**
evaluation of the fine-tuned MobileNetV2 architecture against frozen baseline models.
All calibration models and operating decision thresholds were fixed using the
**Validation Split only**.

---

## 1. Test Split Benchmark Comparison

| Model Architecture | Modality | Accuracy | F1 (Abnormal) | ROC-AUC | 95% CI | Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Class** | None | 0.6018 | 0.7514 | 0.5000 | N/A | N/A |
| **Frozen MobileNetV2** (C=0.01) | Image Only | 0.6484 | 0.6980 | 0.7182 | N/A | N/A |
| **TF-IDF Findings** (C=0.01) | Text Only | 0.9068 | 0.9257 | 0.9676 | N/A | N/A |
| **Fused Multimodal** (C=0.01) | Image + Text | 0.8950 | 0.9119 | 0.9629 | N/A | N/A |
| **Fine-Tuned MobileNetV2** | Image Only | **0.6648** | **0.6697** | **0.7563** | **[0.7180, 0.7921]** | **0.1947** (uncal: 0.2167) |

---

## 2. Validation Split vs Test Split Comparison

| Metric / Setting | Validation Split (n=1090) | Test Split (n=1095) |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.7522 [0.7157, 0.7879] | 0.7563 [0.7180, 0.7921] |
| **Frontal ROC-AUC** | 0.7655 [0.7237, 0.8053] | 0.7587 |
| **Lateral ROC-AUC** | 0.7384 [0.6942, 0.7814] | 0.7558 |
| **Youden Index (J)** | 0.4007 | 0.3810 |
| **Youden Threshold** | `0.6522` | `0.6522` (applied) |
| **Sensitivity @ Youden** | 0.5979 | 0.5645 |
| **Specificity @ Youden** | 0.8028 | 0.8165 |
| **90% Spec Threshold** | `0.7291` | `0.7291` (applied) |
| **Sensitivity @ 90% Spec** | 0.4598 | 0.4537 |
| **Specificity Achieved** | 0.9002 | 0.8784 |
| **Brier Score (Calibrated)** | 0.0000 | 0.1947 |

---

## 3. Projection Breakdown (Test Split)

| Projection | Total Images | Normal Count | Abnormal Count | ROC-AUC | 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Frontal** | 563 | 228 | 335 | 0.7587 | [0.7187, 0.8011] |
| **Lateral** | 532 | 208 | 324 | 0.7558 | [0.7096, 0.7961] |

---

## 4. Per-Finding Breakdown vs Normal (Test Split)

| Finding Token | Abnormal Test Count | ROC-AUC vs Normal | Reliability Flag |
| :--- | :--- | :--- | :--- |
| **Opacity** | 111 | 0.8215 | Sufficient (>=30 images) |
| **Lung** | 136 | 0.8494 | Sufficient (>=30 images) |
| **Pulmonary Atelectasis** | 86 | 0.8546 | Sufficient (>=30 images) |
| **Cardiomegaly** | 79 | 0.9130 | Sufficient (>=30 images) |
| **Calcinosis** | 78 | 0.7974 | Sufficient (>=30 images) |
| **Calcified Granuloma** | 67 | 0.6550 | Sufficient (>=30 images) |
| **Thoracic Vertebrae** | 64 | 0.7940 | Sufficient (>=30 images) |
| **Cicatrix** | 49 | 0.8377 | Sufficient (>=30 images) |
| **Spine** | 63 | 0.7899 | Sufficient (>=30 images) |
| **Pleural Effusion** | 47 | 0.9530 | Sufficient (>=30 images) |

---

## 5. Artifact & Provenance Details

- **Evaluated Split**: `test` (evaluated exactly once)
- **Validation Split Size**: `1090` images
- **Test Split Size**: `1095` images
- **Model Checkpoint SHA-256**: `36115e261d26d18988df3495d4609226c723b5c1997b559c0ecc3d73d1a54232`
- **Weights File SHA-256**: `699147a1037fa08d43d60fa5f60d164491a7a5ce2444945f1c41b32c02d6f571`
- **Evaluation Timestamp**: `2026-10-12T07:13:07.137225+00:00`
- **Seed**: `42`
