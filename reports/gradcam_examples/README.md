# InsightX: Grad-CAM Explainability Examples (Test Split)

> **Research Demo Only - This tool is not a certified medical device.**
> Outputs are AI predictions and saliency maps for research investigation only.
> **Heatmaps are gradient-weighted activations of network feature maps and must NOT be interpreted as clinically verified or diagnostic.**

The following 8 test cases illustrate model visual attention across 4 correct and 4 incorrect predictions under the validation-chosen operating threshold (`0.6522`).

| Filename | Category | Projection | Ground Truth | Predicted Probability | Problems Annotation | Overlay File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1356_IM-0231-1001.dcm.png` | Correct (True Normal) | Frontal | Normal | 0.1127 | normal | `01_correct_tn_1356_IM-0231-1001.dcm.png` |
| `1466_IM-0302-1002.dcm.png` | Correct (True Normal) | Lateral | Normal | 0.1379 | normal | `02_correct_tn_1466_IM-0302-1002.dcm.png` |
| `1887_IM-0575-1001.dcm.png` | Correct (True Abnormal) | Frontal | Abnormal | 0.9844 | Atherosclerosis;Scoliosis;Arthritis;Kyphosis | `03_correct_tp_1887_IM-0575-1001.dcm.png` |
| `772_IM-2317-0001-0002.dcm.png` | Correct (True Abnormal) | Frontal | Abnormal | 0.9791 | Surgical Instruments;Atherosclerosis;Aorta, Thoracic;Cardiomegaly;Calcinosis;Mediastinum | `04_correct_tp_772_IM-2317-0001-0002.dcm.png` |
| `1293_IM-0192-4004.dcm.png` | Incorrect (False Positive) | Frontal | Normal | 0.9566 | normal | `05_incorrect_fp_1293_IM-0192-4004.dcm.png` |
| `519_IM-2131-4001.dcm.png` | Incorrect (False Positive) | Lateral | Normal | 0.9431 | normal | `06_incorrect_fp_519_IM-2131-4001.dcm.png` |
| `229_IM-0873-2001.dcm.png` | Incorrect (False Negative) | Lateral | Abnormal | 0.1582 | Calcinosis;Nodule | `07_incorrect_fn_229_IM-0873-2001.dcm.png` |
| `2497_IM-1022-1001.dcm.png` | Incorrect (False Negative) | Frontal | Abnormal | 0.1718 | Lymph Nodes | `08_incorrect_fn_2497_IM-1022-1001.dcm.png` |
