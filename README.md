<div align="center">

<img src="https://img.shields.io/badge/version-1.0.0-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/status-Backend%20Ready-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Frontend-In%20Progress-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />

---

# 🩻 INSIGHT-X
### *AI-Powered Multimodal Clinical Decision Support System*

> **Intelligence That Drives Decisions**

*Fusing chest X-ray vision with clinical NLP to automate radiological diagnostics*

[🎥 Watch Demo](https://screenapp.io/app/v/CmRiOCmIsx) · [📄 View Presentation](./INSIGHTX.pdf) · [🐛 Report Bug](../../issues)

</div>

---

## 📌 Overview

**INSIGHT-X** is a multimodal AI framework that integrates heterogeneous medical data — chest X-ray images and clinical radiology reports — into a single, unified diagnostic pipeline. Built on the **Indiana University Chest X-ray Dataset**, it fuses deep visual features with specialized medical NLP to create a per-patient diagnostic signature, enabling automated patient clustering and pathology classification.

> This project was developed as part of an AI internship under the guidance of **Mr. Bhuvanesh**, in collaboration with Edunet Foundation, Microsoft, LinkedIn, SAP, and the Ministry of Skill Development.

---

## 🚨 The Problem We're Solving

| Challenge | Description |
|---|---|
| 🔀 **Siloed Modalities** | Traditional medical AI analyzes images and clinical text separately, missing critical cross-modal correlations |
| 🚫 **Destructive Preprocessing** | Standard NLP pipelines strip negation words like *"no," "not," "without"* — flipping a patient's diagnosis entirely |
| 💥 **Computational Bottlenecks** | Extracting high-dimensional features from 7,500+ X-rays on standard hardware causes memory overflow and crashes |

---

## ✅ Our Solution

### 🧠 Medical-Grade NLP Pipeline
A custom text preprocessing module that **preserves clinical negations** (e.g., *"no pneumonia"* ≠ *"pneumonia"*). Standard stopword lists are surgically modified to retain the words that matter most in radiology.

```python
# Negation words are explicitly kept in the vocabulary
negations = {'no', 'not', 'none', 'neither', 'never', 'without', 'negative'}
medical_stop_words = stop_words - negations  # Don't remove these!
```

### ⚡ Optimized Feature Extraction
- **MobileNetV2** (pre-trained on ImageNet) extracts **1,280-dimensional visual embeddings** via its Global Average Pooling layer
- **TF-IDF Vectorization** (top 1,000 features) transforms clinical findings into dense numerical representations
- Both are horizontally fused into a unified patient signature vector

### 🚀 Scalable Batch Inference Pipeline
```python
# tf.data pipeline with prefetching — processes thousands of images
# without crashing on standard hardware
path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
image_ds = path_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
batch_ds = image_ds.batch(64).prefetch(tf.data.AUTOTUNE)
```

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  DATA ACQUISITION│───▶│   PREPROCESSING   │───▶│     FEATURE EXTRACTION    │───▶│   FUSION LAYER  │───▶│   OUTPUT LAYER   │
└─────────────────┘    └──────────────────┘    └───────────────────────────┘    └─────────────────┘    └──────────────────┘
                                                                                                                   
  indiana_reports.csv    Inner Join (uid)       ┌─ TEXT: TF-IDF (1000-dim)  ┐    Fused Patient        K-Means Clusters
  indiana_projections    Clean Text             │                            │    Signature            
  images_normalized/     Medical NLP Filter     └─ IMAGE: MobileNetV2       ┘    (1280 + 1000 dim)    Patient Overview UI
                                                     GAP Layer (1280-dim)         K-Means Clustering
```

---

## 📂 Project Structure

```
InsightX/
│
├── 📁 images_normalized/          # Preprocessed chest X-ray images
├── 📁 insightx-frontend/          # React/Next.js UI (🚧 In Progress)
│
├── 📄 project_code_final.npynb    # Main pipeline notebook (backend complete)
│
├── 📊 indiana_projections.csv     # X-ray metadata (filename, uid, projection)
├── 📊 indiana_reports.csv         # Clinical reports (findings, impressions)
├── 📊 cleaned_multimodel_dat.csv  # Merged & cleaned master dataset
│
├── 🔢 image_features_final.npy    # Pre-extracted MobileNetV2 embeddings (23 MB)
│
├── 🤖 insightx_rf_model.pkl       # Trained Random Forest classifier
├── ⚖️  insightx_scaler.pkl         # Feature scaler (StandardScaler)
│
└── 📋 README.md
```

---

## 🔬 ML Pipeline Deep Dive

### Stage 1 — Data Acquisition & Merging
- Loads `indiana_projections.csv` and `indiana_reports.csv`
- Merges on `uid` (inner join) to align images with their corresponding reports
- Drops rows missing `findings`, `impression`, `comparison`, or `indication`

### Stage 2 — Medical NLP
- Custom stopword list retains clinically critical negation words
- TF-IDF vectorization on processed `findings` text (1,000 features)

### Stage 3 — CNN Feature Extraction
- MobileNetV2 loaded without its classification head (`include_top=False, pooling='avg'`)
- Batch inference with `tf.data` prefetching across 64-image batches
- Produces 1,280-dimensional embedding per image, saved as `.npy`

### Stage 4 — Feature Fusion & Clustering
- Image (1,280-dim) + Text (1,000-dim) vectors concatenated → **2,280-dim fused signature**
- StandardScaler applied before KMeans to equalize modality contributions
- **KMeans (k=5)** groups patients into clinical profiles

### Stage 5 — Classification (Random Forest)
- Labels derived from `impression_proc` (radiologist conclusion) to **prevent label leakage**
- GridSearchCV over `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- 5-fold Stratified Cross-Validation for reliable evaluation
- Metrics: Accuracy, F1-Score, ROC-AUC, Confusion Matrix

### Stage 6 — Explainability (Grad-CAM)
- Separate MobileNetV2 spatial model (no pooling) used for Grad-CAM
- Generates **heatmaps highlighting which X-ray regions** drove the model's decision

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow 2.x, Keras, MobileNetV2 |
| **Classical ML** | scikit-learn (KMeans, Random Forest, GridSearchCV) |
| **NLP** | NLTK, TF-IDF Vectorizer |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Explainability** | Grad-CAM |
| **Frontend** | 🚧 In Progress (insightx-frontend/) |
| **Dataset** | Indiana University Chest X-ray Collection |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for feature extraction)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/insightx.git
cd insightx
```

### 2. Install Dependencies
```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn nltk wordcloud joblib
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
```

### 4. Dataset Setup
Download the Indiana University Chest X-ray Dataset and place files as:
```
insightx/
├── images_normalized/    ← X-ray images (.png)
├── indiana_projections.csv
└── indiana_reports.csv
```

### 5. Run the Pipeline
Open and run `project_code_final.npynb` sequentially. Image features are saved to `image_features_final.npy` so the heavy CNN extraction step only needs to run once.

---

## 📈 Results

The model outputs patient clusters and a binary Normal/Abnormal classification. Key signals the system detects include:

`opacity` · `effusion` · `infiltrate` · `consolidation` · `pneumonia` · `mass` · `nodule` · `cardiomegaly` · `atelectasis` · `edema` · `pleural effusion` · `fracture`

---

## 🔭 Future Scope

| Feature | Description |
|---|---|
| 🔥 **Explainable AI** | Grad-CAM heatmaps integrated into the UI for radiologist review |
| 🤝 **Transformer Fusion** | Cross-Attention (CLIP-style) for simultaneous image-text reasoning |
| 📅 **Time-Series Tracking** | Longitudinal scan comparison to monitor disease progression |
| 📝 **Auto-Reporting** | LLM-powered draft radiology report generation from fused features |
| 📱 **Edge Deployment** | TFLite conversion for offline use in rural clinics on tablets |
| 🔬 **Multi-Pathology Detection** | Move from clustering to direct multi-label disease classification |

---

## 👩‍💻 Team

| Name | Role |
|---|---|
| **Spoorthy S** | ML Pipeline & Feature Fusion |
| **Sindhuja S** | NLP Pipeline & Text Processing |
| **Prema M** | CNN Architecture & Image Processing |
| **Tamilselvi G** | Data Engineering & Evaluation |

**Guide:** Mr. Bhuvanesh  
**Institution:** Sairam Tap, Chennai  
**Program:** Edunet Foundation AI Internship

---

## 🎥 Demo

▶️ [**Watch the full project demo here**](https://screenapp.io/app/v/CmRiOCmIsx)

---

## 📜 License

This project is for educational and research purposes. Dataset usage is subject to the [Indiana University Chest X-ray Dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) terms.

---

<div align="center">

Made with ❤️ and a lot of ☕ by Team INSIGHT-X

*"The best diagnostic AI is one that thinks like a doctor — and sees like one too."*

</div>
