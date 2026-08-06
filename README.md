# 🧠 Dementia Detection using EfficientNetV2 + CBAM

A deep learning project for multi-class dementia stage classification from brain MRI images using **EfficientNetV2B0** and the **Convolutional Block Attention Module (CBAM)**.

---

# Project Overview

This project explores transfer learning for medical image classification. A pretrained EfficientNetV2B0 model is used as the backbone, while CBAM enhances important channel-wise and spatial features before classification.

The complete project includes preprocessing, augmentation, training, evaluation, and model comparison in a modular pipeline.

---

# Features

- EfficientNetV2B0 Transfer Learning
- CBAM Attention Module
- CLAHE Image Enhancement
- Data Augmentation
- Modular Dataset Pipeline
- Modular Training Pipeline
- Evaluation Metrics
- Confusion Matrix
- Baseline vs CBAM Comparison

---

# Project Structure

```text
DEMENTIADETECTION/
│
├── data/                    # Dataset (ignored in Git)
├── logs/                    # Training logs
├── notebooks/               # Development notebooks
├── outputs/                 # Evaluation outputs
├── saved_models/            # Trained models
│
├── src/
│   ├── augmentation/
│   ├── config/
│   ├── dataset/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   └── training/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Model Architecture

```text
Brain MRI
      │
      ▼
CLAHE + Resize + Normalization
      │
      ▼
EfficientNetV2B0
(ImageNet Pretrained)
      │
      ▼
CBAM
(Channel Attention + Spatial Attention)
      │
      ▼
Global Average Pooling
      │
      ▼
Dropout
      │
      ▼
Dense (Softmax)
      │
      ▼
Prediction
```

---

# Workflow

```
MRI Images
      │
      ▼
Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
Dataset Pipeline
      │
      ▼
EfficientNetV2 Baseline
      │
      ▼
CBAM Integration
      │
      ▼
Training
      │
      ▼
Evaluation
      │
      ▼
Performance Comparison
```

---

# Results

| Model                   | Test Accuracy |
| ----------------------- | ------------: |
| EfficientNetV2 Baseline |    **69.18%** |
| EfficientNetV2 + CBAM   |    **85.68%** |

CBAM significantly improved the classification performance over the baseline model.

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Matplotlib

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/DementiaDetection.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The dataset is not included in this repository because of its size.

Place the dataset inside:

```text
data/
```

---

# Future Improvements

- Fine-Tuning EfficientNetV2
- Grad-CAM Explainability
- Streamlit Interface
- FastAPI Deployment

---

# Author

**Raju Kumar**

- LinkedIn: https://www.linkedin.com/in/raju-kumar-577255257/
- GitHub: https://github.com/rajukr9824
