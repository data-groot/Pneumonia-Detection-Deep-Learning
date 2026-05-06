# 🫁 Pneumonia Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

> Detecting pneumonia from chest X-ray images using deep learning — benchmarking CNN, DenseNet121, EfficientNetB3, and ResNet50V2.

---

## 📌 Overview

This project applies deep learning to automatically detect pneumonia from chest X-ray images, training models on a labeled dataset of over 5,000 scans to distinguish between pneumonia-positive and normal lungs. Starting with a custom Convolutional Neural Network (CNN) as the baseline, the project benchmarks three state-of-the-art architectures — DenseNet121, EfficientNetB3, and ResNet50V2 — using transfer learning to achieve high accuracy with strong clinical metrics including precision, recall, and AUC-ROC. This work demonstrates the potential of AI to assist medical professionals in faster, more consistent diagnosis of pulmonary conditions.

---

## 📊 Model Comparison

| Model          | Accuracy    | Precision | Recall | F1-Score | AUC-ROC |
| -------------- | ----------- | --------- | ------ | -------- | ------- |
| Baseline CNN   | 97%         | 96%       | 94%    | 95%      | 0.9949  |
| DenseNet121    | Coming soon | -         | -      | -        | -       |
| EfficientNetB3 | Coming soon | -         | -      | -        | -       |
| ResNet50V2     | Coming soon | -         | -      | -        | -       |

---

## 📁 Project Structure

```
Pneumonia-Detection-Deep-Learning/
├── notebooks/
│   ├── 02_baseline_CNN.ipynb       # Custom CNN baseline model
│   ├── 03_DenseNet121.ipynb        # DenseNet121 transfer learning
│   ├── 04_EfficientNetB3.ipynb     # EfficientNetB3 transfer learning
│   └── 05_ResNet50V2.ipynb         # ResNet50V2 transfer learning
├── models/                         # Saved model files (.keras)
├── results/
│   ├── figures/                    # Confusion matrices, ROC curves
│   └── metrics.csv                 # Aggregated model metrics
├── src/
│   ├── data_loader.py              # Data loading & augmentation
│   ├── train.py                    # Training logic & callbacks
│   └── evaluate.py                 # Evaluation & plotting utilities
├── app/                            # web app (coming soon)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
- **Classes:** NORMAL, PNEUMONIA
- **Total Images:** 5,000+

> Dataset is not included in this repository. Download it from Kaggle and place it in an `archive/` folder at the project root.

---

## 🧠 Models Used

### Baseline — Custom CNN

A custom 4-block CNN architecture built from scratch with progressively increasing filter depths (32 → 64 → 128 → 256), max-pooling, dropout regularization, and a sigmoid output for binary classification.

### DenseNet121

A densely connected network where each layer receives feature maps from all preceding layers. Proven in Stanford's CheXNet — the most cited chest X-ray AI paper — making it highly credible for this exact task.

### EfficientNetB3

A highly efficient architecture that scales depth, width, and resolution together. Widely used in medical imaging competitions with an excellent accuracy-to-compute ratio.

### ResNet50V2

A deep residual network using skip connections to combat the vanishing gradient problem. A strong and well-established baseline in medical imaging research.

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/data-groot/Pneumonia-Detection-Deep-Learning.git
cd Pneumonia-Detection-Deep-Learning
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) and extract to:

```
archive/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
└── val/
```

### 5. Run the notebooks

Open any notebook in `notebooks/` and run all cells.

---

<!-- ## 🔮 Future Work

- [ ] Grad-CAM visualizations — highlight regions the model focuses on
- [ ] Extend to multi-class detection (COVID-19, Tuberculosis)
- [ ] Streamlit web app for live X-ray prediction
- [ ] Deploy to Hugging Face Spaces
- [ ] Hyperparameter tuning with Keras Tuner / Optuna
- [ ] Swin Transformer implementation

--- -->

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Data & Viz:** NumPy, Pandas, Matplotlib, Seaborn
- **ML Utilities:** Scikit-learn

---

<!-- ## 👤 Author

**Krishna Mihir Tatavarthi**
MS Computer Science — UMBC
[GitHub](https://github.com/data-groot) • [LinkedIn](https://www.linkedin.com/in/your-linkedin)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. -->
