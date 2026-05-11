# lung-cancer-classification-gradcam
Python code to predict cancer 
lung-cancer-classification-gradcam/
│
├── lung_dataset/
│   ├── train/
│   │   ├── benign/
│   │   ├── malignant/
│   │   ├── normal/
│   │
│   ├── test/
│       ├── benign/
│       ├── malignant/
│       ├── normal/
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── gradcam_result.png
│
├── models/
│   ├── lung_transfer_model.h5
│
├── screenshots/
│   ├── training.png
│   ├── gradcam.png
│
├── requirements.txt
├── README.md
├── LICENSE
├── train.py
├── gradcam.py
└── .gitignore



# 🫁 Lung Cancer Detection using Deep Learning + Grad-CAM

Prepared by Deepjyoti Das Technologies

## 📌 Overview

This project uses Transfer Learning with ResNet50 to classify lung CT scan/X-ray images into:

- Benign
- Malignant
- Normal

The project also includes:

✅ Grad-CAM Explainable AI Visualization  
✅ Multi-Class ROC Curve  
✅ Confusion Matrix  
✅ F1 Score Evaluation  
✅ Class Balancing using Class Weights

---

# 🚀 Features

- Deep Learning using TensorFlow/Keras
- ResNet50 Transfer Learning
- Data Augmentation
- Grad-CAM Visualization
- Multi-class Classification
- Medical AI Explainability
- High Accuracy CNN Pipeline

---

# 📂 Dataset Structure

```bash
lung_dataset/
│
├── train/
│   ├── benign/
│   ├── malignant/
│   ├── normal/
│
├── test/
│   ├── benign/
│   ├── malignant/
│   ├── normal/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/lung-cancer-classification-gradcam.git
cd lung-cancer-classification-gradcam
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python train.py
```

---

# 🔥 Run Grad-CAM Visualization

```bash
python gradcam.py
```

---

# 🧠 Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

# 📊 Model Architecture

- ResNet50 (ImageNet Pretrained)
- GlobalAveragePooling
- Dense Layer
- Dropout
- Softmax Output

---

# 📈 Evaluation Metrics

- Accuracy
- F1 Score
- ROC-AUC Curve
- Confusion Matrix
- Classification Report

---

# 🖼️ Grad-CAM Explainability

Grad-CAM helps visualize important regions in medical images used by the CNN model for predictions.

---

# 💾 Saved Model

```bash
models/lung_transfer_model.h5
```

---

# 📸 Screenshots

Add screenshots inside:

```bash
screenshots/
```

Examples:
- Training Accuracy Graph
- Confusion Matrix
- Grad-CAM Output

---

# 🔮 Future Improvements

- EfficientNet Integration
- Vision Transformers
- Flask/Streamlit Web App
- Real-time Prediction API
- Medical Report Generation
- Docker Deployment

---



















