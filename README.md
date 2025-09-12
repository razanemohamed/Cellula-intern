# Cellula-Internship
A computer vision internship project

# 🦷 Teeth Classification with CNN (From Scratch)

This repository contains an end-to-end deep learning pipeline for **teeth condition classification** using a **custom convolutional neural network (TinyResNet) trained from scratch**.

---

## ✨ Key Features
- **Balanced training** using a weighted sampler to handle class imbalance  
- **Data augmentation** (random crops, flips, rotations, color jitter) for robust generalization  
- **From-scratch CNN (TinyResNet)** — no pretrained weights  
- **Evaluation** with classification report and confusion matrix  
- **Model interpretability** with Grad-CAM visualizations  

---

## 📂 Dataset Structure
The dataset is organized as follows:

  Teeth_Dataset/
├── Training/
├── Validation/
└── Testing/


⚠️ Non-class folders in the test set (`out`, `output`, `outputs`) are automatically ignored during evaluation.

---

## ⚙️ Setup
Install dependencies:

pip install torch torchvision scikit-learn matplotlib seaborn tqdm pillow

---

## 🚀 Training

To train the model, open the notebook and run all cells:

jupyter notebook Teeth_Classification.ipynb


Training setup:

- **Optimizer:** Adam (lr=1e-3)

- **Loss:** CrossEntropyLoss

- **Batch size:** 32

- **Epochs:** 50

- **Image size:** 224 × 224

- **Best model saved as:** best_model.pth

---

## 📊 Evaluation

The trained model achieved:

- **Accuracy:** ~97% on the test set

Strong performance across all classes

---
## 🔍 Model Interpretability

Grad-CAM heatmaps highlight important regions in the images, showing why the model makes its predictions.

---
## 🌐 Deployment with Streamlit

deployed the EfficientNet-B0 model using Streamlit for real-time image classification.

To run the App:

streamlit run streamlit_app.py

**Features:**

- Upload a teeth image (.jpg, .jpeg, .png)

- Get instant predictions into 7 classes

- View prediction confidence score

- Visualize class probabilities in an interactive bar chart
---