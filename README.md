# Cellula-Internship
A computer vision internship project
# First Task

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

# Second Task

# 💧 Water Segmentation with UNet (From Scratch)

This project focuses on **water body segmentation** from multi-channel satellite imagery using two complementary approaches for water body segmentation from satellite imagery:

**custom UNet architecture (from scratch, no pretrained backbone)**
**Pretrained UNet (ImageNet encoder, 3 channels, frozen encoder with fine-tuning)**

---
### First Approach:
## ✨ Key Features
- **Multi-channel input (12 bands)** including Blue, Green, Red, NIR, SWIR, DEM, WaterProb + computed NDWI & MNDWI  
- **Class imbalance handling** with Dice + BCE loss (optionally focal loss)  
- **Data augmentation** (flips, rotations, elastic transforms, brightness/contrast)  
- **Visualization modules**:  
  - EDA histograms of water fraction per image  
  - Band composites & NDWI/MNDWI visualizations  
  - Normalization stats per-channel  
  - Before/after normalization previews  
- **Evaluation metrics**: IoU, Dice coefficient, classification report  

---

## 📂 Dataset Structure

project_root/
├── data/
│ └── data/
│ ├── images/
│ └── labels/
├── notebooks/
│ └── Water_Segmentation_Enhanced.ipynb

---

## ⚙️ Setup
Install dependencies:


pip install torch torchvision rasterio albumentations scikit-learn matplotlib tqdm

---

## 🚀 Training


Training setup:
- **Architecture:** UNet (custom encoder-decoder with skip connections)

- **Optimizer:** Adam (lr=1e-3)

- **Loss:**Dice + BCE

- **Batch size:** 16

- **Epochs:** 60

- **Image size:** 128 × 128

- **Best model saved under :** experiments_clean/best_model.pth

---
## 📊 Evaluation
The trained model outputs:

- IoU, Dice scores on validation/test sets

- Classification report (precision, recall, F1 for water/background)

- Visual comparisons (input, prediction, ground truth masks)

Example final test performance:

**Test Loss:** ~0.34

**Test IoU:** ~0.54

**Test Dice:** ~0.68

---
## 🔍 Visualization Examples

-Water fraction histogram across dataset

-Example composites (RGB, NIR-SWIR, NDWI, MNDWI)

-Before/after normalization for a sample patch

-Predicted segmentation overlays

---
## 📌 Summary

This project delivers an end-to-end segmentation pipeline for detecting water bodies in satellite imagery.
It covers EDA, preprocessing, custom UNet model training, evaluation, and rich visualizations for both data and predictions.

---
### Second Approach:
## Pretrained UNet with Frozen Encoder

In addition to the custom UNet (11-channel), a second experiment was conducted using transfer learning:

**Architecture:** UNet with a ResNet encoder pretrained on ImageNet

**Input:** adapted to 3 channels (RGB composite subset)

**Training strategy:**

- Encoder initially frozen → only the decoder and segmentation head were trained

- After warm-up, the encoder was unfrozen and fine-tuned end-to-end

**Segmentation head:** the final classifier layer that outputs water/non-water predictions