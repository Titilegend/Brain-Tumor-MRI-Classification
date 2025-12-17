# Brain Tumor MRI Classification 🧠🩻

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-green)
![Keras](https://img.shields.io/badge/Keras-2.14-orange)

---

## Project Overview
This project classifies **MRI brain scans** into **four categories**:

- Glioma  
- Meningioma  
- Pituitary  
- No tumor  

We implement and compare two deep learning architectures: **InceptionResNetV2** and **DenseNet201**, pretrained on ImageNet. The aim is to achieve high accuracy in detecting and classifying brain tumors.

---

## Dataset
- **Source:** Epic and CSCR hospital dataset (or add Kaggle link if applicable)  
- **Number of classes:** 4  
- **Image preprocessing:**  
  - Resize images: 299x299 for InceptionResNetV2, 224x224 for DenseNet201  
  - Normalize pixel values using model-specific preprocessing functions  
- **Split:** Train, Test  

---

## Model Architectures

### InceptionResNetV2
- Pretrained on ImageNet  
- `include_top=False`  
- Classification head:
  - `GlobalAveragePooling2D`  
  - `Dropout(0.3)`  
  - Dense layer (4 units, softmax)  

### DenseNet201
- Pretrained on ImageNet  
- `include_top=False`  
- Similar classification head as InceptionResNetV2  

---

## Training Strategy
1. **Phase 1:** Train top layers while keeping base frozen  
2. **Phase 2:** Fine-tune top layers of the base with a smaller learning rate  
3. **Optimizers:** Adam for initial training, AdamW for fine-tuning  
4. **Loss:** Sparse categorical cross-entropy  
5. **Callbacks:**  
   - `EarlyStopping` (monitor `val_loss`, restore best weights)  
   - `ModelCheckpoint` (save best model)  

---

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score  
- Confusion matrix for per-class performance  
- Comparison between models using test accuracy  

**Example Results (Placeholders):**

| Model               | Test Accuracy | Notes |
|--------------------|---------------|-------|
| InceptionResNetV2   | 94.37%         | Slightly better for Glioma and Pituitary |
| DenseNet201         | 90.02%         | Faster training, smaller model size |

---

## Usage
1. Clone the repository:  
```bash
git clone https://github.com/Titilegend/Brain-Tumor-MRI-Classification.git

2. Navigate to the project folder
3.Install Dependencies
4.Run the notebook to train or evaluate models
jupyter notebook notebooks/brain_tumor_classification.ipynb

