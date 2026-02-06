# Construction Site Safety Detector

**A deep learning image classifier that detects whether construction workers are wearing hard hats, built from scratch with PyTorch and transfer learning.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Val_Accuracy-94.6%25-brightgreen)](#-results)

---

## 🎯 What This Project Does

Upload an image of a construction worker and the model classifies whether they are wearing a hard hat or not, with a confidence score.

- **Binary classification** - Hard Hat vs No Hard Hat
- **Transfer learning** - ResNet18 pre-trained on ImageNet, fine-tuned for our task
- **Interactive demo** - Streamlit web app for real-time predictions

---

## 🖥️ Demo

The Streamlit web app lets you upload any image and get an instant prediction with confidence scores.

```bash
streamlit run app/streamlit_app.py
```

<img width="948" height="803" alt="image" src="https://github.com/user-attachments/assets/0455ab05-7c13-4767-bd05-19786ce5677f" />

---


### Training Strategy

The model is trained in two phases:

1. **Frozen backbone** (10 epochs, lr=0.001) - Only the classifier head trains. The pre-trained ResNet18 features (edges, textures, shapes learned from ImageNet) are preserved.
2. **Fine-tuning** (5 epochs, lr=0.0001) - The last two residual blocks are unfrozen and trained with a lower learning rate, adapting generic features to hard-hat-specific features.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **94.6%** |
| **Architecture** | ResNet18 (11.7M total params) |
| **Trainable (Phase 1 - frozen)** | ~1,026 (classifier head only) |
| **Trainable (Phase 2 - fine-tuned)** | ~5.6M (last 2 residual blocks + head) |
| **Training Time** | ~15 epochs (10 frozen + 5 fine-tuning) |
| **Inference Speed** | <100ms per image on CPU |

---

## 🔍 Known Limitations & Honest Reflection

Through testing, I found the model can be fooled:

| Issue | Severity | What Happens | Root Cause |
|-------|----------|-------------|------------|
| **Color shortcut** | 🔴 | Yellow hair classified as hard hat | Model learned "bright color on head = hat" instead of actual hat shape/structure |
| **Scale sensitivity** | 🔴 | Distant workers flagged as no hat | Training data was mostly close-up/medium shots; small objects lose detail at 224x224 |
| **Single-label** | ⚠️ | Can't handle groups of people | Outputs one label per image - can't say "3/5 workers have hats" |

### How I'd Improve It

- ✅ **Grad-CAM visualization** - Understand what the model actually looks at before trying to fix it
- ✅ **Hard negative mining** - Add training images of yellow wigs, beanies, bike helmets to break the color shortcut
- ✅ **Random grayscale augmentation** - Force the model to learn shape features, not just color
- ✅ **Object detection (YOLOv8)** - The right approach for this problem: detects each person individually, handles groups, works at any distance

> The classification approach was a deliberate learning exercise to understand PyTorch fundamentals, transfer learning, and the full ML pipeline. Object detection is the natural next step for production use.

---

## 🧠 Key Concepts Demonstrated

| Concept | Where | What I Learned |
|---------|-------|---------------|
| **Transfer Learning** | `model.py` | Using ImageNet pre-trained weights as a foundation instead of training from scratch |
| **Two-Phase Training** | `train.py` | Freezing backbone first, then fine-tuning with lower LR prevents destroying learned features |
| **Data Augmentation** | `dataset.py` | Random crops, flips, color jitter, rotation help the model generalize |
| **Custom Dataset** | `dataset.py` | Implementing PyTorch's Dataset interface for loading image/label pairs |
| **Model Evaluation** | `train.py` | Tracking train vs val accuracy to detect overfitting |
| **Web Deployment** | `streamlit_app.py` | Serving a PyTorch model through an interactive UI |

---

## 📁 Project Structure

```
hard-hat-detector/
├── app/
│   └── streamlit_app.py       # Interactive web demo
├── src/
│   ├── dataset.py             # Dataset class, transforms, augmentation
│   ├── model.py               # ResNet18 architecture + transfer learning
│   ├── train.py               # Two-phase training loop
│   └── inference.py           # Prediction on new images
├── notebooks/
│   └── 01_exploration.ipynb   # Data exploration and visualization
├── scripts/                   # Dataset download and preparation helpers
├── data/                      # Dataset (not in repo - see setup below)
├── models/                    # Trained weights (not in repo - see setup below)
├── learningnotes.md           # Detailed learning notes and explanations
└── requirements.txt           # Python dependencies
```

---

## 🚀 Setup

### 1. Clone & Create Environment

```bash
git clone https://github.com/kieranpcremin/hard-hat-detector.git
cd hard-hat-detector
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Download Dataset

Download a hard hat classification dataset and organize it into this structure:

```
data/
├── train/
│   ├── hard_hat/        # ~400 images
│   │   ├── img001.jpg
│   │   └── ...
│   └── no_hard_hat/     # ~400 images
│       ├── img001.jpg
│       └── ...
└── val/
    ├── hard_hat/        # ~100 images
    └── no_hard_hat/     # ~100 images
```

**Dataset sources:**
- [Kaggle - Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- [Roboflow - Hard Hat Classification](https://universe.roboflow.com/search?q=hard+hat+classification)

> **Note:** You may need to reorganize images into the folder structure above depending on the dataset format. See `scripts/` for helper utilities.

### 3. Train the Model

```bash
cd src
python train.py --data_dir ../data --epochs 10 --batch_size 32
```

This will:
- Train with frozen backbone for 10 epochs
- Fine-tune with unfrozen layers for 5 more epochs
- Save the best model to `models/best_model.pth`

### 4. Run the Web App

```bash
streamlit run app/streamlit_app.py
```

### 5. Run Inference from CLI

```bash
cd src
python inference.py path/to/image.jpg --model ../models/best_model.pth
```

---

## 🛠️ Tech Stack

- **PyTorch** - Deep learning framework
- **torchvision** - Pre-trained models and image transforms
- **Streamlit** - Web app framework
- **Pillow** - Image loading and processing
- **matplotlib / seaborn** - Visualization in notebooks
- **scikit-learn** - Model evaluation metrics

---

## 👨‍💻 Author

**Kieran Cremin**
Built with assistance from Claude (Anthropic)

---

## 📄 License

MIT License - Free to use, modify, and distribute.
