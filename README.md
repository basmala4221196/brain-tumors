# 🧠 Brain Tumor Detection & Classification using Deep Learning

An AI-powered healthcare project for automated **brain tumor detection and classification** from MRI images using Deep Learning. The project compares multiple CNN architectures and provides a web-based interface to assist both healthcare professionals and patients.


<p align="left">
<img src="https://img.shields.io/badge/Accuracy-98.86%25-success?style=flat-square"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
</p>

---

## 📌 Overview

Early brain tumor diagnosis is critical for improving patient outcomes, but manual MRI interpretation is time-consuming and depends heavily on radiologists' expertise.

This project introduces an intelligent deep learning system that:

- Detects whether a brain MRI scan contains a tumor
- Classifies tumors into **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**
- Returns a prediction confidence score
- Offers a patient-facing educational website about brain tumors

---

## 🩻 Dataset

Trained on two publicly available Kaggle datasets:

- [Brain Tumor MRI Dataset](#)
- [Crystal Clean Brain Tumor MRI Dataset](#) 

| Split | Images |
|---|---:|
| Training | 21,672 |
| Testing | 7,023 |
| **Total** | **28,695** |

**Classes:** Glioma · Meningioma · Pituitary · No Tumor

---

## 🔬 Pipeline

```
MRI Image
   ↓
Preprocessing → Data Cleaning → Data Augmentation → Feature Extraction
   ↓
Deep Learning Model (VGG16 / ResNet50 / Custom CNN)
   ↓
Prediction → Tumor Type + Confidence Score
```

### Preprocessing Steps
Gaussian noise filtering · duplicate image removal · label correction · resizing (256×256) · histogram equalization · brightness adjustment · rotation · horizontal/vertical flipping · salt & pepper noise augmentation

---

## 📊 Results

### Model Comparison

| Model | Accuracy |
|---|---:|
| **Custom CNN** | **98.86%** |
| VGG16 | 97.80% |
| ResNet50 | 96.99% |

The custom CNN achieved the best overall performance while remaining computationally efficient.

### Custom CNN — Per-Class Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Glioma | 99.25% | 97.90% | 98.57% |
| Meningioma | 97.55% | 99.27% | 98.40% |
| Pituitary | 98.31% | 99.43% | 98.87% |
| No Tumor | 99.75% | 98.40% | 99.07% |

---

## 🌐 Web Application

The companion website allows users to:

- Upload an MRI image
- Get automatic tumor detection + classification
- View prediction confidence
- Read educational content on tumor types and symptoms

`[Add screenshot or demo GIF of the website here]`
`[Add live demo link here, if deployed]`

---

## 🛠️ Tech Stack

**Deep Learning:** ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) — CNN, VGG16, ResNet50

**Computer Vision:** ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) NumPy

**Visualization:** Matplotlib

**Web:** ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)


## 🚀 Future Work

- Train on larger, more diverse MRI datasets
- Support additional tumor types
- Deploy a mobile application
- Add multilingual support
- Explore EfficientNet and DenseNet architectures
- Integrate with clinical workflows

---

## 📚 References

Built on recent advances in deep learning for medical image analysis — CNN-based tumor classification, transfer learning, and publicly available MRI datasets. Full reference list available in the project report.

---

