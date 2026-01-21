# 🎙️ Speech Emotion Recognition using Deep Learning

A deep learning-based **Speech Emotion Recognition (SER)** system that classifies human emotions from raw audio signals using advanced audio feature extraction and neural networks. This project demonstrates an end-to-end pipeline covering data preprocessing, feature engineering, model training, and evaluation.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Features Extracted](#features-extracted)
- [Model Architecture](#model-architecture)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## 🧠 Overview

Speech Emotion Recognition (SER) is a crucial task in **Human-Computer Interaction (HCI)**, **affective computing**, and **AI-driven healthcare systems**. This project focuses on identifying emotions such as *happy, sad, angry, neutral,* etc., directly from speech signals.

The system leverages:
- Audio signal processing techniques
- Deep learning models
- Robust feature extraction methods

---

## 🎯 Motivation

Understanding emotions from speech can enhance:
- Virtual assistants
- Mental health monitoring
- Call center analytics
- Emotion-aware AI systems

This project aims to build a **scalable and interpretable SER pipeline** using modern ML tools.

---

## 📂 Dataset

The project uses a **speech emotion dataset** containing labeled audio samples corresponding to different emotional states.

> ⚠️ Dataset files are not included in this repository due to size constraints.  
> Please download the dataset separately and place it in the appropriate directory.

---

## 🔊 Features Extracted

The following acoustic features are extracted from each audio sample:

- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Chroma Features**
- **Spectral Contrast**
- **Tonnetz**
- **Zero-Crossing Rate**
- **Root Mean Square Energy**

These features help capture both **spectral** and **temporal** characteristics of speech.

---

## 🏗️ Model Architecture

- Input: Extracted audio feature vectors
- Fully Connected Deep Neural Network
- Activation Functions: ReLU
- Output Layer: Softmax (multi-class emotion classification)
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam

---

## 🔄 Project Workflow

1. Load and preprocess audio files
2. Extract meaningful audio features
3. Encode emotion labels
4. Train deep learning model
5. Evaluate performance on test data

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/suvchr105/speech-emotion-recognition.git
cd speech-emotion-recognition


## 📊 Results

- Achieved competitive accuracy on multi-class emotion classification  
- Model shows strong generalization across different speakers  
- Performance varies based on emotion overlap and dataset balance  
- Detailed evaluation metrics are available inside the notebook  

---

## 🧰 Tech Stack

**Programming Language**
- Python  

**Libraries**
- NumPy  
- Pandas  
- Librosa  
- PyTorch  
- Torchaudio  
- Matplotlib  
- Seaborn  
- Hugging Face Transformers  

