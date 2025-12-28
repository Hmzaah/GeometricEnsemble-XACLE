# GeometricEnsemble-XACLE: Heterogeneous Stacking for Audio-Text Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-red)](https://2026.ieeeicassp.org/)

**Official implementation of "Approach 2" for the XACLE Grand Challenge (2nd Place).**

This repository houses the **Geometric Heterogeneous Ensemble**, a dual-stream architecture that combines explicit geometric feature engineering with a Split-Brain prediction head (XGBoost + SVR).

---

## 🏗️ Architecture: The Geometric Heterogeneous Ensemble

Our solution ("Approach 2") ranked **2nd Place** by moving beyond simple end-to-end learning. We employ a **Split-Brain Architecture** that explicitly engineers geometric relationships between modalities before feeding them into a heterogeneous prediction head.

### 1. The 9,220-Dimensional Feature Space
We construct a massive, unified representation vector `(N, 9220)` by concatenating outputs from four state-of-the-art encoders and explicit geometric injections:

| Feature Component | Dimensions | Source / Logic |
| :--- | :---: | :--- |
| **Whisper v2** | 1,280 | Acoustic/Prosodic Features |
| **MS-CLAP** | 2,048 | Coarse-Grained Alignment |
| **LAION-CLAP** | 1,536 | Cross-Modal Embeddings |
| **DeBERTaV3** | 768 | Syntactic/Semantic Text Features |
| **Geometric Injection** | *Variable* | **Explicit Math:** Cosine Similarity ($\cos$), Angular Distance ($\angle$), and $L_1/L_2$ Norms calculated between audio and text tensors. |

### 2. The Split-Brain Predictor
Instead of a single regressor, we route this vector into two mathematically distinct models to balance accuracy and stability:

<p align="center">
  <img src="assets/architecture_diagram.png" alt="Geometric Ensemble Architecture" width="850">
</p>

* **Stream A: XGBoost ($w_1=0.56$)**
    * *Role:* Captures high-frequency, non-linear interactions between the geometric features and embeddings.
    * *Configuration:* Tree-based gradient boosting.
* **Stream B: SVR ($w_2=0.44$)**
    * *Role:* Models the smooth manifold of the alignment score, acting as a regularizer to prevent overfitting to noise.
    * *Configuration:* Radial Basis Function (RBF) kernel.

### 3. Performance
The fusion of these two streams resulted in our final submission score:
> **Final SRCC Score: 0.653**
## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/Snehitc/GeometricEnsemble-XACLE.git](https://github.com/Snehitc/GeometricEnsemble-XACLE.git)
cd GeometricEnsemble-XACLE
pip install -r requirements.txt
