# GeometricEnsemble-XACLE: Heterogeneous Stacking for Audio-Text Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-red)](https://2026.ieeeicassp.org/)

**Official implementation of "Approach 2" for the XACLE Grand Challenge (2nd Place).**

This repository houses the **Geometric Heterogeneous Ensemble**, a dual-stream architecture that combines explicit geometric feature engineering with a Split-Brain prediction head (XGBoost + SVR).

---

## 🏗️ Architecture
Our approach processes a **9,220-dimensional feature vector** through two distinct mathematical pathways.

<p align="center">
  <img src="assets/approach2_flowchart.png" alt="Geometric Ensemble Architecture" width="800">
</p>

### 🔹 Feature Engineering (The "Geometric Injection")
Unlike standard end-to-end models, we explicitly calculate interaction metrics between Audio and Text embeddings:
1.  **Encoders:** Whisper v2, MS-CLAP, LAION-CLAP, DeBERTaV3.
2.  **Geometric Features:**
    * Cosine Similarity ($\cos$)
    * Angular Distance ($\angle$)
    * L1 / L2 Norms ($\| \cdot \|$)
3.  **Fusion:** All features are concatenated into a unified `(N, 9220)` vector.

### 🔹 The Split-Brain Predictor
We employ heterogeneous stacking to capture both sharp decision boundaries and smooth manifold trends:
* **Model A: XGBoost** ($w_1 = 0.56$) - Captures non-linear feature interactions.
* **Model B: SVR (RBF)** ($w_2 = 0.44$) - Regularizes the ranking score.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/Snehitc/GeometricEnsemble-XACLE.git](https://github.com/Snehitc/GeometricEnsemble-XACLE.git)
cd GeometricEnsemble-XACLE
pip install -r requirements.txt
