[![XACLE\_Dataset](https://img.shields.io/badge/GitHub-XACLE-blue)](https://github.com/XACLE-Challenge/the_first_XACLE_challenge_baseline_model)
[![XACLE\_Leaderboard](https://img.shields.io/badge/Leaderboard-XACLE-limegreen)](https://xacle.org/results.html)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9-blue)
[![Paper](https://img.shields.io/badge/Paper-ICASSP--2026-navy?logo=ieee)](https://ieeexplore.ieee.org/document/11461274)


# XACLE-Approach2

This repository implements **Approach 2**: a heterogeneous *split-brain* architecture that combines explicit **geometric feature injection** with deep semantic embeddings to predict audio–text alignment scores.

![Architecture](doc/architecture_diagram.png)

## 🌟 Highlights

* Achieved an SRCC of **0.653** on the official leaderboard.
* **Geometric Injection:** Explicit computation of **Cosine Similarity, Angular Distance, and L1/L2 norms** between audio and text embeddings.
* **Heterogeneous Stacking:** Combines **XGBoost** (tree‑based) and **SVR** (kernel‑based) predictors for stability and accuracy.
* **Massive Feature Space:** **9,220‑dimensional** fused representation from Whisper v2, MS‑CLAP, LAION‑CLAP, and DeBERTaV3.
* **Distribution Matching:** A critical **0–10 Min‑Max scaling fix** derived from validation analysis to reduce MSE.

---

## 🏗️ Methodology

### 1. The 9,220‑Dimensional Feature Space

| Component               | Dimensions | Description                                                                   |
| ----------------------- | ---------: | ----------------------------------------------------------------------------- |
| Whisper v2              |      1,280 | Acoustic / prosodic audio features                                            |
| MS‑CLAP                 |      2,048 | Coarse audio–text alignment                                                   |
| LAION‑CLAP              |      1,536 | Cross‑modal semantic embeddings                                               |
| DeBERTaV3               |        768 | Syntactic & semantic text features                                            |
| **Geometric Injection** |   Variable | Cosine similarity, angular distance, L1/L2 norms between audio & text tensors |

### 2. Split‑Brain Ensemble Predictor

The final prediction is a weighted fusion of two complementary learners:

$$y = 0.56f_{XGB}(x) + 0.44f_{SVR}(x)$$

* **XGBoost (w = 0.56):** Captures high‑frequency nonlinear interactions ($$Depth = 6, LR = 0.01$$)
* **SVR (w = 0.44):** Models the smooth score manifold (RBF kernel, $$C = 0.5, ε = 0.1$$)

### 3. Validation Strategy (Critical)

Standard regressors produced compressed score ranges. We therefore:

1. Analyzed validation ground‑truth distribution (mean ≈ 6.89, range 0–10)
2. Applied **post‑hoc Min‑Max normalization** to map predictions to ([0,10])

This correction significantly reduced MSE and stabilized leaderboard performance.

---

## 🚀 Quick Setup

```bash
git clone https://github.com/Hmzaah/XACLE-Approach2.git
cd XACLE-Approach2
```

```bash
conda create -n GeomEnsemble python=3.9 -y
conda activate GeomEnsemble
pip install -r requirements.txt
```

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 🛠️ Usage

### Feature Extraction

```bash
python features/extract_features.py --data-dir datasets/XACLE_dataset --out-dir features/extracted
```

### Training

```bash
python train.py configs/config_geometric_submission2.json
```

### Inference

```bash
python inference.py outputs/version_geometric_submission2 validation
```

---

## 📊 Results 🥈
<!--
| Version                      |    SRCC ↑ |     LCC ↑ |    KTAU ↑ |     MSE ↓ |
| ---------------------------- | --------: | --------: | --------: | --------: |
| **Submission 2 (This Repo)** | **0.653** | **0.673** | **0.477** | **3.153** |
-->

<table style="text-align: center;">
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="4">Validation</th>
      <th colspan="4">Test</th>
    </tr>
    <tr>
        <td>SRCC $$\uparrow$$</td>
        <td>LCC $$\uparrow$$</td>
        <td>KTAU $$\uparrow$$</td>
        <td>MSE $$\downarrow$$</td>
        <td>SRCC $$\uparrow$$</td>
        <td>LCC $$\uparrow$$</td>
        <td>KTAU $$\uparrow$$</td>
        <td>MSE $$\downarrow$$</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td>0.384</td>
      <td>0.396</td>
      <td>0.264</td>
      <td>4.836</td>
      <td>0.334</td>
      <td>0.342</td>
      <td>0.229</td>
      <td>4.811</td>
    </tr>
    <tr>
      <td><strong>Submission 2<br> (This Repo)</strong></td>
      <td><strong>0.653</strong></td>
      <td><strong>0.673</strong></td>
      <td><strong>0.477</strong></td>
      <td><strong>3.153</strong></td>
      <td><strong>0.616</strong></td>
      <td><strong>0.665</strong></td>
      <td><strong>0.442</strong></td>
      <td><strong>3.023</strong></td>
    </tr>
  </tbody>
</table>




> Validation metrics computed locally; test metrics taken from the official leaderboard.

---

## 💻 Hardware & Performance

* **CPU:** AMD Ryzen 5 (7000 series)
* **GPU:** NVIDIA GeForce RTX 3050 (8 GB VRAM)
* **Runtime:** Feature extraction ≈ 45 min, Training ≈ 15 min

---

## 📂 Directory Structure

```
XACLE-Approach2
│ README.md
│ requirements.txt
│ train.py
│ inference.py
│ evaluate.py
│
├─ features/
│  ├─ geometric_features.py
│  ├─ extract_features.py
│  └─ fusion.py
│
├─ models/
│  ├─ xgboost_model.json
│  └─ svr_model.pkl
│
├─ configs/
│  └─ config_geometric_submission2.json
│
└─ datasets/
   └─ XACLE_dataset/
```

---

## 📜 Citation

> S. B. Chunarkar, K. Hamza and C. -C. Lee, "Cross-Modal Semantic Alignment Via Ensemble Audio-Text Features for XACLE Challenge," ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2026, pp. 21883-21885, doi: 10.1109/ICASSP55912.2026.11461274.
```bibtex
@INPROCEEDINGS{11461274,
  author={Chunarkar, Snehit B. and Hamza, Krishnagiri and Lee, Chi-Chun},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Cross-Modal Semantic Alignment Via Ensemble Audio-Text Features for XACLE Challenge}, 
  year={2026},
  volume={},
  number={},
  pages={21883-21885},
  doi={10.1109/ICASSP55912.2026.11461274}}
```

## Contact

**Hamza** — GitHub: [https://github.com/Hmzaah](https://github.com/Hmzaah)
