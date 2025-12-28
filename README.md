[![XACLE\_Dataset](https://img.shields.io/badge/GitHub-XACLE-blue)](https://github.com/XACLE-Challenge/the_first_XACLE_challenge_baseline_model)
[![Zenodo](https://img.shields.io/badge/Pretrained-GeometricEnsemble-orange?logo=zenodo)](https://zenodo.org/)
[![XACLE\_Leaderboard](https://img.shields.io/badge/Leaderboard-XACLE-limegreen)](https://xacle.org/results.html)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.9-blue)

# GeometricEnsemble-XACLE

> A **distinct, geometry-driven approach** to the XACLE Challenge. This repository contains the **official 2nd Place solution for the ICASSP 2026 XACLE Grand Challenge**, using curvature- and shape-based audio descriptors combined with CLAP-derived audio–text embeddings. This repository presents a fundamentally different modeling strategy centered on geometric audio descriptors, combined with CLAP-derived audio–text embeddings, and an ensemble of regressors for acoustic quality estimation on the XACLE dataset.

![Architecture](https://raw.githubusercontent.com/Hmzaah/GeometricEnsemble-XACLE/main/architecture_diagram.png)

---

## Highlights

* **Second-place solution** in the official XACLE Challenge leaderboard, demonstrating strong generalization and robustness.
* Introduces a **geometry-first feature paradigm**, rather than extending or replicating baseline SVR pipelines.
* Dedicated **Geometric Feature Extractor** (`features/geometric_features.py`) capturing curvature, trajectory, and shape-based audio descriptors.
* Integrates pre-trained CLAP-based models (M2D-CLAP, MGA-CLAP) for complementary audio–text representations.
* Ensemble learner: SVR(s) + LightGBM meta-ensemble (stacking), trained via out-of-fold predictions for stability.
* Notebook-first workflow for feature extraction and rapid experimentation: `train_inference_geometric.ipynb`.

---

## Quick setup

```bash
git clone https://github.com/Hmzaah/GeometricEnsemble-XACLE.git
cd GeometricEnsemble-XACLE
```

Create environment (recommended):

```bash
conda create -n GeomEnsemble python=3.9 -y
conda activate GeomEnsemble
pip install -r requirements.txt
```

Install PyTorch (example for CUDA 11.8):

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## What to download

1. Download XACLE dataset and place it under `datasets/XACLE_dataset/` following the directory structure used in this repo (see `fetch_data.py`).
2. Add CLAP model folders (optional but recommended):

   * **M2D-CLAP**: clone `https://github.com/nttcslab/m2d.git` and place its checkpoint under `m2d/m2d_clap_vit_base-...` as required.
   * **MGA-CLAP**: clone `https://github.com/Ming-er/MGA-CLAP.git` and put pretrained weights into `MGA-CLAP/pretrained_models/`.

> Note: If you only want to run a geometric-only baseline, CLAP models are optional.

---

## Recommended change for MGA-CLAP (inference-only)

In `MGA-CLAP/models/ase_model.py` comment out any heavy imports used only for training/instrumentation (for example `from tools.utils import *`) if you only need inference.

---

## Usage

### Feature extraction (recommended first step)

Use the notebook (recommended):

```
jupyter lab train_inference_geometric.ipynb
```

Or run feature extraction script:

```bash
python features/extract_geometric_and_clap_features.py --data-dir datasets/XACLE_dataset --out-dir features/extracted
```

This script produces a pickled feature table per split: `features/extracted/{train,validation,test}_features.pkl`.

### Training

Train the ensemble (example config):

```bash
python train.py configs/config_geometric_submission2.json
```

This trains base regressors (SVR on geometric, SVR on CLAP, LightGBM) and a stacking meta-learner.

### Inference

```bash
python inference.py <checkpoint_dir> <dataset_key>
```

Example:

```bash
python inference.py outputs/version_geometric_submission2 validation
```

````

Example:

```bash
python inference.py outputs/version_geometric_submission2 validation
````

### Evaluation

```bash
python evaluate.py <inference_csv_path> <ground_truth_csv_path> <save_results_dir>
```

Example:

````bash
python evaluate.py outputs/version_geometric_submission2/inference_result_for_validation.csv datasets/XACLE_dataset/meta_data/validation_average.csv outputs/version_geometric_submission2/
```inference_result_for_validation.csv datasets/XACLE_dataset/meta_data/validation_average.csv outputs/version_geometric_submission2/
````

---

## Directory structure

```
GeometricEnsemble-XACLE
│  README.md
│  requirements.txt
│  train.py
│  inference.py
│  evaluate.py
│  configs/
│  train_inference_geometric.ipynb
│
├─ features
│   ├─ geometric_features.py        # geometric descriptors + utilities
│   ├─ extract_geometric_and_clap_features.py
│   └─ all_feature_dict.py
│
├─ models
│   ├─ regressors.py                # wrappers for SVR, LGBM, stacking
│   └─ load_pretrained_models.py
│
├─ utils
│   └─ utils.py
│
├─ datasets
│   └─ XACLE_dataset
│       ├─ wav
│       │   ├─ train
│       │   └─ validation
│       └─ meta_data
│
├─ m2d
└─ MGA-CLAP
```

---

## Geometric features implemented (high level)

* Short-time spectral shape geometry (framewise centroid/timbre-trajectory curvature)
* Spectral crest/rolloff geometry
* Delta-MFCC curvature and higher-order derivatives
* Harmonicity geometric descriptors (ratio and envelope curvature)
* Temporal zero-crossing geometry (zero-crossing rate slope/curvature)
* Proximity features: pairwise cross-similarity geometry (for short clips)

All features are vectorized by summary statistics: mean, std, skewness, kurtosis, percentiles, and quantized curvature histograms.

---

## Ensemble strategy

1. Extract geometric feature set G and CLAP embedding set C.
2. Train separate SVR models on G and on selected parts of C.
3. Train a LightGBM regressor on concatenated features (optional fast baseline).
4. Stack: meta-learner (Ridge or LightGBM) trained on out-of-fold predictions of base learners.

This design preserves the interpretability of geometric descriptors while benefiting from representation power of CLAP.

---

# Results 🥈

<table style="text-align: center;">
  <thead>
    <tr>
      <th>Version</th>
      <th>SRCC ↑</th>
      <th>LCC ↑</th>
      <th>KTAU ↑</th>
      <th>MSE ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Submission_2 (This repository)</strong></td>
      <td><strong>0.653</strong></td>
      <td><strong>0.673</strong></td>
      <td><strong>0.477</strong></td>
      <td><strong>3.153</strong></td>
    </tr>
  </tbody>
</table>

> **Note**
> - Results correspond to **Submission 2**, the geometry-first approach implemented in this repository.
> - Validation metrics are computed locally.
> - Test metrics are taken directly from the official XACLE leaderboard.


## Hardware & time

* **CPU:** AMD Ryzen 5 (7000 series)
* **GPU:** NVIDIA GeForce RTX 3050 (for CLAP inference)
* **Memory note:** CLAP-based models are VRAM-intensive on consumer GPUs (8 GB). Feature extraction is therefore performed sequentially and cached to disk to prevent out-of-memory errors.
* Training (feature extraction + base learners + stacking): ~60–90 minutes (dataset & hardware dependent)

---

## Reproducibility

* All experiment configs live under `configs/`.
* Use `train.py --config configs/config_geometric_submission2.json` to reproduce our reported run.

---

## License & citation

This project is released under the **MIT License**.

If you use this work, please cite the repository and the original XACLE Challenge dataset accordingly.

---

## Contact

Hamza — GitHub: @Hmzaah
