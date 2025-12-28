[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-red)](https://2026.ieeeicassp.org/)

# GeometricEnsemble-XACLE
**Official implementation of "Approach 2" for the XACLE Grand Challenge (2nd Place).**

This repository houses the **Geometric Heterogeneous Ensemble**, a dual-stream architecture that combines explicit geometric feature engineering with a Split-Brain prediction head (XGBoost + SVR).

## Architecture
Our approach processes a **9,220-dimensional feature vector** through two distinct mathematical pathways.

<p align="center">
  <a href="assets/Flowchart.pdf">
    <img src="assets/architecture_diagram.png" alt="Geometric Ensemble Architecture" width="850">
  </a>
  <br>
  <em>(Click image to view High-Resolution PDF)</em>
</p>

# Setup
### 1. Clone the repository
```bash
git clone [https://github.com/Hmzaah/GeometricEnsemble-XACLE.git](https://github.com/Hmzaah/GeometricEnsemble-XACLE.git)
