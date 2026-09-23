# 🌍 ESG Sentiment Classification: Classical vs. Quantum Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-QML-6929C4.svg)](https://qiskit.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comparative study applying classical and quantum machine learning models to sentiment classification of ESG (Environmental, Social, and Governance) report excerpts — with honest, reproducible negative results.

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Notebook Structure](#notebook-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Scope & Limitations](#scope--limitations)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

## Overview

This project investigates whether quantum machine learning (QML) offers any advantage over classical ML for a real-world NLP classification task: labeling ESG-related text as **risk**, **neutral**, or **opportunity**. The dataset consists of expert-labeled sentences from corporate ESG disclosures.

Classical baselines (Logistic Regression, RBF-kernel SVM) are compared against quantum models (Quantum SVM, Variational Quantum Classifier) built on the same preprocessed feature space, with results validated via k-fold cross-validation.

## Key Findings

- Classical models (Logistic Regression, RBF-SVM) significantly and consistently outperformed the quantum models (QSVM, VQC) across cross-validation folds.
- The performance gap is attributed to:
  - **Information bottleneck** — aggressive dimensionality reduction (TF-IDF → 4D via TruncatedSVD) ahead of quantum encoding
  - **Scaling constraint** — O(n²) quantum kernel computation cost forcing evaluation on small training subsets
  - **Encoding mismatch** — possible mismatch between the chosen feature map / entangling structure and the underlying data geometry
- These results are reported as **honest negative findings**, consistent with current QML literature showing that quantum kernel methods do not yet reliably outperform classical methods on structured/tabular-reduced NLP data at NISQ scale.

## Dataset

[`climatebert/climate_sentiment`](https://huggingface.co/datasets/climatebert/climate_sentiment) — expert-labeled ESG report sentences classified as risk, neutral, or opportunity.

## Pipeline

```
Raw text
  → TF-IDF vectorization
  → TruncatedSVD (dimensionality reduction to 4D)
  → MinMaxScaler (feature scaling to [0, π] for quantum angle encoding)
  → Classical models: Logistic Regression, RBF-SVM
  → Quantum models: QSVM (FidelityQuantumKernel + ZZFeatureMap), VQC (RealAmplitudes ansatz)
  → k-fold cross-validation
  → Comparative analysis
```

## Results

| Model                | Type      | Mean CV Accuracy | Notes |
|-----------------------|-----------|:-----------------:|-------|
| Logistic Regression   | Classical |  63.3% ± 5.7%         | Strong, stable baseline |
| RBF-SVM                | Classical | 60.0% ± 11.3%        | Best-performing overall |
| QSVM (ZZFeatureMap)    | Quantum   |32.2% ± 8.9%        | Limited by kernel scaling |
| VQC (RealAmplitudes)   | Quantum   |35.6% ± 9.0%         | Trained via COBYLA/SPSA |

> Replace the placeholders above with your actual cross-validation numbers from the notebook before publishing.

## Tech Stack

- **Language:** Python
- **Classical ML:** scikit-learn
- **Quantum ML:** Qiskit, `qiskit_algorithms`
  - `ZZFeatureMap`, `FidelityQuantumKernel`, `QSVC`
  - `RealAmplitudes` ansatz with COBYLA / SPSA optimizers (VQC)
- **Data handling:** Pandas, Hugging Face `datasets`
- **Visualization:** Matplotlib
- **Environment:** Jupyter notebooks (VS Code), conda

## Repository Structure

```
├── notebooks/
│   └── esg_sentiment_qml.ipynb   # Main analysis notebook
├── README.md
├── requirements.txt
└── LICENSE
```

> Adjust the structure above to match your actual file layout before pushing.

## Notebook Structure

1. Data loading & preprocessing
2. Classical baseline models
3. Quantum models (QSVM, VQC)
4. Comparative analysis
5. K-fold cross-validation
6. Interpretation
7. Conclusion

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create and activate the conda environment
conda create -n myenvironment python=3.10
conda activate myenvironment

# Install dependencies
pip install -r requirements.txt
```

Key packages: `qiskit`, `qiskit-algorithms`, `scikit-learn`, `pandas`, `datasets`, `pyarrow`, `matplotlib`, `pylatexenc`

## Usage

```bash
jupyter notebook notebooks/esg_sentiment_qml.ipynb
```

Run all cells from top to bottom. If the kernel is restarted, re-run all cells to restore variables, and use `%pip install <package>` inside a notebook cell to install any missing dependency into the active kernel.

## Scope & Limitations

This is an academic/coursework project rather than a novel research contribution. The current scope (aggressive dimensionality reduction, small-scale quantum kernel evaluation) is sufficient to demonstrate a rigorous classical-vs-quantum comparison but would need more extensive statistical testing and richer encoding strategies before being suitable for a peer-reviewed venue.

## Future Work

- Explore alternative feature maps and entangling structures better matched to the data geometry
- Reduce information loss by testing higher-dimensional embeddings with more qubit-efficient encodings
- Add statistical significance testing (e.g., McNemar's test) across model pairs
- Scale kernel evaluation with approximate/sampled quantum kernel methods

## Author

**Faustena** — MSc Data Science

## License

This project is licensed under the [MIT License](LICENSE) — feel free to adapt this to your institution's or your own preferred license before publishing.
