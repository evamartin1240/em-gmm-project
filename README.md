# EM for Gaussian Mixture Models: Initialization Study

This repository contains an implementation of the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMMs), developed from scratch using Python and NumPy. The project investigates how the choice of initialization—random or KMeans-based—affects the performance of EM in terms of convergence and clustering quality.

## Objectives

- Implement the EM algorithm for GMMs without using high-level machine learning libraries.
- Compare random and KMeans initialization strategies.
- Evaluate performance on synthetic data and real biomedical data (Breast Cancer Wisconsin Diagnostic dataset).
- Measure clustering quality using Adjusted Rand Index (ARI) and analyze log-likelihood evolution.
- Assess the role of feature selection in improving numerical stability in high-dimensional data.

## Repository Structure

```
em-gmm-project/
│
├── emgmm/                      # Source code for EM and helper functions
│   ├── gmm.py                  # EM algorithm implementation
│   └── utils.py                # Initialization and utility functions
│
├── notebooks/                 
│   ├── 01_em_gmm_demo.ipynb            # Synthetic data experiments
│   ├── 02_real_data_gmm.ipynb          # Application on real data
│   ├── 03_highdim_gmm.ipynb            # Experiments with full feature set
│   └── 04_subset_features_gmm.ipynb    # Feature selection experiments
│
├── figures/                    # Plots and visualizations
│
├── report/                     # LaTeX report and sections
│   ├── main.tex
│   └── sections/
│
├── requirements.txt            # Project dependencies
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/em-gmm-project.git
   cd em-gmm-project
   ```

2. (Optional) Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Open and run the notebooks using Jupyter or JupyterLab.

## Dependencies

- Python ≥ 3.8
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter

## Credits

This project was developed by Eva Martín as part of the Algorithms and Data Mining course in the MDS program at Universitat Politècnica de Catalunya (UPC).

## License

This repository is intended for academic and research use only. Please contact the author for permission if you plan to reuse or adapt this work in other contexts.

