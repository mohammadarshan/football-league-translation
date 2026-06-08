# Hierarchical Bayesian Football League Translation (HBFLT)

This repository contains the code and results for the paper:  
**"A Hierarchical Bayesian Framework for Cross-League Player Performance Translation in Football"**  
*(Submitted to the Journal of Sports Analytics)*

---

## Repository Structure

```
├── Code/
│   ├── HBFLT.ipynb        # Main modeling notebook (hierarchical Bayesian model, validation, figures)
│   └── HBFLTEDA.ipynb     # Exploratory data analysis notebook
├── results/
│   ├── output.pdf         # Full output of HBFLT.ipynb (model results, figures, validation tables)
│   └── outputeda.pdf      # Full output of HBFLTEDA.ipynb (EDA results and diagnostics)
└── README.md
```

---

## Data

The model uses FBref per-90 player statistics sourced from the Kaggle dataset:  
**"FBRef 2017–2024 (Top Five Leagues)"** by Akshan Krithick  
Available at: [https://www.kaggle.com/datasets/akshankrithick/fbref-2017-2024-for-europes-top-5-leagues](https://www.kaggle.com/datasets/akshankrithick/fbref-2017-2024-for-europes-top-5-leagues)
**"Football Player Stats 2024–2025"** by Hubert Sidorowicz
Available at: [https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025)


Download the dataset and place the CSV files in a directory accessible to the notebooks. Update the data path variable at the top of each notebook accordingly.

---

## Notebooks

### 1. `HBFLTEDA.ipynb` — Exploratory Data Analysis
Run this first. Covers:
- Raw data loading and per-90 computation
- Bridge construction filter funnels
- League, position, season, and age composition
- Log-ratio distribution checks
- Outlier diagnostics
- Inter-statistic correlation matrices

**Expected runtime:** ~2–5 minutes  

> **Note:** The EDA bridge includes only transfers from the four major European feeder leagues (La Liga, Bundesliga, Serie A, Ligue 1) to the Premier League. It does not include within-Premier League transfers. The main modeling notebook additionally includes Premier League-to-Premier League transfers in the training bridge as the baseline reference group for league association parameters, yielding higher training bridge counts (174 ATT+MID, 106 DEF) than those reported in the EDA.

---

### 2. `HBFLT.ipynb` — Main Modeling Notebook
Run after the EDA. Covers:
- Training and test bridge construction
- Hierarchical Bayesian model specification (PyMC, NUTS sampler)
- Convergence diagnostics (Rhat, divergences, ESS)
- Out-of-sample validation (coverage, MAE, RMSE)
- ML baselines (Ridge, GBM)
- Publication figures (forest plots, heatmap, calibration curves, OOS validation)
- Sensitivity analyses (Student-t degrees of freedom, weighting scheme)

**Expected runtime:** ~1.5 hours (GPU recommended)

---

## Dependencies

The following Python libraries are required:

```
pymc
arviz
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
jax
jaxlib
```

The notebooks were developed and run on Google Colab with GPU acceleration. It is recommended to use a GPU runtime to reproduce results within the expected runtime.

---

## Reproducibility

All random seeds are fixed (seed = 42). Results in `results/output.pdf` and `results/outputeda.pdf` were generated from the notebooks as provided and reflect the exact outputs reported in the paper.

---

## Notes

- The two held-out test cohorts (2022/23→2023/24 and 2023/24→2024/25) are excluded from model training via the `EXCLUDED_TRANSFER_SEASONS` parameter in `HBFLT.ipynb`.
- Transfer type (permanent vs. loan) is not distinguished, as the data source does not include transfer type information. This is acknowledged as a limitation in the paper.
- The modeling notebook includes a position random effect guard: when only one position group is present (DEF-only bridge), the position effect is suppressed to avoid identifiability issues alongside the intercept.
