# Dental Caries & Periodontal Disease Risk Assessment

A supervised machine-learning project that predicts **dental caries and periodontal disease risk** from real CDC NHANES clinical survey data using Logistic Regression, Random Forest, and XGBoost — with SMOTE oversampling, SHAP explainability, and publication-quality visualisations.

## Project Objective

This project aims to develop machine learning models for assessing dental caries risk using relevant health and demographic features. The project focuses on data preprocessing, model comparison, evaluation, and explainable machine learning.

## Key Goals

- Analyze dental health-related data
- Identify important risk factors
- Train multiple machine learning models
- Compare model performance
- Use explainable AI to understand predictions

---

## Project Structure

```
dental-caries-risk-assessment/
├── requirements.txt             # Python dependencies
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # NHANES 2017-2018 data downloader & cleaner
│   ├── eda.py                   # Exploratory Data Analysis plots
│   ├── train.py                 # Model training & cross-validation
│   ├── evaluate.py              # ROC / PR curves, confusion matrices, learning curves
│   └── explain.py               # SHAP explainability (TreeExplainer)
├── data/
│   └── raw/
│       └── nhanes_real_dental.csv   # Cleaned NHANES dataset (auto-generated)
├── reports/                     # Generated plots (git-ignored)
└── venv/                        # Virtual environment (git-ignored)
```

## Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python src/data_loader.py       # Download & cache NHANES data
python src/eda.py               # Exploratory Data Analysis (6 plots)
python src/train.py             # Train & compare models (5-Fold CV)
python src/evaluate.py          # Evaluation curves (5 plots)
python src/explain.py           # SHAP explainability (4 plots)
```

All plots are saved to the `reports/` directory.

---

## Dataset: CDC NHANES 2017-2018 Oral Health Survey

This project uses **real clinical data** from the [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/index.htm), a nationally representative survey of the U.S. civilian non-institutionalized population conducted by the CDC's National Center for Health Statistics.

### Source Files

| File | Description | URL |
|------|-------------|-----|
| `DEMO_J.XPT` | Demographics | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT |
| `OHXDEN_J.XPT` | Oral Health Dental Examination | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHXDEN_J.XPT |
| `OHQ_J.XPT` | Oral Health Questionnaire | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHQ_J.XPT |
| `DIQ_J.XPT` | Diabetes Questionnaire | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT |

### Features

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `age` | Integer | DEMO_J (RIDAGEYR) | Age in years at screening |
| `gender` | Binary | DEMO_J (RIAGENDR) | 0 = Male, 1 = Female |
| `bleeding_gums` | Binary | OHQ_J | 0 = No, 1 = Yes |
| `mouth_pain_freq` | Ordinal | OHQ_J | 0 (never) to 4 (very often) |
| `diabetes_status` | Binary | DIQ_J (DIQ010) | 0 = No, 1 = Yes (self-reported) |

### Target

| Variable | Derivation |
|----------|------------|
| `caries_or_perio_risk` | 1 if self-rated oral health condition >= 4 (Fair/Poor), else 0 |

### Citation

> Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS). *National Health and Nutrition Examination Survey (NHANES) 2017-2018.* Hyattsville, MD: U.S. Department of Health and Human Services, CDC.
> https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

---

## Methodology

| Step | Detail |
|------|--------|
| **Data source** | CDC NHANES 2017-2018 (real clinical survey data) |
| **Resampling** | SMOTE (inside each CV fold via `imblearn.Pipeline`) |
| **Cross-validation** | Stratified 5-Fold |
| **Models** | Logistic Regression, Random Forest (200 trees), XGBoost (200 rounds) |
| **Metrics** | PR-AUC (Average Precision), ROC-AUC, Recall |
| **Explainability** | SHAP TreeExplainer (beeswarm, bar, dependence, waterfall) |

---

## Generated Visualisations (15 plots)

### Exploratory Data Analysis
- Feature distributions by risk class
- Pearson correlation heatmap
- Target class balance
- Box plots by risk class
- Violin plots (continuous features)
- Pair plot (scatter matrix)

### Model Evaluation
- ROC curves with mean + std band
- Precision-Recall curves with mean + std band
- Confusion matrices (side-by-side)
- Grouped metric comparison bar chart
- Learning curves (train vs. validation ROC-AUC)

### SHAP Explainability
- Beeswarm summary plot
- Global mean |SHAP| feature importance
- Dependence plots for top features
- Waterfall plot for an individual high-risk patient

---

## Dependencies

- pandas, numpy
- scikit-learn, xgboost, imbalanced-learn
- shap
- matplotlib, seaborn

---

## License

MIT
