# HeartGuard AI - Predictive Health Analytics for Preventive Care

> AI-powered heart disease risk predictor using Random Forest & XGBoost on the UCI dataset. Features an end-to-end ML pipeline with EDA, preprocessing, hyperparameter tuning, and model evaluation. Includes an interactive Streamlit dashboard for real-time risk prediction and preventive care insights.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189C3E?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

---

## Live Demo

**Streamlit Dashboard:** [Heart Disease Risk Predictor](https://heart-disease-risk-predictor--aiml-unfvjfkxxhdusui6pv8epd.streamlit.app/)

> Click the link above to try the live interactive dashboard — no installation required.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models & Results](#models--results)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

Cardiovascular disease is the **leading cause of death globally**, yet many cases are entirely preventable with early detection and lifestyle intervention. **HeartGuard AI** leverages machine learning to identify individuals at risk of heart disease based on routine clinical measurements — enabling proactive, preventive care rather than reactive treatment.

This project delivers:
- A fully documented **Jupyter Notebook** covering every stage of the ML lifecycle
- Two trained classifiers (**Random Forest** and **XGBoost**) with hyperparameter tuning
- Comprehensive **model evaluation** with ROC curves, confusion matrices, and classification reports
- An interactive **Streamlit dashboard** for clinicians or researchers to input patient data and receive real-time risk predictions

---

## Project Structure

```
heartguard-ai/
│
├── heart_disease_predictive_analytics.ipynb  # Main Jupyter Notebook (end-to-end)
├── app.py                                    # Streamlit dashboard
├── requirements.txt                          # Python dependencies
│
├── models/                                   # Saved model artifacts
│   ├── heart_disease_model.pkl               # Best trained model
│   ├── heart_disease_scaler.pkl              # Fitted StandardScaler
│   ├── feature_importance.pkl                # Feature importance rankings
│   └── model_metadata.json                   # Model config & evaluation metrics
│
├── data/
│   └── heart.csv                             # Heart Disease UCI dataset
│
└── outputs/                                  # Generated plots & predictions
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── feature_histograms.html
    ├── feature_boxplots.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    └── test_predictions.csv
```

---

## Dataset

| Property | Value |
|----------|-------|
| Records | 920 (multi-source) / 303 (Cleveland only) |
| Features | 13 clinical features + 1 target |
| Target | `num` binarized to 0 (no disease) / 1 (disease) |
| Missing Values | Present in `ca`, `thal`, `slope` — handled via median imputation |

### Feature Reference

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Continuous |
| `sex` | Sex (Male / Female) | Categorical |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure (mmHg) | Continuous |
| `chol` | Serum cholesterol (mg/dl) | Continuous |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0-2) | Categorical |
| `thalach` | Maximum heart rate achieved | Continuous |
| `exang` | Exercise-induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | Continuous |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0-3) | Ordinal |
| `thal` | Thalassemia type | Categorical |
| **`target`** | **Heart disease present (1) or absent (0)** | **Binary** |

---

## Pipeline

```
Raw CSV
   |
   v
Data Cleaning
(drop irrelevant cols, rename, binarize target)
   |
   v
Exploratory Data Analysis
(distributions, correlations, categorical analysis)
   |
   v
Preprocessing
(string encoding, one-hot encoding, StandardScaler)
   |
   v
Feature Engineering
(age_group, high_chol, hypertension, risk_score)
   |
   v
Model Training
(Random Forest + XGBoost with GridSearchCV)
   |
   v
Evaluation
(Accuracy, Precision, Recall, F1, ROC-AUC)
   |
   v
Predictions + Insights
   |
   v
Streamlit Dashboard
```

---

## Models & Results

Both models were tuned using **GridSearchCV** with **5-fold Stratified Cross-Validation**, optimizing for **ROC-AUC**.

### Performance Comparison

| Metric | Random Forest | XGBoost |
|--------|:------------:|:-------:|
| Accuracy | ~85% | ~86% |
| Precision | ~84% | ~85% |
| Recall | ~87% | ~88% |
| F1-Score | ~85% | ~86% |
| ROC-AUC | ~0.92 | ~0.93 |

### Top Risk Factors

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `thalach` | Maximum heart rate achieved |
| 2 | `ca` | Number of blocked major vessels |
| 3 | `oldpeak` | ST depression on exercise |
| 4 | `cp` | Chest pain type |
| 5 | `thal` | Thalassemia type |
| 6 | `exang` | Exercise-induced angina |
| 7 | `age` | Patient age |
| 8 | `chol` | Serum cholesterol |
| 9 | `trestbps` | Resting blood pressure |
| 10 | `sex` | Patient sex |

---

## Installation

### Prerequisites
- Python 3.10+
- pip
- Git

### Step 1 - Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/heartguard-ai.git
cd heartguard-ai
```

### Step 2 - Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 - Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 - Download the Dataset
Place `heart.csv` in the project root directory.

```bash
# Option A - Kaggle CLI
pip install kaggle
kaggle datasets download -d ronitf/heart-disease-uci
unzip heart-disease-uci.zip

# Option B - Manual download from Kaggle and place heart.csv in root folder
```

---

## Usage

### Run the Jupyter Notebook
```bash
jupyter notebook heart_disease_predictive_analytics.ipynb
```

Run all cells sequentially. The notebook will:
- Train and evaluate both models
- Generate all plots and save them to `outputs/`
- Save model artifacts (`*.pkl`, `model_metadata.json`)

### Launch the Streamlit Dashboard Locally
```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

---

## Streamlit Dashboard

The interactive dashboard provides three tabs:

| Tab | Features |
|-----|----------|
| **Risk Prediction** | Sidebar sliders for all 13 patient parameters, real-time risk probability, risk category (Low / Moderate / High), clinical recommendations, download prediction as CSV |
| **Feature Importance** | Interactive bar chart of top N risk factors, ranked importance table |
| **Dataset Summary** | Class distribution chart, summary statistics, raw data preview, download full dataset as CSV |

---

## Key Insights

Based on model analysis and clinical literature:

- **Low maximum heart rate** (`thalach < 140`) is one of the strongest predictors — indicates poor cardiac reserve
- **Asymptomatic chest pain** paradoxically signals higher risk due to silent ischemia
- **Number of blocked vessels** (`ca`) has a near-linear relationship with disease probability
- **ST depression** (`oldpeak > 2`) under exercise stress is a reliable marker of ischemia
- **Males over 55** face significantly elevated baseline risk
- **High cholesterol** (> 240 mg/dl) and **hypertension** (> 140 mmHg) are compounding modifiable risk factors

> All findings should be validated by qualified medical professionals before clinical application.

---

## Technologies Used

| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Model Persistence | `joblib` |
| Dashboard | `streamlit` |
| Environment | `jupyter`, `python 3.10+` |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

> This project is intended for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

<div align="center">
  Built with Python and Streamlit &nbsp;|&nbsp; Star this repo if you found it useful!
</div>
