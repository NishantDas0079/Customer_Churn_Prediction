# 📱 Telecom Customer Churn Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 🎯 Project Overview

This project implements a **Machine Learning pipeline** to predict customer churn for a telecommunications company. The model identifies at-risk customers, enabling proactive retention strategies and reducing customer attrition.

**Key Features:**
- 📊 Comprehensive EDA with visualizations
- 🤖 Multiple ML models with hyperparameter tuning
- 📈 Business insights generation
- 💾 Production-ready pipeline
- 📝 Detailed technical documentation

## 📋 Dataset

**Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Size:** 7,043 customers × 21 features

**Target Variable:** `Churn` (Yes/No)

**Features:**
- Demographic info (gender, senior citizen status)
- Account information (tenure, contract type)
- Service subscriptions (phone, internet, streaming)
- Billing details (payment method, monthly charges)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Quick Start

```bash
# Clone repository
git clone https://github.com/NishantData/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python src/customer_churn_analysis.py
```



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



# Dependencies
```
# requirements.txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
ipython==8.12.0
joblib==1.2.0
scipy==1.10.0
```

# Project Architecture
```
┌─────────────────────────────────────────────────────┐
│                    DATA PIPELINE                    │
├─────────────────────────────────────────────────────┤
│ 1. Data Collection & Validation                     │
│ 2. Exploratory Data Analysis (EDA)                  │
│ 3. Feature Engineering & Preprocessing              │
│ 4. Model Training & Hyperparameter Tuning           │
│ 5. Model Evaluation & Interpretation                │
│ 6. Business Insights Generation                     │
└─────────────────────────────────────────────────────┘
```

# 📈 Methodology
# 1. Data Preprocessing
Missing value imputation

Categorical feature encoding (One-Hot, Label)

Feature scaling (StandardScaler)

Class imbalance handling

# 2. Models Implemented
✅ Random Forest Classifier

✅ Logistic Regression (Baseline)

✅ Gradient Boosting Classifier

✅ Support Vector Machine

✅ Neural Network (MLP)

# 3. Evaluation Metrics
Accuracy, Precision, Recall, F1-Score

ROC-AUC Score

Confusion Matrix Analysis

Business Cost Analysis

# 📊 Results
```
Model Performance Comparison
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC

Random Forest	0.82	0.78	0.85	0.81	0.89

Gradient Boosting	0.81	0.77	0.83	0.80	0.88

Logistic Regression	0.79	0.75	0.80	0.77	0.86
```

# Key Insights
Top 3 Churn Predictors:

Tenure (customer loyalty duration)
Contract type (month-to-month highest risk)
Monthly charges
High-risk Segment: Month-to-month contract, fiber optic, electronic check payment

Retention Opportunity: 23% of customers identified as high-risk

# 🛠️ Usage
# Full Pipeline
```python
from src.customer_churn_analysis import CustomerChurnAnalyzer

# Initialize analyzer
analyzer = CustomerChurnAnalyzer(data_path='data/raw/telco_churn.csv')

# Run complete analysis
results = analyzer.run_full_pipeline()

# Generate report
analyzer.generate_report('reports/final_report.pdf')
```

# Testing
```
python -m pytest tests/ -v
```

Test coverage includes:

Data loading and validation

Preprocessing transformations

Model predictions consistency

Business logic validations
