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
