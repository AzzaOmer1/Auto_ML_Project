# 🚀 Professional AutoML Platform with MLflow

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg" alt="MLflow">
</p>

---

## 📌 Overview

**Professional AutoML Platform** is an end-to-end Machine Learning web application designed to bridge the gap between static notebooks and production-grade applications. Built with **Streamlit**, **Scikit-Learn**, and **MLflow**, it empowers users to seamlessly go from raw data and Exploratory Data Analysis (EDA) to automated preprocessing, multi-model training, hyperparameter tuning, and model deployment without writing complex boilerplate code.

---

## 🎥 Demo Preview

<p align="center">
  <video src="assets\auto_ml.mp4" width="100%" controls autoplay loop></video>
</p>

---

## 📌 Features

* **Data Ingestion & Validation:** Supports `.csv` and `.xlsx` file uploads with automatic target and problem type detection (Classification vs. Regression).
* **Exploratory Data Analysis (EDA):**
  * Automated summary statistics (`describe()`).
  * Interactive feature distribution plots (Histograms with KDE).
  * Correlation heatmaps for numerical features.
* **Robust Data Preprocessing:**
  * Strict 3-Way Split (Train, Validation, and Test sets).
  * Automated imputation for missing values and robust categorical encoding.
* **Multi-Model Training & Evaluation:**
  * Supports Logistic Regression, Random Forest, Ridge, XGBoost, and LightGBM.
  * Comprehensive validation metrics (Accuracy, F1-Score, MSE, R2-Score).
* **Experiment Tracking:**
  * Full integration with **MLflow** and a local SQLite backend (`mlflow.db`) to track parameters, metrics, and save model artifacts.
* **Hyperparameter Optimization:**
  * Automated tuning using `RandomizedSearchCV`.
* **Model Export:**
  * Single-click download for the best performing production-ready model in `.pkl` format.

---

## 📁 Project Structure

```text
Auto_ML/
│
├── .venv/                 # Virtual Environment (Ignored in Git)
├── models/                # Directory for saved/exported models
├── src/                   # Core Source Code Modules
│   ├── __init__.py        # Package initializer
│   ├── data_loader.py     # Data loading & file format handling
│   ├── eda.py             # Exploratory Data Analysis functions
│   ├── preprocessor.py    # Preprocessing, 3-way split, encoding & scaling
│   ├── model_selector.py  # Training, MLflow logging & hyperparameter tuning
│   └── evaluator.py       # Performance evaluation metrics
│
├── .gitignore             # Untracked files/folders
├── app.py                 # Main Streamlit Application (UI)
├── config.py              # Global configuration & hyperparameters
├── mlflow.db              # Local MLflow database (Ignored in Git)
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```
---
## 💻 How to Run Locally

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/AzzaOmer1/Auto_ML.git
cd Auto_ML
```

### 2. Create & Activate a Virtual Environment

**On Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 MLOps Integration (MLflow)

To view your logged experiments, parameters, and metrics in the MLflow UI, run the following command in a separate terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 🛠️ Built With

* [Python](https://www.python.org/) - Core Programming Language
* [Streamlit](https://streamlit.io/) - Interactive Web Application Framework
* [Scikit-Learn](https://scikit-learn.org/) - Machine Learning Library
* [XGBoost](https://xgboost.ai/) & [LightGBM](https://lightgbm.readthedocs.io/) - Gradient Boosting Algorithms
* [MLflow](https://mlflow.org/) - MLOps Experiment Tracking & Model Registry
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data Manipulation & Processing
* [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Data Visualization
