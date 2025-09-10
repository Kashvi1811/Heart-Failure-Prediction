# Heart Failure Predictor 🫀✨  
Predicting heart failure risk using clinical records with an XGBoost pipeline and an interactive Streamlit app.

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Contact](#-contact)

## 🔎 Overview
Heart failure is a serious health condition where early prediction can save lives. This project leverages clinical data (age, ejection fraction, blood pressure, etc.) and a trained XGBoost pipeline to estimate the probability of heart failure.  

The repository provides:
- A Jupyter notebook with data exploration, preprocessing, and model training.  
- A Streamlit web app for easy, interactive predictions.  
- A pretrained model pipeline (`.pkl`) for instant inference.  

## ✨ Features
- **Data Exploration & Visualization**: Inspect patterns, correlations, and trends in clinical data.  
- **Preprocessing Pipeline**: Clean and transform data consistently for training and inference.  
- **Predictive Modeling**: XGBoost classifier with evaluation metrics.  
- **Interactive App**: Enter patient data via Streamlit UI and receive risk predictions instantly.  
- **Reproducible Workflow**: All steps documented in the notebook for transparency.  

## 📂 Project Structure
```

heart-failure-predictor/
├── app.py                        # Streamlit app for predictions
├── notebooks/
│   └── heartfail.ipynb           # EDA, preprocessing, training, evaluation
├── data/
│   └── heart\_failure\_clinical\_records\_dataset.csv
├── models/
│   └── xgboost\_pipeline.pkl      # Trained model pipeline
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation

````

## ⚡ Getting Started
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/heart-failure-predictor.git
cd heart-failure-predictor
````

2. (Optional) Create a virtual environment:

```bash
python -m venv env
source env/bin/activate       # macOS / Linux
env\Scripts\activate          # Windows PowerShell
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🖥️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to interact with the UI.

Programmatic prediction:

```python
import joblib, pandas as pd
model = joblib.load("models/xgboost_pipeline.pkl")
sample = pd.read_csv("data/heart_failure_clinical_records_dataset.csv").iloc[:1]
prob = model.predict_proba(sample)[:,1]
print(f"Predicted risk: {prob[0]:.2%}")
```

Explore the notebook:

```bash
jupyter notebook
```

Then open `notebooks/heartfail.ipynb`.

## 🛠 Tech Stack

* Python, Jupyter
* pandas, NumPy
* Matplotlib, Seaborn
* scikit-learn, XGBoost
* Streamlit
* joblib

## 🚀 Future Improvements

* Deploy the Streamlit app on **Streamlit Cloud** or **Heroku**.
* Add **REST API endpoints** for predictions.
* Improve accuracy with **hyperparameter tuning**.
* Extend dataset with additional clinical parameters.

## 📜 License

This project is intended for educational and personal use. All rights reserved by **Kashvi1811**.  
*For any usage beyond personal or educational purposes, please contact me in advance.*

## 🤝 Contact & Colloboration

I’m always open to feedback, ideas, and collaboration opportunities! Feel free to reach out:

- **GitHub:** [@Kashvi1811](https://github.com/Kashvi1811)
- **LinkedIn:** [Kashvi on LinkedIn](https://www.linkedin.com/in/kashvisoni1811)
- **Email:** kashvisoni2005@gmail.com

