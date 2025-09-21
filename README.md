## Student-perfomance
## 📌 Project Overview  
This project is an **end-to-end Machine Learning pipeline** designed to predict a student’s **Math score** based on demographic and academic factors such as:  
- Gender  
- Race/Ethnicity  
- Parental level of education  
- Lunch type  
- Test preparation course  
- Reading and writing scores  

The project integrates:  
- **Exploratory Data Analysis (EDA)** and **model training** using Jupyter Notebooks.  
- **Data pipeline** for preprocessing, training, evaluation, and inference.  
- **Flask-based web application** for real-time predictions.  
- **Artifacts storage** for datasets, trained models, preprocessing objects, and evaluation metrics.  

## Installation

1.Clone the repository:
```
git clone https://github.com/your-username/math-score-predictor.git
cd math-score-predictor
```
2.Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
3.Install dependencies:
```
pip install -r requirements.txt
```
## Usage
1. Run the Flask App
```
python app.py
```

## Machine Learning Pipeline

1.Data Collection – Student performance dataset.
2.EDA (Exploratory Data Analysis) – Feature distributions, correlations, and insights.
3.Data Preprocessing – Handling missing values, encoding categorical features, scaling numerical features.
4.Model Training & Evaluation – Multiple regression models tested.
5.Model Selection – Best-performing model finalized.
6.Deployment – Flask web app with saved preprocessing pipeline & model.
  
