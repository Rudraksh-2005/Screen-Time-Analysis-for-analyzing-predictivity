# 📱 Screen Time Analysis for Predicting Productivity

## 📌 Project Overview
This project analyzes user screen usage behavior and predicts productivity levels using machine learning.  
It combines **data analysis, visualization, model training, and an interactive frontend**.

The system takes user inputs such as screen time, app usage, and notification handling, then predicts productivity as:
- Low
- Moderate
- High

---

## 🎯 Objectives
- Analyze screen usage patterns
- Predict productivity using machine learning
- Visualize behavior through charts
- Provide an interactive frontend for users

---

## 🏗️ Project Structure
Screen-Time-Analysis-for-analyzing-predictivity/
│
├── data.csv
├── backend.py
├── frontend.py
├── README.md
│
├── xgb_model.pkl
├── dummy_vars.pkl
├── feature_names.pkl
├── y_labels.pkl


---

## 📊 Dataset Description
The dataset contains demographic and screen usage features.

### Input Features
- Age.Group
- Gender
- Education.Level
- Occupation
- Average.Screen.Time
- Device
- Screen.Activity
- App.Category
- Screen.Time.Period
- Notification.Handling

### Target Variable
- Productivity (categorical)

---

## 📈 Exploratory Data Analysis (EDA)
The backend includes optional EDA with:
- Productivity distribution bar chart
- Screen time vs productivity
- Notification handling vs productivity
- App category vs productivity
- Boxplots for behavioral comparison
- Correlation heatmap (after encoding)
- Feature importance plot (XGBoost)

EDA can be enabled or disabled for performance.

---

## 🤖 Machine Learning Model
- Algorithm: **XGBoost**
- Type: Multi-class classification
- Encoding: One-hot encoding
- Split: 80% training / 20% testing

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## 🌐 Frontend
The frontend is built using **Streamlit**.

### Features
- Sidebar questionnaire (similar to Shiny)
- Button-based prediction
- Probability visualization
- Personalized productivity recommendations

---

## 🧰 Technologies Used

### Python Libraries
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib
- streamlit

### R Libraries (Initial Version)
- tidyverse
- caret
- ggplot2
- shiny
- xgboost

---

## 🔁 R to Python Conversion
The project was initially built in **R (Shiny + caret)** and later fully converted to **Python (Streamlit + scikit-learn)** for better deployment and scalability.

---

## ▶️ How to Run the Project

### 1. Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit

### 2. Run backend
python backend.py

### 3. Run frontend
python -m streamlit run frontend.py

### 4. Open browser
http://localhost:8501
