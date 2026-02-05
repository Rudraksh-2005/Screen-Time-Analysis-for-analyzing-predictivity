📱 Screen Time Analysis for Predicting Productivity
📌 Project Overview

Screen Time Analysis for Predicting Productivity is a data-driven project designed to analyze users’ screen usage behavior and predict their productivity levels.

The project uses machine learning (XGBoost) to classify productivity into multiple levels based on factors such as:

Screen time duration

App usage category

Notification handling behavior

Device type

Usage time period

The system provides:

📊 Exploratory Data Analysis (EDA)

🤖 Multi-class productivity prediction

🎯 Personalized recommendations

🌐 Interactive frontend dashboard

🎯 Objectives

Analyze how screen usage patterns affect productivity

Build a predictive ML model for productivity classification

Visualize behavioral trends using charts and plots

Provide an interactive frontend for user input and prediction

Bridge data science backend with a user-facing interface

🧠 Productivity Classes

The model predicts productivity into three categories:

🔴 Low Productivity – high distraction, excessive screen time

🟡 Moderate Productivity – balanced but improvable usage

🟢 High Productivity – focused and efficient usage habits

🏗️ Project Architecture
Screen-Time-Analysis-for-analyzing-predictivity/
│
├── data.csv                  # Dataset
├── backend.py                # Python backend (EDA + ML training)
├── frontend.py               # Streamlit frontend
├── README.md                 # Project documentation
│
├── xgb_model.pkl             # Trained XGBoost model (Python)
├── dummy_vars.pkl            # One-hot encoder
├── feature_names.pkl         # Encoded feature names
├── y_labels.pkl              # Productivity class labels

📊 Dataset Description

The dataset (data.csv) contains both demographic and screen usage features.

🔑 Input Features
Feature	Description
Age.Group	User age category
Gender	Male / Female
Education.Level	Education background
Occupation	Student / Professional
Average.Screen.Time	Daily screen usage
Device	Primary device used
Screen.Activity	Type of activity
App.Category	App usage category
Screen.Time.Period	Time of usage
Notification.Handling	How notifications are handled
🎯 Target Variable

Productivity (multi-class categorical)

📈 Exploratory Data Analysis (EDA)

EDA is performed in the backend (can be toggled ON/OFF).

Visualizations Included:

📊 Productivity class distribution (bar chart)

📊 Notification handling vs productivity (stacked bar)

📊 Screen time vs productivity (stacked bar)

📊 App category vs productivity

📦 Boxplots for screen time and notification behavior

🔥 Correlation heatmap (after one-hot encoding)

⭐ XGBoost feature importance plot

EDA is run once for insights and then disabled for performance.

🤖 Machine Learning Model
Model Used

XGBoost (Extreme Gradient Boosting)

Multi-class classification (multi:softprob)

Why XGBoost?

Handles categorical data well after encoding

High performance and accuracy

Prevents overfitting using boosting

Industry-standard ML algorithm

Training Pipeline

Data cleaning

Column normalization (R-style compatibility)

One-hot encoding (OneHotEncoder)

Train-test split (80/20)

XGBoost training

Evaluation using Accuracy, Precision, Recall, F1-score

Model serialization (.pkl)

🧪 Model Evaluation Metrics

The model is evaluated using:

Accuracy

Precision (macro)

Recall (macro)

F1-score (macro)

These metrics ensure balanced performance across all productivity classes.

🌐 Frontend (User Interface)
Technology Used

Streamlit (Python)

Why Streamlit?

Python-native frontend

Fast development

Ideal replacement for R Shiny

Interactive & lightweight

Frontend Features

Sidebar questionnaire (like Shiny’s selectInput)

Button-triggered prediction

Probability bar chart for each class

Personalized productivity recommendations

Cached model loading for fast response

🧾 User Workflow

User opens the Streamlit dashboard

Answers screen usage questions

Clicks Analyze My Productivity

Model predicts productivity level

Probabilities are visualized

Actionable recommendations are shown

🧰 Libraries & Tools Used
📦 Python Libraries

pandas – data manipulation

numpy – numerical computation

scikit-learn – preprocessing & evaluation

xgboost – machine learning model

matplotlib & seaborn – visualizations

joblib – model serialization

streamlit – frontend UI

📦 R Libraries (Original Version)

tidyverse

caret

xgboost

ggplot2

shiny

🔁 R to Python Transition

The project was initially developed in R and later fully converted to Python.

Conversion Summary:
Component	R	Python
Backend	caret + xgboost	scikit-learn + xgboost
Encoding	dummyVars	OneHotEncoder
Visualization	ggplot2	matplotlib + seaborn
Frontend	Shiny	Streamlit
Model files	.rds	.pkl

This conversion makes the project more deployable and industry-ready.

▶️ How to Run the Project
1️⃣ Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit

2️⃣ Run backend (train & save model)
python backend.py

3️⃣ Run frontend
python -m streamlit run frontend.py

4️⃣ Open browser
http://localhost:8501
