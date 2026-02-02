📱 Screen Time Analysis for Predicting Productivity
📌 Project Overview

This project analyzes screen usage behavior and predicts a user's productivity level using machine learning.
It combines data visualization, XGBoost classification, and an interactive Shiny dashboard to provide personalized insights and recommendations.

The system:

Analyzes demographic and screen usage patterns

Trains a multi-class XGBoost model

Predicts productivity as Low / Moderate / High

Provides actionable recommendations via a clean UI

🎯 Objectives

Understand how screen habits impact productivity

Visualize behavioral trends using graphs

Build a predictive ML model

Provide personalized productivity feedback

Create a user-friendly interactive dashboard

🛠️ Technologies & Tools Used
🔹 Programming Language

R

🔹 Libraries Used (Backend)
Library	Purpose
tidyverse	Data cleaning & manipulation
caret	Train-test split, dummy encoding, evaluation
xgboost	Machine learning model
MLmetrics	Precision, Recall, F1-Score
ggplot2	Data visualization
reshape2	Data reshaping for heatmap
🔹 Libraries Used (Frontend – Shiny)
Library	Purpose
shiny	Web dashboard
ggplot2	Probability visualization
dplyr	Data handling
xgboost	Prediction
caret	Feature alignment
📂 Project Structure
Screen-Time-Analysis-for-analyzing-predictivity/
│
├── data.csv
├── backend.R
├── frontend.R
├── xgb_model.rds
├── dummy_vars.rds
├── feature_names.Rdata
├── y_labels.Rdata
└── README.md

📊 Dataset Description

The dataset includes:

Demographics: Age, Gender, Education, Occupation

Screen Behavior: Screen time, device type, activity

Usage Patterns: App category, usage time, notifications

Target Variable: Productivity (categorical)

⚙️ Backend (Model Training & Analysis)
1️⃣ Data Loading & Cleaning

CSV file is loaded

Only required columns are selected

Missing values are removed

Target (Productivity) is converted to factor

2️⃣ Exploratory Data Analysis (Graphs)
📌 Productivity Distribution

Bar Plot

Shows how many users fall into each productivity class

📌 Notification Handling vs Productivity

Stacked Bar Plot

Displays how notification habits influence productivity

📌 Screen Time vs Productivity

Stacked Bar Plot

Shows productivity proportions for different screen durations

📌 App Category vs Productivity

Stacked Bar Plot

Highlights which app types are linked to productivity loss or gain

📌 Boxplots

Screen Time vs Productivity

Notification Handling vs Productivity
Shows variation and spread across productivity levels

📌 Correlation Heatmap

Generated after one-hot encoding

Displays relationships between encoded features

Helps detect redundancy and feature dependency

3️⃣ Feature Engineering

One-Hot Encoding using dummyVars

All categorical variables converted to numeric

Encoded feature names saved for frontend consistency

4️⃣ Train-Test Split

80% training, 20% testing

Reproducibility ensured using set.seed(123)

5️⃣ XGBoost Model Training

Multi-class classification (multi:softprob)

Produces probability for each productivity class

Trained using 100 boosting rounds

6️⃣ Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-Score

A confusion matrix is generated to analyze performance.

7️⃣ Feature Importance

XGBoost Gain Plot

Shows which features influence productivity most

Helps interpret the model

8️⃣ Saved Objects

Used by frontend:

xgb_model.rds

dummy_vars.rds

feature_names.Rdata

y_labels.Rdata

🖥️ Frontend (Shiny Dashboard)
🎨 UI Design

Clean and minimal UI

Soft background colors

Sidebar for inputs

Main panel for insights & plots

🧠 User Inputs

Users answer:

Age group

Gender

Education level

Occupation

Average screen time

Device used

Screen activity

App category

Usage time period

Notification handling behavior

🔍 Server Logic

Load trained model & preprocessing objects

Convert user inputs into factors

Apply same dummy encoding as backend

Align features with training data

Predict productivity probabilities

Display results dynamically

📈 Output Components
✅ Predicted Productivity Level

Displayed as text:

Low Productivity

Moderate Productivity

High Productivity

📊 Probability Bar Chart

Shows confidence (%) for each class

Color-coded:

🔴 Low

🟡 Moderate

🟢 High

🌟 Personalized Recommendations

Based on prediction:

🔴 Low Productivity

Reduce entertainment screen time

Disable notifications

Pomodoro technique

Focus mode

Mindfulness habits

🟡 Moderate Productivity

Weekly screen review

Time blocking

Habit stacking

Task planning

🟢 High Productivity

Maintain routines

Prevent burnout

Balance work & rest

Share best practices

▶️ How to Run the Project
Step 1: Install Libraries
install.packages(c(
  "tidyverse", "caret", "xgboost",
  "MLmetrics", "ggplot2", "reshape2", "shiny"
))

Step 2: Run Backend
source("backend.R")

Step 3: Run Shiny App
source("frontend.R")

🚀 Key Highlights

✔ End-to-end ML pipeline
✔ Interpretable visualizations
✔ Real-time prediction
✔ Personalized recommendations
✔ Clean and interactive UI

🧠 Conclusion

This project demonstrates how data science and machine learning can be used to analyze everyday digital habits and provide actionable insights for improving productivity.
It bridges analytics, ML, and human-centered UI design into a single system.
