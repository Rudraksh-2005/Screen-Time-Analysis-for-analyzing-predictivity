
import os
import pandas as pd
import numpy as np
import joblib  #saving ML model & objects
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier


RUN_EDA = True  # Change to True ONLY when you want plots


BASE_PATH = os.path.normpath(
    "C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity"
)

os.makedirs(BASE_PATH, exist_ok=True)


data = pd.read_csv(os.path.join(BASE_PATH, "data.csv"))


data.columns = (
    data.columns
    .str.strip()
    .str.replace(" ", ".", regex=False)
)

target = "Productivity"
features = [
    "Age.Group", "Gender", "Education.Level", "Occupation",
    "Average.Screen.Time", "Device", "Screen.Activity",
    "App.Category", "Screen.Time.Period", "Notification.Handling"
]

data = data[features + [target]].dropna()
data[target] = data[target].astype("category")


productivity_colors = ["#FF4C4C", "#FFD700", "#4CAF50"]

if RUN_EDA:
    # 1. Target Distribution (fixed warning)
    sns.countplot(
        x=data[target],
        hue=data[target],
        palette=productivity_colors,
        legend=False
    )
    plt.title("Distribution of Productivity Classes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Notification Handling vs Productivity
    pd.crosstab(
        data["Notification.Handling"], data[target], normalize="index"
    ).plot(kind="bar", stacked=True, color=productivity_colors)
    plt.title("Notification Handling vs Productivity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Average Screen Time vs Productivity
    pd.crosstab(
        data["Average.Screen.Time"], data[target], normalize="index"
    ).plot(kind="bar", stacked=True, color=productivity_colors)
    plt.title("Average Screen Time vs Productivity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. App Category vs Productivity
    pd.crosstab(
        data["App.Category"], data[target], normalize="index"
    ).plot(kind="bar", stacked=True, color=productivity_colors)
    plt.title("App Category vs Productivity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5. Boxplot - Average Screen Time
    data["Average.Screen.Time.Num"] = data["Average.Screen.Time"].astype("category").cat.codes #This converts each category into a number
    sns.boxplot(x=data[target], y=data["Average.Screen.Time.Num"])
    plt.title("Average Screen Time vs Productivity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 6. Boxplot - Notification Handling
    data["Notification.Num"] = data["Notification.Handling"].astype("category").cat.codes
    sns.boxplot(x=data[target], y=data["Notification.Num"])
    plt.title("Notification Handling vs Productivity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X = encoder.fit_transform(data[features])
feature_names = encoder.get_feature_names_out(features) #Keeps track of generated dummy variables

if RUN_EDA:
    corr = pd.DataFrame(X, columns=feature_names).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of One-Hot Encoded Features")
    plt.tight_layout()
    plt.show()

y = data[target].cat.codes
y_labels = list(data[target].cat.categories)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

model = XGBClassifier(
    objective="multi:softprob", #multi-class classification
    num_class=len(y_labels),
    n_estimators=50, #number of trees
    max_depth=5, #controls overfitting
    learning_rate=0.1, #step size
    subsample=0.8, #randomness
    colsample_bytree=0.8, #randomness
    eval_metric="mlogloss",
    n_jobs=-1
)
model.fit(X_train, y_train)

if RUN_EDA:
    importance = model.get_booster().get_score(importance_type="gain"  #Shows which features contribute most) # get_score() It extracts feature importance scores from the trained model.
    imp_df = pd.DataFrame({
        "Feature": importance.keys(),
        "Gain": importance.values()
    }).sort_values("Gain", ascending=False).head(20)

    sns.barplot(x="Gain", y="Feature", data=imp_df, color="#4CAF50")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()


y_pred = model.predict(X_test)

print("\n📊 XGBoost Evaluation")
print("Accuracy :", np.mean(y_pred == y_test))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall   :", recall_score(y_test, y_pred, average="macro"))
print("F1-score :", f1_score(y_test, y_pred, average="macro"))

MODEL_PATH = os.path.join(BASE_PATH, "xgb_model.pkl")
ENCODER_PATH = os.path.join(BASE_PATH, "dummy_vars.pkl")
FEATURES_PATH = os.path.join(BASE_PATH, "feature_names.pkl")
LABELS_PATH = os.path.join(BASE_PATH, "y_labels.pkl")

joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)
joblib.dump(feature_names, FEATURES_PATH)
joblib.dump(y_labels, LABELS_PATH)

assert os.path.exists(MODEL_PATH), "❌ Model file NOT saved!"

print("\n✅ Backend complete & files saved")
print("📁 Files created:")
print(" -", MODEL_PATH)
print(" -", ENCODER_PATH)
print(" -", FEATURES_PATH)
print(" -", LABELS_PATH)

