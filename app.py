import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_PATH, "xgb_model.pkl"))
encoder = joblib.load(os.path.join(BASE_PATH, "dummy_vars.pkl"))
feature_names = joblib.load(os.path.join(BASE_PATH, "feature_names.pkl"))
y_labels = joblib.load(os.path.join(BASE_PATH, "y_labels.pkl"))

st.title("📱 Personalized Productivity Analysis Dashboard")

st.sidebar.header("🧠 Answer a few quick questions")

def select(label, options):
    return st.sidebar.selectbox(label, options)

user_input = {
    "Age.Group": select("Your Age Group", ["Below 18", "18–24", "25–34", "35–44", "45 and above"]),
    "Gender": select("Gender", ["Male", "Female"]),
    "Education.Level": select("Education Level", ["High school or below", "Undergraduate", "Graduate"]),
    "Occupation": select("Occupation", ["Student", "Professional"]),
    "Average.Screen.Time": select("Avg Screen Time", ["Less than 2", "2–4", "4–6", "6–8", "8-10", "More than 10"]),
    "Device": select("Device", ["Smartphone", "Laptop/PC", "Tablet", "Television"]),
    "Screen.Activity": select("Screen Activity", ["Entertainment (gaming, streaming, social media, etc.)", "Academic/Work-related"]),
    "App.Category": select("App Category", [
        "Social Media (e.g., Facebook, Instagram, LinkedIn, Twitter)",
        "Streaming (e.g., YouTube, Netflix)",
        "Productivity (e.g., Microsoft Office, Notion)",
        "Messaging (e.g., WhatsApp, Messenger)",
        "Gaming"
    ]),
    "Screen.Time.Period": select("Screen Time Period", [
        "Morning (6 AM–12 PM)", "Afternoon (12 PM–6 PM)",
        "Evening (6 PM–10 PM)", "Late night (10 PM–6 AM)"
    ]),
    "Notification.Handling": select("Notification Handling", [
        "Check them briefly and resume my work",
        "Ignore them until my task is completed",
        "Spend time interacting with the notifications",
        "Turn off notifications altogether"
    ])
}

if st.sidebar.button("Analyze My Productivity"):
    df = pd.DataFrame([user_input])
    X_user = encoder.transform(df)
    probs = model.predict_proba(X_user)[0]

    pred_idx = np.argmax(probs)
    st.subheader("📌 Your Predicted Productivity Level")
    st.success(y_labels[pred_idx])

    st.bar_chart(pd.DataFrame({
        "Probability (%)": probs * 100
    }, index=y_labels))

    st.subheader("🌟 Recommendations")

    if pred_idx == 0:
        st.error("🔴 Reduce distractions, limit screen time, use focus modes.")
    elif pred_idx == 1:
        st.warning("🟡 Improve consistency using time blocking.")
    else:
        st.success("🟢 Maintain your strong productivity habits.")
