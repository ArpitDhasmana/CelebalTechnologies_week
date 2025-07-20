import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Inject custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f4f6f8;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: #1e1e1e;
    }
    h1, h2, h3, h4 {
        color: #003366;
    }
    p, div, span {
        color: #1e1e1e;
    }
    .css-1aumxhk, .st-bx {
        background-color: #e8f0fe !important;
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] {
        background-color: #e8f0fe !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #222222 !important;
    }
    .stSlider > div {
        color: #ff4b4b !important;
    }
    </style>
""", unsafe_allow_html=True)

# Set Streamlit page configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🔬 Breast Cancer Detection with ML")
st.markdown("""
This interactive web app uses a trained machine learning model to predict whether a breast tumor is **malignant** or **benign** based on input features from a digitized image.
""")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")

def user_input_features():
    mean_radius = st.sidebar.slider("Mean Radius", 5.0, 30.0, 14.0)
    mean_texture = st.sidebar.slider("Mean Texture", 5.0, 40.0, 20.0)
    mean_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 200.0, 90.0)
    mean_area = st.sidebar.slider("Mean Area", 100.0, 2500.0, 500.0)
    mean_smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.2, 0.1)
    worst_radius = st.sidebar.slider("Worst Radius", 5.0, 40.0, 20.0)

    data = {
        'mean radius': mean_radius,
        'mean texture': mean_texture,
        'mean perimeter': mean_perimeter,
        'mean area': mean_area,
        'mean smoothness': mean_smoothness,
        'worst radius': worst_radius
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("📥 Entered Data")
st.write(input_df)

# Preprocess input
scaled_input = scaler.transform(input_df)

# Make prediction
prediction = model.predict(scaled_input)
pred_proba = model.predict_proba(scaled_input)

st.subheader("🧠 Model Prediction")
result = "Benign" if prediction[0] == 1 else "Malignant"
st.write(f"### 🔎 Predicted Tumor Type: **{result}**")

st.write("#### 🔢 Prediction Probabilities")
proba_df = pd.DataFrame(pred_proba, columns=["Malignant", "Benign"])
st.bar_chart(proba_df.T)

# Feature Importance
st.subheader("📈 Feature Importance")
st.markdown("Model insights based on training data:")

try:
    importances = model.feature_importances_
    feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'worst radius']
    importance_dict = dict(zip(feature_names, importances))
    sorted_items = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots()
    sns.barplot(x=list(sorted_items.values()), y=list(sorted_items.keys()), ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)
except AttributeError:
    st.info("Feature importance not available for this model.")

st.markdown("---")
