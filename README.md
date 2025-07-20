

# 🔬 Breast Cancer Detection with Machine Learning

This interactive web application uses a trained machine learning model to predict whether a breast tumor is **malignant** or **benign** based on input features from a digitized image.

[🌐 View the Live App](https://celebaltechnologiesweek7.streamlit.app/)

---

## 🧠 Model Overview

The model was trained using the [Breast Cancer Wisconsin Diagnostic Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), focusing on a subset of the most relevant features to ensure interpretability and real-time performance.

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: StandardScaler
- **Features Used**:
  - Mean Radius
  - Mean Texture
  - Mean Perimeter
  - Mean Area
  - Mean Smoothness
  - Worst Radius

---

## 🛠️ Features

- 📊 Real-time prediction using interactive sliders
- 🔢 Probability distribution of outcomes
- 📈 Visualized feature importances
- 🌐 Deployed using [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🚀 How to Run Locally

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📁 Files in This Project

| File               | Description                             |
| ------------------ | --------------------------------------- |
| `app.py`           | Main Streamlit app                      |
| `model.pkl`        | Trained Random Forest Classifier        |
| `scaler.pkl`       | Fitted StandardScaler for preprocessing |
| `requirements.txt` | Python dependencies                     |
| `README.md`        | This file                               |

---

## ❤️ Credits

Built with:

* [Streamlit](https://streamlit.io/)
* [scikit-learn](https://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)

---
```

---

Would you like me to automatically write this `README.md` file to your project folder as well?
```
