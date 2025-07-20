from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load and select 6 features
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df = df[['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'worst radius']]
X = df
y = data.target

# Train-test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale and train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
