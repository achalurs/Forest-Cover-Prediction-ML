# ================================
# Forest Cover Type Prediction
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("train.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())

# -------------------------------
# 2. Separate Features & Target
# -------------------------------
X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]

print("\nTarget Variable Distribution:")
print(y.value_counts())

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 5. Train Random Forest Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# 6. Model Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 7. Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 8. Confusion Matrix
# -------------------------------
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 9. Save Model
# -------------------------------
joblib.dump(model, "forest_cover_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and Scaler Saved Successfully!")
