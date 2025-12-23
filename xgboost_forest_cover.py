import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("train.csv")

# Drop Id column (not useful for prediction)
data.drop("Id", axis=1, inplace=True)

X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"] - 1  # XGBoost expects labels starting from 0

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 3. Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 4. XGBoost Model
# -------------------------------
xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=7,
    eval_metric="mlogloss",
    random_state=42
)

# -------------------------------
# 5. Hyperparameter Tuning
# -------------------------------
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# -------------------------------
# 6. Evaluation
# -------------------------------
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nXGBoost Accuracy:", accuracy * 100, "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. Save Model
# -------------------------------
joblib.dump(best_model, "xgboost_forest_model.pkl")
joblib.dump(scaler, "xgboost_scaler.pkl")

print("\nXGBoost model saved successfully!")
