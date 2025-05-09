# TODO: Evaluate model performance using modern algorithms like SVM (Support Vector Machine) 
#       based on the previous analysis. Compare metrics with current approaches and perform
#       hyperparameter tuning for optimal results.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_directory("outputs")

# ------------------ Load Data ------------------ #
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")
target = "attack_detected"

X = data.drop(columns=[target, "Label"], errors="ignore")
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Hyperparameter Tuning ------------------ #
param_grid = {
    'num_leaves': [51, 70],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.1, 0.05],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"\nBest Parameters: {grid_search.best_params_}")

# ------------------ Train Model ------------------ #
model = best_model
model.fit(X_train, y_train)
# ------------------ Evaluate ------------------ #
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------ Save Model ------------------ #
model.booster_.save_model("Phase 3/outputs/lgb_model.txt")

print("\n Done: Model trained, evaluated, and explained.")

