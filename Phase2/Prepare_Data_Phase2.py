import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
import shap
from sklearn.metrics import accuracy_score 
from tabulate import tabulate

def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def evaluate_model(y_true, y_pred, model_name, is_classifier=False):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print("+------------+---------+")
    print("| Metric     | Value   |")
    print("+============+=========+")
    print(f"| MAE        | {mae:.4f}  |")
    print(f"| MSE        | {mse:.4f}  |")
    print(f"| RMSE       | {rmse:.4f}  |")
    print(f"| R²         | {r2:.4f}  |")
    
    if is_classifier:
        accuracy = accuracy_score(y_true, y_pred.round()) * 100
        print(f"| Accuracy   | {accuracy:.2f}% |")
    else:
        threshold = 0.1
        correct_predictions = np.sum(np.abs(y_true - y_pred) <= (threshold * np.abs(y_true)))
        percentage_correct = (correct_predictions / len(y_true)) * 100
        print(f"| Within {threshold*100:.0f}% | {percentage_correct:.2f}% |")
    
    print("+------------+---------+")

def plot_feature_importance(model, feature_names, title, filename):
    """Plot and save feature importance."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    
    plt.figure(figsize=(12, 6))
    importances.nlargest(10).plot(kind='bar', title=title)
    plt.savefig(filename)
    plt.show()

def plot_residuals(y_test, y_pred, model_name, filename):
    """Plot residual distribution."""
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, label=model_name, alpha=0.6)
    plt.axhline(0, color='black', linewidth=1)
    plt.legend()
    plt.title(f"{model_name} Residual Distribution")
    plt.savefig(filename)
    plt.show()

def plot_shap_summary(model, X_test, filename):
    """Plot SHAP summary for feature impact analysis."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test)
    plt.savefig(filename)
    plt.show()

# --------------Load & Prepare Data---------------------
# Create directory for saving plots
create_directory("Phase2/Plots")

# Load dataset
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")

# Define main target variable
target_column = "attack_detected"
X = data.drop(columns=[target_column, "Label"])
y = data[target_column]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------Train Models-------------------------------

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
lgb_model = LGBMClassifier(random_state=42)

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# --------------Evaluate Models------------------------------
# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)

# Evaluate each model
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_lgb, "LightGBM Classifier",is_classifier=True)



#-------------- Feature Importance Visualization----------------
plot_feature_importance(rf_model, X.columns, "Random Forest Feature Importance", "Phase2/Plots/feature_importance_rf.png")
plot_feature_importance(gb_model, X.columns, "Gradient Boosting Feature Importance", "Phase2/Plots/feature_importance_gb.png")
plot_feature_importance(lgb_model, X.columns, "LightGBM Feature Importance", "Phase2/Plots/feature_importance_lgb.png")



# --------------Residual Analysis--------------------------------
plot_residuals(y_test, y_pred_rf, "Random Forest", "Phase2/Plots/residual_rf.png")
plot_residuals(y_test, y_pred_gb, "Gradient Boosting", "Phase2/Plots/residual_gb.png")
plot_residuals(y_test, y_pred_lgb, "LightGBM", "Phase2/Plots/residual_lgb.png")



#-------------- Train LightGBM for Session Duration Prediction-----
# Define new target variable
target_column = "session_duration"
X_new = data.drop(columns=[target_column, "Label", "attack_detected"])
y_new = data[target_column]

# Train-test split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Train LightGBM regressor
lgb_regressor = LGBMRegressor(random_state=42)
lgb_regressor.fit(X_train_new, y_train_new)



#-------------- Feature Importance (Session Duration)----------------
plot_feature_importance(lgb_regressor, X_new.columns, "LightGBM Feature Importance (Session Duration)", "Phase2/Plots/lgb_feature_importance_session_duration.png")

# SHAP Summary for Session Duration
plot_shap_summary(lgb_regressor, X_test_new, "Phase2/Plots/shap_summary_session_duration.png")

print("Model Training, Evaluation, and SHAP Analysis complete!")
