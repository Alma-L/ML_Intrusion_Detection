import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Changed to classifiers
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from lightgbm import LGBMClassifier, LGBMRegressor
import shap
import os
import numpy as np

def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def evaluate_model(y_true, y_pred, model_name, problem_type='classification'):
    """Enhanced evaluation function with proper metrics for each problem type"""
    if problem_type == 'classification':
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        
        print(f"\n{model_name} Classification Performance:")
        print("+-------------+----------+")
        print("| Metric      | Value    |")
        print("+=============+==========+")
        print(f"| Accuracy    | {acc:.4f}  |")
        print(f"| F1-Score    | {f1:.4f}  |")
        print(f"| ROC AUC     | {roc_auc:.4f}  |")
        print("+-------------+----------+")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(f"Phase2/Plots/cm_{model_name.replace(' ', '_')}.png")
        plt.show()
        
    else:  # Regression
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{model_name} Regression Performance:")
        print("+------------+----------+")
        print("| Metric     | Value    |")
        print("+============+==========+")
        print(f"| MAE        | {mae:.4f}  |")
        print(f"| MSE        | {mse:.4f}  |")
        print(f"| RMSE       | {rmse:.4f}  |")
        print(f"| R²         | {r2:.4f}  |")
        print("+------------+----------+")

def plot_feature_importance(model, feature_names, title, filename):
    """Plot and save feature importance."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    
    plt.figure(figsize=(12, 6))
    importances.nlargest(10).plot(kind='bar', title=title)
    plt.savefig(filename)
    plt.show()

def plot_residuals(y_test, y_pred, model_name, filename):
    """Plot residual distribution (for regression only)."""
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
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# -------------- Load & Prepare Data ---------------------
create_directory("Phase2/Plots")
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")

# ------------- Intrusion Detection (Classification) ------
target_column = "attack_detected"
X = data.drop(columns=[target_column, "Label"])
y = data[target_column]

# Split dataset (no scaling needed for tree-based models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classification models
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
lgb_model = LGBMClassifier(random_state=42)

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Evaluate classifiers
evaluate_model(y_test, rf_model.predict(X_test), "Random Forest", 'classification')
evaluate_model(y_test, gb_model.predict(X_test), "Gradient Boosting", 'classification')
evaluate_model(y_test, lgb_model.predict(X_test), "LightGBM", 'classification')

# Feature Importance Visualization
plot_feature_importance(rf_model, X.columns, "Random Forest Feature Importance", "Phase2/Plots/feature_importance_rf.png")
plot_feature_importance(gb_model, X.columns, "Gradient Boosting Feature Importance", "Phase2/Plots/feature_importance_gb.png")
plot_feature_importance(lgb_model, X.columns, "LightGBM Feature Importance", "Phase2/Plots/feature_importance_lgb.png")

# SHAP Analysis for best classifier
plot_shap_summary(lgb_model, X_test, "Phase2/Plots/shap_summary_lgb.png")

# ------------- Session Duration Prediction (Regression) ------
target_column = "session_duration"
X_new = data.drop(columns=[target_column, "Label", "attack_detected"])
y_new = data[target_column]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

# Train and evaluate regression model
lgb_regressor = LGBMRegressor(random_state=42)
lgb_regressor.fit(X_train_new, y_train_new)
y_pred_reg = lgb_regressor.predict(X_test_new)

evaluate_model(y_test_new, y_pred_reg, "Session Duration Prediction", 'regression')
plot_residuals(y_test_new, y_pred_reg, "Session Duration", "Phase2/Plots/residual_session_duration.png")
plot_feature_importance(lgb_regressor, X_new.columns, 
    "LightGBM Feature Importance (Session Duration)", 
    "Phase2/Plots/lgb_feature_importance_session_duration.png"
)
plot_shap_summary(lgb_regressor, X_test_new, "Phase2/Plots/shap_summary_session_duration.png")

print("Analysis complete!")
