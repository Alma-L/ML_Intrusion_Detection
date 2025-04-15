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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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
    sns.barplot(x=importances.nlargest(10), y=importances.nlargest(10).index, palette='viridis')
    plt.title(title, fontsize=14, pad=15, weight='bold')
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(y_test, y_pred, model_name, filename):
    """Plot residual distribution (for regression only)."""
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='teal', alpha=0.6, stat='density')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Line')
    plt.title(f"{model_name} Residual Distribution", fontsize=14, pad=15, weight='bold')
    plt.xlabel("Residuals", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_summary(model, X_test, filename):
    """Plot SHAP summary for feature impact analysis."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.title("SHAP Feature Importance", fontsize=14, pad=15, weight='bold')
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# -------------- Load & Prepare Data ---------------------
create_directory("Phase2/Plots")
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")

# ------------- Intrusion Detection (Classification) ------
target_column = "attack_detected"
X = data.drop(columns=[target_column, "Label"])
y = data[target_column]

# Print columns used for prediction
print("Columns used for intrusion detection prediction:")
print(X.columns.tolist())

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

# Print columns used for regression prediction
print("\nColumns used for session duration prediction:")
print(X_new.columns.tolist())

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

print(f"\nFeature counts:")
print(f"- Intrusion detection: {len(X.columns)} features")
print(f"- Session duration prediction: {len(X_new.columns)} features")


#-----------------------------Unsupervised Algorithms----------------------
# ---------------- Isolation Forest for Anomaly Detection -----------------

# Train only on **normal** data 
X_train_normal = X_train[y_train == 0]

# Tune contamination based on actual attack ratio
contamination_rate = (y_train == 1).sum() / len(y_train)

iso_forest = IsolationForest(
    contamination=contamination_rate,  # Approx % of anomalies
    random_state=42,
    n_estimators=100,
    max_samples='auto'
)

# Fit model on normal samples
iso_forest.fit(X_train_normal)

# Predict on test set
# -1 = anomaly, 1 = normal
anomaly_preds = iso_forest.predict(X_test)

# Map predictions to match `attack_detected` (1=attack, 0=normal)
mapped_preds = np.where(anomaly_preds == -1, 1, 0)

# Evaluate 
evaluate_model(y_test, mapped_preds, "Isolation Forest", 'classification')

# Optional: Anomaly Score Distribution
scores = iso_forest.decision_function(X_test)
plt.figure(figsize=(10, 5))
sns.histplot(scores, bins=50, kde=True)
plt.title("Isolation Forest Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.savefig("Phase2/Plots/Isolation-Forest/Isolation-Forest-Anomaly-Scores")
plt.show()

# ---------------- K-Means Clustering ------------------
# ---------------- Feature Selection ------------------
selected_features = [
    'Duration', 'SourcePort', 'DestinationPort', 'PacketCount', 'ByteCount',
    'network_packet_size', 'login_attempts', 'session_duration',
    'ip_reputation_score', 'failed_logins', 'unusual_time_access'
]

X_train_kmeans = X_train[selected_features]
X_test_kmeans = X_test[selected_features]

# ---------------- Scaling the Data ------------------
# Standardizing the features before applying KMeans
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_kmeans)
X_test_scaled = scaler.transform(X_test_kmeans)

# ---------------- KMeans Clustering ------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)

# Predict clusters on the test data
kmeans_preds = kmeans.predict(X_test_scaled)

# ---------------- Map KMeans clusters to match anomaly labels ------------------
labels = np.zeros_like(kmeans_preds)
for i in range(2):
    mask = (kmeans_preds == i)
    labels[mask] = np.bincount(mapped_preds[mask]).argmax()  # Adjusting labels as per true class distribution

# ---------------- Evaluation ------------------
silhouette_avg = silhouette_score(X_test_scaled, kmeans_preds)
print(f"\nSilhouette Score for K-Means Clustering (Selected Features): {silhouette_avg:.4f}")

# ---------------- Evaluation Function ------------------
evaluate_model(mapped_preds, labels, "K-Means", 'classification')

# ---------------- Cluster Visualization ------------------
# Replace cluster labels with descriptive names for visualization
label_names = np.array(['Normal', 'Anomaly'])
visual_labels = label_names[labels]

# session_duration vs ByteCount scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_test_kmeans['session_duration'],
    y=X_test_kmeans['ByteCount'],
    hue=visual_labels,
    palette="viridis"
)
plt.title("K-Means Clustering - session_duration vs ByteCount")
plt.xlabel("Session Duration")
plt.ylabel("Byte Count")
plt.legend(title="Traffic Type")
plt.tight_layout()
plt.savefig("Phase2/Plots/K-Means/kmeans_clusters_session_bytecount.png")
plt.show()

# ---------------- PCA for Cluster Visualization ------------------
pca = PCA(n_components=2)
# Using the already scaled test data (X_test_scaled)
reduced_data = pca.fit_transform(X_test_scaled)

# PCA scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_preds, cmap='viridis')
plt.title("K-Means Clustering Visualization (PCA)")
plt.xlabel("Traffic Type Variation Axis")
plt.ylabel("Attack vs. Normal Variance Axis")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("Phase2/Plots/K-Means/clustering-Visualization-PCA.png")


print("\nAnalysis complete!")

