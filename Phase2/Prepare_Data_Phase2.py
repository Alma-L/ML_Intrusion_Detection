# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Sklearn Modules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMClassifier, LGBMRegressor

# --------------------------------- Helper Functions --------------------------------- #

def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def evaluate_model(y_true, y_pred, model_name, problem_type='classification'):
    if problem_type == 'classification':
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)

        print(f"\n{model_name} Classification Performance:")
        print("+-------------+----------+")
        print("| Metric      | Value    |")
        print("+=============+==========+")
        print(f"| Accuracy    | {acc:.4f}  |")
        print(f"| F1-Score    | {f1:.4f}  |")
        print(f"| ROC AUC     | {roc_auc:.4f}  |")
        print(f"| Precision   | {prec:.4f}  |")
        print(f"| Recall      | {rec:.4f}  |")
        print("+-------------+----------+")

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{model_name} Confusion Matrix")
        model_dir = f"Phase2/Plots/Classification/{model_name.replace(' ', '')}"    
        create_directory(model_dir)
        plt.savefig(os.path.join(model_dir, f"cm_{model_name.replace(' ', '_')}.png"))

    else:
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
    importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances.nlargest(10), y=importances.nlargest(10).index, palette='viridis')
    plt.title(title, fontsize=14, pad=15, weight='bold')
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(y_test, y_pred, model_name, filename):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='teal', alpha=0.6, stat='density')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Line')
    plt.title(f"{model_name} Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_summary(model, X_test, filename):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------------- Load & Prepare Data --------------------------------- #
create_directory("Phase2/Plots")
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")

# --------------------------------- Classification Phase --------------------------------- #
target_column = "attack_detected"
X = data.drop(columns=[target_column, "Label"])
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
lgb_model = LGBMClassifier(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

evaluate_model(y_test, rf_model.predict(X_test), "Random Forest")
evaluate_model(y_test, gb_model.predict(X_test), "Gradient Boosting")
evaluate_model(y_test, lgb_model.predict(X_test), "LightGBM")

plot_feature_importance(rf_model, X.columns, "Random Forest Feature Importance", "Phase2/Plots/Classification/RandomForest/feature_importance_rf.png")
plot_feature_importance(gb_model, X.columns, "Gradient Boosting Feature Importance", "Phase2/Plots/Classification/GradientBoosting/feature_importance_gb.png")
plot_feature_importance(lgb_model, X.columns, "LightGBM Feature Importance", "Phase2/Plots/Classification/LightGBM/feature_importance_lgb.png")

plot_shap_summary(lgb_model, X_test, "Phase2/Plots/Classification/LightGBM/shap_summary_lgb.png")

# --------------------------------- Regression Phase --------------------------------- #
target_column = "session_duration"
X_new = data.drop(columns=[target_column, "Label", "attack_detected"])
y_new = data[target_column]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

lgb_regressor = LGBMRegressor(random_state=42)
lgb_regressor.fit(X_train_new, y_train_new)
y_pred_reg = lgb_regressor.predict(X_test_new)

evaluate_model(y_test_new, y_pred_reg, "Session Duration Prediction", 'regression')
plot_residuals(y_test_new, y_pred_reg, "Session Duration", "Phase2/Plots/Regression/LightGBM/residual_session_duration.png")
plot_feature_importance(lgb_regressor, X_new.columns, "LightGBM Feature Importance (Session Duration)", "Phase2/Plots/Regression/LightGBM/lgb_feature_importance_session_duration.png")
plot_shap_summary(lgb_regressor, X_test_new, "Phase2/Plots/Regression/LightGBM/shap_summary_session_duration.png")

# --------------------------------- Isolation Forest --------------------------------- #
X_train_normal = X_train[y_train == 0]
contamination_rate = (y_train == 1).sum() / len(y_train)

iso_forest = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)
iso_forest.fit(X_train_normal)
anomaly_preds = iso_forest.predict(X_test)
mapped_preds = np.where(anomaly_preds == -1, 1, 0)
evaluate_model(y_test, mapped_preds, "Isolation Forest")

scores = iso_forest.decision_function(X_test)
plt.figure(figsize=(10, 5))
sns.histplot(scores, bins=50, kde=True)
plt.title("Isolation Forest Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.savefig("Phase2/Plots/Isolation-Forest/Isolation-Forest-Anomaly-Scores")
plt.show()

# --------------------------------- K-Means Clustering --------------------------------- #
selected_features = [
    'Duration', 'PacketCount', 'ByteCount',
    'network_packet_size', 'login_attempts', 'session_duration',
    'ip_reputation_score', 'failed_logins', 'unusual_time_access'
]
X_train_kmeans = X_train[selected_features]
X_test_kmeans = X_test[selected_features]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_kmeans)
X_test_scaled = scaler.transform(X_test_kmeans)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
kmeans_preds = kmeans.predict(X_test_scaled)

labels = np.zeros_like(kmeans_preds)
for i in range(2):
    mask = (kmeans_preds == i)
    labels[mask] = np.bincount(mapped_preds[mask]).argmax()

silhouette_avg = silhouette_score(X_test_scaled, kmeans_preds)
print(f"\nSilhouette Score for K-Means Clustering (Selected Features): {silhouette_avg:.4f}")
evaluate_model(mapped_preds, labels, "K-Means")

visual_labels = np.array(['Normal', 'Anomaly'])[labels]
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_test_kmeans['session_duration'],
    y=X_test_kmeans['ByteCount'],
    hue=visual_labels,
    palette="viridis"
)
plt.title("K-Means Clustering - session_duration vs ByteCount")
plt.tight_layout()
plt.savefig("Phase2/Plots/K-Means/kmeans_clusters_session_bytecount.png")
plt.show()

# --------------------------------- DBSCAN Clustering --------------------------------- #
dbscan_features = [
    'ip_reputation_score', 'failed_logins', 'network_packet_size',
    'session_duration', 'ByteCount', 'unusual_time_access'
]
X_dbscan = X_test[dbscan_features].copy()
scaler_db = RobustScaler()
X_dbscan_scaled = scaler_db.fit_transform(X_dbscan)

neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(X_dbscan_scaled)
distances, _ = neighbors_fit.kneighbors(X_dbscan_scaled)
distances = np.sort(distances[:, 3])

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-Distance Graph for Epsilon Selection')
plt.tight_layout()
plt.savefig("Phase2/Plots/DBSCAN/k_distance_graph.png")
plt.close()

optimal_eps = 0.85
min_samples = 2 * X_dbscan.shape[1]

dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_dbscan_scaled)
print("\nDBSCAN cluster distribution:")
print(pd.Series(dbscan_labels).value_counts())

cluster_stats = pd.DataFrame({'cluster': dbscan_labels, 'true_label': y_test.values})\
    .groupby('cluster')['true_label'].value_counts(normalize=True)
print("\nCluster-to-label mapping statistics:")
print(cluster_stats)

normal_cluster = cluster_stats.loc[:, 0].idxmax()
mapped_dbscan_labels = np.where(dbscan_labels == normal_cluster, 0, 1)
evaluate_model(y_test, mapped_dbscan_labels, "DBSCAN")