import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned data
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")

# Define features and target variable
target_column = "attack_detected"
X = data.drop(columns=[target_column, "Label"])  # Removing Label(not needed)
y = data[target_column]

# Split data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}\n")

evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

# Feature Importance Visualization
feature_importances_rf = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances_gb = pd.Series(gb_model.feature_importances_, index=X.columns)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
feature_importances_rf.nlargest(10).plot(kind='bar', title='Random Forest Feature Importance')
plt.subplot(1, 2, 2)
feature_importances_gb.nlargest(10).plot(kind='bar', title='Gradient Boosting Feature Importance')
plt.show()

# Residual Analysis
residuals_rf = y_test - y_pred_rf
residuals_gb = y_test - y_pred_gb

plt.figure(figsize=(12, 5))
sns.histplot(residuals_rf, kde=True, bins=30, label='Random Forest', color='blue', alpha=0.6)
sns.histplot(residuals_gb, kde=True, bins=30, label='Gradient Boosting', color='red', alpha=0.6)
plt.legend()
plt.title("Residual Distribution")
plt.show()

print("Model Training, Feature Importance Visualization, and Residual Analysis")
