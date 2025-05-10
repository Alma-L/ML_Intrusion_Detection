import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, 
                                   Conv1D, MaxPooling1D, 
                                   GlobalMaxPooling1D, LSTM)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_directory("outputs")

# ------------------ Load Data ------------------ #
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")
target = "attack_detected"

X = data.drop(columns=[target, "Label"], errors="ignore")
y = data[target]

# ------------------ Data Preparation ------------------ #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling for SVM and Neural Networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNNs/LSTMs
X_train_3d = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_3d = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# ================== LightGBM ================== #
print("\nTraining LightGBM...")
lgbm_params = {
    'num_leaves': [51, 70],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.1, 0.05],
    'n_estimators': [100, 200]
}

lgbm_grid = GridSearchCV(
    LGBMClassifier(random_state=42),
    param_grid=lgbm_params,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)
lgbm_grid.fit(X_train, y_train)
best_lgbm = lgbm_grid.best_estimator_
y_pred_lgbm = best_lgbm.predict(X_test)

print(f"\nLightGBM Accuracy: {accuracy_score(y_test, y_pred_lgbm):.4f}")
print(classification_report(y_test, y_pred_lgbm))
best_lgbm.booster_.save_model("outputs/lgb_model.txt")

# ================== SVM ================== #
print("\nTraining SVM...")
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid=svm_params,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)

print(f"\nSVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm))

# ================== CNN ================== #
print("\nTraining CNN...")
model_cnn = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train_3d.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_3d, y_train, epochs=15, batch_size=128, validation_split=0.1, verbose=1)
y_pred_cnn = (model_cnn.predict(X_test_3d) > 0.5).astype(int)

print(f"\nCNN Accuracy: {accuracy_score(y_test, y_pred_cnn):.4f}")
print(classification_report(y_test, y_pred_cnn))
model_cnn.save("outputs/cnn_model.h5")

# ================== LSTM ================== #
print("\nTraining LSTM...")
model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_3d.shape[1], 1)),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train_3d, y_train, epochs=15, batch_size=128, validation_split=0.1, verbose=1)
y_pred_lstm = (model_lstm.predict(X_test_3d) > 0.5).astype(int)

print(f"\nLSTM Accuracy: {accuracy_score(y_test, y_pred_lstm):.4f}")
print(classification_report(y_test, y_pred_lstm))
model_lstm.save("outputs/lstm_model.h5")

# ================== Autoencoder ================== #
print("\nTraining Autoencoder...")
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_scaled[y_train==0], X_train_scaled[y_train==0], 
              epochs=50, batch_size=128, validation_split=0.1, verbose=1)

# Calculate threshold
reconstructions = autoencoder.predict(X_train_scaled[y_train==0])
mse = np.mean(np.square(X_train_scaled[y_train==0] - reconstructions), axis=1)
threshold = np.quantile(mse, 0.95)

# Evaluate
test_recon = autoencoder.predict(X_test_scaled)
test_mse = np.mean(np.square(X_test_scaled - test_recon), axis=1)
y_pred_ae = (test_mse > threshold).astype(int)

print(f"\nAutoencoder Accuracy: {accuracy_score(y_test, y_pred_ae):.4f}")
print(classification_report(y_test, y_pred_ae))
autoencoder.save("outputs/autoencoder.h5")

# ================== Model Comparison ================== #
results = pd.DataFrame({
    'Model': ['LightGBM', 'SVM', 'CNN', 'LSTM', 'Autoencoder'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lgbm),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_cnn),
        accuracy_score(y_test, y_pred_lstm),
        accuracy_score(y_test, y_pred_ae)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lgbm),
        f1_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_cnn),
        f1_score(y_test, y_pred_lstm),
        f1_score(y_test, y_pred_ae)
    ],
    'ROC AUC': [
        roc_auc_score(y_test, best_lgbm.predict_proba(X_test)[:, 1]),
        roc_auc_score(y_test, best_svm.predict_proba(X_test_scaled)[:, 1]),
        roc_auc_score(y_test, model_cnn.predict(X_test_3d)),
        roc_auc_score(y_test, model_lstm.predict(X_test_3d)),
        roc_auc_score(y_test, test_mse)  # Using MSE as anomaly score
    ]
})

print("\n================== Final Comparison ==================")
print(results.sort_values('Accuracy', ascending=False))

# Save comparison results
results.to_csv("outputs/model_comparison.csv", index=False)
print("\nAll models trained and evaluated. Results saved in outputs/ directory.")