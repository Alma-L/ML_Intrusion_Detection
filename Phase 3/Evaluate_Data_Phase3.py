import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_directory("outputs")

# Load Data
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")
target = "attack_detected"

X = data.drop(columns=[target, "Label"], errors="ignore")
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_3d = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_3d = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

# For Torch
X_train_tensor = torch.FloatTensor(X_train_3d).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test_3d).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1).to(device)

# For Autoencoder
X_train_ae = torch.FloatTensor(X_train_scaled).to(device)
X_test_ae = torch.FloatTensor(X_test_scaled).to(device)

# Validation split for deep learning models
val_split = 0.1
val_size = int(len(X_train_tensor) * val_split)
train_tensor = TensorDataset(X_train_tensor[:-val_size], y_train_tensor[:-val_size])
val_tensor = TensorDataset(X_train_tensor[-val_size:], y_train_tensor[-val_size:])

train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=64)

# ========== SVM ==========
try:
    print("\nTraining SVM (subset + GridSearch)...")
    X_svm, y_svm = resample(X_train_scaled, y_train, n_samples=5000, stratify=y_train, random_state=42)
    svm_params = {'C': [1], 'kernel': ['rbf'], 'gamma': ['scale']}
    svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=3, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_svm, y_svm)
    best_svm = svm_grid.best_estimator_
    y_pred_svm = best_svm.predict(X_test_scaled)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))
    
    # Save the SVM model and report
    joblib.dump(best_svm, "outputs/svm_model.pkl")
    with open("outputs/svm_model_details.txt", "w") as f:
        f.write(f"SVM Model Configuration:\n")
        f.write(f"Best parameters: {svm_grid.best_params_}\n")
        f.write(f"Best score: {svm_grid.best_score_}\n")
        f.write(f"Number of support vectors: {best_svm.n_support_}\n")
        f.write(f"Classes: {best_svm.classes_}\n")
        f.write(f"Intercept: {best_svm.intercept_}\n")
        f.write(f"Gamma: {best_svm.gamma}\n")
    
    report_svm = classification_report(y_test, y_pred_svm)
    with open("outputs/svm_report.txt", "w") as f:
        f.write(report_svm)

except Exception as e:
    print("SVM Error:", e)
    import traceback
    traceback.print_exc()

# ========== CNN ==========
try:
    print("\nTraining CNN...")
    class CNN(nn.Module):
        def __init__(self, input_shape):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, 3)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, 3)
            self.gpool = nn.AdaptiveMaxPool1d(1)
            self.fc1 = nn.Linear(64, 32)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.gpool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return torch.sigmoid(self.fc2(x))

    model_cnn = CNN(X_train_3d.shape[1:]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_cnn.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    best_val_loss = float('inf')
    patience = 3
    counter = 0
    min_delta = 1e-4

    for epoch in range(20):
        model_cnn.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model_cnn(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        model_cnn.eval()
        with torch.no_grad():
            val_loss = 0
            for x_val, y_val in val_loader:
                val_loss += criterion(model_cnn(x_val.to(device)), y_val.to(device)).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopped CNN at epoch", epoch)
                    break

    model_cnn.eval()
    with torch.no_grad():
        y_pred_cnn = (model_cnn(X_test_tensor) > 0.5).float().cpu().numpy()
    print("CNN Accuracy:", accuracy_score(y_test, y_pred_cnn))
    print(classification_report(y_test, y_pred_cnn))
    
    # Save the CNN model details to a text file
    with open("outputs/cnn_model_details.txt", "w") as f:
        f.write("CNN Model Architecture:\n")
        f.write(str(model_cnn) + "\n\n")
        f.write("Model State Dictionary:\n")
        for name, param in model_cnn.state_dict().items():
            f.write(f"{name}: {param.size()}\n")
            if param.numel() < 10:  # Only write small tensors completely
                f.write(f"{param}\n")
            else:
                f.write(f"[Tensor of size {tuple(param.size())} - too large to display]\n")
    
    # Save the model weights
    torch.save(model_cnn.state_dict(), "outputs/cnn_model.pth")

except Exception as e:
    print("CNN Error:", e)
    import traceback
    traceback.print_exc()

# ========== LSTM ==========
try:
    print("\nTraining LSTM...")
    class LSTMNet(nn.Module):
        def __init__(self, input_shape):
            super(LSTMNet, self).__init__()
            self.lstm = nn.LSTM(input_shape[1], 32, batch_first=True)
            self.fc1 = nn.Linear(32, 16)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return torch.sigmoid(self.fc2(x))

    model_lstm = LSTMNet(X_train_3d.shape[1:]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_lstm.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    best_val_loss = float('inf')
    patience = 3
    counter = 0
    min_delta = 1e-4

    for epoch in range(20):
        model_lstm.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model_lstm(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        model_lstm.eval()
        with torch.no_grad():
            val_loss = 0
            for x_val, y_val in val_loader:
                val_loss += criterion(model_lstm(x_val.to(device)), y_val.to(device)).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopped LSTM at epoch", epoch)
                    break

    model_lstm.eval()
    with torch.no_grad():
        y_pred_lstm = (model_lstm(X_test_tensor) > 0.5).float().cpu().numpy()
    print("LSTM Accuracy:", accuracy_score(y_test, y_pred_lstm))
    print(classification_report(y_test, y_pred_lstm))
    
    # Save the LSTM model details to a text file
    with open("outputs/lstm_model_details.txt", "w") as f:
        f.write("LSTM Model Architecture:\n")
        f.write(str(model_lstm) + "\n\n")
        f.write("Model State Dictionary:\n")
        for name, param in model_lstm.state_dict().items():
            f.write(f"{name}: {param.size()}\n")
            if param.numel() < 10:  # Only write small tensors completely
                f.write(f"{param}\n")
            else:
                f.write(f"[Tensor of size {tuple(param.size())} - too large to display]\n")
    
    # Save the model weights
    torch.save(model_lstm.state_dict(), "outputs/lstm_model.pth")

except Exception as e:
    print("LSTM Error:", e)
    import traceback
    traceback.print_exc()

# ========== Autoencoder ==========
try:
    print("\nTraining Autoencoder...")
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    ae_model = Autoencoder(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae_model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Validation split for Autoencoder
    val_split = 0.1
    val_size = int(len(X_train_ae) * val_split)
    train_ae_dataset = TensorDataset(X_train_ae[:-val_size])
    val_ae_dataset = TensorDataset(X_train_ae[-val_size:])
    train_ae_loader = DataLoader(train_ae_dataset, batch_size=64, shuffle=True)
    val_ae_loader = DataLoader(val_ae_dataset, batch_size=64)

    train_losses = []
    val_losses = []

    for epoch in range(50):
        ae_model.train()
        train_loss = 0
        for x_batch in train_ae_loader:
            x_batch = x_batch[0].to(device)
            optimizer.zero_grad()
            output = ae_model(x_batch)
            loss = criterion(output, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_ae_loader)
        train_losses.append(train_loss)

        ae_model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_batch in val_ae_loader:
                x_batch = x_batch[0].to(device)
                output = ae_model(x_batch)
                val_loss += criterion(output, x_batch).item()
            val_loss /= len(val_ae_loader)
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()

    ae_model.eval()
    with torch.no_grad():
        X_test_reconstructed = ae_model(X_test_ae)
    print("Autoencoder Reconstruction Error:", np.mean((X_test_ae.cpu().numpy() - X_test_reconstructed.cpu().numpy())**2))

    # Save the Autoencoder model details to a text file
    with open("outputs/autoencoder_model_details.txt", "w") as f:
        f.write("Autoencoder Model Architecture:\n")
        f.write(str(ae_model) + "\n\n")
        f.write("Model State Dictionary:\n")
        for name, param in ae_model.state_dict().items():
            f.write(f"{name}: {param.size()}\n")
            if param.numel() < 10:  # Only write small tensors completely
                f.write(f"{param}\n")
            else:
                f.write(f"[Tensor of size {tuple(param.size())} - too large to display]\n")
        f.write("\nTraining Metrics:\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
    
    # Save the model weights
    torch.save(ae_model.state_dict(), "outputs/autoencoder_model.pth")

    # Plotting Training and Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/autoencoder_loss.png')
    plt.close()

except Exception as e:
    print("Autoencoder Error:", e)
    import traceback
    traceback.print_exc()