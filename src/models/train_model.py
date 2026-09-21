import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import sys
import torch.nn.functional as F

# Import model
from model import ECG_CNN_LSTM

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Alpha should be a tensor of weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

PROCESSED_DATA_DIR = 'data/processed'
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        RR_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'RR_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
        RR_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'RR_test.npy'))
    except FileNotFoundError:
        print("Data files not found. Please run make_dataset.py first.")
        sys.exit(1)

    # Normalize data (simple standardization)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Normalize RR features
    rr_mean = np.mean(RR_train, axis=0)
    rr_std = np.std(RR_train, axis=0) + 1e-8 # prevent division by zero
    RR_train = (RR_train - rr_mean) / rr_std
    RR_test = (RR_test - rr_mean) / rr_std

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    RR_train_t = torch.tensor(RR_train, dtype=torch.float32)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    RR_test_t = torch.tensor(RR_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, RR_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, RR_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model, Loss, Optimizer
    model = ECG_CNN_LSTM(num_classes=5).to(device)
    
    # Use "Soft" Class Weights: Square root scaling prevents extreme weight disparities
    counts = np.bincount(y_train)
    soft_weights = np.sqrt(np.max(counts) / counts)
    class_weights_tensor = torch.tensor(soft_weights, dtype=torch.float32).to(device)
    
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, rr_features, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            rr_features = rr_features.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, rr_features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    # 4. Evaluation
    print("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, rr_features, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            rr_features = rr_features.to(device)
            outputs = model(inputs, rr_features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Print metrics
    target_names = ['N', 'S', 'V', 'F', 'Q']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    main()
