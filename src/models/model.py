import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG_CNN_LSTM(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG_CNN_LSTM, self).__init__()
        
        # CNN Feature Extractor
        # Input shape: (Batch, 1, 180)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Shape: (Batch, 32, 90)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Shape: (Batch, 64, 45)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Shape: (Batch, 128, 22) 
        
        # LSTM Layer
        # LSTM input expects (Batch, Sequence_length, Features)
        # So we permute the CNN output from (Batch, 128, 22) -> (Batch, 22, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully Connected Classifier
        # Bidirectional LSTM with hidden size 64 -> 128 output size
        # + 3 for RR features -> 131
        self.fc1 = nn.Linear(131, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x, rr_features=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Add channel dimension
            
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Permute for LSTM: (Batch, Features, Seq_Len) -> (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        
        # Pass through LSTM
        # lstm_out shape: (Batch, Seq_Len, Hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # We only need the output of the final time step
        # lstm_out[:, -1, :] gets the last sequence element
        last_out = lstm_out[:, -1, :]
        
        if rr_features is not None:
            last_out = torch.cat((last_out, rr_features), dim=1)
        
        out = F.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
