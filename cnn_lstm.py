#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:54:53 2024

@author: poulimenos
"""

# ============================================
# Βιβλιοθήκες
# ============================================
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================
# 1. Φόρτωση και προεπεξεργασία δεδομένων
# ============================================

# Φόρτωση του dataset
df = pd.read_csv("/home/poulimenos/project/features.csv")

# Αφαίρεση εγγραφών με ελλιπή δεδομένα
df = df.dropna()

# Μετατροπή των κειμένων της στήλης 'disease' σε αριθμητικούς στόχους
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['disease'])

# Διαχωρισμός δεδομένων σε χαρακτηριστικά (X) και στόχους (y)
X = df.drop(columns=['koa', 'nm', 'pd', 'id', 'disease', 'target'], errors='ignore')
y = df['target']

# Κανονικοποίηση (Scaling) των χαρακτηριστικών για να βρίσκονται σε παρόμοια κλίμακα
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Αναδιάταξη των δεδομένων για να έχουν μορφή 3D (απαραίτητο για CNN-LSTM)
X = X.reshape(X.shape[0], 1, X.shape[1])

# ============================================
# 2. Δημιουργία Dataset και DataLoader
# ============================================
class TimeSeriesDataset(Dataset):
    """Dataset για διαχείριση των χαρακτηριστικών και στόχων."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================
# 3. Δημιουργία CNN-LSTM Μοντέλου
# ============================================
class CNNLSTM(nn.Module):
    """Το μοντέλο CNN-LSTM για επεξεργασία δεδομένων χρονικής σειράς."""
    def __init__(self, input_dim, hidden_dim, lstm_layers, output_dim):
        super(CNNLSTM, self).__init__()
        # CNN για εξαγωγή χαρακτηριστικών
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # LSTM για εκμάθηση σχέσεων χρονικής αλληλουχίας
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        # Fully connected layer για ταξινόμηση
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.cnn(x)  # Εφαρμογή CNN
        x = x.permute(0, 2, 1)  # Αναδιάταξη για LSTM (batch, seq_len, features)
        x, _ = self.lstm(x)  # Εφαρμογή LSTM
        x = x[:, -1, :]  # Επιλέγουμε την τελευταία έξοδο της LSTM
        x = self.fc(x)  # Τελικό επίπεδο ταξινόμησης
        return x

# ============================================
# 4. Εκπαίδευση με Stratified K-Fold Cross-Validation
# ============================================
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Λίστες για αποθήκευση μετρικών
fold_accuracies, fold_precisions, fold_recalls, fold_f1_scores = [], [], [], []

# Βρόχος εκπαίδευσης για κάθε fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}/{k_folds}")

    # Διαχωρισμός δεδομένων
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Επαναφορά scaling μετά την κανονικοποίηση
    X_train_fold_2d = X_train_fold.reshape(X_train_fold.shape[0], X_train_fold.shape[2])
    X_val_fold_2d = X_val_fold.reshape(X_val_fold.shape[0], X_val_fold.shape[2])

    X_train_fold_2d = scaler.fit_transform(X_train_fold_2d)
    X_val_fold_2d = scaler.transform(X_val_fold_2d)

    X_train_fold = X_train_fold_2d.reshape(X_train_fold.shape[0], 1, X_train_fold_2d.shape[1])
    X_val_fold = X_val_fold_2d.reshape(X_val_fold.shape[0], 1, X_val_fold_2d.shape[1])

    # Μετατροπή δεδομένων σε Tensors
    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32)
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32)
    y_train_fold = torch.tensor(y_train_fold.values, dtype=torch.long)
    y_val_fold = torch.tensor(y_val_fold.values, dtype=torch.long)

    # Δημιουργία DataLoader
    train_loader_fold = DataLoader(TimeSeriesDataset(X_train_fold, y_train_fold), batch_size=32, shuffle=True)
    val_loader_fold = DataLoader(TimeSeriesDataset(X_val_fold, y_val_fold), batch_size=32, shuffle=False)

    # Ορισμός του μοντέλου και των παραμέτρων
    model = CNNLSTM(input_dim=X_train_fold.shape[2], hidden_dim=32, lstm_layers=2, output_dim=len(np.unique(y)))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Εκπαίδευση
    for epoch in range(10):  # 10 εποχές
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader_fold:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Αξιολόγηση
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader_fold:
            preds = model(X_batch)
            y_pred.extend(torch.argmax(preds, axis=1).tolist())
            y_true.extend(y_batch.tolist())

    # Μετρικές
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    fold_accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1_scores.append(f1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix Fold {fold + 1}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Μέσες τιμές μετρικών
print(f"Average Accuracy: {np.mean(fold_accuracies):.2f}")
print(f"Average Precision: {np.mean(fold_precisions):.2f}")
print(f"Average Recall: {np.mean(fold_recalls):.2f}")
print(f"Average F1-Score: {np.mean(fold_f1_scores):.2f}")
