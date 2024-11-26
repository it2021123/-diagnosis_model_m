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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================
# 1. Φόρτωση και προεπεξεργασία δεδομένων
# ============================================

# Φόρτωση του dataset
df_nm = pd.read_csv("/home/poulimenos/project/nm_features.csv")
df_nm_s1 = pd.read_csv("/home/poulimenos/project/synthetic_data.csv")
df_nm_s2 = pd.read_csv("/home/poulimenos/project/synthetic_data1.csv")
df_pd = pd.read_csv("/home/poulimenos/project/pd_features.csv")
df_koa  = pd.read_csv("/home/poulimenos/project/koa_features.csv")

# Αφαίρεση εγγραφών με ελλιπή δεδομένα
df_nm = df_nm.dropna()
df_nm = df_nm.drop(columns=['ID'])
df_nm_s1 = df_nm_s1.dropna()
df_nm_s2 = df_nm_s2.dropna()
df_koa = df_koa.dropna()
df_koa = df_koa.drop(columns=['ID'])
df_pd=df_pd.dropna()
df_pd = df_pd.drop(columns=['ID'])

# Επιλογή τυχαίων 1000 μοναδικών γραμμών
#new_df_koa = df_koa.sample(n=1000, replace=False, random_state=42)

# Συνδυασμός (κατακόρυφος)
df = pd.concat([df_nm, df_nm_s1,df_nm_s2,df_koa,df_pd], axis=0, ignore_index=True)

# Μετατροπή των κειμένων της στήλης 'Disease' σε αριθμητικούς στόχους
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['Disease'])

# Ανακάτεμα γραμμών
shuffled_df = df.sample(frac=1, random_state=42)  # Ορισμός `random_state` για επαναληψιμότητα (προαιρετικό)


# Διαχωρισμός δεδομένων σε χαρακτηριστικά (X) και στόχους (y)
# Καθορίζουμε τα ονόματα των στηλών που θέλουμε να επιλέξουμε
columns_to_select = [
    'emd_mean_right_rotation_shoulder', 'emd_std_right_rotation_shoulder', 'emd_energy_right_rotation_shoulder',
    'emd_mean_right_rotation_hip', 'emd_std_right_rotation_hip', 'emd_energy_right_rotation_hip',
    'emd_mean_right_rotation_knee', 'emd_std_right_rotation_knee', 'emd_energy_right_rotation_knee',
    'emd_mean_right_abduction_adduction_shoulder', 'emd_std_right_abduction_adduction_shoulder', 'emd_energy_right_abduction_adduction_shoulder',
    'emd_mean_right_abduction_adduction_hip', 'emd_std_right_abduction_adduction_hip', 'emd_energy_right_abduction_adduction_hip',
    'emd_mean_right_abduction_adduction_knee', 'emd_std_right_abduction_adduction_knee', 'emd_energy_right_abduction_adduction_knee',
    'emd_mean_right_flexion_extension_shoulder', 'emd_std_right_flexion_extension_shoulder', 'emd_energy_right_flexion_extension_shoulder',
    'emd_mean_right_flexion_extension_hip', 'emd_std_right_flexion_extension_hip', 'emd_energy_right_flexion_extension_hip',
    'emd_mean_right_flexion_extension_knee', 'emd_std_right_flexion_extension_knee', 'emd_energy_right_flexion_extension_knee',
    
    'emd_mean_left_rotation_shoulder', 'emd_std_left_rotation_shoulder', 'emd_energy_left_rotation_shoulder',
    'emd_mean_left_rotation_hip', 'emd_std_left_rotation_hip', 'emd_energy_left_rotation_hip',
    'emd_mean_left_rotation_knee', 'emd_std_left_rotation_knee', 'emd_energy_left_rotation_knee',
    'emd_mean_left_abduction_adduction_shoulder', 'emd_std_left_abduction_adduction_shoulder', 'emd_energy_left_abduction_adduction_shoulder',
    'emd_mean_left_abduction_adduction_hip', 'emd_std_left_abduction_adduction_hip', 'emd_energy_left_abduction_adduction_hip',
    'emd_mean_left_abduction_adduction_knee', 'emd_std_left_abduction_adduction_knee', 'emd_energy_left_abduction_adduction_knee',
    'emd_mean_left_flexion_extension_shoulder', 'emd_std_left_flexion_extension_shoulder', 'emd_energy_left_flexion_extension_shoulder',
    'emd_mean_left_flexion_extension_hip', 'emd_std_left_flexion_extension_hip', 'emd_energy_left_flexion_extension_hip',
    'emd_mean_left_flexion_extension_knee', 'emd_std_right_flexion_extension_knee', 'emd_energy_left_flexion_extension_knee',
    
    'RIGHT_CLOSED_TO_CAMERA', 'LEFT_CLOSED_TO_CAMERA'
]

# Επιλέγουμε τις στήλες από το DataFrame
X = df[columns_to_select]

y = df['target']

scaler=StandardScaler()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Αναδιάταξη των δεδομένων για να έχουν μορφή 3D (απαραίτητο για CNN-LSTM)
X_array = X.values  
# Μετατροπή του DataFrame σε NumPy array
X_reshaped = X_array.reshape(X_array.shape[0], 1, X_array.shape[1])


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

    # Διαχωρισμός δεδομένων για το fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Κανονικοποίηση χαρακτηριστικών
    scaler = StandardScaler()
    X_train_fold_2d = scaler.fit_transform(X_train_fold)  # Από 2D
    X_val_fold_2d = scaler.transform(X_val_fold)

    # Reshape για CNN-LSTM
    X_train_fold = X_train_fold_2d.reshape(X_train_fold_2d.shape[0], 1, X_train_fold_2d.shape[1])
    X_val_fold = X_val_fold_2d.reshape(X_val_fold_2d.shape[0], 1, X_val_fold_2d.shape[1])

    # Μετατροπή δεδομένων σε Tensors
    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32)
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32)
    y_train_fold = torch.tensor(y_train_fold.values, dtype=torch.long)
    y_val_fold = torch.tensor(y_val_fold.values, dtype=torch.long)

    # Δημιουργία DataLoader
    train_loader_fold = DataLoader(TimeSeriesDataset(X_train_fold, y_train_fold), batch_size=128, shuffle=True)
    val_loader_fold = DataLoader(TimeSeriesDataset(X_val_fold, y_val_fold), batch_size=128, shuffle=False)

    # Ορισμός του μοντέλου και των παραμέτρων
    model = CNNLSTM(input_dim=X_train_fold.shape[2], hidden_dim=32, lstm_layers=2, output_dim=len(np.unique(y)))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Εκπαίδευση
    for epoch in range(16):  #  εποχές
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


