#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:36:02 2024

@author: poulimenos
"""
import pandas as pd
import numpy as np

import pandas as pd

def swap(df,f1,f2):
 # Ανάθεση των ονομάτων των στηλών σε μια λίστα
 columns = df.columns.tolist()

 # Ανάθεση των στηλών που θες να ανταλλάξεις
 col1, col2 = f1, f2  # Παράδειγμα στηλών που θες να ανταλλάξεις

 # Εύρεση των index των στηλών που θες να ανταλλάξεις
 index1, index2 = columns.index(col1), columns.index(col2)

 # Ανταλλαγή των στηλών στο DataFrame
 columns[index1], columns[index2] = columns[index2], columns[index1]

 # Δημιουργία ενός νέου DataFrame με την ανταλλαγή
 df_swapped = df[columns]
 return df_swapped

# Φόρτωση δεδομένων
df = pd.read_csv("/home/poulimenos/project/nm_features.csv")

# Εξαγωγή χαρακτηριστικών και κανονικοποίηση στο [-1, 1]
features = df.drop(columns=["Disease", "ID"])
features=features.dropna()
print(features.head())
# Προσθήκη θορύβου στα δεδομένα (data augmentation)
noise_factor = 0.01  # Παράγοντας θορύβου, μπορείς να το τροποποιήσεις ανάλογα με τις ανάγκες σου
noise = np.random.normal(0, noise_factor, features.shape)  # Δημιουργία θορύβου
print(noise)
augmented_features = features + noise  # Προσθήκη θορύβου στα χαρακτηριστικά
print(augmented_features)
data_dim = augmented_features.shape[1]  # Διαστάσεις χαρακτηριστικών
print(data_dim)

#μορφοποίηση του dataset 
# Μετατροπή float τιμών σε ακέραιους (0 ή 1)
augmented_features["RIGHT_CLOSED_TO_CAMERA"] = (augmented_features["RIGHT_CLOSED_TO_CAMERA"] >= 0.5).astype(int)  # 0 αν < 0.5, αλλιώς 1
augmented_features["LEFT_CLOSED_TO_CAMERA"] = (augmented_features["LEFT_CLOSED_TO_CAMERA"] >= 0.5).astype(int)
augmented_features['Disease']= 'NM'
# Ανταλλαγή (swap) των στηλών 

augmented_features = swap(augmented_features,"RIGHT_CLOSED_TO_CAMERA","Disease")
print(augmented_features.head())
augmented_features = swap(augmented_features,"LEFT_CLOSED_TO_CAMERA","RIGHT_CLOSED_TO_CAMERA")

# Αποθήκευση των συνθετικών δεδομένων με θόρυβο
augmented_features.to_csv("synthetic_data1.csv", index=False)
print("Συνθετικά δεδομένα αποθηκεύτηκαν στο synthetic_data.csv")



