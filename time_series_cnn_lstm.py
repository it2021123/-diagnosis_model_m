import os
import re
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Φορτώνει τα δεδομένα από ένα αρχείο CSV
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_windows(data, window_size=10):
    """
    Δημιουργεί χρονικά παράθυρα από τα δεδομένα
    """
    windows = [data.iloc[i:i + window_size] for i in range(0, len(data), window_size)]
    return windows

def extract_features_from_window(window):
    """
    Εξάγει χαρακτηριστικά από κάθε παράθυρο
    """
    features = {
        'mean_flexion_knee': window['flexion_extension_knee'].mean(),
        'mean_flexion_hip': window['flexion_extension_hip'].mean(),
        'mean_flexion_shoulder': window['flexion_extension_shoulder'].mean(),
        'std_flexion_knee': window['flexion_extension_knee'].std(),
        'std_flexion_hip': window['flexion_extension_hip'].std(),
        'std_flexion_shoulder': window['flexion_extension_shoulder'].std(),
        'rate_of_change_flexion_knee': window['flexion_extension_knee'].diff().sum(),
        'rate_of_change_flexion_hip': window['flexion_extension_hip'].diff().sum(),
        'rate_of_change_flexion_shoulder': window['flexion_extension_shoulder'].diff().sum(),
        'angle_difference_flexion_knee': window['flexion_extension_knee'].iloc[-1] - window['flexion_extension_knee'].iloc[0],
        'angle_difference_flexion_hip': window['flexion_extension_hip'].iloc[-1] - window['flexion_extension_hip'].iloc[0],
        'angle_difference_flexion_shoulder': window['flexion_extension_shoulder'].iloc[-1] - window['flexion_extension_shoulder'].iloc[0]
    }
    return features

# Δημιουργία του LSTM-CNN μοντέλου
def build_lstm_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # 3 classes: KOA, NM, PD
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Αξιολόγηση του μοντέλου
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

# Κύρια διαδικασία επεξεργασίας και εκπαίδευσης
def process_files_nm(root_folders):
    for root_folder in root_folders:
        if not root_folder.exists():
            print(f"Folder not found: {root_folder}")
            continue  # Skip folder if not found
        csv_files = list(root_folder.rglob("*.csv"))

        for csv_file in csv_files:
            print(f"Found CSV file: {csv_file}")
            filename = os.path.basename(csv_file)
            
            # Εύρεση ID και target στο όνομα αρχείου
            match = re.search(r"(\d{3})_(\w+)_(\d{1})_(\d{2})", filename)

            if match:
                part1 = match.group(1)  # "001"
                part2 = match.group(2)  # "PD"
                part3 = match.group(3)  # "02"
                target = part2  # KOA, NM, PD
            else:
                print(f"Invalid filename format for {filename}")
                continue  # Skip file if format is incorrect
            
            # Φόρτωση δεδομένων
            data = load_data(csv_file)
            if data is None:
                continue
            
            if 'DISEASE' not in data.columns:
                print(f"Column 'DISEASE' not found in {filename}")
                continue

            # Χώρισε τα δεδομένα σε παράθυρα
            window_size = 10
            windows = create_windows(data, window_size)

            # Εξαγωγή χαρακτηριστικών
            series_list = []
            for window in windows:
                features = extract_features_from_window(window)
                series_list.append(features)
            
            # Δημιουργία DataFrame για χαρακτηριστικά
            columns = [f'feature_{i}' for i in range(len(features))]
            series_df = pd.DataFrame(series_list, columns=columns)
            print(series_df)

            # Λάβε τις ετικέτες (labels) από τη στήλη DISEASE
            labels = data['DISEASE'].iloc[::window_size][:len(series_df)]  # Χρησιμοποιούμε το DISEASE ως στόχο

            # Κωδικοποίηση των ετικετών
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
"""
            # Εκτέλεση LO-SO-CV ή LAGO-CV
            group_kfold = GroupKFold(n_splits=5)
            for train_idx, test_idx in group_kfold.split(series_df, labels_encoded, groups=data['Subject']):
                X_train, X_test = series_df.iloc[train_idx], series_df.iloc[test_idx]
                y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]

                # Αναμόρφωση των δεδομένων για το LSTM
                X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

                # Δημιουργία του LSTM-CNN μοντέλου
                model = build_lstm_cnn_model(input_shape=(X_train.shape[1], 1))

                # Εκπαίδευση του μοντέλου
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

                # Αξιολόγηση του μοντέλου
                accuracy = evaluate_model(model, X_test, y_test)
                print(f"Model accuracy: {accuracy * 100:.2f}%")
"""



def process_files_koa_pd(root_folders):
    """
    Διαβάζει και επεξεργάζεται τα αρχεία .csv σε έναν ή περισσότερους φακέλους
    """
    for root_folder in root_folders:
        if not root_folder.exists():
            print(f"Folder not found: {root_folder}")
            continue  # Αν η διαδρομή δεν υπάρχει, παραλείπει τον φάκελο
        csv_files = list(root_folder.rglob("*.csv"))

        for csv_file in csv_files:
            print(f"Found CSV file: {csv_file}")
            filename = os.path.basename(csv_file)
            print(filename)
            
            # Εύρεση ID στο όνομα αρχείου
            match = re.search(r"(\d{3})(\w+)_(\w+)_(\d{2})", filename)

            if match:
                part1 = match.group(1)  # Αποθηκεύει τα πρώτα τρία ψηφία, π.χ., "001"
                part2 = match.group(2)  # Αποθηκεύει τον δεύτερο τμήμα, π.χ., "PD"
                part3 = match.group(3)  # Αποθηκεύει το τρίτο τμήμα, π.χ., "02"
                print("First part:", part1)
                print("Second part:", part2)
                print("Third part:", part3)
            else:
                print(f"Invalid filename format for {filename}")
                continue  # Αγνόηση αρχείου αν δεν ταιριάζει το πρότυπο
            
            # Φόρτωση δεδομένων από .csv
            data = load_data(csv_file)
            if data is None:
                continue
            
            # Έλεγχος αν η στήλη `Disease` υπάρχει στο DataFrame
            if 'Disease' not in data.columns:
                print(f"Column 'Disease' not found in {filename}")
                continue
            
            # Χώρισε τα δεδομένα σε παράθυρα
            window_size = 10
            windows = create_windows(data, window_size)

            # Εξαγωγή χαρακτηριστικών από τα παράθυρα
            series_list = []
            for window in windows:
                features = extract_features_from_window(window)
                series_list.append(features)
            
            # Δημιουργία DataFrame για τα χαρακτηριστικά
            series_df = pd.DataFrame(series_list)
            print(series_df)

            # Λάβε τις ετικέτες (labels) από τη στήλη DISEASE και χωρίσε τις ανά παράθυρο
            labels = data['Disease'].iloc[::window_size][:len(series_df)]  # Χρησιμοποιούμε το DISEASE ως στόχο

            # Κωδικοποίηση των ετικετών
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)

"""
            # Εκτέλεση LO-SO-CV ή LAGO-CV
            group_kfold = GroupKFold(n_splits=5)
            for train_idx, test_idx in group_kfold.split(series_df, labels_encoded, groups=data['Subject']):
                X_train, X_test = series_df.iloc[train_idx], series_df.iloc[test_idx]
                y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]

                # Αναμόρφωση των δεδομένων για το LSTM
                X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

                # Δημιουργία του LSTM-CNN μοντέλου
                model = build_lstm_cnn_model(input_shape=(X_train.shape[1], 1))

                # Εκπαίδευση του μοντέλου
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

                # Αξιολόγηση του μοντέλου
                accuracy = evaluate_model(model, X_test, y_test)
                print(f"Model accuracy: {accuracy * 100:.2f}%")
                """


# Ορισμός φακέλων
root_folders =  Path("/home/poulimenos/project/Updated_NM_CSVs/"), 
root_folders1= [ Path("/home/poulimenos/project/KOA_EL_CSV/"), 
  Path("/home/poulimenos/project/KOA_MD_CSV/"), 
  Path("/home/poulimenos/project/KOA_SV_CSV/"), 
  Path("/home/poulimenos/project/PD/")   ]


# Επεξεργασία αρχείων
process_files_nm(root_folders)
process_files_koa_pd(root_folders1)
