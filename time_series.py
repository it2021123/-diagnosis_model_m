import os
import re
from pathlib import Path
import pandas as pd
from PyEMD import EMD
import numpy as np


def load_data(file_path):
    """
    Φορτώνει δεδομένα από ένα αρχείο CSV.
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
    Δημιουργεί χρονικά παράθυρα από τα δεδομένα.
    """
    return [data.iloc[i:i + window_size] for i in range(0, len(data), window_size)]


def extract_features_from_window(window, i):
    """
    Εξάγει χαρακτηριστικά από ένα χρονικό παράθυρο δεδομένων χρησιμοποιώντας EMD και στατιστικά.

    Παράμετροι:
        window: DataFrame - Το χρονικό παράθυρο.
        i: int - Δείκτης (για υποβοήθηση του κώδικα).

    Επιστρέφει:
        dict - Λεξικό με τα εξαγόμενα χαρακτηριστικά.
    """
    features = {}
    joint_columns = [
        'left_flexion_extension_knee', 'left_flexion_extension_hip', 'left_flexion_extension_shoulder',
        'left_abduction_adduction_knee', 'left_abduction_adduction_hip', 'left_abduction_adduction_shoulder',
        'left_rotation_knee', 'left_rotation_hip', 'left_rotation_shoulder',
        'right_flexion_extension_knee', 'right_flexion_extension_hip', 'right_flexion_extension_shoulder',
        'right_abduction_adduction_knee', 'right_abduction_adduction_hip', 'right_abduction_adduction_shoulder',
        'right_rotation_knee', 'right_rotation_hip', 'right_rotation_shoulder'
    ]

    for joint in joint_columns:
        if joint in window.columns:
            try:
                # Αφαίρεση NaN και έλεγχος για άδειες τιμές
                joint_values = window[joint].dropna().values
                if joint_values.size == 0:
                    raise ValueError("Empty or NaN values in joint data.")

                # EMD Χαρακτηριστικά
                imfs = EMD().emd(joint_values)
                if len(imfs) > 0:
                    features[f'emd_mean_{joint}'] = np.mean(imfs[0])
                    features[f'emd_std_{joint}'] = np.std(imfs[0])
                    features[f'emd_energy_{joint}'] = np.sum(imfs[0]**2)
                else:
                    features[f'emd_mean_{joint}'] = 0
                    features[f'emd_std_{joint}'] = 0
                    features[f'emd_energy_{joint}'] = 0

                # Στατιστικά Χαρακτηριστικά
                features[f'mean_{joint}'] = np.mean(joint_values)
                features[f'median_{joint}'] = np.median(joint_values)
                features[f'std_{joint}'] = np.std(joint_values)
                features[f'min_{joint}'] = np.min(joint_values)
                features[f'max_{joint}'] = np.max(joint_values)
                features[f'range_{joint}'] = np.ptp(joint_values)  # max - min
            except Exception as e:
                print(f"Error processing joint {joint}: {e}")
        else:
            print(f"Joint column {joint} not found in window.")

    # Στατικά χαρακτηριστικά
    additional_columns = ['Disease', 'ID', 'RIGHT_CLOSED_TO_CAMERA', 'LEFT_CLOSED_TO_CAMERA']
    for col in additional_columns:
        if col in window.columns:
            features[col] = window[col].iloc[0]

    return features




def process_files(root_folders, regex_pattern):
    """
    Επεξεργάζεται όλα τα αρχεία CSV στους φακέλους και εξάγει χαρακτηριστικά.

    Παράμετροι:
        root_folders: list[Path] - Λίστα με φακέλους για επεξεργασία.
        regex_pattern: str - Regex μοτίβο για την ανάλυση των ονομάτων αρχείων.

    Επιστρέφει:
        DataFrame - Συνδυασμένο DataFrame με όλα τα χαρακτηριστικά.
    """
    all_features = []

    for root_folder in root_folders:
        if not root_folder.exists():
            print(f"Folder not found: {root_folder}")
            continue

        csv_files = list(root_folder.rglob("*.csv"))
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            match = re.search(regex_pattern, filename)

            if not match:
                print(f"Invalid filename format for {filename}")
                continue

            data = load_data(csv_file)
            if data is None:
                continue

            if 'Disease' not in data.columns:
                print(f"'Disease' column not found in {filename}")

            windows = create_windows(data, window_size=10)
            for window in windows:
                features = extract_features_from_window(window, 0)
                all_features.append(features)

    return pd.DataFrame(all_features)


# Επεξεργασία NM, KOA και PD
#nm_df = process_files([Path("/home/poulimenos/project/NM/")], r"(\d{3})_(\w+)_(\d{1})_(\d{2})")
#koa_df = process_files([Path("/home/poulimenos/project/KOA/")], r"(\d{3})(\w+)_(\w+)_(\d{2})")
pd_df = process_files([Path("/home/poulimenos/project/PD/")], r"(\d{3})_(\w+)_(\w+)_(\d{2})")

# Αποθήκευση σε CSV
#nm_df.to_csv("/home/poulimenos/project/nm_features.csv", index=False)
#koa_df.to_csv("/home/poulimenos/project/koa_features.csv", index=False)
pd_df.to_csv("/home/poulimenos/project/pd_features.csv", index=False)

print("Feature extraction completed and saved to CSV files.")
