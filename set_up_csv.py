# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Ορισμός φακέλων
root_folders = [
    Path("/home/poulimenos/project/KOA/"),
    Path("/home/poulimenos/project/PD/"),
    Path("/home/poulimenos/project/NM/")
]

# Αναζήτηση για .csv αρχεία και ενημέρωση των αντίστοιχων .csv αρχείων
for root_folder in root_folders:
    if not root_folder.exists():
        print(f"Folder not found: {root_folder}")
        continue  # Αν η διαδρομή δεν υπάρχει, παραλείπει τον φάκελο
    csv_files = list(root_folder.rglob("*.csv"))

    for csv_file in csv_files:
        print(f"Found CSV file: {csv_file}")
        filename = os.path.basename(csv_file)
        
        # Εύρεση ID στο όνομα αρχείου
        match = re.search(r"(\d{3})(\w+)_(\w+)_(\d{2})", filename)

        if match:
            part1 = match.group(1)  # Αποθηκεύει τα πρώτα τρία ψηφία, π.χ., "001"
            part2 = match.group(2)  # Αποθηκεύει τον δεύτερο τμήμα, π.χ., "PD"
            part3 = match.group(3)  # Αποθηκεύει το τρίτο τμήμα, π.χ., "02"
            part4= match.group(4)
            print("First part:", part1)
            print("Second part:", part2)
            print("Third part:", part3)
            print("4th part:",part4)
        else:
            print(f"Invalid filename format for {filename}")
            continue  # Αγνόηση αρχείου αν δεν ταιριάζει το πρότυπο
        
        # Φόρτωση δεδομένων από .csv
        try:
            data = pd.read_csv(csv_file)
            print(f"Loaded data from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

        # Έλεγχος αν η στήλη `Disease` υπάρχει στο DataFrame
        if 'Disease' not in data.columns:
            print(f"Column 'Disease' not found in {filename}")
            continue

        # Δημιουργία νέων στηλών στόχου NM, KOA, PD βάσει της στήλης `Disease`
         # Κωδικοποίηση της στήλης στόχου 'disease' σε ακέραιους αριθμούς (0, 1, 2 για KOA, NM, PD)
        data['target'] = data['Disease'].map({'KOA': 0, 'NM': 1, 'PD': 2})

        
        data['LEFT_CLOSED_TO_CAMERA'] = (data['Side'] == 2).astype(int)
        data['RIGHT_CLOSED_TO_CAMERA'] = (data['Side'] == 1).astype(int)

        data['frame_count'] = (data['frame_time'] / max(data['frame_time']) * 100).astype(int)

        # Δημιουργία του scaler
        scaler = StandardScaler()

        
        # Ονόματα στηλών γωνιών
        angles_columns = ['left_flexion_extension_knee', 'left_flexion_extension_hip', 'left_flexion_extension_shoulder',
                          'left_abduction_adduction_knee', 'left_abduction_adduction_hip', 'left_abduction_adduction_shoulder',
                          'left_rotation_knee', 'left_rotation_hip', 'left_rotation_shoulder',
                          'right_flexion_extension_knee', 'right_flexion_extension_hip', 'right_flexion_extension_shoulder',
                                            'right_abduction_adduction_knee', 'right_abduction_adduction_hip', 'right_abduction_adduction_shoulder',
                                            'right_rotation_knee', 'right_rotation_hip', 'right_rotation_shoulder']

        # Κανονικοποίηση των γωνιών με το StandardScaler
        data[angles_columns] = scaler.fit_transform(data[angles_columns])

        # Ενημέρωση του ίδιου αρχείου CSV με τις τροποποιήσεις
        try:
            data.to_csv(csv_file, index=False)  # Αποθήκευση στο ίδιο αρχείο χωρίς να συμπεριλάβουμε το index
            print(f"Updated {csv_file}")
        except Exception as e:
            print(f"Error saving updated file {csv_file}: {e}")



