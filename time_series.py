#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:31:23 2024

@author: poulimenos
"""
import os
import re
from pathlib import Path
import pandas as pd



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

def extract_features_from_window(window,i):
    """
    Εξάγει χαρακτηριστικά από κάθε παράθυρο
    """
    try:
        if(i==0):
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
            'angle_difference_flexion_shoulder': window['flexion_extension_shoulder'].iloc[-1] - window['flexion_extension_shoulder'].iloc[0],
            
               }
         
         # Προσθήκη επιπλέον χαρακτηριστικών αν υπάρχουν οι στήλες
          additional_columns = ['Disease','ID','RIGHT_CLOSED_TO_CAMERA', 'LEFT_CLOSED_TO_CAMERA', 'NM', 'KOA', 'PD']
        else :
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
             'angle_difference_flexion_shoulder': window['flexion_extension_shoulder'].iloc[-1] - window['flexion_extension_shoulder'].iloc[0],
             
                }
          
          # Προσθήκη επιπλέον χαρακτηριστικών αν υπάρχουν οι στήλες
           additional_columns = ['Disease','ID','RIGHT_CLOSED_TO_CAMERA', 'LEFT_CLOSED_TO_CAMERA', 'NM', 'KOA', 'PD']
       
           
        for col in additional_columns:
            if col in window.columns:
                # Έλεγχος ότι το παράθυρο έχει τουλάχιστον 2 γραμμές
                features[col.lower()] = window[col].iloc[1] if len(window) > 1 else None
            else:
                features[col.lower()] = None  # Αν η στήλη δεν υπάρχει
        
        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return {}




def process_files_nm(root_folders):
    all_features = []  # Λίστα για αποθήκευση όλων των χαρακτηριστικών
    for root_folder in root_folders:
        if not root_folder.exists():
            print(f"Folder not found: {root_folder}")
            continue
        csv_files = list(root_folder.rglob("*.csv"))

        for csv_file in csv_files:
            print(f"Found CSV file: {csv_file}")
            filename = os.path.basename(csv_file)
            
            match = re.search(r"(\d{3})_(\w+)_(\d{1})_(\d{2})", filename)

            if match:
                part1 = match.group(1)
                part2 = match.group(2)
                target = part2
            else:
                print(f"Invalid filename format for {filename}")
                continue
            
            data = load_data(csv_file)
            if data is None:
                continue
            i=0
            if 'DISEASE' not in data.columns:
                print(f"Column 'DISEASE' not found in {filename}")
                i=1

            window_size = 10
            windows = create_windows(data, window_size)

            for window in windows:
                features = extract_features_from_window(window,i)
                all_features.append(features)

    # Δημιουργία συνολικού DataFrame με όλα τα χαρακτηριστικά
    series_df = pd.DataFrame(all_features)
    return series_df

def process_files_koa_pd(root_folders):
    all_features = []  # Λίστα για αποθήκευση όλων των χαρακτηριστικών
    for root_folder in root_folders:
        if not root_folder.exists():
            print(f"Folder not found: {root_folder}")
            continue
        csv_files = list(root_folder.rglob("*.csv"))

        for csv_file in csv_files:
            print(f"Found CSV file: {csv_file}")
            filename = os.path.basename(csv_file)
            
            match = re.search(r"(\d{3})(\w+)_(\w+)_(\d{2})", filename)

            if match:
                part1 = match.group(1)
                part2 = match.group(2)
            else:
                print(f"Invalid filename format for {filename}")
                continue
            
            data = load_data(csv_file)
            if data is None:
                continue
            
            if 'Disease' not in data.columns:
                print(f"Column 'Disease' not found in {filename}")
                continue

            window_size = 10
            windows = create_windows(data, window_size)

            for window in windows:
                features = extract_features_from_window(window,0)
                all_features.append(features)

    # Δημιουργία συνολικού DataFrame με όλα τα χαρακτηριστικά
    series_df = pd.DataFrame(all_features)
    return series_df


# Επεξεργασία αρχείων
nm_df = process_files_nm([Path("/home/poulimenos/project/Updated_NM_CSVs/")])
dis_df = process_files_koa_pd([
    Path("/home/poulimenos/project/KOA_EL_CSV/"), 
    Path("/home/poulimenos/project/KOA_MD_CSV/"), 
    Path("/home/poulimenos/project/KOA_SV_CSV/"), 
    Path("/home/poulimenos/project/PD/")
])

# Συνδυασμός των δύο DataFrame
if nm_df is not None and dis_df is not None:
    combined_df = pd.concat([nm_df, dis_df], ignore_index=True)
    
    # Αποθήκευση του DataFrame σε αρχείο CSV
    output_path = "/home/poulimenos/project/combined_features.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Combined DataFrame saved to {output_path}")
else:
    print("One or both DataFrames are empty. Cannot create combined CSV.")
