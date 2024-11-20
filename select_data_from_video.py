#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 00:01:18 2024

@author: poulimenos
"""
import pandas as pd
import cv2
import mediapipe as mp
import os
import re
from pathlib import Path
import numpy as np

# Αρχικοποίηση της MediaPipe για αναγνώριση θέσης σώματος (Pose)
mp_pose = mp.solutions.pose

# Ορισμός του φακέλου που θα ψάξουμε για τα αρχεία .MOV
#Αλλάζοντας το path θα μπορούμε να κάνουμε το ίδιο για τις υπόλοιπες κατηγορίες
root_folder = Path("/home/poulimenos/Downloads/44pfnysy89-1/KOA/")

# Αναζήτηση όλων των αρχείων .MOV μέσα στον φάκελο και τους υποφακέλους
mov_files = list(root_folder.rglob("*.MOV"))

# Ορισμός των επιθυμητών σημείων αναφοράς του σώματος
desired_landmarks = {
    'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
    'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
    'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
    'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE,
    'LEFT_HEEL': mp_pose.PoseLandmark.LEFT_HEEL,
    'RIGHT_HEEL': mp_pose.PoseLandmark.RIGHT_HEEL,
    'LEFT_FOOT_INDEX': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    'RIGHT_FOOT_INDEX': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
}

# Συνάρτηση για τον υπολογισμό του διανύσματος μεταξύ δύο σημείων
def calculate_vector(x1, y1, x2, y2):
    return (x2 - x1, y2 - y1)

# Συνάρτηση για τον υπολογισμό της γωνίας μεταξύ τριών σημείων (χρησιμοποιώντας το διάνυσμα)
def calculate_angle_new_method(x1, y1, x2, y2, x3, y3):
    vector1 = calculate_vector(x1, y1, x2, y2)
    vector2 = calculate_vector(x2, y2, x3, y3)
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cos_theta = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# Συνάρτηση για να υπολογίσετε τις γωνίες σε διάφορους αρθρώσεις (γόνατο, ισχίο, ώμος κ.λπ.)
def calculate_return_all_angles(landmark_knee, landmark_hip, landmark_shoulder, landmark_elbow, landmark_ankle):
    flexion_extension_knee = calculate_angle_new_method(landmark_hip.x, landmark_hip.y, landmark_knee.x, landmark_knee.y, landmark_ankle.x, landmark_ankle.y)
    flexion_extension_hip = calculate_angle_new_method(landmark_knee.x, landmark_knee.y, landmark_hip.x, landmark_hip.y, landmark_shoulder.x, landmark_shoulder.y)
    flexion_extension_shoulder = calculate_angle_new_method(landmark_elbow.x, landmark_elbow.y, landmark_shoulder.x, landmark_shoulder.y, landmark_hip.x, landmark_hip.y)
    abduction_adduction_knee = calculate_angle_new_method(landmark_hip.y, landmark_hip.z, landmark_knee.y, landmark_knee.z, landmark_ankle.y, landmark_ankle.z)
    abduction_adduction_hip = calculate_angle_new_method(landmark_knee.y, landmark_knee.z, landmark_hip.y, landmark_hip.z, landmark_shoulder.y, landmark_shoulder.z)
    abduction_adduction_shoulder = calculate_angle_new_method(landmark_elbow.y, landmark_elbow.z, landmark_shoulder.y, landmark_shoulder.z, landmark_hip.y, landmark_hip.z)
    rotation_knee = calculate_angle_new_method(landmark_hip.x, landmark_hip.z, landmark_knee.x, landmark_knee.z, landmark_ankle.x, landmark_ankle.z)
    rotation_hip = calculate_angle_new_method(landmark_knee.x, landmark_knee.z, landmark_hip.x, landmark_hip.z, landmark_shoulder.x, landmark_shoulder.z)
    rotation_shoulder = calculate_angle_new_method(landmark_elbow.x, landmark_elbow.z, landmark_shoulder.x, landmark_shoulder.z, landmark_hip.x, landmark_hip.z)
    
    # Επιστροφή όλων των υπολογισμένων γωνιών σε ένα λεξικό
    return {
        "flexion_extension_knee": flexion_extension_knee,
        "flexion_extension_hip": flexion_extension_hip,
        "flexion_extension_shoulder": flexion_extension_shoulder,
        "abduction_adduction_knee": abduction_adduction_knee,
        "abduction_adduction_hip": abduction_adduction_hip,
        "abduction_adduction_shoulder": abduction_adduction_shoulder,
        "rotation_knee": rotation_knee,
        "rotation_hip": rotation_hip,
        "rotation_shoulder": rotation_shoulder
    }

# Επεξεργασία κάθε αρχείου βίντεο .MOV
for mov_file in mov_files:
    cap = cv2.VideoCapture(str(mov_file))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Λήψη των καρέ ανά δευτερόλεπτο
    filename = os.path.basename(mov_file)

    # Έλεγχος αν το όνομα του αρχείου ακολουθεί το σωστό φορμά
    match = re.search(r"(\d{3})_(\w+)_([LR]|\d{2})", filename)
    if not match:
        print(f"Invalid filename format for {filename}")
        cap.release()
        continue  # Αν δεν είναι σωστό το φορμά, παραλείπουμε το αρχείο

    # Ανάλυση του ονόματος του αρχείου για εξαγωγή πληροφοριών
    video_id, disease, left_or_right = match.groups()
    level = '0'  # Προκαθορισμένο επίπεδο (μπορεί να προσαρμοστεί αν υπάρχουν περισσότερα δεδομένα)

    # Αρχικοποίηση του πίνακα για την αποθήκευση των δεδομένων πόζας
    pose_data = []

    # Επεξεργασία κάθε καρέ του βίντεο και αναγνώριση θέσης σώματος με την MediaPipe
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Σταματάμε αν το βίντεο τελειώσει

            # Επεξεργασία κάθε δεύτερου καρέ για να μειώσουμε τη φόρτωση
            if frame_count % 2 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    # Δημιουργία ενός λεξικού με τα δεδομένα του κάθε καρέ
                    frame_data = {
                        'frame_time': cap.get(cv2.CAP_PROP_POS_MSEC),
                        'ID': video_id,
                        'Disease': disease,
                        'LEFT_OR_RIGHT_CLOSED_TO_CAMERA': left_or_right,
                        'LEVEL': level
                    }

                    # Προσθήκη δεδομένων για κάθε σημείο αναφοράς (landmark) του σώματος
                    for name, idx in desired_landmarks.items():
                        landmark = results.pose_landmarks.landmark[idx]
                        frame_data[f'{name}_x'] = landmark.x
                        frame_data[f'{name}_y'] = landmark.y
                        frame_data[f'{name}_z'] = landmark.z
                        frame_data[f'{name}_visibility'] = landmark.visibility

                    # Υπολογισμός των γωνιών και προσθήκη τους στα δεδομένα του καρέ
                    angles = calculate_return_all_angles(
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                    )
                    frame_data.update(angles)
                    pose_data.append(frame_data)  # Προσθήκη των δεδομένων στο σύνολο

            frame_count += 1

    # Αποθήκευση των δεδομένων σε αρχείο CSV
    output_csv = f"{video_id}_{disease}_{level}_{left_or_right}.csv"
    df = pd.DataFrame(pose_data)
    df.to_csv(output_csv, index=False)
    print(f"Processed {filename}, results saved to {output_csv}")

    cap.release()  # Απελευθέρωση του πόρου για το βίντεο
