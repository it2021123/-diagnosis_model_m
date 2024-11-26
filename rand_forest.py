from sklearn.model_selection import train_test_split, LeaveOneOut, LeaveOneGroupOut, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
new_df_koa = df_koa.sample(n=1000, replace=False, random_state=42)

# Συνδυασμός (κατακόρυφος)
df = pd.concat([df_nm, df_nm_s1,df_nm_s2,new_df_koa,df_pd], axis=0, ignore_index=True)

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
    'Disease']

# Επιλέγουμε τις στήλες από το DataFrame
X = df.drop(columns=columns_to_select)

y = df['target']

scaler=StandardScaler()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Δημιουργία του μοντέλου RandomForest
clf = RandomForestClassifier(
    n_estimators=500, 
    max_depth=20, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    max_features='log2', 
    bootstrap=True
)

# ============================================
# 1. Hold-Out Validation: Διαχωρισμός των δεδομένων σε εκπαίδευση και τεστ
# ============================================
print("1. Hold-Out Validation:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

holdout_accuracy = accuracy_score(y_test, y_pred)
holdout_precision = precision_score(y_test, y_pred, average='weighted')
holdout_recall = recall_score(y_test, y_pred, average='weighted')
holdout_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Hold-Out Validation Accuracy: {holdout_accuracy:.2f}")
print(f"Hold-Out Precision: {holdout_precision:.2f}")
print(f"Hold-Out Recall: {holdout_recall:.2f}")
print(f"Hold-Out F1-Score: {holdout_f1:.2f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f"Confusion Matrix Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

