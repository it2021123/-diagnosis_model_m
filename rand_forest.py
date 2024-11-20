from sklearn.model_selection import train_test_split, LeaveOneOut, LeaveOneGroupOut, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Φόρτωση των δεδομένων από το αρχείο CSV
df = pd.read_csv("/home/poulimenos/project/features.csv")

# Αφαίρεση γραμμών που περιέχουν NaN
df = df.dropna()

# Κωδικοποίηση της στήλης στόχου 'disease' σε ακέραιους αριθμούς (0, 1, 2 για KOA, NM, PD)
df['target'] = df['disease'].map({'KOA': 0, 'NM': 1, 'PD': 2})

# Δημιουργία των χαρακτηριστικών (X) και του στόχου (y)
X = df.drop(columns=['koa', 'nm', 'pd', 'id', 'disease', 'target'], errors='ignore')  # Χαρακτηριστικά
y = df['target']  # Στόχος είναι η στήλη 'target'

# Δημιουργία groups για LAGO-CV (χρησιμοποιούμε μόνο την 'id' για να κατατάξουμε τα δεδομένα σε ομάδες)
groups = df['id']  # Χρησιμοποιούμε μόνο το 'id' ως ομάδα

# Δημιουργία του μοντέλου RandomForest
clf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    max_features='sqrt', 
    bootstrap=True, 
    random_state=64
)

# ============================================
# 1. Hold-Out Validation: Διαχωρισμός των δεδομένων σε εκπαίδευση και τεστ
# ============================================
print("1. Hold-Out Validation:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
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


# ============================================
# 2. Stratified K-Fold Cross-Validation
# ============================================
print("2. Stratified K-Fold Cross-Validation:")
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_accuracies = []
stratified_precisions = []
stratified_recalls = []
stratified_f1_scores = []

for train_index, test_index in stratified_kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    stratified_accuracies.append(accuracy_score(y_test, y_pred))
    stratified_precisions.append(precision_score(y_test, y_pred, average='weighted'))
    stratified_recalls.append(recall_score(y_test, y_pred, average='weighted'))
    stratified_f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

print(f"Stratified K-Fold Accuracy: {np.mean(stratified_accuracies):.2f} ± {np.std(stratified_accuracies):.2f}")
print(f"Stratified K-Fold Precision: {np.mean(stratified_precisions):.2f} ± {np.std(stratified_precisions):.2f}")
print(f"Stratified K-Fold Recall: {np.mean(stratified_recalls):.2f} ± {np.std(stratified_recalls):.2f}")
print(f"Stratified K-Fold F1-Score: {np.mean(stratified_f1_scores):.2f} ± {np.std(stratified_f1_scores):.2f}\n")


from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import numpy as np

# ============================================
# 3. Leave-A-Group-Out Cross-Validation (LAGO-CV) 
# ============================================
print("3. Leave-A-Group-Out Cross-Validation (LAGO-CV) - Using 'id':")
lago = LeaveOneGroupOut()  # Δημιουργία του LAGO-CV
lago_accuracies = []  # Λίστα για αποθήκευση των ακρίβειών

# Επανάληψη για κάθε ομάδα με την 'id'
for train_index, test_index in lago.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    lago_accuracies.append(accuracy_score(y_test, y_pred))

print(f"LAGO-CV Accuracies for each group (using 'id'): {lago_accuracies}")
print(f"Mean LAGO Accuracy (using 'id'): {np.mean(lago_accuracies):.2f}\n")




#======================================================================================
#------Πως επηρρεαζεται απο τα δεδομενα απο αντικείμενα που ανήκουν στην nm κατηγορια==
#======================================================================================
df1=df[df['target']>0]
# Δημιουργία των χαρακτηριστικών (X) και του στόχου (y)
X = df1.drop(columns=['koa', 'nm', 'pd', 'id', 'disease', 'target'], errors='ignore')  # Χαρακτηριστικά
y = df1['target']  # Στόχος είναι η στήλη 'target'


# ----------------------------------------------------------------------------------
#  Hold-Out Validation: Διαχωρισμός των δεδομένων σε εκπαίδευση και τεστ

print("1. Hold-Out Validation:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
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