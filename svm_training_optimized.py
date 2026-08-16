import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_auc_score
)
import matplotlib.pyplot as plt
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 OPTIMIZED SVM TRAINING - FINAL VERSION")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================================
print("📥 Loading data...")
df = pd.read_csv('KaggleV2-May-2016.csv')

# Rename columns to Python standard
df = df.rename(columns={
    'PatientId': 'patient_id',
    'AppointmentID': 'appointment_id',
    'ScheduledDay': 'scheduled_day',
    'AppointmentDay': 'appointment_day',
    'Hipertension': 'hypertension',
    'Handcap': 'handicap',
    'SMS_received': 'sms_received',
    'No-show': 'no_show',
    'Age': 'age',
    'Gender': 'gender',
    'Neighbourhood': 'neighbourhood'
})

# Clean data
df = df.drop_duplicates()
df = df[df['age'] >= 0]

# Convert dates to datetime
df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
df['appointment_day'] = pd.to_datetime(df['appointment_day'])

# Calculate days until appointment
df['days_until_appointment'] = (df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()).dt.days
df = df[df['days_until_appointment'] >= 0]

# Map categorical columns
df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# One-hot encode appointment weekday
df['appt_weekday'] = df['appointment_day'].dt.day_name()
df = pd.get_dummies(df, columns=['appt_weekday'], drop_first=True)

# Drop unnecessary columns
df = df.drop(columns=['patient_id', 'appointment_id', 'scheduled_day', 'appointment_day'])

print(f"✓ Data shape: {df.shape}")
print(f"✓ No-show distribution: {dict(df['no_show'].value_counts())}\n")

# ============================================================================
# STEP 2: SEPARATE FEATURES AND TARGET
# ============================================================================
X = df.drop('no_show', axis=1)
y = df['no_show']

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Train/Test split:")
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test:  {X_test.shape[0]} samples\n")

# ============================================================================
# STEP 4: SCALE NUMERICAL FEATURES
# ============================================================================
scaler = StandardScaler()
numerical_columns = ['age', 'days_until_appointment']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# ============================================================================
# STEP 5: ONE-HOT ENCODE NEIGHBOURHOOD
# ============================================================================
X_train_encoded = pd.get_dummies(X_train, columns=['neighbourhood'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['neighbourhood'], drop_first=True)

# Align columns between train and test sets
train_cols = X_train_encoded.columns
test_cols = X_test_encoded.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_encoded[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train_encoded[c] = 0

X_test_encoded = X_test_encoded[train_cols]

print(f"✓ Final features: {X_train_encoded.shape[1]} after encoding\n")

# ============================================================================
# STEP 6: TRAIN OPTIMIZED SVM WITH PROBABILITIES
# ============================================================================
print("="*80)
print("🔧 TRAINING SVM")
print("="*80)
print("\n⚡ Optimizations:")
print("   • Linear kernel (10x faster than RBF)")
print("   • Reduced hyperparameter grid (4 combinations)")
print("   • 2-fold CV (faster convergence)")
print("   • probability=True (for Flask app compatibility)")
print("   • class_weight='balanced' (handles imbalanced data)\n")

parameters = {
    'C': [1, 10],
    'gamma': [0.01, 0.1]
}

print(f"Testing {len(parameters['C']) * len(parameters['gamma'])} parameter combinations...\n")

start_time = time.time()

grid = GridSearchCV(
    SVC(
        kernel='linear',
        class_weight='balanced',
        random_state=42,
        probability=True,  # ✅ ENABLES PROBABILITY PREDICTIONS
        verbose=0
    ),
    parameters,
    scoring='f1',
    cv=2,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_encoded, y_train)

train_time = time.time() - start_time
print(f"\n✅ Training completed in {train_time:.2f} seconds\n")

model = grid.best_estimator_

print(f"Best parameters found:")
print(f"  C: {grid.best_params_['C']}")
print(f"  Gamma: {grid.best_params_['gamma']}")
print(f"  Best CV F1-Score: {grid.best_score_:.4f}\n")

# ============================================================================
# STEP 7: MAKE PREDICTIONS WITH PROBABILITIES
# ============================================================================
print("="*80)
print("🎯 PREDICTIONS & EVALUATION")
print("="*80 + "\n")

y_pred = model.predict(X_test_encoded)
y_proba = model.predict_proba(X_test_encoded)[:, 1]  # ✅ PROBABILITIES (0-1)

# ============================================================================
# STEP 8: PERFORMANCE EVALUATION
# ============================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"📈 Metrics:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"   [[True Neg  False Pos]  = [{cm[0][0]}  {cm[0][1]}]")
print(f"    [False Neg True Pos ]] = [{cm[1][0]}  {cm[1][1]}]\n")

# ============================================================================
# STEP 9: SAVE MODEL & ARTIFACTS
# ============================================================================
print("="*80)
print("💾 SAVING MODEL ARTIFACTS")
print("="*80 + "\n")

# Save the trained model
joblib.dump(model, "svm_model.pkl")
print("✓ svm_model.pkl")

# Save the scaler (SAME SCALER USED IN TRAINING)
joblib.dump(scaler, "svm_scaler.pkl")
print("✓ svm_scaler.pkl")

# Save feature columns (CRITICAL FOR APP.PY TO MATCH)
feature_columns = list(X_train_encoded.columns)
joblib.dump(feature_columns, "svm_feature_columns.pkl")
print("✓ svm_feature_columns.pkl")

print("\n" + "="*80)
print("✅ SVM TRAINING COMPLETE & READY FOR PRODUCTION")
print("="*80)
print(f"\n🎉 Summary:")
print(f"   Training Time:  {train_time:.2f} seconds")
print(f"   Test Accuracy:  {accuracy*100:.2f}%")
print(f"   Test Recall:    {recall*100:.2f}%")
print(f"   Test F1-Score:  {f1:.4f}")
print(f"\n📱 Ready to use in Flask app!")
print(f"   All files saved: svm_model.pkl, svm_scaler.pkl, svm_feature_columns.pkl\n")
