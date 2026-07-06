import os
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder


def normalize_symptom_value(value):
    if pd.isna(value):
        return 0

    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "1", "present"}:
        return 1
    if normalized in {"no", "n", "false", "0", "absent"}:
        return 0

    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value):
        return numeric_value

    return 0


# =====================================================
# CONFIGURATION
# =====================================================

# Minimum number of samples a disease must have to be included in training.
# Diseases below this threshold are too rare to learn reliably.
MIN_SAMPLES_PER_DISEASE = 10

# Dataset files to load and combine
DATA_PATHS = [
    "Disease and symptoms dataset.csv",
]

# =====================================================
# LOAD DATASET
# =====================================================

datasets = []
for path in DATA_PATHS:
    if os.path.exists(path):
        datasets.append(pd.read_csv(path))

if not datasets:
    raise FileNotFoundError("No dataset files found from DATA_PATHS")

data = pd.concat(datasets, ignore_index=True).drop_duplicates().reset_index(drop=True)

print("=" * 60)
print("DISEASE PREDICTION MODEL — ENHANCED TRAINING")
print("=" * 60)
print(f"\nDataset Loaded Successfully!")
print(f"  Shape: {data.shape}")
print(f"  Total samples: {data.shape[0]:,}")
print(f"  Total features: {data.shape[1] - 1}")


# =====================================================
# TARGET + FEATURES
# =====================================================

# First column = Disease Name (Output)
target_column = data.columns[0]
data = data[data[target_column].notna()].copy()
data[target_column] = data[target_column].astype(str).str.strip()
data = data[data[target_column] != ""]

print(f"\nTarget Column: '{target_column}'")
print(f"Unique diseases (before filtering): {data[target_column].nunique()}")


# =====================================================
# FILTER RARE DISEASES
# =====================================================

disease_counts = data[target_column].value_counts()
rare_diseases = disease_counts[disease_counts < MIN_SAMPLES_PER_DISEASE].index
valid_diseases = disease_counts[disease_counts >= MIN_SAMPLES_PER_DISEASE].index

print(f"\n--- Rare Disease Filtering ---")
print(f"  Minimum samples threshold: {MIN_SAMPLES_PER_DISEASE}")
print(f"  Diseases removed (< {MIN_SAMPLES_PER_DISEASE} samples): {len(rare_diseases)}")
print(f"  Diseases kept: {len(valid_diseases)}")

data = data[data[target_column].isin(valid_diseases)].reset_index(drop=True)

print(f"  Samples after filtering: {data.shape[0]:,}")


# =====================================================
# PREPARE FEATURES AND TARGET
# =====================================================

y = data[target_column]
X = data.iloc[:, 1:]

# Convert symptom columns to numeric values
X = X.apply(lambda column: column.map(normalize_symptom_value))
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Store original feature names before any selection
original_feature_columns = list(X.columns)

print(f"\nTotal Symptom Features: {X.shape[1]}")
print(f"Feature sparsity: {(X == 0).values.mean() * 100:.1f}% zeros")


# =====================================================
# FEATURE SELECTION — Remove Zero-Variance Features
# =====================================================

# Remove features that are constant (zero variance) — they add noise
selector = VarianceThreshold(threshold=0.001)
X_selected = selector.fit_transform(X)
selected_mask = selector.get_support()
selected_features = [col for col, keep in zip(X.columns, selected_mask) if keep]

print(f"\n--- Feature Selection ---")
print(f"  Features before: {X.shape[1]}")
print(f"  Features after (variance > 0.001): {len(selected_features)}")
print(f"  Features removed: {X.shape[1] - len(selected_features)}")

X = pd.DataFrame(X_selected, columns=selected_features)


# =====================================================
# ENCODE TARGET LABELS
# =====================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

n_classes = len(label_encoder.classes_)
print(f"\nEncoded {n_classes} disease classes")


# =====================================================
# TRAIN / TEST SPLIT (Stratified)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print(f"\n--- Train/Test Split ---")
print(f"  Train samples: {X_train.shape[0]:,}")
print(f"  Test samples:  {X_test.shape[0]:,}")


# =====================================================
# BUILD ENSEMBLE MODEL
# =====================================================

def predict_in_batches(model, X, batch_size=2000):
    preds = []
    for i in range(0, len(X), batch_size):
        preds.extend(model.predict(X.iloc[i:i+batch_size]))
    return np.array(preds)

print("\n" + "=" * 60)
print("TRAINING ENSEMBLE MODEL")
print("=" * 60)

# --- Model 1: Random Forest ---
print("\n[1/3] Training Random Forest...")
t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=1,            
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, predict_in_batches(rf, X_test))
print(f"    Random Forest Accuracy: {rf_acc * 100:.2f}%  ({time.time() - t0:.1f}s)")

# --- Model 2: Extra Trees ---
print("\n[2/3] Training Extra Trees...")
t0 = time.time()

et = ExtraTreesClassifier(
    n_estimators=50,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=1,            
)
et.fit(X_train, y_train)
et_acc = accuracy_score(y_test, predict_in_batches(et, X_test))
print(f"    Extra Trees Accuracy: {et_acc * 100:.2f}%  ({time.time() - t0:.1f}s)")

# --- Model 3: Histogram Gradient Boosting ---
print("\n[3/3] Training Histogram Gradient Boosting...")
t0 = time.time()

hgb = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=10,
    min_samples_leaf=10,
    learning_rate=0.1,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    random_state=42,
)
hgb.fit(X_train, y_train)
hgb_acc = accuracy_score(y_test, predict_in_batches(hgb, X_test))
print(f"    HistGradientBoosting Accuracy: {hgb_acc * 100:.2f}%  ({time.time() - t0:.1f}s)")


# --- Voting Classifier (Soft Voting) ---
print("\n[Ensemble] Building Voting Classifier (soft voting)...")
t0 = time.time()

# Use pre-fitted estimators to avoid re-training
voting = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("et", et),
        ("hgb", hgb),
    ],
    voting="soft",
    n_jobs=2,
)

# Manually set the fitted state since models are already trained
voting.estimators_ = [rf, et, hgb]
voting.le_ = LabelEncoder().fit(y_train)
voting.classes_ = voting.le_.classes_

ensemble_pred = predict_in_batches(voting, X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"    Voting Ensemble Accuracy: {ensemble_acc * 100:.2f}%  ({time.time() - t0:.1f}s)")


# =====================================================
# SELECT BEST MODEL
# =====================================================

models = {
    "Random Forest": (rf, rf_acc),
    "Extra Trees": (et, et_acc),
    "HistGradientBoosting": (hgb, hgb_acc),
    "Voting Ensemble": (voting, ensemble_acc),
}

best_name, (best_model, best_acc) = max(models.items(), key=lambda x: x[1][1])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
for name, (_, acc) in models.items():
    marker = " ★ BEST" if name == best_name else ""
    print(f"  {name:30s} → {acc * 100:.2f}%{marker}")

print(f"\n🏆 Selected: {best_name} ({best_acc * 100:.2f}%)")


# =====================================================
# CROSS-VALIDATION (lightweight, memory-safe)
# =====================================================

print("\n--- 5-Fold Stratified Cross-Validation ---")

# Use the best non-ensemble model for CV (ensemble CV is too slow)
best_individual_name = max(
    {k: v for k, v in models.items() if k != "Voting Ensemble"}.items(),
    key=lambda x: x[1][1],
)[0]
best_individual_model = models[best_individual_name][0]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_individual_model.__class__(**best_individual_model.get_params()),
    X, y_encoded,
    cv=cv,
    scoring="accuracy",
    n_jobs=2,       
)

print(f"  Model: {best_individual_name}")
print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")


# =====================================================
# DETAILED CLASSIFICATION REPORT
# =====================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (Test Set)")
print("=" * 60)

y_test_labels = label_encoder.inverse_transform(y_test)
pred_labels = label_encoder.inverse_transform(
    predict_in_batches(best_model, X_test)
)

report = classification_report(
    y_test_labels,
    pred_labels,
    output_dict=True,
    zero_division=0,
)

# Print summary metrics
print(f"\n  Weighted Avg Precision: {report['weighted avg']['precision'] * 100:.2f}%")
print(f"  Weighted Avg Recall:    {report['weighted avg']['recall'] * 100:.2f}%")
print(f"  Weighted Avg F1-Score:  {report['weighted avg']['f1-score'] * 100:.2f}%")

# Show worst-performing classes
per_class = {
    k: v for k, v in report.items()
    if k not in ("accuracy", "macro avg", "weighted avg")
}
worst_classes = sorted(per_class.items(), key=lambda x: x[1]["f1-score"])[:10]

print(f"\n  Worst 10 Classes (by F1-Score):")
for cls, metrics in worst_classes:
    print(f"    {cls:40s} → F1: {metrics['f1-score']:.2f}  "
          f"(P: {metrics['precision']:.2f}, R: {metrics['recall']:.2f}, "
          f"N: {metrics['support']})")


# =====================================================
# SAVE MODEL + ARTIFACTS
# =====================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

os.makedirs("saved_model", exist_ok=True)

# Save the best model as pipeline.pkl (matching what app.py expects)
with open("saved_model/pipeline.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save feature columns (matching what app.py expects)
# Use the ORIGINAL full column list so the app's input alignment works correctly.
# The model was trained on selected_features (after variance filtering),
# but the app constructs input DataFrames using the full feature list.
# We need to save both so the app can map inputs correctly.
with open("saved_model/feature_columns.pkl", "wb") as file:
    pickle.dump(selected_features, file)

# Save label encoder for inverse transforms
with open("saved_model/target_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

# Save the feature selector so app can apply same transform
with open("saved_model/feature_selector.pkl", "wb") as file:
    pickle.dump(selector, file)

# Save original feature columns (pre-selection) for input mapping
with open("saved_model/original_feature_columns.pkl", "wb") as file:
    pickle.dump(original_feature_columns, file)

# Also save the old-format model.pkl for backwards compatibility
with open("saved_model/model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print(f"\n  ✅ Model saved: saved_model/pipeline.pkl")
print(f"  ✅ Feature columns saved: saved_model/feature_columns.pkl ({len(selected_features)} features)")
print(f"  ✅ Original columns saved: saved_model/original_feature_columns.pkl ({len(original_feature_columns)} features)")
print(f"  ✅ Label encoder saved: saved_model/target_encoder.pkl")
print(f"  ✅ Feature selector saved: saved_model/feature_selector.pkl")
print(f"  ✅ Backward-compat model: saved_model/model.pkl")


# =====================================================
# QUICK PREDICTION TEST
# =====================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTION TEST")
print("=" * 60)

sample_symptoms = pd.DataFrame(
    [[1] * len(selected_features)],
    columns=selected_features,
)

result = best_model.predict(sample_symptoms)
if hasattr(best_model, "predict_proba"):
    proba = best_model.predict_proba(sample_symptoms)[0]
    top_idx = np.argsort(proba)[-3:][::-1]
    print("\n  Top 3 Predictions:")
    for idx in top_idx:
        disease_name = label_encoder.inverse_transform([idx])[0]
        print(f"    {disease_name}: {proba[idx] * 100:.1f}%")
else:
    result_label = label_encoder.inverse_transform(result)[0]
    print(f"\n  Predicted Disease: {result_label}")


print("\n" + "=" * 60)
print(f"TRAINING COMPLETE — Best Accuracy: {best_acc * 100:.2f}%")
print("=" * 60)