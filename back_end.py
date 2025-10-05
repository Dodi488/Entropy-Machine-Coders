import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, 
                             roc_auc_score, accuracy_score)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

print("=" * 70)
print("EXOPLANET CLASSIFIER: FUSED KEPLER/K2 TRAINING + TESS TESTING")
print("=" * 70)

# --- Step 1: Load Fused Training Data ---
FUSED_FILE = './fused_kepler_k2_data.csv'
TESS_FILE = '/home/rodrigo/Desktop/nasa/Entropy-Machine-Coders/raw_data/TOI_2025.10.04_08.55.28.csv'

print("\n--- Loading Training Data ---")
try:
    train_df = pd.read_csv(FUSED_FILE)
    print(f"✓ Fused Kepler/K2: {len(train_df)} rows, {len(train_df.columns)} columns")
except FileNotFoundError:
    print(f"✗ ERROR: {FUSED_FILE} not found")
    print("Please run the fusion script first to generate 'fused_kepler_k2_data.csv'")
    sys.exit(1)

print("\n--- Loading Test Data (TESS) ---")
try:
    tess_df = pd.read_csv(TESS_FILE, comment='#')
    print(f"✓ TESS TOI: {len(tess_df)} rows, {len(tess_df.columns)} columns")
except FileNotFoundError:
    print(f"✗ ERROR: {TESS_FILE} not found")
    print("Please download TESS data and save as './raw_data/toi_2025.10.04_08.55.41.csv'")
    sys.exit(1)

# --- Step 2: Prepare Training Features ---
print("\n--- Preparing Training Data ---")

# Separate features from target
if 'disposition' not in train_df.columns:
    print("ERROR: 'disposition' column not found in fused data")
    sys.exit(1)

# Get feature columns (exclude disposition and source)
exclude_cols = ['disposition', 'source']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"Available features: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")

# Remove features with too many missing values (>80% missing)
good_features = []
for col in feature_cols:
    missing_pct = train_df[col].isna().sum() / len(train_df)
    if missing_pct < 0.8:
        good_features.append(col)

print(f"Features with <80% missing: {len(good_features)}")

feature_cols = good_features

# Prepare X and y
X_train = train_df[feature_cols].copy()
y_train_labels = train_df['disposition']

# Fill missing values with median
print("\nFilling missing values with median...")
for col in feature_cols:
    if X_train[col].isna().any():
        X_train[col].fillna(X_train[col].median(), inplace=True)

# Check class distribution
print("\nTraining class distribution:")
for label, count in y_train_labels.value_counts().items():
    print(f"  {label}: {count} ({count/len(y_train_labels)*100:.1f}%)")

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)

print(f"\nTraining samples: {len(X_train)}")
print(f"Training features: {len(feature_cols)}")
print(f"Classes: {le.classes_}")

# --- Step 3: Scale Features ---
print("\n--- Scaling Features ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- Step 4: Train Model ---
print("\n--- Training Random Forest Model ---")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',
    verbose=0
)
model.fit(X_train_scaled, y_train)
print("✓ Training complete")

# --- Step 5: Feature Importance ---
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 15 Most Important Features ---")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# --- Step 6: Prepare TESS Test Data ---
print("\n--- Preparing TESS Test Data ---")

# Find TESS disposition column
tess_disp_cols = [col for col in tess_df.columns if 'disp' in col.lower()]
if tess_disp_cols:
    tess_disp_col = tess_disp_cols[0]
    print(f"Using TESS disposition column: {tess_disp_col}")
else:
    print("WARNING: No disposition column found in TESS data")
    tess_disp_col = None

# Map TESS disposition to binary
def map_tess_disposition(disp):
    if pd.isna(disp):
        return None
    disp_str = str(disp).upper()
    # TESS uses different terminology
    if any(word in disp_str for word in ['CP', 'CONFIRMED', 'PC']):
        return 'CONFIRMED'
    else:
        return 'CANDIDATE'

if tess_disp_col:
    tess_df['binary_disposition'] = tess_df[tess_disp_col].apply(map_tess_disposition)
    tess_test = tess_df.dropna(subset=['binary_disposition']).copy()
    
    print(f"TESS samples with labels: {len(tess_test)}")
    if len(tess_test) > 0:
        print("TESS class distribution:")
        for label, count in tess_test['binary_disposition'].value_counts().items():
            print(f"  {label}: {count} ({count/len(tess_test)*100:.1f}%)")
else:
    print("Cannot evaluate on TESS without disposition labels")
    tess_test = tess_df.copy()
    tess_test['binary_disposition'] = 'CANDIDATE'

# Map TESS features to training features
print("\n--- Mapping TESS Features ---")

# Common feature mappings
tess_to_kepler_mapping = {
    'toi': 'pl_name',
    'pl_orbper': 'pl_orbper',
    'pl_rade': 'pl_rade',
    'st_teff': 'st_teff',
    'pl_radj': 'pl_radj',
    'pl_bmassj': 'pl_bmassj',
    'pl_orbsmax': 'pl_orbsmax',
    'pl_orbeccen': 'pl_orbeccen',
    'st_rad': 'st_rad',
    'st_mass': 'st_mass',
    'st_logg': 'st_logg',
    'st_met': 'st_met',
}

X_test = pd.DataFrame()
matched_features = 0
missing_features = []

for train_col in feature_cols:
    # Try direct match first
    if train_col in tess_test.columns:
        X_test[train_col] = tess_test[train_col]
        matched_features += 1
    # Try mapped match
    elif train_col in tess_to_kepler_mapping.values():
        tess_col = [k for k, v in tess_to_kepler_mapping.items() if v == train_col]
        if tess_col and tess_col[0] in tess_test.columns:
            X_test[train_col] = tess_test[tess_col[0]]
            matched_features += 1
        else:
            X_test[train_col] = X_train[train_col].median()
            missing_features.append(train_col)
    else:
        # Feature not found, use training median
        X_test[train_col] = X_train[train_col].median()
        missing_features.append(train_col)

print(f"Matched features: {matched_features}/{len(feature_cols)} ({matched_features/len(feature_cols)*100:.1f}%)")
if len(missing_features) > 0:
    print(f"Missing features (using median): {len(missing_features)}")
    if len(missing_features) <= 10:
        print(f"  {missing_features}")

# Fill missing values
for col in feature_cols:
    if X_test[col].isna().any():
        X_test[col].fillna(X_train[col].median(), inplace=True)

# Prepare labels
y_test_labels = tess_test['binary_disposition']
y_test = le.transform(y_test_labels)

# Scale test data
X_test_scaled = scaler.transform(X_test)

print(f"\nTest samples: {len(X_test)}")

# --- Step 7: Evaluate Model ---
print("\n" + "="*70)
print("CROSS-DATASET EVALUATION: KEPLER/K2 → TESS")
print("="*70)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', 
                           pos_label=le.transform(['CONFIRMED'])[0], zero_division=0)
recall = recall_score(y_test, y_pred, average='binary', 
                     pos_label=le.transform(['CONFIRMED'])[0], zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', 
             pos_label=le.transform(['CONFIRMED'])[0], zero_division=0)

try:
    confirmed_idx = list(le.classes_).index('CONFIRMED')
    auc = roc_auc_score(y_test, y_pred_proba[:, confirmed_idx])
except Exception as e:
    auc = "N/A"
    print(f"Could not calculate AUC: {e}")

print(f"\nTest Performance (TESS):")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc if isinstance(auc, str) else f'{auc:.4f}'}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Classes: {le.classes_}")

# Per-class metrics
print(f"\nDetailed Per-Class Metrics:")
for idx, class_name in enumerate(le.classes_):
    mask = y_test == idx
    if mask.sum() > 0:
        tp = np.sum((y_test == idx) & (y_pred == idx))
        fp = np.sum((y_test != idx) & (y_pred == idx))
        fn = np.sum((y_test == idx) & (y_pred != idx))
        
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n  {class_name}:")
        print(f"    True Positives:  {tp}")
        print(f"    False Positives: {fp}")
        print(f"    False Negatives: {fn}")
        print(f"    Precision: {class_precision:.4f}")
        print(f"    Recall:    {class_recall:.4f}")
        print(f"    Support:   {mask.sum()}")

print("="*70)
print("Model ready for predictions!")
print("="*70)

# Store globals for API
FEATURE_COLUMNS = feature_cols

# --- API Endpoints ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = []
        for col in FEATURE_COLUMNS:
            if col in data:
                input_data.append(data[col])
            else:
                input_data.append(X_train[col].median())
        
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        input_scaled = scaler.transform(input_df)
        
        prediction_encoded = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        prediction_decoded = le.inverse_transform(prediction_encoded)
        
        probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(le.classes_, prediction_proba[0])
        }
        
        return jsonify({'prediction': prediction_decoded[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
