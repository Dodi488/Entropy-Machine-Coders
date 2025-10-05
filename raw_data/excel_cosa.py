import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- Step 1: Load the Training and Testing Datasets ---
print("Loading datasets...")
# Load the dataset from the research paper for training
try:
    train_df_raw = pd.read_csv('/home/rodrigo/Desktop/nasa/Entropy-Machine-Coders/raw_data/cumulative_2025.10.04_08.55.33.csv', comment="#")
except FileNotFoundError:
    print("Ensure 'cumulative_2025.10.04_08.55.33.csv' is in the same directory.")
    exit()

# Load your raw TESS data for prediction
test_df_raw = pd.read_csv('./raw_data/TOI_2025.10.04_08.55.28.csv', comment='#')
print("Datasets loaded successfully.")


# --- Step 2: Prepare the Training Data Based on the Research Paper ---
print("Preparing training data according to the paper's methodology...")

# 2a. Remove columns as specified in the paper
columns_to_remove = [
    'rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition',
    'koi_score', 'koi_teq_err1', 'koi_teq_err2'
]
train_df = train_df_raw.drop(columns=columns_to_remove, errors='ignore')

train_df.to_csv("~/Desktop/nasa/Entropy-Machine-Coders/purbeaaaa.csv")

# 2b. Filter for 'CONFIRMED' and 'CANDIDATE' dispositions
dispositions_to_keep = ['CONFIRMED', 'CANDIDATE']
train_df = train_df[train_df['koi_disposition'].isin(dispositions_to_keep)]

# 2c. Select and rename features that are common with the TESS data
# This is crucial for the model to work on your prediction data
feature_map = {
    'koi_period': 'orbital_period',
    'koi_prad': 'planet_radius',
    'koi_steff': 'stellar_temp'
}
# Also include the target column
train_df = train_df[list(feature_map.keys()) + ['koi_disposition']].copy()
train_df.rename(columns=feature_map, inplace=True)

# 2d. Handle missing values
# Fill NaNs with the median value of each column for robustness
for col in train_df.select_dtypes(include=np.number).columns:
    train_df[col].fillna(train_df[col].median(), inplace=True)
train_df.dropna(inplace=True) # Drop any remaining non-numeric NaNs

# 2e. Prepare features (X) and labels (y)
X_train = train_df.drop('koi_disposition', axis=1)
y_train_labels = train_df['koi_disposition']

# 2f. Encode labels and Scale features
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels) # CONFIRMED -> 0, CANDIDATE -> 1

# Use StandardScaler as per the paper's methodology
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("Training data prepared successfully.")


# --- Step 3: Train the AI Model ---
print("Training model on preprocessed Kepler data...")
# Using RandomForest as it's one of the ensemble methods from the paper
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Model training complete.")


# --- Step 4: Preprocess the TESS Data for Prediction ---
print("Preprocessing TESS data...")
# Select the same columns as used in training
test_df = test_df_raw[['tid', 'pl_orbper', 'pl_rade', 'st_teff', 'tfopwg_disp']].copy()
test_df.rename(columns={
    'tid': 'star_id',
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius',
    'st_teff': 'stellar_temp',
    'tfopwg_disp': 'disposition'
}, inplace=True)
test_df.dropna(inplace=True)

# Standardize disposition labels for evaluation
test_df['disposition_standardized'] = test_df['disposition'].apply(lambda x: 'CANDIDATE' if x in ['CP', 'PC'] else 'CONFIRMED')


# --- Step 5: Prepare the TESS Features for Evaluation ---
X_test = test_df.drop(['star_id', 'disposition', 'disposition_standardized'], axis=1)
y_test_labels = test_df['disposition_standardized']

# --- Step 6: Full Evaluation on TESS data ---
print("\n--- Full Evaluation on TESS Data ---")
X_test_scaled = scaler.transform(X_test)
y_test = le.transform(y_test_labels)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy on TESS data: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# --- Step 7: Function to Check an Individual Row ---
print("\n--- Individual Exoplanet Prediction ---")
def check_exoplanet(row_data):
    """
    Takes a single row of data (as a pandas Series), processes it,
    and predicts if it's an exoplanet.
    """
    if isinstance(row_data, pd.Series):
        row_data = row_data.to_frame().T

    # 1. Scale the data using the already-fitted scaler
    row_scaled = scaler.transform(row_data)

    # 2. Make a prediction (returns [0] or [1])
    prediction_encoded = model.predict(row_scaled)

    # 3. Decode the prediction back to 'CONFIRMED' or 'CANDIDATE'
    prediction_decoded = le.inverse_transform(prediction_encoded)

    return prediction_decoded[0]

# --- Step 8: Apply Prediction to each row of TESS data ---
# Create final_data with the features needed for prediction
final_data = test_df_raw[['tid', 'pl_orbper', 'pl_rade', 'st_teff', 'tfopwg_disp']].copy()
final_data.dropna(inplace=True)
final_data['is_exoplanet_candidate'] = ''

# Iterate through each row and apply the check_exoplanet function
for index, row in final_data.iterrows():
    # Create a Series with only the features used in training
    features = row[['pl_orbper', 'pl_rade', 'st_teff']]
    result = check_exoplanet(features)
    final_data.at[index, 'is_exoplanet_candidate'] = 'yes' if result == 'CANDIDATE' else 'no'

# Save to CSV
output_path = "tess_predictions.csv"
final_data.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")