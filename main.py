# Step 1: Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# --- Step 2: Load the Training and Testing Datasets ---
print("Loading datasets...")
train_df = pd.read_csv('./fused_kepler_k2_data.csv')
test_df_raw = pd.read_csv('./raw_data/TOI_2025.10.04_08.55.28.csv', comment='#')
print("Datasets loaded successfully.")


# --- Step 3: Prepare the Training Data ---
print("Preparing training data...")
X_train = train_df.drop(['star_id', 'disposition', 'source'], axis=1)
y_train_labels = train_df['disposition']
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)


# --- Step 4: Train the AI Model ---
print("Training model on Kepler and K2 data...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Model training complete.")


# --- Step 5: Preprocess the TESS Data for Use ---
print("Preprocessing TESS data...")
test_df = test_df_raw[['tid', 'pl_orbper', 'pl_rade', 'st_teff', 'tfopwg_disp']].copy()
test_df.rename(columns={
    'tid': 'star_id',
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius',
    'st_teff': 'stellar_temp',
    'tfopwg_disp': 'disposition'
}, inplace=True)
test_df.dropna(inplace=True)
test_df['disposition'] = test_df['disposition'].apply(lambda x: 'PLANET' if x in ['CP', 'PC'] else 'FALSE POSITIVE')


# --- Step 6: Prepare the TESS Features ---
# We keep the features and true labels separate for our examples
X_test = test_df.drop(['star_id', 'disposition'], axis=1)
y_test_labels = test_df['disposition']


# --- (Optional) Step 7: Full Evaluation on TESS data ---
print("\n--- Full Evaluation on TESS Data ---")
X_test_scaled = scaler.transform(X_test)
y_test = le.transform(y_test_labels)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy on TESS data: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# --- Step 8: Function to Check an Individual Row ---
print("\n--- Individual Exoplanet Prediction ---")

def check_exoplanet(row_data):
    """
    Takes a single row of data (as a pandas Series or DataFrame),
    processes it, and predicts if it's an exoplanet.
    """
    # Ensure the input is a DataFrame with the correct feature names
    if isinstance(row_data, pd.Series):
        row_data = row_data.to_frame().T
    
    # 1. Scale the data using the already-fitted scaler
    row_scaled = scaler.transform(row_data)
    
    # 2. Make a prediction (returns [0] or [1])
    prediction_encoded = model.predict(row_scaled)
    
    # 3. Decode the prediction back to 'PLANET' or 'FALSE POSITIVE'
    prediction_decoded = le.inverse_transform(prediction_encoded)
    
    # 4. Return the result
    return prediction_decoded[0]

# --- Example Usage ---

# Example 1: Let's test the 5th object from our TESS dataset
print("Checking a sample row from TESS...")
sample_row_features = X_test.iloc[[5]]
true_label = y_test_labels.iloc[5]

prediction = check_exoplanet(sample_row_features)

print(f"Data for object being checked:\n{sample_row_features}")
print(f"\nModel Prediction: '{prediction}'")
print(f"Actual Label:     '{true_label}'")

print("-" * 30)

# Example 2: Let's test the 20th object
print("\nChecking another sample row...")
sample_row_features_2 = X_test.iloc[[20]]
true_label_2 = y_test_labels.iloc[20]

prediction_2 = check_exoplanet(sample_row_features_2)

print(f"Data for object being checked:\n{sample_row_features_2}")
print(f"\nModel Prediction: '{prediction_2}'")
print(f"Actual Label:     '{true_label_2}'")