import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)  # This allows a frontend to communicate with this backend

# --- 1. Train the AI Model ONCE on Startup (using logic from excel_cosa.py) ---

print("--- Initializing Model Training ---")

# --- Step 1a: Define File and Column Names ---
TRAINING_DATA_FILE = './raw_data/k2pandc_2025.10.04_08.55.22.csv'
DISPOSITION_COL = "disposition"
ORBITAL_PERIOD_COL = "pl_orbper"
PLANET_RADIUS_COL = "pl_rade"
STELLAR_TEMP_COL = "st_teff"

# --- Step 1b: Load the Training Dataset ---
print(f"Loading training data from '{TRAINING_DATA_FILE}'...")
try:
    # Load the raw Kepler/K2 data for training
    train_df_raw = pd.read_csv(TRAINING_DATA_FILE, comment='#')
except FileNotFoundError:
    print(f"FATAL ERROR: Training data file not found at '{TRAINING_DATA_FILE}'.")
    print("Please make sure the file exists in the correct directory.")
    sys.exit()
print("Dataset loaded successfully.")

# --- Step 1c: Prepare the Training Data ---
print("Preparing training data...")
# Select only the columns we need for the model
train_df = train_df_raw[[ORBITAL_PERIOD_COL, PLANET_RADIUS_COL, STELLAR_TEMP_COL, DISPOSITION_COL]].copy()

# Rename columns to be more user-friendly
train_df.rename(columns={
    ORBITAL_PERIOD_COL: 'orbital_period',
    PLANET_RADIUS_COL: 'planet_radius',
    STELLAR_TEMP_COL: 'stellar_temp',
    DISPOSITION_COL: 'disposition'
}, inplace=True)

# Handle missing numerical data by filling with the median value of each column
for col in train_df.select_dtypes(include=np.number).columns:
    train_df[col].fillna(train_df[col].median(), inplace=True)

# Drop any remaining rows that might have missing data (e.g., in the disposition column)
train_df.dropna(inplace=True)

# Separate features (X) from the target label (y)
X_train = train_df.drop('disposition', axis=1)
y_train_labels = train_df['disposition']

# Encode the text labels ('CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE') into numbers
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)

# Create and fit the scaler on the entire training data. We will reuse this exact scaler for predictions.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("Data preparation complete.")

# --- Step 1d: Train the Model ---
print("Training RandomForest model...")
# Initialize and train the model on the full, prepared dataset
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Model trained and ready to accept requests!")


# --- 2. Define API Endpoints ---

# This endpoint serves the main HTML page (you would need an 'index.html' in a 'templates' folder)
@app.route('/')
def home():
    """Serves the frontend user interface."""
    return render_template('index.html')

# This endpoint handles prediction requests from the frontend
@app.route('/predict', methods=['POST'])
def predict():
    """Receives input data, makes a prediction, and returns it."""
    try:
        # Get the JSON data sent from the frontend
        data = request.get_json()

        # Create a pandas DataFrame from the input data
        # The column order must match the order used during training
        input_df = pd.DataFrame([[data['orbital_period'],
                                  data['planet_radius'],
                                  data['stellar_temp']]],
                                columns=['orbital_period', 'planet_radius', 'stellar_temp'])

        # Use the *same scaler* that was fitted on the training data to transform the new input
        input_scaled = scaler.transform(input_df)

        # Use the trained model to make a prediction
        prediction_encoded = model.predict(input_scaled)

        # Decode the numeric prediction back to its original text label (e.g., 'CONFIRMED')
        prediction_decoded = le.inverse_transform(prediction_encoded)

        # Return the result in JSON format
        return jsonify({'prediction': prediction_decoded[0]})

    except Exception as e:
        # Handle potential errors, like missing keys in the input data
        return jsonify({'error': str(e)}), 400


# --- 3. Run the Server ---
if __name__ == '__main__':
    # This will start a local development server on port 5000
    # The 'debug=True' flag allows the server to auto-reload when you save changes
    app.run(debug=True, port=5000)  