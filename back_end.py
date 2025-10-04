import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)  # This allows the frontend to communicate with the backend

# --- 1. Train the AI Model ONCE on Startup ---
print("Loading training data (Kepler & K2)...")
train_df = pd.read_csv('fused_kepler_k2_data.csv')

print("Preparing training data...")
# Separate features from labels
X_train = train_df.drop(['star_id', 'disposition', 'source'], axis=1)
y_train_labels = train_df['disposition']

# Encode the text labels ('PLANET', 'FALSE POSITIVE') into numbers (1, 0)
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)

# Create and fit the scaler on the training data. We will reuse this exact scaler.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training model...")
# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Model trained and ready!")


# --- 2. Define API Endpoints ---

# This endpoint serves the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# This endpoint handles prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    fused_df = pd.read_csv('fused_kepler_k2_data.csv')
    
    star_id = data.get('star_id')
    if star_id and star_id in fused_df['star_id'].values:
        star_data = fused_df[fused_df['star_id'] == star_id].iloc[0]
        input_df = pd.DataFrame([[star_data['orbital_period'], 
                                star_data['planet_radius'], 
                                star_data['stellar_temp']]], 
                              columns=['orbital_period', 'planet_radius', 'stellar_temp'])
    else:
        input_df = pd.DataFrame([[data['orbital_period'], 
                                data['planet_radius'], 
                                data['stellar_temp']]], 
                              columns=['orbital_period', 'planet_radius', 'stellar_temp'])
    
    input_scaled = scaler.transform(input_df)
    prediction_encoded = model.predict(input_scaled)
    prediction_decoded = le.inverse_transform(prediction_encoded)

    return jsonify({'prediction': prediction_decoded[0]})

# --- 3. Run the Server ---
if __name__ == '__main__':
    # This will start a local development server
    app.run(debug=True, port=5000)
