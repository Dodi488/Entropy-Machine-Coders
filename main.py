# Step 1: Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Load the Training and Testing Datasets ---
print("Loading datasets...")
# The fused Kepler+K2 data is our training set
train_df = pd.read_csv('./fused_kepler_k2_data.csv')
# The TESS data is our test set
test_df_raw = pd.read_csv('./raw_data/TOI_2025.10.04_08.55.28.csv', comment='#')
print("Datasets loaded successfully.")


# --- Step 3: Prepare the Training Data ---
print("Preparing training data...")
# Separate features from labels in the training set
X_train = train_df.drop(['star_id', 'disposition', 'source'], axis=1)
y_train_labels = train_df['disposition']

# Encode the text labels ('PLANET', 'FALSE POSITIVE') into numbers (1, 0)
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)

# Scale the training features to a range between 0 and 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)


# --- Step 4: Train the AI Model ---
print("Training model on Kepler and K2 data...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Model training complete.")


# --- Step 5: Preprocess the TESS Data for Testing ---
print("Preprocessing TESS data for testing...")
# Apply the exact same preprocessing steps as before
test_df = test_df_raw[['tid', 'pl_orbper', 'pl_rade', 'st_teff', 'tfopwg_disp']].copy()
test_df.rename(columns={
    'tid': 'star_id',
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius',
    'st_teff': 'stellar_temp',
    'tfopwg_disp': 'disposition'
}, inplace=True)

# Drop rows with missing values to ensure the test set is clean
test_df.dropna(inplace=True)

# Standardize the disposition labels
test_df['disposition'] = test_df['disposition'].apply(lambda x: 'PLANET' if x in ['CP', 'PC'] else 'FALSE POSITIVE')


# --- Step 6: Prepare the TESS Test Data ---
# Separate the features from the true labels
X_test = test_df.drop(['star_id', 'disposition'], axis=1)
y_test_labels = test_df['disposition']

# IMPORTANT: Use the SAME encoder and scaler that were fitted on the training data
# This ensures the data is transformed in the exact same way.
y_test = le.transform(y_test_labels)
X_test_scaled = scaler.transform(X_test)


# --- Step 7: Make Predictions and Evaluate the Model on TESS Data ---
print("\n--- Evaluating Model on Unseen TESS Data ---")

# Use the trained model to make predictions
y_pred = model.predict(X_test_scaled)
print("Checar el y_pred")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on TESS data: {accuracy * 100:.2f}%")

# Print a detailed classification report
print("\nClassification Report:")
target_names = le.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

# Display a confusion matrix to visualize performance
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix for a clear visual
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Model Performance on TESS Dataset')
plt.show()