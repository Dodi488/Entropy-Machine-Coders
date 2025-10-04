# Solo combinar los kepplers para el entrenamiento del modelo.

# Step 1: Import pandas and load the datasets
import pandas as pd

# It's good practice to define filenames in variables
k2_file = ('/home/rodrigo/Desktop/nasa/Entropy-Machine-Coders/raw_data/k2pandc_2025.10.04_08.55.22.csv')
kepler_file = ('/home/rodrigo/Desktop/nasa/Entropy-Machine-Coders/raw_data/cumulative_2025.10.04_08.55.33.csv')

print("Loading Kepler and K2 datasets...")
# The 'comment' parameter automatically skips the header comments
df_k2 = pd.read_csv(k2_file, comment='#')
df_kepler = pd.read_csv(kepler_file, comment='#')
print("Datasets loaded successfully.")

# --- Step 2: Preprocessing Each Dataset ---
print("Preprocessing datasets...")

# 2a. Preprocess the K2 data
k2_processed = df_k2[['tic_id', 'pl_orbper', 'pl_rade', 'st_teff', 'disposition']].copy()
k2_processed.rename(columns={
    'tic_id': 'star_id',
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius',
    'st_teff': 'stellar_temp'
}, inplace=True)
k2_processed['source'] = 'K2'
# Standardize disposition: 'CONFIRMED' is treated as a planet.
k2_processed['disposition'] = k2_processed['disposition'].apply(lambda x: 'PLANET' if x == 'CONFIRMED' else 'FALSE POSITIVE')


# 2b. Preprocess the Kepler data
kepler_processed = df_kepler[['kepid', 'koi_period', 'koi_prad', 'koi_steff', 'koi_disposition']].copy()
kepler_processed.rename(columns={
    'kepid': 'star_id',
    'koi_period': 'orbital_period',
    'koi_prad': 'planet_radius',
    'koi_steff': 'stellar_temp',
    'koi_disposition': 'disposition'
}, inplace=True)
kepler_processed['source'] = 'Kepler'
# Standardize disposition: 'CONFIRMED' and 'CANDIDATE' are treated as planets.
kepler_processed['disposition'] = kepler_processed['disposition'].apply(lambda x: 'PLANET' if x in ['CONFIRMED', 'CANDIDATE'] else 'FALSE POSITIVE')

print("Preprocessing complete.")


# --- Step 3: Fuse the Datasets ---
print("Fusing Kepler and K2 datasets...")
# Since Kepler and K2 don't share a common ID, we concatenate them (stack them).
fused_data = pd.concat([k2_processed, kepler_processed], ignore_index=True)
print("Fusing complete.")


# --- Step 4: Final Cleaning of the Fused Dataset ---
print("Cleaning fused dataset...")

# Handle missing numeric values by filling them with the median of their respective column.
fused_data['orbital_period'].fillna(fused_data['orbital_period'].median(), inplace=True)
fused_data['planet_radius'].fillna(fused_data['planet_radius'].median(), inplace=True)
fused_data['stellar_temp'].fillna(fused_data['stellar_temp'].median(), inplace=True)

# Drop any rows that still don't have a disposition label
fused_data.dropna(subset=['disposition'], inplace=True)

# Save the final, fused, and cleaned CSV file
output_filename = 'fused_kepler_k2_data.csv'
fused_data.to_csv(output_filename, index=False)

print(f"Final dataset saved to '{output_filename}'")
print("\nFirst 5 rows of the new fused dataset:")
print(fused_data.head())
print("\nLast 5 rows of the new fused dataset:")
print(fused_data.tail())
print("\nInfo about the final dataset:")
fused_data.info()