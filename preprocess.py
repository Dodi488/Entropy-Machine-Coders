# Este codigo es una version antigua y muy posiblemente lo borarre.

# Step 1: Import pandas and load the datasets
import pandas as pd

# It's good practice to define filenames in variables
toi_file = './TOI_2025.10.04_08.55.28.csv'
k2_file = './k2pandc_2025.10.04_08.55.22.csv'
kepler_file = './cumulative_2025.10.04_08.55.33.csv'

print("Loading datasets...")
# The 'comment' parameter automatically skips the header comments
df_toi = pd.read_csv(toi_file, comment='#')
df_k2 = pd.read_csv(k2_file, comment='#')
df_kepler = pd.read_csv(kepler_file, comment='#')
print("Datasets loaded successfully.")

# --- Step 2: Preprocessing Each Dataset ---
print("Preprocessing datasets...")

# 2a. Preprocess the TESS (TOI) data
toi_processed = df_toi[['tid', 'pl_orbper', 'pl_rade', 'st_teff', 'tfopwg_disp']].copy()
toi_processed.rename(columns={
    'tid': 'star_id',
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius',
    'st_teff': 'stellar_temp',
    'tfopwg_disp': 'disposition'
}, inplace=True)
toi_processed['source'] = 'TESS'
toi_processed['disposition'] = toi_processed['disposition'].apply(lambda x: 'PLANET' if x in ['CP', 'PC'] else 'FALSE POSITIVE')

# 2b. Preprocess the K2 data
k2_processed = df_k2[['tic_id', 'pl_orbper', 'pl_rade', 'st_teff', 'disposition']].copy()
k2_processed.rename(columns={
    'tic_id': 'star_id',
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius',
    'st_teff': 'stellar_temp'
}, inplace=True)
k2_processed['source'] = 'K2'
k2_processed['disposition'] = k2_processed['disposition'].apply(lambda x: 'PLANET' if x == 'CONFIRMED' else 'FALSE POSITIVE')

# 2c. Preprocess the Kepler data
kepler_processed = df_kepler[['kepid', 'koi_period', 'koi_prad', 'koi_steff', 'koi_disposition']].copy()
kepler_processed.rename(columns={
    'kepid': 'star_id',
    'koi_period': 'orbital_period',
    'koi_prad': 'planet_radius',
    'koi_steff': 'stellar_temp',
    'koi_disposition': 'disposition'
}, inplace=True)
kepler_processed['source'] = 'Kepler'
kepler_processed['disposition'] = kepler_processed['disposition'].apply(lambda x: 'PLANET' if x in ['CONFIRMED', 'CANDIDATE'] else 'FALSE POSITIVE')

print("Preprocessing complete.")


# --- Step 3: Fix Data Types and Fuse the Datasets ---
print("Fixing data types for merge...")

# **THE FIX**: Convert 'star_id' columns to a consistent numeric type.
# We use pd.to_numeric with errors='coerce' to turn any non-numeric IDs into 'NaN' (Not a Number).
toi_processed['star_id'] = pd.to_numeric(toi_processed['star_id'], errors='coerce')
k2_processed['star_id'] = pd.to_numeric(k2_processed['star_id'], errors='coerce')

# Drop any rows where the ID could not be converted (they became NaN)
toi_processed.dropna(subset=['star_id'], inplace=True)
k2_processed.dropna(subset=['star_id'], inplace=True)

# Now, ensure they are both integers for a clean merge.
toi_processed['star_id'] = toi_processed['star_id'].astype('int64')
k2_processed['star_id'] = k2_processed['star_id'].astype('int64')


print("Fusing datasets...")
merged_tess_k2 = pd.merge(toi_processed, k2_processed, on='star_id', how='outer', suffixes=('_tess', '_k2'))

# Intelligently combine the columns from the merged data
merged_tess_k2['orbital_period'] = merged_tess_k2['orbital_period_tess'].fillna(merged_tess_k2['orbital_period_k2'])
merged_tess_k2['planet_radius'] = merged_tess_k2['planet_radius_tess'].fillna(merged_tess_k2['planet_radius_k2'])
merged_tess_k2['stellar_temp'] = merged_tess_k2['stellar_temp_tess'].fillna(merged_tess_k2['stellar_temp_k2'])
merged_tess_k2['disposition'] = merged_tess_k2['disposition_tess'].fillna(merged_tess_k2['disposition_k2'])
merged_tess_k2['source'] = merged_tess_k2['source_tess'].fillna(merged_tess_k2['source_k2'])

# Select only the newly created, combined columns
merged_tess_k2 = merged_tess_k2[['star_id', 'orbital_period', 'planet_radius', 'stellar_temp', 'disposition', 'source']]

# Concatenate the Kepler data to the bottom of the merged TESS/K2 data
final_fused_data = pd.concat([merged_tess_k2, kepler_processed], ignore_index=True)

print("Fusing complete.")


# --- Step 4: Final Cleaning of the Fused Dataset ---
print("Cleaning fused dataset...")

# Handle missing numeric values by filling them with the median
final_fused_data['orbital_period'].fillna(final_fused_data['orbital_period'].median(), inplace=True)
final_fused_data['planet_radius'].fillna(final_fused_data['planet_radius'].median(), inplace=True)
final_fused_data['stellar_temp'].fillna(final_fused_data['stellar_temp'].median(), inplace=True)

# Drop any rows that still don't have a disposition label
final_fused_data.dropna(subset=['disposition'], inplace=True)

# Save the final, fused, and cleaned CSV file
output_filename = 'fused_exoplanet_data.csv'
final_fused_data.to_csv(output_filename, index=False)

print(f"Final dataset saved to '{output_filename}'")
print("\nFirst 5 rows of the new fused dataset:")
print(final_fused_data.head())
print("\nLast 5 rows of the new fused dataset (showing Kepler data):")
print(final_fused_data.tail())
print("\nInfo about the final dataset:")
final_fused_data.info()