import pandas as pd
import numpy as np

# File paths
k2_file = '/home/rodrigo/Desktop/nasa/Entropy-Machine-Coders/raw_data/k2pandc_2025.10.04_08.55.22.csv'
kepler_file = '/home/rodrigo/Desktop/nasa/Entropy-Machine-Coders/raw_data/cumulative_2025.10.04_08.55.33.csv'

print("Loading Kepler and K2 datasets...")
df_k2 = pd.read_csv(k2_file, comment='#')
df_kepler = pd.read_csv(kepler_file, comment='#')
print(f"K2 loaded: {len(df_k2)} rows, {len(df_k2.columns)} columns")
print(f"Kepler loaded: {len(df_kepler)} rows, {len(df_kepler.columns)} columns")

# --- Step 1: Find Common Columns ---
print("\n--- Analyzing Common Features ---")

# Get all columns from both datasets
k2_cols = set(df_k2.columns)
kepler_cols = set(df_kepler.columns)

# Find common columns (these exist in both datasets)
common_cols = k2_cols & kepler_cols
print(f"Common columns between datasets: {len(common_cols)}")
print(f"Sample common columns: {list(common_cols)[:10]}")

# --- Step 2: Identify Disposition Columns ---
k2_disp_col = 'disposition'
kepler_disp_col = 'koi_pdisposition'  # Kepler uses this column name

# --- Step 3: Get All Numerical Features ---
print("\n--- Extracting Numerical Features ---")

# Get numerical columns from K2 (excluding IDs and error columns)
k2_numerical = df_k2.select_dtypes(include=[np.number]).columns.tolist()
exclude_patterns = ['id', 'err', '_err', 'flag', 'rowid', 'name']
k2_features = [col for col in k2_numerical 
               if not any(pattern in col.lower() for pattern in exclude_patterns)]

# Get numerical columns from Kepler
kepler_numerical = df_kepler.select_dtypes(include=[np.number]).columns.tolist()
kepler_features = [col for col in kepler_numerical 
                   if not any(pattern in col.lower() for pattern in exclude_patterns)]

print(f"K2 numerical features: {len(k2_features)}")
print(f"Kepler numerical features: {len(kepler_features)}")

# Find matching features (features that exist in both datasets)
matching_features = list(set(k2_features) & set(kepler_features))
print(f"Matching features: {len(matching_features)}")
print(f"Matching features: {matching_features[:15]}")

# --- Step 4: Create Column Mapping ---
# Map Kepler columns to K2 columns for features that don't match by name
column_mapping = {
    'koi_period': 'pl_orbper',
    'koi_prad': 'pl_rade',
    'koi_steff': 'st_teff',
    'koi_sma': 'pl_orbsmax',
    'koi_teq': 'pl_eqt',
    'koi_insol': 'pl_insol',
    'koi_srad': 'st_rad',
    'koi_smass': 'st_mass',
    'koi_sage': 'st_age',
    'koi_slogg': 'st_logg',
    'koi_smet': 'st_met',
}

# --- Step 5: Process K2 Data ---
print("\n--- Processing K2 Data ---")

# Select disposition and all numerical features
k2_cols_to_keep = [k2_disp_col] + k2_features
k2_processed = df_k2[k2_cols_to_keep].copy()

# Standardize column names
k2_processed.columns = ['disposition'] + k2_features

# Map disposition values
k2_processed['disposition'] = k2_processed['disposition'].apply(
    lambda x: 'CONFIRMED' if str(x).upper() == 'CONFIRMED' else 'CANDIDATE'
)

k2_processed['source'] = 'K2'

# --- Step 6: Process Kepler Data ---
print("--- Processing Kepler Data ---")

# Select disposition and numerical features
kepler_cols_to_keep = [kepler_disp_col] + kepler_features

# Only keep columns that exist
kepler_cols_to_keep = [col for col in kepler_cols_to_keep if col in df_kepler.columns]
kepler_processed = df_kepler[kepler_cols_to_keep].copy()

# Rename columns to match K2 naming
rename_dict = {'koi_pdisposition': 'disposition'}
rename_dict.update(column_mapping)
kepler_processed.rename(columns=rename_dict, inplace=True)

# Map disposition values
kepler_processed['disposition'] = kepler_processed['disposition'].apply(
    lambda x: 'CONFIRMED' if str(x).upper() in ['CONFIRMED', 'CANDIDATE'] else 'CANDIDATE'
)

kepler_processed['source'] = 'Kepler'

# --- Step 7: Align Columns ---
print("\n--- Aligning Columns ---")

# Get common columns between processed datasets
k2_cols = set(k2_processed.columns)
kepler_cols = set(kepler_processed.columns)
final_common_cols = list(k2_cols & kepler_cols)

print(f"Final common columns: {len(final_common_cols)}")
print(f"Features to use: {len(final_common_cols) - 2}")  # -2 for disposition and source

# --- Step 8: Fuse Datasets ---
print("\n--- Fusing Datasets ---")

# Only keep common columns
k2_aligned = k2_processed[final_common_cols]
kepler_aligned = kepler_processed[final_common_cols]

# Concatenate
fused_data = pd.concat([k2_aligned, kepler_aligned], ignore_index=True)

print(f"Fused dataset: {len(fused_data)} rows, {len(fused_data.columns)} columns")

# --- Step 9: Clean Fused Dataset ---
print("\n--- Cleaning Fused Dataset ---")

# Remove disposition column temporarily
disposition_col = fused_data['disposition']
source_col = fused_data['source']
feature_data = fused_data.drop(['disposition', 'source'], axis=1)

# Show feature coverage
print("\nFeature coverage before cleaning:")
for col in feature_data.columns[:10]:
    coverage = feature_data[col].notna().sum() / len(feature_data)
    print(f"  {col}: {coverage*100:.1f}%")

# Fill missing values with median
for col in feature_data.columns:
    if feature_data[col].isna().any():
        feature_data[col].fillna(feature_data[col].median(), inplace=True)

# Reassemble dataset
fused_data = feature_data.copy()
fused_data['disposition'] = disposition_col
fused_data['source'] = source_col

# Drop rows with missing disposition
fused_data.dropna(subset=['disposition'], inplace=True)

# --- Step 10: Save ---
output_filename = 'fused_kepler_k2_data.csv'
fused_data.to_csv(output_filename, index=False)

print(f"\n✓ Final dataset saved to '{output_filename}'")
print(f"✓ Total samples: {len(fused_data)}")
print(f"✓ Total features: {len(fused_data.columns) - 2}")  # -2 for disposition and source

print("\nClass distribution:")
print(fused_data['disposition'].value_counts())

print("\nSource distribution:")
print(fused_data['source'].value_counts())

print("\nFirst 5 rows:")
print(fused_data.head())

print("\nDataset info:")
fused_data.info()