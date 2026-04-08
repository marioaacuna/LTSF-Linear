import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the data
root_path = './DATA_MA'
data_path = 'xNormF_PCA_s.csv'
df = pd.read_csv(os.path.join(root_path, data_path))

# Store the date column
date_column = df.iloc[:, 0]

# Get all feature columns (F1 to F66)
feature_columns = df.iloc[:, 1:]

# Normalize each feature to [-1, 1]
normalized_features = pd.DataFrame()
for column in feature_columns.columns:
    series = feature_columns[column]
    min_val = series.min()
    max_val = series.max()
    range_val = max_val - min_val
    
    # Apply normalization formula: 2 * ((x - min) / range) - 1
    normalized_features[column] = 2 * ((series - min_val) / range_val) - 1

# Verify normalization
print("Verification of normalized ranges:")
for column in normalized_features.columns:
    print(f"{column} - Min: {normalized_features[column].min():.3f}, Max: {normalized_features[column].max():.3f}")

# Create final dataframe with date and normalized features
final_df = pd.concat([date_column, normalized_features], axis=1)

# Save normalized data
output_path = os.path.join(root_path, 'normalized_dataF_PCA_s.csv')
final_df.to_csv(output_path, index=False)
print(f"\nNormalized data saved to: {output_path}")

# Optional: Plot histograms for a few features to verify distribution
plt.figure(figsize=(15, 5))

# Original data (first 3 features)
plt.subplot(1, 2, 1)
for column in feature_columns.columns[:3]:
    plt.hist(feature_columns[column], bins=20, alpha=0.5, label=column)
plt.title('Original Data Distribution (First 3 Features)')
plt.legend()

# Normalized data (first 3 features)
plt.subplot(1, 2, 2)
for column in normalized_features.columns[:3]:
    plt.hist(normalized_features[column], bins=50, alpha=0.5, label=column)
plt.title('Normalized Data Distribution (First 3 Features)')
plt.legend()

plt.tight_layout()
plt.show()
