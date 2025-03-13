import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/intrusion_traffic_data.csv')

print("\n### Data Types ###")
print(df.dtypes)

print("\n### Data Quality Overview ###")
print(df.info())

print("\n### Number of Complete (Non-Null) Values ###")
print(df.count())

print("\n### Number of Null (Missing) Values ###")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['number'])

Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Define outlier mask only in numeric columns 
outlier_mask = (numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))

outlier_counts = outlier_mask.sum()
print("\n### Number of Outliers per Column ###")
print(outlier_counts)

# Remove rows with outliers
df_cleaned = df[~outlier_mask.any(axis=1)]

print(f"\n### Data Shape Before Removing Outliers: {df.shape} ###")
print(f"### Data Shape After Removing Outliers: {df_cleaned.shape} ###")

# Analyze skewness of numerical features
numerical_features = ['Duration', 'PacketCount', 'ByteCount', 'network_packet_size', 'session_duration', 'ip_reputation_score', 'failed_logins']
print("\nSkewness of numerical features:")
skewness_values = {feature: df_cleaned[feature].skew() for feature in numerical_features}
print(tabulate(skewness_values.items(), headers=['Feature', 'Skewness'], tablefmt='pretty', floatfmt=".4f"))

# Visualize the distribution of numerical features before cleaning
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.1)
plt.savefig('Plots/numerical_distributions.png')
plt.show()
plt.close()

# Visualize the distribution of numerical features after cleaning
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_cleaned[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature} (Cleaned)")
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.1)
plt.savefig('Plots/numerical_distributions_cleaned.png')
plt.show()
plt.close()
