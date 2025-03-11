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

#  We find outliers only in numeric columns (e.g., not in the 'Protocol' column)
outlier_mask = (numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))

outlier_counts = outlier_mask.sum()
print("\n### Number of Outliers per Column ###")
print(outlier_counts)


# Analyze skewness of numerical features
numerical_features = ['Duration', 'PacketCount', 'ByteCount', 'network_packet_size', 'session_duration', 'ip_reputation_score', 'failed_logins']
print("\nSkewness of numerical features:")
skewness_values = {feature: df[feature].skew() for feature in numerical_features}
print(tabulate(skewness_values.items(), headers=['Feature', 'Skewness'], tablefmt='pretty', floatfmt=".4f"))
