import pandas as pd

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
