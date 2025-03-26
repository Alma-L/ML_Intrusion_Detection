import pandas as pd

# Load datasets and merge them
merged_df_cleaned = pd.merge(
    pd.read_csv("Data/intrusion_raw_data.csv"),
    pd.read_csv("Data/network_traffic_raw_data.csv", sep=";"),
    left_on="Protocol", right_on="protocol_type", how="inner"
).dropna()

merged_df_cleaned = merged_df_cleaned.sample(n=90000)

merged_df_cleaned.to_csv("Data/intrusion_traffic_data.csv", index=False)

print("Cleaned dataset saved as intrusion_traffic_data.csv with sampled rows.")
