# TODO: Evaluate model performance using modern algorithms like SVM (Support Vector Machine) 
#       based on the previous analysis. Compare metrics with current approaches and perform
#       hyperparameter tuning for optimal results.
import pandas as pd

# --------------------------------- Load & Prepare Data --------------------------------- #
data = pd.read_csv("Data/cleaned_intrusion_traffic_data.csv")

# Drop rows with missing target or key features
df = data.dropna(subset=['target_column'])

# Split features and target
X = df.drop('target_column', axis=1)
y = df['target_column']