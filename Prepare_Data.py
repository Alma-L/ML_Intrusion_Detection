import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import median_abs_deviation, skew, boxcox

# Suppress FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Create a directory to save the plots
if not os.path.exists('Plots'):
    os.makedirs('Plots')

# Load the dataset
df = pd.read_csv('Data/intrusion_traffic_data.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))

# Check for missing values
print("\nMissing values in the dataset:")
missing_values = df.isnull().sum()
print(tabulate(missing_values.reset_index(), headers=['Column', 'Missing Values'], tablefmt='pretty'))

# Handle missing values (drop or impute)
df_cleaned = df.dropna()  # Drop rows with missing values
print("\nAfter Handling Missing Values:")
print(f"Shape before: ({df.shape[0]:05d}, {df.shape[1]:02d})")
print(f"Shape after: ({df_cleaned.shape[0]:05d}, {df_cleaned.shape[1]:02d})")

# Check for duplicate rows
print("\nNumber of duplicate rows:")
duplicates = df_cleaned.duplicated().sum()
print(tabulate([['Duplicates', duplicates]], headers=['Metric', 'Count'], tablefmt='pretty'))

# Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()
print("\nAfter Removing Duplicates:")
print(f"Shape before: ({df.shape[0]:05d}, {df.shape[1]:02d})")
print(f"Shape after: ({df_cleaned.shape[0]:05d}, {df_cleaned.shape[1]:02d})")

# Check for unique values in categorical columns
categorical_columns = ['Protocol', 'encryption_used', 'browser_type']
print("\nUnique values in categorical columns:")
unique_values = {col: df_cleaned[col].nunique() for col in categorical_columns}
print(tabulate(unique_values.items(), headers=['Column', 'Unique Values'], tablefmt='pretty'))

# Analyze skewness of numerical features
numerical_features = ['Duration', 'PacketCount', 'ByteCount', 'network_packet_size', 'session_duration', 'ip_reputation_score', 'failed_logins']
print("\nSkewness of numerical features (Before Transformation):")
skewness_values_before = {feature: df_cleaned[feature].skew() for feature in numerical_features}
print(tabulate(skewness_values_before.items(), headers=['Feature', 'Skewness'], tablefmt='pretty', floatfmt=".4f"))

# Function to calculate Modified Z-Score
def modified_zscore(series):
    median = np.median(series)
    mad = median_abs_deviation(series, scale='normal')
    return 0.6745 * (series - median) / mad

# Function to remove outliers using Modified Z-Score and IQR
def remove_outliers_combined(df, columns, z_threshold=3.5, iqr_multiplier=1.5):  # Adjusted IQR multiplier
    outlier_mask = pd.Series(False, index=df.index)
    for feature in columns:
        # Modified Z-Score
        mod_z_scores = modified_zscore(df[feature])
        z_outliers = (mod_z_scores > z_threshold) | (mod_z_scores < -z_threshold)
        
        # IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        iqr_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        
        # Combine outliers
        feature_outliers = z_outliers | iqr_outliers
        outlier_mask |= feature_outliers
    df_cleaned = df[~outlier_mask]
    return df_cleaned

# Apply combined method to specific columns
specific_columns = ['session_duration', 'network_packet_size']
df_cleaned = remove_outliers_combined(df_cleaned, specific_columns)

# Verify that no specific outliers remain
print("\nVerifying no specific outliers remain:")
specific_outliers_after_removal = {}
for feature in specific_columns:
    mod_z_scores = modified_zscore(df_cleaned[feature])
    Q1 = df_cleaned[feature].quantile(0.25)
    Q3 = df_cleaned[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR  # Adjusted IQR multiplier
    upper_bound = Q3 + 1.5 * IQR  # Adjusted IQR multiplier
    z_outliers = ((mod_z_scores > 3.5) | (mod_z_scores < -3.5)).sum()
    iqr_outliers = ((df_cleaned[feature] < lower_bound) | (df_cleaned[feature] > upper_bound)).sum()
    specific_outliers_after_removal[feature] = z_outliers + iqr_outliers
    print(f"{feature}: {z_outliers} Z-Score outliers, {iqr_outliers} IQR outliers")

# Print the number of specific outliers remaining after removal
print("\nSpecific Outliers Remaining After Removal:")
specific_outliers_remaining_table = pd.DataFrame(specific_outliers_after_removal.items(), columns=['Column', 'Outliers Remaining'])
print(tabulate(specific_outliers_remaining_table, headers='keys', tablefmt='pretty'))

# Function to cap outliers using IQR
def cap_outliers_iqr(df, feature, iqr_multiplier=1.5):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    df[feature] = df[feature].clip(lower_bound, upper_bound)  # Cap values
    return df

# Cap outliers for 'failed_logins'
df_cleaned = cap_outliers_iqr(df_cleaned, 'failed_logins', iqr_multiplier=1.5)

# Function to fix skewness using Box-Cox or Yeo-Johnson transformation
def fix_skewness(df, columns):
    for feature in columns:
        if df[feature].dtype in ['int64', 'float64']:  # Only apply to numeric columns
            if df[feature].min() > 0:  # Box-Cox requires positive values
                df[feature], _ = boxcox(df[feature] + 1)  # Add 1 to handle zeros
            else:  # Use Yeo-Johnson for negative or zero values
                pt = PowerTransformer(method='yeo-johnson')
                df[feature] = pt.fit_transform(df[feature].values.reshape(-1, 1))
            
            # Ensure non-negative values after transformation
            df[feature] = df[feature] - df[feature].min()  # Shift to make all values non-negative
    return df

# Fix skewness for numerical features
df_cleaned = fix_skewness(df_cleaned, numerical_features)

# Analyze skewness of numerical features (After Transformation)
print("\nSkewness of numerical features (After Transformation):")
skewness_values_after = {feature: df_cleaned[feature].skew() for feature in numerical_features}
print(tabulate(skewness_values_after.items(), headers=['Feature', 'Skewness'], tablefmt='pretty', floatfmt=".4f"))

# Function to count and display the number of remaining outliers
def count_remaining_outliers(df, columns, z_threshold=3.5, iqr_multiplier=1.5):
    outlier_counts = {}
    for feature in columns:
        # Modified Z-Score
        mod_z_scores = modified_zscore(df[feature])
        z_outliers = ((mod_z_scores > z_threshold) | (mod_z_scores < -z_threshold)).sum()
        
        # IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        iqr_outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
        
        # Combine outliers
        total_outliers = z_outliers + iqr_outliers
        outlier_counts[feature] = total_outliers
    
    # Print the number of outliers for each feature
    print("\nNumber of Outliers Remaining for Each Numerical Feature:")
    print(tabulate(outlier_counts.items(), headers=['Feature', 'Outliers Remaining'], tablefmt='pretty'))
    
    return outlier_counts

# Count and display remaining outliers for all numerical features
outlier_counts = count_remaining_outliers(df_cleaned, numerical_features)

# Visualize the distribution of numerical features (After Preprocessing)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_cleaned[feature], kde=True, bins=30, color='green', label='After Cleaning and Transformation')
    plt.title(f"Distribution of {feature} (After Preprocessing)")
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.1)
plt.savefig('Plots/numerical_distributions_after_cleaning_and_transformation.png')
plt.show()
plt.close()

# Visualize the distribution of categorical features (After Cleaning)
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, data=df_cleaned, palette='viridis')
    plt.title(f"Distribution of {col} (After Preprocessing)")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.subplots_adjust(hspace=0.8, bottom=0.1)
plt.savefig('Plots/categorical_distributions_after_cleaning.png')
plt.show()
plt.close()

# Check the distribution of the target variable (After Cleaning)
plt.figure(figsize=(6, 4))
sns.countplot(x='Label', data=df_cleaned, palette='Set2')
plt.title("Distribution of Labels (After Preprocessing)")
plt.subplots_adjust(bottom=0.2)
plt.savefig('Plots/label_distribution_after_cleaning.png')
plt.show()
plt.close()

# Check for outliers using boxplots (After Cleaning and Transformation)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df_cleaned[feature], palette='Set3')
    plt.title(f"Boxplot of {feature} (After Preprocessing)")
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.1)
plt.savefig('Plots/boxplots_after_cleaning_and_transformation.png')
plt.show()
plt.close()

# Correlation matrix for numerical features (After Cleaning and Transformation)
plt.figure(figsize=(12, 8))
corr_matrix_after = df_cleaned[numerical_features].corr()
sns.heatmap(corr_matrix_after, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Numerical Features (After Preprocessing)")
plt.subplots_adjust(bottom=0.2)
plt.savefig('Plots/correlation_matrix_after_cleaning_and_transformation.png')
plt.show()
plt.close()

# Preprocessing: Encode categorical variables and prepare features/target
df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_columns, drop_first=True)  # One-hot encoding

# Drop non-numeric columns that are not useful for modeling
non_numeric_columns = ['SourceIP', 'DestinationIP', 'session_id', 'protocol_type']
df_cleaned = df_cleaned.drop(columns=non_numeric_columns)

# Prepare features and target
X = df_cleaned.drop('Label', axis=1)  # Features
y = df_cleaned['Label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the distribution of the target variable before SMOTE
print("\nDistribution of Labels Before SMOTE:")
before_smote_distribution = y_train.value_counts().reset_index()
before_smote_distribution.columns = ['Label', 'Count']
print(tabulate(before_smote_distribution, headers='keys', tablefmt='pretty'))

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Check the distribution of the target variable after SMOTE
print("\nDistribution of Labels After SMOTE:")
after_smote_distribution = pd.Series(y_train_res).value_counts().reset_index()
after_smote_distribution.columns = ['Label', 'Count']
print(tabulate(after_smote_distribution, headers='keys', tablefmt='pretty'))

# Create a table to compare the distribution before and after SMOTE
comparison_table = pd.merge(before_smote_distribution, after_smote_distribution, on='Label', suffixes=('_Before', '_After'))
print("\nComparison of Class Distribution Before and After SMOTE:")
print(tabulate(comparison_table, headers='keys', tablefmt='pretty'))

# Compare original dataset and SMOTE-applied dataset
original_stats = {
    'Dataset': ['Original', 'After SMOTE'],
    'Shape': [X_train.shape, X_train_res.shape],
    'Class Distribution': [y_train.value_counts().to_dict(), pd.Series(y_train_res).value_counts().to_dict()]
}
print("\nComparison of Original Dataset and SMOTE-Applied Dataset:")
print(tabulate(original_stats, headers='keys', tablefmt='pretty'))