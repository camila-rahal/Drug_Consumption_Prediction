import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore


# fetch dataset
drug_consumption_quantified = fetch_ucirepo(id=373)

# data (as pandas dataframes)
X = drug_consumption_quantified.data.features
y = drug_consumption_quantified.data.targets

# metadata
print(drug_consumption_quantified.metadata)

# variable information
print(drug_consumption_quantified.variables)

# Print features (X)
print("\nFeatures (X):")
print(X)

# Print targets (y)
print("\nTargets (y):")
print(y)

"""# **Violin plot:**"""

# Change the name of column impulsive
X.rename(columns={'impuslive': 'impulsive'}, inplace=True)

# Check the drug use for each target variable on the dataset
for column in y.columns:
    print(f"{column}:\n", y[column].value_counts())


# Reorder the classification of drug use from CL0 to CL6
ordered_categories = ['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6']
for column in y.columns:
    y.loc[:, column] = pd.Categorical(y[column], categories=ordered_categories, ordered=True)

# Violin plots for all columns
plt.figure(figsize=(18, 10))
for i, column in enumerate(y.columns):
    plt.subplot(4, 5, i + 1)
    sns.violinplot(y=y[column])
    plt.title(f'{column}')
plt.tight_layout()
plt.savefig('violin_plots_all_columns.png', dpi=300)  # Save plot
plt.show()


"""#**Merging X and y:**

"""

# Check if 'index' is already the DataFrame index
if 'index' not in X.columns and 'index' not in y.columns:
    # Merge using the index directly if possible
    merged_df = pd.merge(X, y, left_index=True, right_index=True)
else:
    # Merge on 'index' and prevent suffixes
    merged_df = pd.merge(X, y, on='index', suffixes=('', ''))

# Display the merged DataFrame
print("Merged DataFrame:")
print(merged_df.head())

"""#**Droping columns and excluding rows with "users" of Semeron:**"""

# Assuming 'merged_df' is your merged DataFrame containing the Semeron column

# Filter rows where the Semeron column equals 'C0'
filtered_df = merged_df[merged_df['semer'] == 'CL0']

# Optionally, check the number of rows dropped
rows_dropped = len(merged_df) - len(filtered_df)
print(f"Number of rows dropped: {rows_dropped}")

# Drop the Semeron and Chocolate column from the filtered DataFrame
filtered_df = filtered_df.drop(columns=['semer', 'choc', 'alcohol', 'caff', 'cannabis', 'nicotine'])


# Display the resulting DataFrame
print("Filtered DataFrame:")
print(filtered_df.head())

"""# **Designing the Target Variable:**
- ## Consumer vs Non-consumer
"""

# List of drug columns in the dataset
drug_columns = [
    'amphet', 'amyl', 'benzos', 'heroin',
    'coke', 'crack', 'ecstasy', 'ketamine',
    'meth', 'mushrooms', 'vsa', 'legalh', 'lsd',
]

# Classify participants into User (1) and Non-user (0)
# Non-user: All drug usage levels are CL0 CL1, CL2
# User: Any drug usage level is CL3 or higher
filtered_df['Target'] = filtered_df[drug_columns].apply(
    lambda row: 1 if any(value not in ['CL0', 'CL1', 'CL2'] for value in row) else 0, axis=1
)

# Drop the drug usage columns
filtered_df = filtered_df.drop(columns=drug_columns)

# Display the first few rows of the updated DataFrame
print("DataFrame with Target Variable:")
print(filtered_df[['Target']].head())

# Verify the classification
print("Target Variable Distribution:")
print(filtered_df['Target'].value_counts())

print(filtered_df.columns)

"""---

# **Exploratory Data Analysis:**


---

- ## Correlation Matrix

A positive correlation indicates that as a feature's value increases, the likelihood of the target being 1 (e.g., "User") also increases, such as impulsivity with a correlation of 0.6. Conversely, a negative correlation means that as the feature's value increases, the likelihood of the target being 1 decreases, such as education level with a correlation of -0.4. Features with correlations near 0 show little to no linear relationship with the target and may not be useful for prediction.
"""

# Compute correlation matrix for numerical columns
correlation_matrix = filtered_df.corr()

# Plot the correlation matrix with color scale fixed to [-1, 1]
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    vmin=-1,  # Minimum value for the color scale
    vmax=1    # Maximum value for the color scale
)
plt.title("Correlation Matrix (Scaled -1 to 1)")
plt.show()

# List of numerical columns for distribution plots
numerical_columns = ['age', 'ethnicity', 'country', 'education', 'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss']

# Plot distributions for each numerical column
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(filtered_df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Skewness and Kurtosis
print("Skewness:\n", filtered_df.skew())
print("Kurtosis:\n", filtered_df.kurt())

'''Outliers count'''

# Identify outliers using z-scores
z_scores = filtered_df[['age', 'gender', 'education', 'country', 'nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss']].apply(zscore)
outliers = (z_scores.abs() > 3).sum()  # Count outliers
print("Number of Outliers Detected in Each Variable:")
print(outliers)

"""# **Table 1: Using metadata to present true values and create the Demographic Table with the distribution between consumer and non-consumer:**"""

print(filtered_df.columns)

# Metadata mappings based on the information provided
age_mapping = {
    -0.95197: '18-24 years',
    -0.07854: '25-34 years',
    0.49788: '35-44 years',
    1.09449: '45-54 years',
    1.82213: '55-64 years',
    2.59171: '65+ years'
}

gender_mapping = {
    0.48246: 'Female',
    -0.48246: 'Male'
}

education_mapping = {
    -2.43591: 'Left school before 16 years',
    -1.73790: 'Left school at 16 years',
    -1.43719: 'Left school at 17 years',
    -1.22751: 'Left school at 18 years',
    -0.61113: 'Some college or university (no certificate)',
    -0.05921: 'Professional certificate/diploma',
    0.45468: 'University degree',
    1.16365: 'Master’s degree',
    1.98437: 'Doctorate degree'
}

country_mapping = {
    -0.09765: 'Australia',
    0.24923: 'Canada',
    -0.46841: 'New Zealand',
    -0.28519: 'Other',
    0.21128: 'Republic of Ireland',
    0.96082: 'UK',
    -0.57009: 'USA'
}

ethnicity_mapping = {
    -0.50212: 'Asian',
    -1.10702: 'Black',
    1.90725: 'Mixed-Black/Asian',
    0.12600: 'Mixed-White/Asian',
    -0.22166: 'Mixed-White/Black',
    0.11440: 'Other',
    -0.31685: 'White'
}

filtered_df['age_category'] = filtered_df['age'].map(age_mapping)
filtered_df['gender_category'] = filtered_df['gender'].map(gender_mapping)
filtered_df['education_category'] = filtered_df['education'].map(education_mapping)
filtered_df['country_category'] = filtered_df['country'].map(country_mapping)
filtered_df['ethnicity_category'] = filtered_df['ethnicity'].map(ethnicity_mapping)

# Apply mappings to replace values with their real-world labels
# Function to summarize demographics by the target variable
def summarize_demographics_by_target(df, target_column, demographic_columns):
    # Initialize a summary dictionary
    summary = {}

    for col in demographic_columns:
        grouped = filtered_df.groupby(target_column)[col].value_counts().unstack(fill_value=0)
        summary[col] = grouped

    # Combine all demographic summaries into one DataFrame
    concise_summary = pd.concat(summary, axis=1)

    return concise_summary

# Define demographic columns to summarize
demographic_columns = ['age_category', 'gender_category', 'education_category',
                       'country_category', 'ethnicity_category']

# Create a concise summary table
concise_summary_table = summarize_demographics_by_target(filtered_df, 'Target', demographic_columns)

# Transpose for better readability
concise_summary_table = concise_summary_table.T

# Display the summarized table
print("Demographics Summary Table by Target:")
print(concise_summary_table)

# Save the concise summary to an HTML file
html_output_path = "Demographics_Summary_By_Target.html"
concise_summary_table.to_html(html_output_path)
print(f"Concise summary saved to {html_output_path}")

# Save the concise summary to a CSV file
csv_output_path = "Demographics_Summary_By_Target.csv"
concise_summary_table.to_csv(csv_output_path)
print(f"Concise summary saved to {csv_output_path}")

# List of columns to drop
columns_to_drop = ['age_category', 'gender_category', 'education_category',
                   'country_category', 'ethnicity_category']

# Create summary statistics for numerical features
numerical_summary = filtered_df.describe().T  # Transpose for better readability
print("Summary Statistics for Numerical Features:")
print(numerical_summary)

# Save the concise summary to an HTML file
html_output_2_path = "Summary_Statistics.html"
numerical_summary.to_html(html_output_2_path)
print(f"Summary statistics saved to {html_output_2_path}")


# Total Frequency table for categorical variables
categorical_columns = ['age_category', 'gender_category', 'education_category', 'country_category', 'ethnicity_category']
for col in categorical_columns:
    print(f"Frequency Table for {col}:")
    print(filtered_df[col].value_counts())

# Drop the columns
filtered_df = filtered_df.drop(columns=columns_to_drop, errors='ignore')

# Verify the DataFrame after dropping
print("Columns after dropping:")
print(filtered_df.columns)

"""# **Oversampling the minority class with SMOTE:**"""

# Separate features and target
X = filtered_df.drop(columns=['Target'])
y = filtered_df['Target']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verify the new class distribution
print("New Target Distribution:")
print(pd.Series(y_resampled).value_counts())

# Combine resampled features and target into a single DataFrame
balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                         pd.DataFrame(y_resampled, columns=['Target'])], axis=1)

# Display the balanced DataFrame
print(balanced_df.head())

# Save the balanced dataset to a CSV file
balanced_df.to_csv('balanced_dataset.csv', index=False)
print("Balanced dataset saved to 'balanced_dataset.csv'")
