####
####
#### PLEASE READ DISCLAIMER BELOW:
#### THIS CODE WAS NOT USED 
#### This was the approach we first had to identify the true label of the demographic information, back then we didn't know we could 
#### access through the metadata (as an output), so we design this code to assume the demographic labels according to the values. 
#### This is part of our work, however, this code is not being used, is just here as evidence of our work process.
####
####
####
####
import pandas as pd

# Load the dataset
data_path = r'drug_consumption_data.csv'
df = pd.read_csv(data_path)

# Define assumed means and standard deviations for all variables
assumed_means = {
    "age": 40,  # Example: mean age in years
    "education": 3,  # Example: mean education level on a 0-5 scale
    "country": 1,  # Example: assuming "1" is a middle-range value
    "ethnicity": 2,  # Example: a numeric code for ethnicity
    "nscore": 0,  # Assumed mean for personality traits
    "escore": 0,
    "oscore": 0,
    "ascore": 0,
    "cscore": 0,
    "impuslive": 0,
    "ss": 0,
}

assumed_stds = {
    "age": 10,  # Example: standard deviation for age
    "education": 1.5,  # Example: standard deviation for education
    "country": 0.5,  # Example: assuming variability in country codes
    "ethnicity": 0.8,  # Example: variability for ethnicity codes
    "nscore": 1,  # Assumed standard deviation for personality traits
    "escore": 1,
    "oscore": 1,
    "ascore": 1,
    "cscore": 1,
    "impuslive": 1,
    "ss": 1,
}

# De-standardize all variables
for col in assumed_means.keys():
    if col in df.columns:
        df[f"{col}_assumed"] = df[col] * assumed_stds[col] + assumed_means[col]

# Map 'gender' to readable categories
df["gender_readable"] = df["gender"].apply(lambda x: "Male" if x > 0 else "Female")

# Save the processed dataset
#output_path = r'C:\Users\lenovo\Documents\HEALTH SCIENCES\SUPERVISED MACHINE LEARNING\INSTALL ORIGINAL DATA\processed_drug_consumption_data.csv'
#df.to_csv(output_path, index=False)

# Display the first few rows of the processed dataset
print(df.head())
# Inspect means and standard deviations
print("Assumed Means:")
print(assumed_means)
print("Assumed Standard Deviations:")
print(assumed_stds)
for col in assumed_means.keys():
    if f"{col}_assumed" in df.columns:
        print(f"Variable: {col}_assumed")
        print(f"Min: {df[f'{col}_assumed'].min()}, Max: {df[f'{col}_assumed'].max()}")

df.rename(columns={f"{col}_assumed": f"{col}_real" for col in assumed_means.keys()}, inplace=True)
