import pandas as pd

# Load your dataset
df = pd.read_csv('us_racial.csv')  # Replace 'us_racial.csv' with the actual file path

# Columns to keep
columns_to_keep = [
    'Date', 'State', 'Cases_Total', 'Cases_White', 'Cases_Black',
    'Cases_Latinx', 'Cases_Asian', 'Cases_AIAN', 'Cases_NHPI',
    'Cases_Multiracial', 'Cases_Other', 'Cases_Unknown'
]

# Drop irrelevant columns by keeping only the columns listed above
df_preprocessed = df[columns_to_keep]

# Convert 'Date' column to datetime format
df_preprocessed['Date'] = pd.to_datetime(df_preprocessed['Date'], format='%Y%m%d')

# Handle missing values - replacing NaN with 0 for simplicity
# Note: Perform this operation directly on df_preprocessed to keep everything in one DataFrame
df_preprocessed.fillna(0, inplace=True)

# Now df_preprocessed contains the changes, including the filled NaN values
# Optionally, save the preprocessed data to a new CSV file
df_preprocessed.to_csv('us_racial_processed.csv', index=False)

# Show the first few rows of the preprocessed dataframe
print(df_preprocessed.head())
