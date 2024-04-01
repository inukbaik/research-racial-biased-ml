# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('datasets/completed_data.csv')

# Filter the dataset for COVID-19 deaths data
covid_deaths_data = data[data['Indicator'] == 'Distribution of COVID-19 deaths (%)']

# Fill missing values with 0
covid_deaths_data_filled = covid_deaths_data.fillna(0)

# Define features and target variable
# Adjust target variable if necessary
X = covid_deaths_data_filled[['Non-Hispanic White', 'Non-Hispanic Black', 'Non-Hispanic American Indian or Alaska Native', 'Non-Hispanic Asian', 'Non-Hispanic Native Hawaiian or Other Pacific Islander', 'Hispanic', 'Other', 'Urban Rural Description', 'Income_per_Capita']]
y = covid_deaths_data_filled['Total deaths']

# Define categorical features for encoding and numeric features
categorical_features = ['Urban Rural Description']
numeric_features = X.columns.difference(categorical_features).tolist()

# Set up preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline with preprocessor and a RandomForestRegressor
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_pipeline.fit(X_train, y_train)

# Extract feature names after one-hot encoding and feature importances
ohe_feature_names = rf_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = list(numeric_features) + list(ohe_feature_names)
feature_importances = rf_pipeline.named_steps['regressor'].feature_importances_

# Create a DataFrame for feature importances and sort it
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df.sort_values(by='Importance', ascending=True), palette='viridis')
plt.title('Feature Importances for COVID-19 Death Rates')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
