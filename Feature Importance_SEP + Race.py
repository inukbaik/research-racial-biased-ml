# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
file_path = 'datasets/AH_Deaths_by_Educational.csv'
data = pd.read_csv(file_path)

# Filter out irrelevant years and unknown categories
data = data[data['Year'] != 2019]
data = data[(data['Education Level'] != 'Unknown') & (data['Race or Hispanic Origin'] != 'Unknown')]

# Save the processed data to a new CSV file
data.to_csv('procdata.csv', index=False)

# Load the cleaned dataset
df = pd.read_csv('datasets/procdata.csv')

# Further clean the dataset by dropping unnecessary columns
df_cleaned = df.drop(columns=['Data as of'])


# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in ['Education Level', 'Race or Hispanic Origin', 'Sex', 'Age Group']:
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    label_encoders[column] = le

# Split the data into features (X) and target (y)
features_to_include = ['Education Level', 'Race or Hispanic Origin', 'Sex', 'Age Group']  # List the column names of features to include
X = df_cleaned[features_to_include]
y = df_cleaned['COVID-19 Deaths']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Extract feature importances
feature_importance = rf_model.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

# Visualize the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importance for Predicting COVID-19 Deaths')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
