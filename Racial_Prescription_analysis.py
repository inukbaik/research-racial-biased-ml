# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'datasets/race_prescription.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns and filter the dataset
columns_to_drop = ['AnalysisDate', 'Start Date', 'End Date', 'Jurisdiction of Occurrence']
data.drop(columns=columns_to_drop, inplace=True)
data = data[data['Date Of Death Year'] != 2019]

# Clean categorical data
data['Sex'] = data['Sex'].replace({'M': 'Male', 'F': 'Female'})

# Calculate COVID-19 risk score and define labels based on the threshold
data['COVID_Deaths'] = data['COVID-19 (U071, Multiple Cause of Death)'] + data['COVID-19 (U071, Underlying Cause of Death)']
data['Total_Deaths'] = data['AllCause']
data['COVID_Risk_Score'] = data['COVID_Deaths'] / data['Total_Deaths']
threshold = data['COVID_Risk_Score'].mean()
data['Risk_Label'] = (data['COVID_Risk_Score'] > threshold).astype(int)

# Encode categorical variables and split the data
X = pd.get_dummies(data[['AgeGroup', 'Race/Ethnicity']], columns=['AgeGroup', 'Race/Ethnicity'])
y = data['Risk_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM classifier and evaluate accuracy
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Heatmap for risk assessment by race/ethnicity and age group
heatmap_data_race = data.pivot_table(index='AgeGroup', columns='Race/Ethnicity', values='Risk_Label', aggfunc=np.mean)
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data_race, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Proportion of Risky Individuals by Race/Ethnicity and Age Group')
plt.ylabel('Age Group')
plt.xlabel('Race/Ethnicity')
plt.xticks(rotation=45)
plt.show()

# Encode 'Sex' for a different analysis and split the data
le = LabelEncoder()
X = le.fit_transform(data['Sex']).reshape(-1, 1)
y = data['Risk_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a new SVM classifier for the 'Sex' feature
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Heatmap for risk assessment by sex and age group
heatmap_data_sex = data.pivot_table(index='AgeGroup', columns='Sex', values='Risk_Label', aggfunc=np.mean)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_sex, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Proportion of Risky Individuals by Sex and Age Group')
plt.ylabel('Age Group')
plt.xlabel('Sex')
plt.xticks(rotation=45)
plt.show()
