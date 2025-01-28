import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Import data
file_path = '/Users/charlottewatson/Downloads/Establishment REDACTED UNI.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Data Preprocessing for predictive model
# making the assumption that employees retire at 30 years of service or at age 65 (Typical force policy)
df['Retirement_Age'] = np.where(df['Length of Service'] >= 30, 'Retired', np.where(df['Age'] >= 65, 'Retired', 'Not Retired'))

# converted 'Retirement_Age' to binary: 1 for retired, 0 for not retired - to make the model work
df['Retired'] = df['Retirement_Age'].apply(lambda x: 1 if x == 'Retired' else 0)

# Dropped rows with missing 'Age' or 'Length of Service' data
df = df.dropna(subset=['Age', 'Length of Service'])

#  relevant features for the predictive model
X = df[['Age', 'Length of Service']]  # Independent variables
y = df['Retired']  # Dependent variable (Retirement status)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Logistic Regression model - training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report for Predictive Model:")
print(classification_report(y_test, y_pred))

# Get predicted probabilities of retirement (for all employees, staff and officers)
df['Predicted Retirement Probability'] = model.predict_proba(X)[:, 1]

# Analysis 1: Predicted Retirement Probability by Department
# Group by 'Department' and calculate the average predicted retirement probability
department_retirement = df.groupby('Department')['Predicted Retirement Probability'].mean().reset_index()

# Sorted the departments by the predicted retirement probability and selecting the top 3 (too many otherwise)
department_retirement_sorted = department_retirement.sort_values(by='Predicted Retirement Probability', ascending=False).head(3)

# Displayed the top 3 departments with the highest predicted retirement probabilities
print("Top 3 Department Predicted Retirement Probability:")
print(department_retirement_sorted)

# Visualisation: Predicted Retirement Probability by Top 3 Departments (again otherwise too many)
plt.figure(figsize=(12, 8))
sns.barplot(x='Predicted Retirement Probability', y='Department', data=department_retirement_sorted)
plt.title('Top 3 Departments with Highest Predicted Retirement Probability')
plt.xlabel('Predicted Retirement Probability')
plt.ylabel('Department')
plt.show()

# Analysis 2: Predicted Retirement Probability by Job Role
# Group by Job Role and calculate the average predicted retirement probability
role_retirement = df.groupby('Job Role')['Predicted Retirement Probability'].mean().reset_index()

# Sorted the roles by the predicted retirement probability and selecting the top 10 (as otherwise too many)
role_retirement_sorted = role_retirement.sort_values(by='Predicted Retirement Probability', ascending=False).head(10)

# Displayed the top 10 job roles with the highest predicted retirement probabilities
print("Top 10 Job Role Predicted Retirement Probability:")
print(role_retirement_sorted)

# Visualisation: Predicted Retirement Probability by Top 10 Job Roles
plt.figure(figsize=(12, 8))
sns.barplot(x='Predicted Retirement Probability', y='Job Role', data=role_retirement_sorted)
plt.title('Top 10 Job Roles with Highest Predicted Retirement Probability')
plt.xlabel('Predicted Retirement Probability')
plt.ylabel('Job Role')
plt.show()
