import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import resample

# First, I load my data from an excel file
file_path = '/Users/charlottewatson/Downloads/Establishment REDACTED UNI.xlsx'
df = pd.read_excel(file_path)

# 1. Data Cleansing: I handle any missing values by dropping rows with missing 'Age' or 'Length of Service' values
df = df.dropna(subset=['Age', 'Length of Service'])

# Here I create a target variable based on age and length of service.
# I make the assumption that employees over 55 years old with more than 30 years of service are likely to retire soon
df['Retirement Likely'] = ((df['Age'] > 55) & (df['Length of Service'] > 30)).astype(int)

# 2. Feature Selection: I select factors which I think will help predict retirement.
features = ['Age', 'Length of Service', 'FTE', 'Working Hours']
X = df[features]
y = df['Retirement Likely']

# 3. Data Split: I split the data into training and testing sets (80%, 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# I scale the features for better performance with logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Logistic Regression Model: I create my logistic regression model and train it
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Model Evaluation: I predict with the test data and output the classification report and confusion matrix
y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# The model achieves a high overall accuracy of 98%, but due to significant class imbalance, it fails to predict employees likely to retire.
# This results in a precision and recall of 0.00 for the minority class (retirees). The confusion matrix shows the model correctly identifies non-retiring employees but misses all likely retirees.

# To address the imbalance in the data, I will oversample the minority class (retirees)
# I combine the data back to resample the minority class
df_balanced = pd.concat([X, y], axis=1)

# I separate the majority and minority classes
df_majority = df_balanced[df_balanced['Retirement Likely'] == 0]
df_minority = df_balanced[df_balanced['Retirement Likely'] == 1]

# Oversample the minority class (number of retirees)
df_minority_oversampled = resample(df_minority, 
                                   replace=True,    # Sample with replacement
                                   n_samples=len(df_majority),  # Match number of majority class
                                   random_state=42)

# Combine the majority and oversampled minority class
df_oversampled = pd.concat([df_majority, df_minority_oversampled])

# Separate the features and target variable again
X_oversampled = df_oversampled[features]
y_oversampled = df_oversampled['Retirement Likely']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size=0.2, random_state=42)

# Scale the features for better performance
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict with the test data
y_pred = model.predict(X_test_scaled)

# Output the classification report and confusion matrix
classification_report_output = classification_report(y_test, y_pred)
confusion_matrix_output = confusion_matrix(y_test, y_pred)

print("Classification Report after Oversampling:")
print(classification_report_output)
print("\nConfusion Matrix after Oversampling:")
print(confusion_matrix_output)

# Feature Importance Analysis: I will now plot the feature importance from the logistic regression model
feature_importance = abs(model.coef_[0])
sorted_idx = feature_importance.argsort()

# Bar plot for feature importance
plt.figure(figsize=(8, 6))
plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Predicting Retirement Likelihood')
plt.show()

# Demographic analysis - Box plot - Age distribution for Retirement Likely vs Not Likely
# Grouping the data by 'Retirement Likely' and plotting the boxplot manually using Matplotlib
plt.figure(figsize=(8, 6))

# Plot the boxplot for each retirement group
df.boxplot(column='Age', by='Retirement Likely', vert=False)

# Set labels and title
plt.title('Age Distribution for Retirement Likely vs Not Likely')
plt.suptitle('')  # Remove the default title from the boxplot
plt.xlabel('Age')
plt.ylabel('Retirement Likely')

# Set custom xticks labels to correspond to the retirement categories
plt.xticks([0, 1], ['Not Likely', 'Likely'])

plt.show()

# Additional analysis ....................
# Analyse job roles/ departments
job_role_analysis = df.groupby('Job Role')['Retirement Likely'].mean().sort_values(ascending=False)
department_analysis = df.groupby('Department')['Retirement Likely'].mean().sort_values(ascending=False)

# Display results
print("Job Role Analysis:\n", job_role_analysis)
print("\nDepartment Analysis:\n", department_analysis)

# Cost analysis.................
# Calculate the average age and length of service for employees likely to retire
retirement_stats = df[df['Retirement Likely'] == 1][['Age', 'Length of Service', 'FTE']].mean()

# Display the results
print("\nRetirement Stats:\n", retirement_stats)

# Assumptions - cost estimation
avg_recruitment_cost = 5000  # £5,000 per hire
avg_training_cost = 2000  # £2,000 per employee for training

# Count the number of employees likely to retire
num_retirees = df[df['Retirement Likely'] == 1].shape[0]

# Estimate recruitment and training costs
recruitment_cost = num_retirees * avg_recruitment_cost
training_cost = num_retirees * avg_training_cost

print(f"\nEstimated Recruitment Cost: £{recruitment_cost}")
print(f"Estimated Training Cost: £{training_cost}")

# Job role analysis..........................
# Grouped by job role / calculate the percentage of employees likely to retire
job_role_retirement = df.groupby('Job Role')['Retirement Likely'].mean().sort_values(ascending=False)

# Calculated the number of retirees per job role
job_role_retirement_count = df.groupby('Job Role')['Retirement Likely'].sum().sort_values(ascending=False)

# Calculated the average length of service for each job role
job_role_service = df.groupby('Job Role')['Length of Service'].mean().sort_values(ascending=False)

# Combined the results into a single dataframe to compare
job_role_analysis = pd.DataFrame({
    'Retirement Likely (%)': job_role_retirement * 100,
    'Number of Retirees': job_role_retirement_count,
    'Average Length of Service (years)': job_role_service
})

# Display the top 3 job roles with the highest retirement likelihood
top_3_retiring_roles = job_role_analysis.head(3)
print("\nTop 3 Retiring Roles:\n", top_3_retiring_roles)

# Department analysis.......................
# Group by department and calculate the retirement likelihood
department_retirement = df.groupby('Department')['Retirement Likely'].mean().sort_values(ascending=False)

# Calculated the number of retirees per department
department_retirement_count = df.groupby('Department')['Retirement Likely'].sum().sort_values(ascending=False)

# Calculated the average length of service for each department
department_service = df.groupby('Department')['Length of Service'].mean().sort_values(ascending=False)

# Combined the results into a single dataframe
department_analysis = pd.DataFrame({
    'Retirement Likely (%)': department_retirement * 100,
    'Number of Retirees': department_retirement_count,
    'Average Length of Service (years)': department_service
})

# Display the results
print("\nDepartment Retirement Analysis:\n", department_analysis)
