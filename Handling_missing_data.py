import pandas as pd
from sklearn.impute import SimpleImputer

# Sample dataset with missing values
data = {'Age': [25, 30, None, 35, None, 40], 'Salary': [50000, 60000, 55000, None, 65000, None]}
df = pd.DataFrame(data)

# Handling missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can use 'mean', 'median', 'most_frequent'
df['Age'] = imputer.fit_transform(df[['Age']])
df['Salary'] = imputer.fit_transform(df[['Salary']])

print(df)
