import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load the CSV file with semi colon delimiter
df = pd.read_csv('wine+quality/winequality-red.csv', delimiter=';')

print(df.head())


# missing_values = df.isnull().sum()
# no missing values

# duplicate_rows = df[df.duplicated()]

#there are duplicate rows but it is duplicates of some features, not the entire row i believe

# Drop density as the variance between rows is neglient
dropDensityCol = 'density'
df = df.drop(columns=[dropDensityCol])

# did not handle "outliers" as that assumes I know the relevant range or variance of the data, no data jumps out of the range excessively. Outliers can be registered by plotting later on?

# Standardize each numerical features using z-score normalization
scaler = StandardScaler()
df[['fixed acidity']] = scaler.fit_transform(df[['fixed acidity' ]])
df[['volatile acidity']] = scaler.fit_transform(df[['volatile acidity' ]])
df[['citric acid']] = scaler.fit_transform(df[['citric acid' ]])
df[['residual sugar']] = scaler.fit_transform(df[['residual sugar' ]])
df[['chlorides']] = scaler.fit_transform(df[['chlorides' ]])
df[['free sulfur dioxide']] = scaler.fit_transform(df[['free sulfur dioxide' ]])
df[['total sulfur dioxide']] = scaler.fit_transform(df[['total sulfur dioxide' ]])
df[['sulphates']] = scaler.fit_transform(df[['sulphates' ]])
df[['total sulfur dioxide']] = scaler.fit_transform(df[['total sulfur dioxide' ]])
df[['alcohol']] = scaler.fit_transform(df[['alcohol' ]])
df[['quality']] = scaler.fit_transform(df[['quality' ]])

# did not correct for errors or inconsistencies as that requires domain knowledge of what data points aren't right, plotting or modelling the data may help here.

print("Cleaned dataset:")
print(df.head())

