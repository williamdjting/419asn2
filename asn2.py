import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the CSV file with semi colon delimiter
df = pd.read_csv('wine+quality/winequality-red.csv', delimiter=';')

print(df.head())

X = df.iloc[:, :-1]  # Features = everything but last column
y = df.iloc[:, -1]   # Labels = last column, i.e. quality of red wine

# missing_values = df.isnull().sum()
# no missing values

# duplicate_rows = df[df.duplicated()]

#there are duplicate rows but it is duplicates of some features, not the entire row i believe

# Drop density as the variance between rows is neglient
dropDensityCol = 'density'
X = X.drop(columns=[dropDensityCol])

# did not handle "outliers" as that assumes I know the relevant range or variance of the data, no data jumps out of the range excessively. Outliers can be registered by plotting later on?

# Standardize each numerical features using z-score normalization
scaler = StandardScaler()
X[['fixed acidity']] = scaler.fit_transform(X[['fixed acidity' ]])
X[['volatile acidity']] = scaler.fit_transform(X[['volatile acidity' ]])
X[['citric acid']] = scaler.fit_transform(X[['citric acid' ]])
X[['residual sugar']] = scaler.fit_transform(X[['residual sugar' ]])
X[['chlorides']] = scaler.fit_transform(X[['chlorides' ]])
X[['free sulfur dioxide']] = scaler.fit_transform(X[['free sulfur dioxide' ]])
X[['total sulfur dioxide']] = scaler.fit_transform(X[['total sulfur dioxide' ]])
X[['sulphates']] = scaler.fit_transform(X[['sulphates' ]])
X[['total sulfur dioxide']] = scaler.fit_transform(X[['total sulfur dioxide' ]])
X[['alcohol']] = scaler.fit_transform(X[['alcohol' ]])


# did not correct for errors or inconsistencies as that requires domain knowledge of what data points aren't right, plotting or modelling the data may help here.

print("cleaned dataset")
print(X.head())

# Split the data into training and testing sets
# random 80/20 split is chosen as its typical
# random state 42 is chosen as its typical
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from dataset: "These datasets can be viewed as classification or regression tasks.  The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones)."

# train logistic regression model
logistic_regression_model = LogisticRegression(max_iter=100)
logistic_regression_model.fit(X_train, y_train)

# valuate logistic regression model
logistic_regression_predictions = logistic_regression_model.predict(X_test)
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)


# train random forest classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# evaluate random forest classifier
random_forest_predictions = random_forest_model.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print("Random Forest Accuracy:", random_forest_accuracy)

# approximately
# Logistic Regression Accuracy: 0.578125
# Random Forest Accuracy: 0.6625