import numpy as np
import pandas as pd
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score



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
# unsure if we should do PCA scoring to increase the accuracy score?

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
logistic_regression_model = LogisticRegression(max_iter=50)
logistic_regression_model.fit(X_train, y_train)

# valuate logistic regression model
logistic_regression_predictions = logistic_regression_model.predict(X_test)
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)


# train random forest classifier
random_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest_model.fit(X_train, y_train)

# evaluate random forest classifier
random_forest_predictions = random_forest_model.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print("Random Forest Accuracy:", random_forest_accuracy)

# Plot Precision-Recall Curve for Logistic Regression
# plt.figure(figsize=(8, 6))
# for i in range(len(np.unique(y))):
#     precision, recall, _ = precision_recall_curve(y_test == i, logistic_regression_model.predict_proba(X_test)[:, i])
#     plt.plot(recall, precision, marker='.', label=f'Class {i} (Logistic Regression)')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve (Logistic Regression)')
# plt.legend()
# plt.show()

# Plot Precision-Recall Curve for Random Forest
# plt.figure(figsize=(8, 6))
# for i in range(len(np.unique(y))):
#     precision, recall, _ = precision_recall_curve(y_test == i, random_forest_model.predict_proba(X_test)[:, i])
#     plt.plot(recall, precision, marker='.', label=f'Class {i} (Random Forest)')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve (Random Forest)')
# plt.legend()
# plt.show()


# approximately
# Logistic Regression Accuracy: 0.578125
# Random Forest Accuracy: 0.6625

# so far the final model = random_forest_model

finalModel = random_forest_model
finalModelAccuracy = random_forest_accuracy

y_pred = finalModel.predict(X_test)

# baseline classifier - random forest outputs
accuracyY = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)

print(f'Accuracy: {accuracyY}')
print('Confusion Matrix:')
print(cm)
# prints confusion matrix, not sure how to interpret


# part 2
# slice to first 10 rows of the dataset
part2data = df.sample(frac=0.1, random_state=42).reset_index(drop=True)  # Reset index after sampling

# cleaning
X2 = part2data.iloc[:, :-1]  # Features = everything but last column
y2 = part2data.iloc[:, -1]   # Labels = last column, i.e. quality of red wine

dropDensityCol = 'density'
X2 = X2.drop(columns=[dropDensityCol])

scaler = StandardScaler()
X2[['fixed acidity']] = scaler.fit_transform(X2[['fixed acidity' ]])
X2[['volatile acidity']] = scaler.fit_transform(X2[['volatile acidity' ]])
X2[['citric acid']] = scaler.fit_transform(X2[['citric acid' ]])
X2[['residual sugar']] = scaler.fit_transform(X2[['residual sugar' ]])
X2[['chlorides']] = scaler.fit_transform(X2[['chlorides' ]])
X2[['free sulfur dioxide']] = scaler.fit_transform(X2[['free sulfur dioxide' ]])
X2[['total sulfur dioxide']] = scaler.fit_transform(X2[['total sulfur dioxide' ]])
X2[['sulphates']] = scaler.fit_transform(X2[['sulphates' ]])
X2[['total sulfur dioxide']] = scaler.fit_transform(X2[['total sulfur dioxide' ]])
X2[['alcohol']] = scaler.fit_transform(X2[['alcohol' ]])

# used below algorithm with slight modifications from this source: https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
cv = LeaveOneOut()

# enumerate splits
y_true, y_pred = list(), list()
part2array = []
for i in range(10):
  # leave out one data point at a time
  X_train = X2.drop(index=i)

  y_train = y2.drop(index=i)
    
  # fit model
  part2model = RandomForestClassifier(random_state=42)
  part2model.fit(X_train, y_train)
    
  # evaluate model on left-out data point and calculate the scoring
  y_true = y2[i:i+1]
  y_pred = part2model.predict(X2[i:i+1])
    
  acc = accuracy_score(y_true, y_pred)
  print('Accuracy for iteration {}: {:.3f}'.format(i+1, acc))
  
  part2array.append(acc)

average_accuracy = sum(part2array) / len(part2array)
print("Average Accuracy Score:", average_accuracy)


# this is another method using cross_val_score but it generates "Part 2 Accuracy: [1. 1. 1. 0. 1. 1. 1. 0. 0. 1.]" which is not right
part2Model = random_forest_model
# for i in range(10):
scores = cross_val_score(part2Model, X2, y2, scoring="accuracy", cv=cv, n_jobs=-1)
print('Part 2 Accuracy:', scores)