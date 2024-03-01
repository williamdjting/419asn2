import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
from itertools import permutations


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
logistic_regression_model = LogisticRegression(max_iter=1000)
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
# dataset shrink to 10% for comput reasons
# part2data = df.sample(frac=0.1, random_state=42).reset_index(drop=True)  # Reset index after sampling

# # iterate 10 times for 10 LOO iterations
# for i in range(10):
#   # for each iteration, we randomly remove a row from the dataset and then drop it not in place, then the model is trained and tested with this row removed    
#   random_index = np.random.randint(0, len(part2data))
#   new_part2data = part2data.drop(random_index, inplace=False)
#   # cleaning
#   X2 = new_part2data.iloc[:, :-1]  # Features = everything but last column
#   y2 = new_part2data.iloc[:, -1]   # Labels = last column, i.e. quality of red wine

#   dropDensityCol = 'density'
#   X2 = X2.drop(columns=[dropDensityCol])

#   scaler = StandardScaler()
#   X2[['fixed acidity']] = scaler.fit_transform(X2[['fixed acidity' ]])
#   X2[['volatile acidity']] = scaler.fit_transform(X2[['volatile acidity' ]])
#   X2[['citric acid']] = scaler.fit_transform(X2[['citric acid' ]])
#   X2[['residual sugar']] = scaler.fit_transform(X2[['residual sugar' ]])
#   X2[['chlorides']] = scaler.fit_transform(X2[['chlorides' ]])
#   X2[['free sulfur dioxide']] = scaler.fit_transform(X2[['free sulfur dioxide' ]])
#   X2[['total sulfur dioxide']] = scaler.fit_transform(X2[['total sulfur dioxide' ]])
#   X2[['sulphates']] = scaler.fit_transform(X2[['sulphates' ]])
#   X2[['total sulfur dioxide']] = scaler.fit_transform(X2[['total sulfur dioxide' ]])
#   X2[['alcohol']] = scaler.fit_transform(X2[['alcohol' ]])

#   # used algorithm with slight modifications from this source: https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
#   cv = LeaveOneOut()

#   # enumerate splits
#   y_true, y_pred = list(), list()
#   # Enumerate splits

#   # for i in range(10):
#   for train_ix, test_ix in cv.split(X2):
#         # Extract training and test data based on indices
#         X_train, X_test = X2.iloc[train_ix], X2.iloc[test_ix]
#         y_train, y_test = y2.iloc[train_ix], y2.iloc[test_ix]
        
#         # Fit model
#         model = RandomForestClassifier(random_state=42)
#         model.fit(X_train, y_train)
        
#         # Evaluate model
#         yhat = model.predict(X_test)
        
#         # Store true and predicted values
#         y_true.append(y_test.iloc[0])  # Assuming y_test is a Series
#         y_pred.append(yhat[0])

#     # calculate accuracy 
#   acc = accuracy_score(y_true, y_pred)
#   print('Accuracy: %.3f' % acc)

# Part 3
def leave_group_out_influence(X_train, y_train, X_test, y_test, model):
    # Fit the model on the entire training set
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy on the test set predictions before removing the group
    y_pred_test_before = model.predict(X_test)
    accuracy_before = accuracy_score(y_test, y_pred_test_before)
    
    # Initialize lists to store group sizes and influence scores
    group_sizes = []
    influence_scores = []
    
    # Loop through different group sizes
    for size in np.arange(0.1, 1.1, 0.1):  # Group sizes from 10% to 100%
        # Calculate the number of samples in the group
        group_size = int(len(X_train) * size)
        
        # Select a random group of data points
        group_indices = np.random.choice(len(X_train), group_size, replace=False)
        X_group = X_train.iloc[group_indices]
        y_group = y_train.iloc[group_indices]
        
        # Retrain the model without the left-out group
        X_train_left_out = X_train.drop(X_train.index[group_indices])
        y_train_left_out = y_train.drop(y_train.index[group_indices])
        
        # Check if training set is empty
        if len(X_train_left_out) == 0:
            print(f"Skipping group size {size}: Empty training set after leaving out group")
            continue
        
        model.fit(X_train_left_out, y_train_left_out)
        
        # Evaluate the model on the group left out
        y_pred_left_out = model.predict(X_group)
        accuracy_after = accuracy_score(y_group, y_pred_left_out)
        
        # Calculate the drop in accuracy
        accuracy_drop = accuracy_before - accuracy_after
        
        # Append the group size and influence score to the lists
        group_sizes.append(group_size)
        influence_scores.append(accuracy_drop)
    
    return group_sizes, influence_scores

# Call the function for logistic regression model
group_sizes_lr, influence_scores_lr = leave_group_out_influence(X_train, y_train, X_test, y_test, logistic_regression_model)

# Call the function for random forest model
group_sizes_rf, influence_scores_rf = leave_group_out_influence(X_train, y_train, X_test, y_test, random_forest_model)

# Print group sizes and influence scores for logistic regression
print("Logistic Regression:")
for size, score in zip(group_sizes_lr, influence_scores_lr):
    print(f"Group Size: {size}, Influence Score: {score}")

# Print group sizes and influence scores for random forest
print("\nRandom Forest:")
for size, score in zip(group_sizes_rf, influence_scores_rf):
    print(f"Group Size: {size}, Influence Score: {score}")

# Plot group size vs. influence for logistic regression and random forest
plt.figure(figsize=(10, 6))
plt.plot(group_sizes_lr, influence_scores_lr, marker='o', label='Logistic Regression')
plt.plot(group_sizes_rf, influence_scores_rf, marker='o', label='Random Forest')
plt.xlabel('Group Size')
plt.ylabel('Influence (Accuracy Drop)')
plt.title('Group Size vs. Influence on Model Performance')
plt.legend()
plt.grid(True)
plt.show()

# Part 4
# Function to compute Shapley values for a single observation
def compute_shapley_value_single(observation, model, X_train, y_train, n_permutations=10):
    shapley_values = np.zeros(len(observation))
    
    # Create permutations of feature indices
    feature_indices = np.arange(len(observation))
    perms = permutations(feature_indices)
    
    # Compute marginal contribution for each permutation
    for perm in perms:
        marginal_contributions = []
        for i in range(1, len(perm) + 1):
            subset = perm[:i]
            obs_subset = observation.copy()
            obs_subset[subset] = 0 
            pred_subset = model.predict_proba([obs_subset])[0]
            pred = model.predict_proba([observation])[0]
            marginal_contribution = np.abs(pred_subset - pred)
            max_marginal_contribution = np.max(marginal_contribution)
            marginal_contributions.append(max_marginal_contribution)

        shapley_value = np.mean(marginal_contributions)

        for idx in perm:
            shapley_values[idx] += shapley_value
    
    shapley_values /= n_permutations
    
    return shapley_values

# Function to compute Shapley values for all observations in the training data
def compute_shapley_values(X_train, model, n_permutations=10):
    shapley_values_all = []
    for i in range(len(X_train)):
        observation = X_train.iloc[i]
        shapley_values = compute_shapley_value_single(observation, model, X_train, y_train, n_permutations)
        shapley_values_all.append(shapley_values)
    return np.array(shapley_values_all)

shapley_values_rf = compute_shapley_values(X_train, random_forest_model)

# Plot the distribution of Shapley values
plt.figure(figsize=(10, 6))
plt.hist(shapley_values_rf.flatten(), bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Shapley Value')
plt.ylabel('Frequency')
plt.title('Distribution of Shapley Values (Random Forest)')
plt.grid(True)
plt.show()