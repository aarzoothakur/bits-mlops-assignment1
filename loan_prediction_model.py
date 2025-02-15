import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score
    )
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import GridSearchCV
import mlflow

# Step 1: Load the CSV data
data = pd.read_csv('./data/loan_approval_dataset.csv')
data.head()

data.shape
"""Will look for the data description"""
data.info()

# Print the unique values in the object variables
print('Married: ' + str(data['Married'].unique()))
print('Dependents: ' + str(data['Dependents'].unique()))
print('Education: ' + str(data['Education'].unique()))
print('Self_Employed: ' + str(data['Self_Employed'].unique()))
print('Property_Area: ' + str(data['Property_Area'].unique()))

print('Loan status', data['Loan_Status'].value_counts())

"""The data set is not balanced in outputs"""

data.describe()

"""# Checking the missing values in the dataset"""

# Will look for the missing values in the data
data.isnull().sum().sort_values(ascending=False)

# Handling missing values
# Fill missing Credit_History with the most common
# value (since it's likely categorical)
data['Credit_History'] = data['Credit_History'].fillna(1.0)

# Fill missing Self_Employed with the most common category (No)
data['Self_Employed'] = data['Self_Employed'].fillna('No')

# Fill missing LoanAmount with the mean of the column (since it's numerical)
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())

# Fill missing values in the 'Dependents' column with the most common value (0)
data['Dependents'] = data['Dependents'].fillna('0')

# Check if there are any remaining missing values
data.isnull().sum().sort_values(ascending=False)

# Drop any remaining rows with missing values
data.dropna(inplace=True)

# Print the unique values in the object variables
print('Married: ' + str(data['Married'].unique()))
print('Dependents: ' + str(data['Dependents'].unique()))
print('Education: ' + str(data['Education'].unique()))
print('Self_Employed: ' + str(data['Self_Employed'].unique()))
print('Property_Area: ' + str(data['Property_Area'].unique()))

data.describe()

"""# Handling the outliers from the dataset"""
# Creating a new variable called total income by adding applicant income +
# coapplicant income
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

# Drop Loan_ID column
data = data.drop(columns=['Loan_ID'], axis=1)

# Make all other columns numerical as well
data['Married'] = np.where((data['Married'] == 'Yes'), 1, 0)
data['Gender'] = np.where((data['Gender'] == 'Female'), 1, 0)
data['Education'] = np.where((data['Education'] == 'Graduate'), 1, 0)
data['Self_Employed'] = np.where((data['Self_Employed'] == 'Yes'), 1, 0)
data['Dependents'] = np.where((data['Dependents'] == '0'), 0, 1)
data['Loan_Status'] = np.where((data['Loan_Status'] == 'Y'), 1, 0)

# Label encoding for 'Property_Area' column
le = preprocessing.LabelEncoder()
data['Property_Area'] = le.fit_transform(data['Property_Area'])

# Separate features and target
y = data['Loan_Status']
X = data.drop(columns=['Loan_Status'])

"""Selecting Important features"""
# Apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concatenate two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Name of the column', 'Score']
print(featureScores.nlargest(10, 'Score'))

# Select important features
X = X[['Married', 'Credit_History', 'TotalIncome', 'CoapplicantIncome',
       'LoanAmount', 'ApplicantIncome']]

X.head()

X.shape, y.shape

y.value_counts()

"""Dataset is not balanced, so we use SMOTE algorithm to balance the output"""
sm = SMOTE(random_state=2)
X, y = sm.fit_resample(X, y)
y.value_counts()

# Splitting the test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=6)

x_train.shape, x_test.shape

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [500, 1000, 2000, ]
}

# Initialize Logistic Regression
log_reg = LogisticRegression()

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
mlflow.set_experiment("Logistic Regression Loan Prediction")
with mlflow.start_run():
    # Log parameters for the experiment
    mlflow.log_param("C_values", param_grid['C'])
    mlflow.log_param("solvers", param_grid['solver'])
    mlflow.log_param("max_iter_values", param_grid['max_iter'])

# Fit the grid search to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters and best score from the grid search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Hyperparameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score}')
# Log the best score as a metric
mlflow.log_metric("best_score", best_score)

# Use the best model found by grid search
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(x_test)
# Check mse
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)

# Accuracy on the test dataset
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test dataset: {accuracy}')

# Confusion matrix and classification report
print(metrics.confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
mlflow.sklearn.log_model(best_model, "logistic_regression_model")

# Saving the tuned model to joblib file
joblib.dump(best_model, 'loan_prediction_tuned_model.joblib')
