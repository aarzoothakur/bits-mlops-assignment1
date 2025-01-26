import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

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

# Fitting Logistic regression
log = LogisticRegression(max_iter=1000)  # Increase max_iter

log.fit(x_train, y_train)

# Saving the model to joblib file
joblib.dump(log, 'loan_prediction_model.joblib')

# Make predictions with the trained model on test data
pred = log.predict(x_test)

# Accuracy on the test dataset
accuracy = accuracy_score(y_test, pred)
print(f'Accuracy on test dataset: {accuracy}')

# Printing confusion matrix and classification report
print(metrics.confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
