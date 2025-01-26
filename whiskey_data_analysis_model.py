import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Start MLflow experiment
mlflow.start_run()
# Set a relative path to store the model
# artifacts (inside the working directory)
artifact_dir = os.path.join(os.getcwd(), "mlruns")

print("Experiment started...")

# Example whiskey data processing (replace with actual data loading)
whiskey_data = pd.read_csv('whiskey_data.csv')
print("Data loaded successfully")

whiskey_data.rename(
    columns={'ï»¿acidity_level': 'acidity_level'}, inplace=True
)
whiskey_data['acidity_level_log'] = np.log1p(whiskey_data['acidity_level'])
whiskey_data['fruitiness_level_log'] = np.log1p(
    whiskey_data['fruitiness_level']
)
whiskey_data['citrus_content_log'] = np.log1p(whiskey_data['citrus_content'])
whiskey_data = whiskey_data.drop(
    ['acidity_level', 'fruitiness_level', 'citrus_content'], axis=1
)

# Encoding the target variable
print("Encoding target variable...")
if whiskey_data['whiskey_quality'].dtype == 'object':
    le = LabelEncoder()
    whiskey_data[
        'whiskey_quality'
    ] = le.fit_transform(whiskey_data['whiskey_quality'])
print("Target variable encoded successfully")

# Features and target
X = whiskey_data.drop('whiskey_quality', axis=1)
y = whiskey_data['whiskey_quality']

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split successfully")

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling completed")

# Log hyperparameters
n_estimators = 100
max_depth = None
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)

# Train the model
print("Training model...")
rf_model = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
rf_model.fit(X_train, y_train)
print("Model training completed")

# Predictions
print("Making predictions...")
y_pred = rf_model.predict(X_test)

# Log metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy}, F1 Score: {f1}")
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1_score", f1)

# Log confusion matrix elements (optional)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print(f"Confusion Matrix - TN: {tn},FP: {fp}, FN: {fn}, TP: {tp}")
mlflow.log_metric("True_Negative", tn)
mlflow.log_metric("False_Positive", fp)
mlflow.log_metric("False_Negative", fn)
mlflow.log_metric("True_Positive", tp)

# Log model
print("Logging model...")
mlflow.sklearn.log_model(rf_model, "rf_model", artifact_path=artifact_dir)

# Save model locally
joblib.dump(rf_model, 'whiskey_model.pkl')

# End MLflow run
print("Ending MLflow run...")
mlflow.end_run()

print("Execution completed!")
