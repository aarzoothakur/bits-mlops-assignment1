import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Start MLflow experiment
mlflow.start_run()

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

# Log dataset info in MLflow
mlflow.log_param("dataset", "Iris")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train.values.ravel())

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics and model parameters in MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_param("max_iter", 200)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Log the trained model in MLflow
mlflow.sklearn.log_model(model, "logistic_regression_model")

# Save the model using joblib (for external use)
joblib.dump(model, 'model.pkl')

# End MLflow run
mlflow.end_run()
