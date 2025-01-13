import pandas as pd
import numpy as np
import joblib
import pytest
from sklearn.metrics \
    import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing \
    import StandardScaler, LabelEncoder

# Constants
MODEL_PATH = 'whiskey_model.pkl'
DATA_PATH = './whiskey_data.csv'
ACCURACY_THRESHOLD = 0.35
F1_SCORE_THRESHOLD = 0.35


# Load the saved model
@pytest.fixture(scope="module")
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


# Load and preprocess the test data
@pytest.fixture(scope="module")
def preprocess_data():
    whiskey_data = pd.read_csv(DATA_PATH)
    # Rename columns for easier access
    whiskey_data.rename(
        columns={'ï»¿acidity_level': 'acidity_level'}, inplace=True
    )
    # Apply log transformation for skewed features
    whiskey_data[
        'acidity_level_log'
    ] = np.log1p(
        whiskey_data['acidity_level']
    )
    whiskey_data[
        'fruitiness_level_log'
    ] = np.log1p(whiskey_data['fruitiness_level'])
    whiskey_data[
        'citrus_content_log'
    ] = np.log1p(whiskey_data['citrus_content'])
    # Drop original columns after log transformation
    whiskey_data = whiskey_data.drop(
        ['acidity_level', 'fruitiness_level', 'citrus_content'], axis=1
    )
    # Split into features and target
    X = whiskey_data.drop('whiskey_quality', axis=1)
    y = whiskey_data['whiskey_quality']
    # Apply Label Encoding to 'whiskey_quality'
    # to convert categorical labels to numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Apply scaling (same scaler used during training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_encoded, label_encoder


# Test the accuracy of the model
def test_accuracy(load_model, preprocess_data):
    X_scaled, y_true, _ = preprocess_data
    model = load_model

    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y_true, y_pred)

    # Assert the accuracy is above a reasonable threshold
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Accuracy is below expected threshold: {accuracy:.2f}"
    )


# Test the classification report for
# specific performance metrics
def test_classification_report(load_model, preprocess_data):
    X_scaled, y_true, label_encoder = preprocess_data
    model = load_model

    y_pred = model.predict(X_scaled)
    report = classification_report(
        y_true,y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    # Assert that F1-score for each class is above a threshold
    for whiskey_class in report:
        if whiskey_class not in ['accuracy', 'macro avg', 'weighted avg']:
            f1_score = report[whiskey_class]['f1-score']
            assert f1_score >= F1_SCORE_THRESHOLD, (
                f"F1-score for class {whiskey_class} is below expected threshold: {f1_score:.2f}"
            )


# Test the confusion matrix
def test_confusion_matrix(load_model, preprocess_data):
    X_scaled, y_true, _ = preprocess_data
    model = load_model

    y_pred = model.predict(X_scaled)
    conf_matrix = confusion_matrix(y_true, y_pred)
   
    # Assert that there are no zero values
    # in the diagonal (meaning no class was missed completely)
    diagonal_values = np.diag(conf_matrix)
    assert all(diagonal_values > 0), (
        f"Confusion matrix has zero values on the diagonal: {diagonal_values}"
    )
