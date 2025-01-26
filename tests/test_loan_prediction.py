import joblib
import pytest


# Load the model from joblib
@pytest.fixture
def model():
    model = joblib.load('loan_prediction_tuned_model.joblib')
    return model


# Test prediction for a given input
def test_prediction(model):
    input_data = [[1, 1.0, 459.0, 0.0, 25.0, 3459]]

    # Make prediction
    prediction = model.predict(input_data)
    # Assert that the prediction is a valid class (either 0 or 1)
    assert prediction in [0, 1], f"Unexpected prediction class: {prediction}"
