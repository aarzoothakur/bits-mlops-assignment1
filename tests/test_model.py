import joblib
import numpy as np


def test_model_prediction():
    # Load the trained model
    model = joblib.load('model.pkl')

    # Define test sample (using average feature values of the Iris dataset)
    test_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Make a prediction
    prediction = model.predict(test_sample)

    # Assert that the prediction is one of the valid classes (0, 1, or 2)
    assert prediction[0] in [0, 1, 2], f"Unexpected prediction: {prediction[0]}"
