import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Define a test sample (using average feature values of the Iris dataset)
test_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Make a prediction
prediction = model.predict(test_sample)

# Check if the prediction is correct (should be one of the species classes)
print(f"Predicted class: {prediction[0]}")
