from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_prediction_tuned_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Prepare the input data for prediction
    features = np.array([data['features']])
    
    # Predict the loan approval status
    prediction = model.predict(features)
    
    # Return the result as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
