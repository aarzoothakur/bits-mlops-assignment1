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
    input_data = np.array(data['input']).reshape(1, -1)
    # Predict the loan approval status
    prediction = model.predict(input_data)
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5200)
