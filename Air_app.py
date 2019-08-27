import numpy as np
from flask import Flask, jsonify, request
import pickle

# Model

model = pickle.load(open('tues_model2.pkl', 'rb'))

app = Flask(__name__)

@app.route('/api', methods=['POST'])

def make_predict():
    # Get the data.
    data = request.get_json(force=True)
    # Transform the data.
    predict_request = [data['neighborhood'], data['room_type'],
    data['accommodates'], data['bathrooms'], data['bedrooms'],
    data['number_of_reviews'], data['wifi'], data['cable_tv'],
    data['washer'], data['kitchen']]
    
    
    # Parse the data.
    predict_request = np.array(predict_request).reshape(1, -1)
    # Predictions
    y_hat = model.predict(predict_request)
    # Send back to the browser.
    output = {'y_hat': int(y_hat[0])}
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
