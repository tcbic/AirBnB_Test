import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle
import xgboost as xgb

# Model

model = pickle.load(open('tues_model4.pkl', 'rb'))

app = Flask(__name__)

@app.route('/api', methods=['POST'])

def make_predict():
    # Get the data.
    data = request.get_json(force=True)
    # Transform the data.
    
    predict_request = [data['neighborhood'], data['room_type'],
    data['accommodates'], data['bedrooms'],
    data['number_of_reviews'], data['wifi'], data['cable_tv'],
    data['washer'], data['kitchen']]
    
    # Parse the data.
    predict_request = np.array(predict_request).reshape(1, -1)
    #predict_request = xgb.DMatrix(predict_request)
    #predict_request = pd.DataFrame.from_dict(predict_request)

    # Predictions
    y_hat = model.predict(predict_request)
    # Send back to the browser.
    output = {'y_hat': int(y_hat[0])}
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
