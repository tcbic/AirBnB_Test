import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
import pickle

# Model
model = pickle.load(open('wednesday_model_4_keras.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods=['POST'])

def make_predict():
    # Get the data.
    data = request.get_json(force=True)

    

    # Transform data.

    predict_request = [data['neighborhood'],
                       data['room_type'],
                       data['accommodates'],
                       data['bedrooms'],
                       data['number_of_reviews'],
                       data['wifi'],
                       data['cable_tv'],
                       data['washer'],
                       data['kitchen']]

    data.update((x, [y]) for x, y in data.items())
    
    # Parse data.
    data_df = pd.DataFrame.from_dict(predict_request)

    print(data_df)

    # Predictions
    y_hat = model.predict(data_df)

    # Send back to the browser.

    output = {'y_hat': int(y_hat[0])}

    # Return the data.
    return jsonify(results=output)

#if __name__ == '__main__':
    #app.run(port = 9000, debug=True)
