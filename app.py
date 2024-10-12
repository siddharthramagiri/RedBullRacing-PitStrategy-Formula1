from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

best_clf = pickle.load(open('RF_Strategy.pkl','rb'))
FEATURES = ['EventName','Stint','meanAirTemp','meanTrackTemp','meanHumid','Rainfall','GridPosition','lapNumberAtBeginingOfStint']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        print(f"Circuit -------- > {data['circuit']}")
        d1 = float(data['circuit'])
        d2 = float(data['stint'])
        d3 = float(data['meanAirTemp'])
        d4 = float(data['meanTrackTemp'])
        d5 = float(data['meanHumid'])
        d6 = float(data['Rainfall'])
        d7 = float(data['GridPosition'])
        d8 = float(data['lapNumber'])

        arr = np.array([[d1,d2,d3,d4,d5,d6,d7,d8]])
        print("Array is -------- > ",arr)
        
        df = pd.DataFrame(arr, columns=FEATURES)
        
        Y_predict = best_clf.predict(df)
        print(f"Prediction --> {Y_predict}")
        return jsonify({'prediction' : Y_predict.tolist()})
        
    except Exception as e :
        print(f"Error : {str(e)}")
        return jsonify({'Err' : str(e)}) , 500


if __name__ == '__main__':
    app.run(debug=True,port=8000)