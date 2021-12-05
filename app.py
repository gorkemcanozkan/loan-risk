import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from jinja2 import Template
import pickle
from joblib import load

app=Flask(__name__)
model=pickle.load(open('model_pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():
    '''
    For rendering the results on HTML GUI
    '''
    if request.method=="POST":
        features=[float(x) for x in request.form.values()]
        scaler = load('scaler_risk.joblib')
        final_features=scaler.transform([features])
        prediction=model.predict(final_features)
        output="Is Low!"
        if prediction==1:
            output="Is High!"
        return render_template('index.html',prediction_text="Default Risk {}".format(output))
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
