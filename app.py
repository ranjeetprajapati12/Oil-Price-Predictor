import joblib
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

model=joblib.load(open('Oil_Price_Prediction.pkl','rb'))
scaler=joblib.load(open('opp.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)

    df2=pd.DataFrame([data],columns=['Date1'])
    df3=df2['Date1'].astype('datetime64')
    value=scaler.transform([df3])
    output=model.predict(value)
    print(output[0][0])
    return jsonify(output[0][0])

@app.route('/predict',methods=['POST'])
def predict():
    data=request.form.values()
    df2=pd.DataFrame([data],columns=['Date1'])
    df3=df2['Date1'].astype('datetime64')
    value=scaler.transform([df3])
    output=model.predict(value)[0][0]

    return render_template("home.html",prediction_text="The predicted Price of oil is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)