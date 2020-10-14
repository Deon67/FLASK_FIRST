import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('linear_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        feature=[[int(x) for x in request.form.values()]]
        prediction=model.predict(feature)
        output = round(prediction[0], 2)
    except Exception as e:
        return render_template('index.html', prediction_text='Please Enter a Integer')
    else:
        return render_template('index.html', prediction_text='GPA is {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)