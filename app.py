 # Importing essential libraries
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from datetime import datetime

# Load the LogisticRegression model
model = pickle.load(open('Insurance_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['a'])
        sex = request.form.get('b')
        bmi = float(request.form['c'])
        children = int(request.form['d'])    
        smoker = request.form.get('e')
        region = request.form.get('f')
       

        data = [[age,sex,bmi,children,smoker,region]]
          

        df = pd.DataFrame(data, columns=['age', 'sex' ,'bmi','children','smoker','region'])            
        
        
        
        
        
        my_prediction = model.predict(df)
        
        a = np.array(my_prediction)
        lis = my_prediction.tolist()
        my_prediction = round(lis[0],2)
        
        a.tofile('sample1.csv',sep=',')
        return render_template('result.html', prediction=my_prediction)
        

if __name__ == '__main__':
     app.run(debug=True)
