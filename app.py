from flask import Flask

import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

## import ridge regressor model and standard scaler model..
rf_classifier=pickle.load(open('models/rf_classifier.pkl','rb'))
standard_scaler=pickle.load(open('models/std.pkl','rb'))
crop={1:'rice',2:'maize',3:'jute',4:'cotton',5:'coconut',6:'papaya',7:'orange',
            8:'apple',9:'muskmelon',10:'watermelon',11:'grapes',12:'mango',13:'banana',
            14:'pemogranate',15:'lentil',16:'blackgram',17:'mungbean',18:'mothbeans',
            19:'pigeonpeas',20:'kidneybeans',21:'chickpea',22:'coffee'}

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')




@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        N = float(request.form.get('N'))
        P = float(request.form.get('P'))
        K = float(request.form.get('K'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        rainfall = float(request.form.get('rainfall'))
        ph = float(request.form.get('ph'))
        

        new_data_scaled=standard_scaler.transform([[N,P,K,temperature,humidity,rainfall,ph]])
        result=rf_classifier.predict(new_data_scaled)

   
    
   

        return render_template('home.html',result=crop[result[0]])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
