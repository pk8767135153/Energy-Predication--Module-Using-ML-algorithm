import pickle 
import numpy as np
import pandas as pd
from flask import Flask, request, render_template,jsonify

app = Flask(__name__)
# Load model 
regmodel = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Home Page
@app.route('/')
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/feedback_form")
def feedback_form():
    return render_template("feedback.html")


# Prediction Page
@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route('/submit', methods=["POST"])
def submit():
    building_types = request.form['building_types']
    square_footage = request.form['square_footage']
    number_of_occupants = request.form['number_of_occupants']
    no_of_appliances = request.form['no_of_appliances']
    avg_temperature = request.form['avg_temperature']
    week_days = request.form['week_days']
    print(building_types,square_footage,number_of_occupants,no_of_appliances,avg_temperature,week_days)
    
    if building_types == "residential":
        building_types = 1
    elif building_types == "commercial":
        building_types = 2
    elif building_types == "industrial":
        building_types = 3
    else: 
        building_types = 0
    
    if week_days == "weekend" or week_days or "sunday":
        week_days = 0
    else: 
        week_days = 1
        
    data = {
        'Building Types': building_types,
        'Square Footage': square_footage,
        'Number of Occupants': number_of_occupants,
        'Number of Appliances': no_of_appliances,
        'Average Temperature': avg_temperature,
        'Week Days': week_days
    }
    values = data.values()
    l = [[int(i) for i in values]]
    new_data = scaler.transform(l)
    output = regmodel.predict(new_data)
    return f"Data {data} \n Values {values} l {l}"
if __name__ == "__main__":
    app.run(debug=True)
