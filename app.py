from flask import Flask, render_template, request
import joblib 
import pandas as pd 

app = Flask (__name__)

##joblib iis used to load the model, scaler and feature columns that were saved during the training phase.
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

def build_input_row(form): 
  #here the raw input data is transformed into model ready data 

#dictionary of the inputs
  row = {
    'age': int(form['age']), 
    'gender': 1 if form['gender'] =='M' else 0, 
    'scholarship': int(form.get('scholarship', 0)), 
    'hypertension': int(form.get('hypertension', 0)),
    'diabetes': int(form.get('diabetes', 0)),
    'alcoholism': int(form.get('alcoholism', 0)), 
    'handicap': int(form.get('handicap', 0)), 
    'sms_received': int(form['sms_received']),
    'days_until_appointment': int(form['days_until_appointment']),
  }

#one hot encoding for the weekday given by the user. 

  weekday = form['appt_weekday']
  for col in feature_columns:
    if col.startswith('appt_weekday_'):
      row[col] = 1 if col == f'appt_weekday_{weekday}' else 0


  input_df = pd.DataFrame([row], columns=feature_columns)
  input_df = input_df.reindex(columns=feature_columns, fill_value=0)

#classifying the risk of whether the patient will attend or not based on the predicted probability.
def classify_risk(proba):
    ##it returns three variables being the risk level, the color we want displayed and a message
    if proba < 0.3:
        return "Low", "green", "Patient likely to attend"
    elif proba < 0.6:
        return "Medium", "orange", "Consider a reminder call"
    else:
        return "High", "red", "Recommend SMS reminder / overbook this slot"


@app.route('/', methods=['GET', 'POST'])

def index():
    context = {}

    if request.method == 'POST':
        input_df = build_input_row(request.form)

        # Scale using the SAME fitted scaler from training 
        input_scaled = scaler.transform(input_df)

        # this retrieves one of the proabbility numbers from the two categories. 
        proba = model.predict_proba(input_scaled)[0][1]

        risk_level, color, message = classify_risk(proba)

        context = {
            'proba': round(proba * 100, 1),
            'risk_level': risk_level,
            'color': color,
            'message': message,
        }

    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True)

