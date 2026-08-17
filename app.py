from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# joblib is used to load the model, scaler and feature columns that were saved during the training phase.
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# The scaler was only fit on these two numerical columns during training
numerical_columns = ['age', 'days_until_appointment']

# Build the list of neighbourhood options directly from the saved feature
# columns, so the form always matches whatever the model was actually trained on.
neighbourhood_options = sorted([
    col.replace('neighbourhood_', '') for col in feature_columns
    if col.startswith('neighbourhood_')
])


def build_input_row(form):
    # here the raw input data is transformed into model ready data

    # dictionary of the inputs
    row = {
        'age': int(form['age']),
        'gender': 1 if form['gender'] == 'M' else 0,
        'scholarship': int(form.get('scholarship', 0)),
        'hypertension': int(form.get('hypertension', 0)),
        'diabetes': int(form.get('diabetes', 0)),
        'alcoholism': int(form.get('alcoholism', 0)),
        'handicap': int(form.get('handicap', 0)),
        'sms_received': int(form['sms_received']),
        'days_until_appointment': int(form['days_until_appointment']),
    }

    # one hot encoding for the weekday given by the user.
    weekday = form['appt_weekday']
    for col in feature_columns:
        if col.startswith('appt_weekday_'):
            row[col] = 1 if col == f'appt_weekday_{weekday}' else 0

    # one hot encoding for the neighbourhood given by the user.
    neighbourhood = form['neighbourhood']
    for col in feature_columns:
        if col.startswith('neighbourhood_'):
            row[col] = 1 if col == f'neighbourhood_{neighbourhood}' else 0

    # neighbourhood columns that weren't the selected one). reindex() below
    # is what correctly fills every remaining expected column with 0.
    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    return input_df


def classify_risk(proba):
    # it returns three variables being the risk level, the color we want displayed and a message
    if proba < 0.3:
        return "Low", "green", "Patient likely to attend"
    elif proba < 0.6:
        return "Medium", "orange", "Consider a reminder call"
    else:
        return "High", "red", "Recommend SMS reminder / overbook this slot"


@app.route('/', methods=['GET', 'POST'])
def index():
    context = {'neighbourhood_options': neighbourhood_options}

    if request.method == 'POST':
        input_df = build_input_row(request.form)

        # this retrieves one of the probability numbers from the two categories.
        proba = model.predict_proba(input_df)[0][1]

        risk_level, color, message = classify_risk(proba)

        context.update({
            'proba': round(proba * 100, 1),
            'risk_level': risk_level,
            'color': color,
            'message': message,
        })

    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True)
