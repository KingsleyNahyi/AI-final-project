from flask import Flask, render_template, request
import joblib 
import pandas as pd 

app = Flask(__name__)

# Load the SVM model, scaler, and feature columns that were saved during training
print("Loading SVM model artifacts...")
model = joblib.load('svm_model.pkl')
scaler = joblib.load('svm_scaler.pkl')
feature_columns = joblib.load('svm_feature_columns.pkl')
print("✓ Model loaded successfully!\n")

def build_input_row(form): 
    """Transform raw input data into model-ready data"""
    
    # Dictionary of the inputs
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

    # One-hot encoding for the weekday given by the user
    weekday = form['appt_weekday']
    for col in feature_columns:
        if col.startswith('appt_weekday_'):
            row[col] = 1 if col == f'appt_weekday_{weekday}' else 0

    # Create DataFrame with all feature columns
    input_df = pd.DataFrame([row], columns=feature_columns)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    return input_df


def classify_risk(proba):
    """Classify no-show risk based on predicted probability"""
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
        # Build the input DataFrame
        input_df = build_input_row(request.form)

        # Scale using the SAME fitted scaler from training 
        numerical_columns = ['age', 'days_until_appointment']
        input_df_scaled = input_df.copy()
        input_df_scaled[numerical_columns] = scaler.transform(input_df[numerical_columns])

        # Get probability of no-show from SVM model
        # predict_proba returns [[prob_no, prob_yes]], we want prob_yes (index 1)
        proba = model.predict_proba(input_df_scaled)[0][1]

        # Classify risk level
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
