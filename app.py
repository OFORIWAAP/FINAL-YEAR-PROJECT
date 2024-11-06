from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Load the model and preprocessor
model = load_model('student_gpa_predictor.h5')
preprocessor = joblib.load('preprocessor.pkl')

# Map text inputs to numerical values
def preprocess_input(data):
    data['Gender'] = 1 if data['Gender'].lower() == 'male' else 0
    data['Tutoring'] = 1 if data['Tutoring'].lower() == 'yes' else 0
    data['ParentalSupport'] = 1 if data['ParentalSupport'].lower() == 'yes' else 0
    data['Extracurricular'] = 1 if data['Extracurricular'].lower() == 'yes' else 0
    data['Sports'] = 1 if data['Sports'].lower() == 'yes' else 0
    data['Music'] = 1 if data['Music'].lower() == 'yes' else 0
    data['Volunteering'] = 1 if data['Volunteering'].lower() == 'yes' else 0
    return data

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract and preprocess data
        data = {
            'Age': int(request.form['Age']),
            'StudyTimeWeekly': float(request.form['StudyTimeWeekly']),
            'Absences': int(request.form['Absences']),
            'Gender': request.form['Gender'],
            'Ethnicity': int(request.form['Ethnicity']),
            'ParentalEducation': int(request.form['ParentalEducation']),
            'Tutoring': request.form['Tutoring'],
            'ParentalSupport': request.form['ParentalSupport'],
            'Extracurricular': request.form['Extracurricular'],
            'Sports': request.form['Sports'],
            'Music': request.form['Music'],
            'Volunteering': request.form['Volunteering'],
            'first_gpa': float(request.form['first_gpa']),
            'second_gpa': float(request.form['second_gpa']),
            'third_gpa': float(request.form['third_gpa']),
            'fourth_gpa': float(request.form['fourth_gpa'])
        }

        data = preprocess_input(data)
        input_data = pd.DataFrame([data])
        input_data_transformed = preprocessor.transform(input_data)

        prediction = model.predict(input_data_transformed)[0][0]

        # Redirect with the prediction as a query parameter
        return redirect(url_for('result', prediction=prediction))

    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)