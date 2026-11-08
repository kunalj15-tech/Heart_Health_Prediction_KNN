
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Get the current project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load our trained ML files
model = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        age = int(request.form["age"])
        sex = request.form["sex"]
        chest_pain = request.form["chest_pain"]

        resting_bp = int(request.form["resting_bp"])
        cholesterol = int(request.form["cholesterol"])
        fasting_bs = int(request.form["fasting_bs"])

        resting_ecg = request.form["resting_ecg"]
        max_hr = int(request.form["max_hr"])
        exercise_angina = request.form["exercise_angina"]

        oldpeak = float(request.form["oldpeak"])
        st_slope = request.form["st_slope"]

        # Numerical input values
        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak
        }

        # Categorical input values
        categories = {
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingECG": resting_ecg,
            "ExerciseAngina": exercise_angina,
            "ST_Slope": st_slope
        }

        # Create one-hot encoded columns
        for col in columns:
            for category, value in categories.items():

                if col.startswith(category + "_"):
                    input_data[col] = int(
                        col == category + "_" + value
                    )

        # Create DataFrame in training column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(
            columns=columns,
            fill_value=0
        )

        # Scale the input
        scaled_data = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_data)[0]

        # Display result
        if prediction == 1:
            result = "High Risk of Heart Disease"
            status = "danger"
        else:
            result = "Low Risk of Heart Disease"
            status = "success"

        return render_template(
            "index.html",
            result=result,
            status=status
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
