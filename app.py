from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("/Users/anvitha/Downloads/Crop_Recommendation_App/crop_model (1).pkl", "rb") as f:
    scaler, model = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        input_data = np.array([data])
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        return render_template("result.html", prediction=prediction.capitalize())
    except:
        return "⚠️ Error in input. Please enter valid values."

if __name__ == "__main__":
    app.run(debug=True)
