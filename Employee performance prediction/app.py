from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature order
model = joblib.load(open("model.pkl", "rb"))
features = joblib.load(open("feature_order.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def predict():
    return render_template("predict.html", features=features)

@app.route('/submit', methods=["POST"])
def submit():
    inputs = [float(request.form[feature]) for feature in features]
    df = pd.DataFrame([inputs], columns=features)
    result = model.predict(df)[0]
    return render_template("submit.html", result=round(result, 4))

if __name__ == '__main__':
    app.run(debug=True)
