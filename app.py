
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load our trained ML model
model = joblib.load("friction_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        int(data["clicks"]),
        int(data["errors"]),
        int(data["time_seconds"]),
        int(data["backtracks"]),
        int(data["help_requests"]),
        int(data["task_completed"])
    ]]

    prediction = int(model.predict(features)[0])

    if prediction == 1:
        result = "High Friction"
    else:
        result = "Low Friction"

    return jsonify({
        "prediction": prediction,
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)
