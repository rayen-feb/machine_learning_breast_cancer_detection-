from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from PIL import Image

app = Flask(__name__)

feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]
cnn_model = load_model("saved_models/cnn_model.keras", compile=False)

# ----------------------
# Load models once
# ----------------------
scaler = joblib.load("saved_models/scaler.pkl")
mlp_model = load_model("saved_models/mlp.keras", compile=False)

# ----------------------
# Prediction logic
# ----------------------
def predict_numeric(features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = mlp_model.predict(X_scaled, verbose=0)[0][0]
    return "Malignant" if prob >= 0.5 else "Benign"

# ----------------------
# Routes
# ----------------------
@app.route("/result", methods=["POST"])
def result():
    method = request.form.get("method")
    prediction = None

    if method == "manual":
        features = [float(request.form.get(f"f{i+1}", 0)) for i in range(30)]
        prediction = predict_numeric(features)

    elif method == "paste":
        row = request.form.get("row")
        try:
            features = [float(x.strip()) for x in row.split(",")]
            if len(features) == 30:
                prediction = predict_numeric(features)
            else:
                prediction = "Error: must be 30 values"
        except Exception:
            prediction = "Error: invalid format"

    elif method == "image":
        file = request.files["image"]
        if file:
            img = Image.open(file).convert("L").resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 128, 128, 1)
            prediction = "Image prediction not yet implemented"

    return render_template("result.html", prediction=prediction)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        method = request.form.get("method")
        prediction = None

        if method == "manual":
            # Collect 30 features from form inputs
            features = [float(request.form.get(f"f{i+1}", 0)) for i in range(30)]
            prediction = predict_numeric(features)

        elif method == "paste":
            row = request.form.get("row")
            try:
                features = [float(x.strip()) for x in row.split(",")]
                if len(features) == 30:
                    prediction = predict_numeric(features)
                else:
                    prediction = "Error: must be 30 values"
            except Exception:
                prediction = "Error: invalid format"

        elif method == "image":
            file = request.files["image"]
            if file:
                img = Image.open(file).convert("L").resize((128, 128))
                img_array = np.array(img) / 255.0
                img_array = img_array.reshape(1, 128, 128, 1)
                # Placeholder until CNN model is added
                prediction = "Image prediction not yet implemented"

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

# ----------------------
# Run server
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
