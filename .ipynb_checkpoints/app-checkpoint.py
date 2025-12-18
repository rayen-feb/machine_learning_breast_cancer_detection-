import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from sklearn.datasets import load_breast_cancer

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("Breast Cancer Prediction App")
st.write("Predict whether a tumor is malignant or benign using multiple models.")

# ----------------------
# Load scaler (USED BY ALL NON-GRU MODELS)
# ----------------------
scaler = joblib.load("saved_models/scaler.pkl")

# ----------------------
# Load models
# ----------------------
gru_encoder = load_model("saved_models/gru_encoder.keras", compile=False)
gru_svm = joblib.load("saved_models/gru_svm.pkl")

# Extract GRU embedding
gru_embedding_model = models.Model(
    inputs=gru_encoder.input,
    outputs=gru_encoder.get_layer("embedding").output
)

mlp_model     = load_model("saved_models/mlp.keras", compile=False)
svm_model     = joblib.load("saved_models/SVM_model.pkl")
linear_model  = joblib.load("saved_models/linear_regression_model.pkl")
softmax_model = joblib.load("saved_models/softmax_model.pkl")
l1_knn_model  = joblib.load("saved_models/L1_KNN_model.pkl")
l2_knn_model  = joblib.load("saved_models/L2_KNN_model.pkl")

# ----------------------
# Model selection
# ----------------------
model_name = st.selectbox(
    "Select model:",
    ["GRU+SVM", "MLP", "SVM", "Linear Regression", "Softmax", "L1-KNN", "L2-KNN"]
)

# ----------------------
# Feature input
# ----------------------
input_method = st.radio(
    "Feature input method:",
    ["Manual input", "Paste row of 30 numbers"]
)

features = None

if input_method == "Manual input":
    if "features" not in st.session_state:
        st.session_state.features = [0.0] * 30

    cols = st.columns(6)
    for i in range(30):
        with cols[i % 6]:
            st.session_state.features[i] = st.number_input(
                f"F{i+1}",
                value=st.session_state.features[i],
                key=f"f{i}"
            )
    features = st.session_state.features

else:
    input_str = st.text_area("Paste a row of 30 numbers separated by commas")
    if input_str:
        try:
            features = [float(x.strip()) for x in input_str.split(",")]
            if len(features) != 30:
                st.error("Exactly 30 values are required.")
                features = None
        except:
            st.error("Invalid input format.")
            features = None

# ----------------------
# Prediction logic (FIXED)
# ----------------------
def predict(features, model_name):
    X = np.array(features).reshape(1, -1)

    if model_name == "GRU+SVM":
        # ❗ GRU WAS TRAINED ON RAW FEATURES
        X_seq = X.reshape((1, 30, 1))
        embedding = gru_embedding_model.predict(X_seq, verbose=0)
        pred = gru_svm.predict(embedding)[0]

    else:
        # ❗ ALL OTHER MODELS USE SCALED FEATURES
        X_scaled = scaler.transform(X)

        if model_name == "MLP":
            prob = mlp_model.predict(X_scaled, verbose=0)[0][0]
            pred = int(prob >= 0.5)

        elif model_name == "Softmax":
            pred = softmax_model.predict(X_scaled)[0]

        elif model_name == "SVM":
            pred = svm_model.predict(X_scaled)[0]

        elif model_name == "Linear Regression":
            pred = int(linear_model.predict(X_scaled)[0] >= 0.5)

        elif model_name == "L1-KNN":
            pred = l1_knn_model.predict(X_scaled)[0]

        elif model_name == "L2-KNN":
            pred = l2_knn_model.predict(X_scaled)[0]

    return "Malignant" if pred == 0 else "Benign"


# ----------------------
# Predict button
# ----------------------
if st.button("Predict") and features is not None:
    prediction = predict(features, model_name)
    st.success(f"Predicted class: **{prediction}**")

# ----------------------
# Honest test (sklearn malignant sample)
# ----------------------
if st.button("Test known malignant"):
    data = load_breast_cancer()
    X, y = data.data, data.target
    malignant_index = np.where(y == 0)[0][0]  # 0 = malignant
    st.write("Prediction:", predict(X[malignant_index], model_name))
