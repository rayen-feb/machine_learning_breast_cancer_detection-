import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from PIL import Image
import io

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Styling (clinical + friendly)
# ----------------------
st.markdown("""
    <style>
    .css-18e3th9 {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #007ACC;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Breast Cancer Prediction App")
st.write("Predict whether a tumor is malignant or benign using multiple models. You can input numeric features or upload an image.")

# ----------------------
# Load scaler and models
# ----------------------
scaler = joblib.load("saved_models/scaler.pkl")

# GRU+SVM
gru_encoder = load_model("saved_models/gru_encoder.keras", compile=False)
gru_svm = joblib.load("saved_models/gru_svm.pkl")
gru_embedding_model = models.Model(
    inputs=gru_encoder.input,
    outputs=gru_encoder.get_layer("embedding").output
)

# Other models
mlp_model     = load_model("saved_models/mlp.keras", compile=False)
svm_model     = joblib.load("saved_models/SVM_model.pkl")
linear_model  = joblib.load("saved_models/linear_regression_model.pkl")
softmax_model = joblib.load("saved_models/softmax_model.pkl")
l1_knn_model  = joblib.load("saved_models/L1_KNN_model.pkl")
l2_knn_model  = joblib.load("saved_models/L2_KNN_model.pkl")

# ----------------------
# Sidebar: Model selection
# ----------------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Select model:",
    ["GRU+SVM", "MLP", "SVM", "Linear Regression", "Softmax", "L1-KNN", "L2-KNN"]
)

input_method = st.sidebar.radio(
    "Feature input method:",
    ["Manual input", "Paste row of 30 numbers", "Upload tumor image"]
)

# ----------------------
# Feature input
# ----------------------
features = None
img_array = None

if input_method == "Manual input":
    if "features" not in st.session_state:
        st.session_state.features = [0.0] * 30
    st.subheader("Enter 30 numeric features")
    cols = st.columns(6)
    for i in range(30):
        with cols[i % 6]:
            st.session_state.features[i] = st.number_input(
                f"F{i+1}", value=st.session_state.features[i], key=f"f{i}"
            )
    features = st.session_state.features

elif input_method == "Paste row of 30 numbers":
    st.subheader("Paste a row of 30 numbers separated by commas")
    input_str = st.text_area("Input features here")
    if input_str:
        try:
            features = [float(x.strip()) for x in input_str.split(",")]
            if len(features) != 30:
                st.error("Exactly 30 values are required.")
                features = None
        except:
            st.error("Invalid input format.")
            features = None

elif input_method == "Upload tumor image":
    st.subheader("Upload tumor image (PNG/JPG)")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("L")  # grayscale
        img = img.resize((128,128))
        img_array = np.array(img)/255.0
        img_array = img_array.reshape(1,128,128,1)
        st.image(img, caption="Uploaded Image", use_column_width=True)

# ----------------------
# Prediction logic
# ----------------------
def predict_numeric(features, model_name):
    X = np.array(features).reshape(1, -1)
    if model_name == "GRU+SVM":
        X_seq = X.reshape((1,30,1))
        embedding = gru_embedding_model.predict(X_seq, verbose=0)
        pred = gru_svm.predict(embedding)[0]
    else:
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

# Placeholder for CNN image prediction
def predict_image(img_array):
    # Replace cnn_model with your trained CNN
    # cnn_pred = cnn_model.predict(img_array, verbose=0)[0][0]
    # return "Malignant" if cnn_pred>=0.5 else "Benign"
    return "Feature not implemented yet"

# ----------------------
# Prediction buttons
# ----------------------
if st.button("Predict"):
    if features is not None:
        prediction = predict_numeric(features, model_name)
        st.success(f"Predicted class: **{prediction}**")
    elif img_array is not None:
        prediction = predict_image(img_array)
        st.success(f"Predicted class: **{prediction}**")
    else:
        st.warning("Please provide input features or upload an image.")

# ----------------------
# Test known samples
# ----------------------
if st.button("Test known MALIGNANT"):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    i = np.where(y == 0)[0][0]
    st.write("True label: Malignant")
    st.write("Prediction:", predict_numeric(X[i], model_name))

if st.button("Test known BENIGN"):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    i = np.where(y == 1)[0][0]
    st.write("True label: Benign")
    st.write("Prediction:", predict_numeric(X[i], model_name))
