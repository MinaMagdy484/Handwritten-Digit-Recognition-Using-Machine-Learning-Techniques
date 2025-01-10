import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load the models
models = {
    "Linear Regression": joblib.load("linear_regression.pkl"),
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Naive Bayes": joblib.load("naive_bayes.pkl"),
    #"Neural Network": joblib.load("neural_network.pkl")
}

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    img = image.convert('L')  # Convert to grayscale
    img_resized = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img_resized).flatten()  # Flatten to 1D
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Sidebar navigation
st.sidebar.title("Features")
page = st.sidebar.radio("Go to", ["About the Project", "Image Classification"])

if page == "About the Project":
    st.title("About the Project")
    st.write("""
    This Image Classification app is designed to classify images into digits from 0 to 9.

    ### Features
    - Upload an image to classify it as a digit.
    - Choose from multiple machine learning models.

    ### Technology Used
    - **Streamlit** for building the interactive web app.
    - **Scikit-learn** for the machine learning pipeline.
    - **Joblib** for model serialization.

    ### Author
    Developed by Bavley Adel, Potros Atia, Mariam Essam, Bishoy Adel, Mina Magdy.
    """)

elif page == "Image Classification":
    st.title("Image Classification")

    # Model selection
    model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))
    selected_model = models[model_choice]

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict using the selected model
        prediction = selected_model.predict([processed_image])
        st.success(f"Predicted Digit using {model_choice}: {prediction[0]}")
