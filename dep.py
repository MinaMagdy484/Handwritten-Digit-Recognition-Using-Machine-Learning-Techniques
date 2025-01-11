import streamlit as st
import numpy as np
import joblib
from PIL import Image
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Load the models
models = {
    "Linear Regression": joblib.load("linear_regression_(ovr).pkl"),
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Naive Bayes": joblib.load("naive_bayes.pkl"),
    "Neural Network": joblib.load("NeuralNetwork.pkl")
}

def create_neural_network():
    model = Sequential()
    model.add(Input(shape=(784,)))  # Input layer for flattened 28x28 grayscale images
    model.add(Dense(512, activation='relu'))  # First hidden layer
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(256, activation='relu'))  # Second hidden layer
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(10, activation='softmax'))  # Output layer (10 classes)
    return model

# Load model weights
def load_model_weights(model, weights_file):
    with open(weights_file, "rb") as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    return model

neural_network = create_neural_network()
model = load_model_weights(neural_network, "NeuralNetwork.pkl")
models["Neural Network"] = neural_network

# Preprocess the uploaded image
def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    img = image.convert('L')  # Convert to grayscale
    img_resized = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img_resized)  # Keep as 2D array for display
    img_flattened = img_array.flatten()  # Flatten to 1D
    img_normalized = img_flattened / 255.0  # Normalize to [0, 1]
    return img_array, img_normalized  # Return both 2D image and flattened array


def get_highest_probability(prediction):
    """Function to find the digit with the highest probability and print the result."""
    predicted_digit = np.argmax(prediction)
    highest_probability = prediction[0][predicted_digit]
    st.write(f"The predicted digit is: {predicted_digit}")
    st.write(f"Probability: {highest_probability:.8f}")


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
    - *Streamlit* for building the interactive web app.
    - *Scikit-learn* and *TensorFlow/Keras* for the machine learning pipeline.
    - *Joblib* and *Pickle* for model serialization.

    ### Authors
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

        # Preprocess the image
        img_2d, processed_image = preprocess_image(image)

        # Display the images side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", width=150)

        with col2:
            st.image(Image.fromarray(img_2d), caption="Preprocessed Image (28x28 Grayscale)", width=150)

        # Predict using the selected model
        if model_choice == "Neural Network":
             prediction = model.predict(np.expand_dims(processed_image, axis=0))
             get_highest_probability(prediction)
        else:
            prediction = selected_model.predict([processed_image])

        st.success(f"Predicted Digit using {model_choice}: {prediction[0]}" if model_choice != "Neural Network" else f"Predicted Digit using {model_choice}: {prediction}")
