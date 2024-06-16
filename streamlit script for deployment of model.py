import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model
model = tf.keras.models.load_model('VGG-model.h5')

def classify_image(image):
    # Preprocess the image as required by your model
    image = image.resize((224, 224)) # Resize the image to the expected input size of your model
    image = np.array(image) # Convert the image to a numpy array
    image = image / 255.0 # Normalize the image to [0,1]
    
    # Ensure the input has the correct shape, including the batch dimension
    image = np.expand_dims(image, axis=0) # Add an extra dimension for the batch size
    
    # Make a prediction
    predictions = model.predict(image)
    # Assuming the model returns a single prediction for the class
    prediction_class = np.argmax(predictions)
    # Map the prediction class to a human-readable label
    if prediction_class == 0:
        return "Cat"
    elif prediction_class == 1:
        return "Dog"
    else:
        return "Unknown"

st.title('Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button('Classify'):
        prediction_label = classify_image(image)
        st.write(f"The image is classified as: {prediction_label}")