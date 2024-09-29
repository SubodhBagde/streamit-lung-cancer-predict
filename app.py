import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Lung Cancer Prediction App", layout="wide")

# Load the model
@st.cache_resource
def load_prediction_model():
    return load_model('chest_cancer_model_fine_tuned.h5')

# Define class names
class_names = ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']

# Function to preprocess the image
def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Function to make prediction
def predict(image):
    model = load_prediction_model()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0]

# Function to plot confidence scores
def plot_confidence_scores(predictions):
    fig, ax = plt.subplots()
    sns.barplot(x=predictions, y=class_names, ax=ax)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Prediction Confidence Scores')
    return fig

# Function to display cancer information
def display_cancer_info(cancer_type):
    info = {
        "Adenocarcinoma": """
        - Most common type of lung cancer
        - Often found in outer areas of the lung
        - Tends to grow slower than other types
        - Common in both smokers and non-smokers
        """,
        "Large cell carcinoma": """
        - Tends to grow and spread quickly
        - Can appear in any part of the lung
        - Often diagnosed at later stages
        - Accounts for about 10-15% of lung cancers
        """,
        "Squamous cell carcinoma": """
        - Often linked to a history of smoking
        - Usually found in the central part of the lungs
        - Tends to grow slower than other types
        - Accounts for about 25-30% of all lung cancers
        """,
        "Normal": """
        - No signs of cancerous cells
        - Regular lung structure and function
        - Important for early detection and comparison
        """
    }
    st.sidebar.write(info[cancer_type])

# Main function
def main():
    st.title("Lung Cancer Prediction from CT Scan Images")
    st.write("Upload a Chest CT Scan image to predict the type of lung cancer.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CT Scan image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded CT Scan", use_column_width=True)

        # Preprocess the image and make prediction
        img = image.load_img(uploaded_file, target_size=(224, 224))
        predictions = predict(img)

        # Display prediction results
        with col2:
            st.subheader("Prediction Results")
            predicted_class = class_names[np.argmax(predictions)]
            st.write(f"Predicted Class: **{predicted_class}**")
            st.write("Confidence Scores:")
            for class_name, confidence in zip(class_names, predictions):
                st.write(f"{class_name}: {confidence:.2%}")
            
            # Plot confidence scores
            fig = plot_confidence_scores(predictions)
            st.pyplot(fig)

        # User feedback
        st.subheader("Provide Feedback")
        feedback = st.radio("Was this prediction correct?", ("Yes", "No", "Unsure"))
        if st.button("Submit Feedback"):
            # Here you would typically save this feedback to a database
            st.success("Thank you for your feedback! It will help us improve our model.")

    # Educational section
    st.sidebar.title("Learn About Lung Cancer Types")
    cancer_type = st.sidebar.selectbox("Select a cancer type to learn more:", class_names)
    display_cancer_info(cancer_type)

    # Footer
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Disclaimer: This app is for educational purposes only. Always consult with a healthcare professional for medical advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()