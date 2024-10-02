# Streamlit Lung Cancer Prediction

This project is a machine learning web application built using Streamlit to predict lung adenocarcinoma disease based on Chest CT-Scan images. The model classifies the input into one of four categories: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, or Normal.

The application has been containerized using Docker to make it easier to deploy and run across different environments.

## Project Overview

* Input: CT-scan image of the chest
* Output: Prediction of lung disease type with confidence scores
* Model: Pre-trained MobileNet model fine-tuned for lung cancer classification
* Classes: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal

## Features

* CT-Scan Image Upload: Users can upload a chest CT-scan image to get a prediction.
* Confidence Scores: The application returns confidence levels for each disease category.
* Educational Information: Provides a brief explanation of each cancer type.
* Comparative Analysis: Users can compare any 2 uploaded Chest CT-scan images to get better understanding.
* Downloadable Report: Users can download their analysis report in PDF Format.
 
## Repository Structure

* `app.py`
   Contains the Streamlit app logic that handles image uploads, makes predictions using the model, and displays results.

* `chest_cancer_model_fine_tuned.h5`
   The fine-tuned MobileNet model used to predict the type of lung cancer.

* `Dockerfile`
   The Dockerfile for containerizing the application, ensuring it runs in a consistent environment.

* `requirements.txt`
   Contains the list of dependencies needed to run the app, including Streamlit, TensorFlow, and other required libraries.

* `Chest_CT_Scans/`
   A folder for storing CT-scan images (sample or testing).

## How to Run the Project

### Local Setup
1. Clone the repository:
   
   ```
   git clone https://github.com/yourusername/streamlit-lung-cancer-predict.git
   cd streamlit-lung-cancer-predict
   ```
2. Install dependencies:

   You can install the required dependencies using the `requirements.txt` file:

   ```
   pip install -r requirements.txt
   ```
3. Run the app::

   ```
   streamlit run app.py
   ```
### Docker Setup
1. Pull the Docker image:

   Pull the pre-built Docker image from Docker Hub using the following command:

   ```
   docker pull subodhb57/my-streamlit-app
   ```
2. Run the Docker container:

   Once the image is pulled, run the Docker container:

   ```
   docker run -p 8501:8501 subodhb57/my-streamlit-app
   ```
3. Access the app:

   Open your browser and navigate to `http://localhost:8501` to interact with the web application.

### Model Details
* Model Architecture: MobileNet
* Dataset: The dataset consists of 4183 CT-scan images classified into 4 categories: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal. It has been split into training, testing, and 
  validation sets.
* Model File: `chest_cancer_model_fine_tuned.h5`

## Contributing
Feel free to open a pull request if you'd like to contribute. Any feedback or suggestions are welcome!
