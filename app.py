import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and scaler
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a function for prediction
def predict_cancer(features):
    # Standardize the features
    scaled_features = scaler.transform([features])
    # Make prediction
    prediction = model.predict(scaled_features)
    return prediction[0]

# Streamlit app
st.title('Breast Cancer Prediction')

st.write('Enter the values for the following features to predict the breast cancer diagnosis:')

# Input features
radius_mean = st.number_input('Radius Mean', min_value=0.0, step=0.1)
texture_mean = st.number_input('Texture Mean', min_value=0.0, step=0.1)
perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, step=0.1)
area_mean = st.number_input('Area Mean', min_value=0.0, step=0.1)
smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, step=0.1)
compactness_mean = st.number_input('Compactness Mean', min_value=0.0, step=0.1)
concavity_mean = st.number_input('Concavity Mean', min_value=0.0, step=0.1)
concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, step=0.1)
symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, step=0.1)
fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, step=0.1)

# Predict button
if st.button('Predict'):
    features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]
    prediction = predict_cancer(features)
    
    if prediction == 1:
        st.write('**Prediction: Malignant**')
    else:
        st.write('**Prediction: Benign**')
