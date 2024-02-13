import numpy as np 

import streamlit as st
import requests
from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# function to load a cached model in order to avoid loading the model everytime the app is being used - git LFS bandwidth depletes very fast

@st.experimental_singleton
def load_cached_model(model_url):
    response = requests.get(model_url)
    model_content = BytesIO(response.content)
    return load_model(model_content)

# function to get an image, transform it to array and RGB (if necessary), preprocess, apply the model and predict the results

def xray_image(img, model):
    '''
    This is a function to get an image, transform it to array and RGB (if necessary), preprocess, apply the model 
    and make predictions. 
    Input: img = image location/path
    Output: A dictionary with the labels and its predictions
    '''
    img_array = image.img_to_array(img)

    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
        
    img_array = np.expand_dims(image.img_to_array(img_array), axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    labels = ['Elbow fracture', 'Fingers fracture', 'Forearm fracture', 'Wrist fracture', 'Humerus Fracture', 'Shoulder fracture']

    return dict(zip(labels, predictions[0]))
