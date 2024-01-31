import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# function to get an image, transform it to array and RGB (if necessary), preprocess, apply the model and predict the results

def xray_image(img):
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

    model = load_model('models/best_model_-10.h5')
    predictions = model.predict(img_array)

    labels = ['Elbow fracture', 'Fingers fracture', 'Forearm fracture', 'Wrist fracture', 'Humerus Fracture', 'Shoulder fracture']

    return dict(zip(labels, predictions[0]))
