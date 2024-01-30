import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


def xray_image(img):
    '''
    '''
    img_rgb = tf.image.grayscale_to_rgb(image.img_to_array(img))
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = preprocess_input(img_array)

    model = load_model('models/best_model_-10.h5')
    predictions = model.predict(img_array)

    labels = ['Elbow fracture', 'Fingers fracture', 'Forearm fracture', 'Wrist fracture', 'Humerus Fracture', 'Shoulder fracture']

    return dict(zip(labels, predictions[0]))
