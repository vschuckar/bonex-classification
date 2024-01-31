import os
import hashlib
import re

import numpy as np 

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns

# function to generate a short name for the image and .txt files, with the first character of the name being the first character of the .txt file

def generate_short_name(file_name):
    '''
    This function receives a file name and return a short random 8 characters name, unique for each file. 
    Input: File name
    Output: New random unique file name with 8 characters
    '''
    return hashlib.md5(file_name.encode('utf-8')).hexdigest()[:8]

def rename_images_with_classification(folder_path):
    '''
    This function receives a folder path and, inside of the folder, looks for the files that end with .jpg and .txt.
    Then, it gets the first character inside for the .txt file, which becomes the first character of the new .jpg file name, 
    the rest of the .jpg file name is a short random 8 characters string. 
    Input: Folder path
    Output: New name for the .jpg, with a specific first character and new 8 random characters
    '''
    for subdir in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subdir)):
            images_path = os.path.join(folder_path, subdir, 'images')
            labels_path = os.path.join(folder_path, subdir, 'labels')

            for file_name in os.listdir(images_path):
                if file_name.endswith('.jpg'):
                    image_path = os.path.join(images_path, file_name)

                    label_file_path = os.path.join(labels_path, file_name.replace('.jpg', '.txt'))

                    if not os.path.exists(label_file_path):
                        print(f"Label file not found for {file_name}. Skipping.")
                        continue

                    with open(label_file_path, 'r') as label_file:
                        classification = label_file.read(1)

                        if not classification:
                            print(f"Skipping {label_file_path} as it is empty.")
                            continue

                    short_name = generate_short_name(file_name)

                    new_file_name = f"{classification}_{short_name}.jpg"

                    os.rename(image_path, os.path.join(images_path, new_file_name))

# function to delete the empty txt files

def delete_empty_txt_files(folder_path):
    '''
    This function receives a folder path and access its .txt files. If it is empty, it deletes the file. 
    Input: Folder path
    Output: Deletes empty .txt files
    '''
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith(".txt") and os.path.getsize(file_path) == 0:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# creating a function to change the class number to another number 

def rename_images_with_new_class(folder_path, old_class, new_class):
    '''
    This function receives a folder path, access its files and, if it is a .jpg file, changes the first character of its name to another.
    Input: Folder path, old first character of the file name, new first character of the file name
    Output: Changed first character of the .jpg file name 
    '''
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(folder_path, file_name)
            
            class_match = re.match(r'^(\d+)_', file_name)
            
            if class_match:
                class_label = class_match.group(1)
                
                if class_label == str(old_class):
                    new_file_name = f"{new_class}_{file_name[len(class_label) + 1:]}"
                    
                    os.rename(image_path, os.path.join(folder_path, new_file_name))
            else:
                print(f"Skipping {file_name} as it does not follow the expected pattern.")

# function to receive and load an image, predicting its fracture class and plotting a graph for it

def xray_image(img_path, model):
    '''
    This function receives the location path of an image and a model. It loads and transforms the image in order to apply 
    a model on it in order to make predictions. Then it shows the image and its predictions graph side by side.
    Input: Image path (location) and model (optional)
    Output: The image and the graph of its predictions side by side
    '''
    model = load_model('best_model_-10.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    img = image.load_img(img_path, target_size=(299, 299))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    labels = ['Elbow fracture', 'Fingers fracture', 'Forearm fracture', 'Wrist fracture', 'Humerus Fracture', 'Shoulder fracture']
    image_to_plot = img

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [2, 5]})
    axes[0].imshow(image_to_plot, cmap='gray')
    axes[0].set_title('X-ray')

    sns.barplot(x=labels, y=predictions[0], hue=labels, palette='Set2', ax=axes[1])
    axes[1].set_title('Predictions')
    axes[1].set_ylim([0, 1])  
    axes[1].set_ylabel('Probability')  
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)

    return plt.show()

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
