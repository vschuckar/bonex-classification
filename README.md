# Bone Fracture Classification 

Anatomical site studied: **Upper limbs**

Computer Vision Project 

Original dataset: https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/data

## Introduction 

* Diagnostic errors in medicine, including radiology, indicate inadequate patient care and can vary from minimal to potentially life-threatening.
* Radiology accounts for a significant number of malpractice claims, with errors in interpretation leading to missed diagnoses in nearly 75% of cases.
* Fractures can lead to impaired life quality, disability, health decline, and are a serious economic burden to individuals, their families, societies, and health-care systems. 

In order to aid patient care & support medical diagnostics tools, this dataset will be used to develop a bone fracture classification & detection model. 

In this interaction, the classification part will be done and on the next one, the detection will also be introduced. 

## Methodology

### Data "cleaning" and preprocessing 

Data structure: 
* Three folders: train, valid, and test.
  * Inside each folder, two folders: images and labels.
    * images folder: x-ray images of different fracture sites.
    * labels folder: .txt files with the following structure -> class number and bounding box or pixel-level segmentation masks to indicate the location and extent of the detected fracture.
  
Problems with the dataset: 
* The names of the image files and txt files were the same for each different file but did not have any useful information on it.
* The classification of the images were the following numbers: 0, 1, 2, 4, 5, and 6 which can lead to confusion due to the missing 3.
* The fracture names are: Elbow Positive, Fingers Positive, Forearm Fracture, Humerus Fracture, Shoulder Fracture, and Wrist Positive but there were not 
  clearly connected in the data source with the class numbers. 
* A lot of the txt were empty, meaning there was a hidden 'class' in the images. This will be discussed later.

Solutions: 
1. Extract the class number from the txt file and put it on the image file name and the txt file name, obviously matching image/txt file names for each file.
2. Change the number of class 6 to class 3. 
3. Investigate the images and confirm to each fracture name each class belong. 
4. Skip empty .txt files while renaming. This will be discussed later. 

Solution 3. 
* Class 0 - Elbow Positive 
* Class 1 - Fingers Positive 
* Class 2 - Forearm Positive 
* Class 3 - Wrist Positive 
* Class 4 - Humerus Fracture
* Class 5 - Shoulder Fracture

### Data "transformation", modelling and analysis

1. After some changes to the dataset (more information on the notebooks), all the sets were transformed to a dataframe.
2. Data augmentation through image data generator
3. Pre-trained models for training
4. Predictions
5. Metrics analysis

## Challenges

* There was no detailed information about the dataset setup.
* After inspection I've realised that besides the body parts classes, there was another class that I believe it is the "not fractured" one.
* In order for that class to be introduced, I also need to introduce the object detection part of the project.
* Since the project is time constrained (and the model training is time consuming), I will present what I've done so far and work on the changes & improvements right afterwards. 

Below a summary of the models used, the fine-tunning applied and the overall results. 

![InceptionV3](https://github.com/vschuckar/final_project/assets/149705224/bd6204b1-e2fd-4d66-b757-dc22d4da3890)

## Conclusion & Next Steps 

* Bone Fracture Classification - Maximum Accuracy reached: 81%
* Increase the model capability introducing object detection
 
