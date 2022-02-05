# install dependencies
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import cv2
import matplotlib.pyplot as plt
import math
from scipy import stats
import operator
import argparse
import sys, getopt


parser = argparse.ArgumentParser(description='Poseconnect')
parser.add_argument('--training_dir', type=str, default='../training/',
                    help='location of training files')
parser.add_argument('--testing_dir', type=str, default='../testing/',
                    help='location of testing files')

args = parser.parse_args()

# get dataframe with viable filepaths and the associated name of subject

# get filepaths for for training and test datasets – test dataset contains one image per subject, training set
# contains remaining images

# Input for training directory
# rootdir = args.training_dir
rootdir = ""


paths = []
names = []
start = time.time()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if subdir != rootdir and file != '.DS_Store':
            filepath = str(os.path.join(subdir, file))
            paths.append(filepath)
            name = subdir.split("TD/")[1]
            names.append(name)

# create pandas dataframe containing the filepath of each image and their associated subject name
dataframe_dict = {}
dataframe_dict['names'] = names
dataframe_dict['filepath'] = paths

all_data_df = pd.DataFrame.from_dict(dataframe_dict)

# list containing encodings for training faces
known_face_encodings = []

# list containing labels for training faces
known_face_names = []

not_recognized = []

# iterate through each training image, learn encoding, append true label to known_face_names
train_paths = all_data_df['filepath'].tolist()
train_names = all_data_df['names'].tolist()

start = time.time()
for image_num in range(len(train_paths)):
    image = face_recognition.load_image_file(train_paths[image_num])
    boxes = face_recognition.face_locations(image, model="cnn")
    try:
        face_encoding = face_recognition.face_encodings(image, boxes)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(train_names[image_num])
    except:
        not_recognized.append(image_num)
end = time.time()

# Input for testing directory
# rootdir = args.testing_dir
rootdir = ""


kinectaa_paths = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file != '.DS_Store':
            filepath = str(os.path.join(subdir, file))
            kinectaa_paths.append(filepath)

kinectbb_paths = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file != '.DS_Store':
            filepath = str(os.path.join(subdir, file))
            kinectbb_paths.append(filepath)

kinectcc_paths = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file != '.DS_Store':
            filepath = str(os.path.join(subdir, file))
            kinectcc_paths.append(filepath)

video_filepaths = kinectaa_paths + kinectbb_paths + kinectcc_paths
video_filepaths.sort()
true_names = ['bertrand']*len(video_filepaths)

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    """
    Converts Euclidean distance between two faces (one known, one unknown) into a percentage match score
    (from: https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage)
    """
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

# list containing predicted names for test faces
predicted = []

# list containing true labels for test faces
true_labels = []

# list containing all images that a face was detected in, for prediction
used_filepaths = []

# list containing prediction confidence scores for each prediction, calculated from Euclidean distance
distances = []

# iterate through each image in test set and generate predictions

for test_image_num in range(len(video_filepaths)):
    unknown_image = face_recognition.load_image_file(video_filepaths[test_image_num])
    face_locations = face_recognition.face_locations(unknown_image, model="cnn")
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    try:
        face_encoding = face_encodings[0]

        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # 
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            used_filepaths.append(video_filepaths[test_image_num])
            distances.append(face_distance_to_conf(np.min(face_distances)))
        
        # update predictions and true labels listts
        predicted.append(name)
        true_labels.append(true_names[test_image_num])
    
    except:
        pass
end = time.time()
print('Elapsed time: (seconds)', (end-start))


test_accuracy = 1 - (sum(np.array(predicted) != np.array(true_labels))/(len(np.array(predicted))))
print("Test accuracy: ", test_accuracy)
print("Predicted Result: ", np.array(predicted))