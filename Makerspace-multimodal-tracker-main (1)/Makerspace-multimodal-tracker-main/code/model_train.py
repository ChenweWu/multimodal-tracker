# install dependencies
import face_recognition
import numpy as np
import os
import pandas as pd
import time
import argparse
import pickle
from pathlib import Path

def training_model(training_path, weight_dir):
    # Input for training directory
    # rootdir = args.training_dir
    rootdir = training_path

    paths = []
    names = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if subdir != rootdir and file != '.DS_Store':
                filepath = str(os.path.join(subdir, file))
                paths.append(filepath)
                name = subdir.split("training/")[1]
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

    for image_num in range(len(train_paths)):
        image = face_recognition.load_image_file(train_paths[image_num])
        boxes = face_recognition.face_locations(image, model="cnn")
        try:
            face_encoding = face_recognition.face_encodings(image, boxes)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(train_names[image_num])
        except:
            not_recognized.append(image_num)

    print(known_face_names)
    filePath = 'code/faces_encodings.dat'
    filePath2 = 'code/faces_names.dat'
    # As file at filePath is deleted now, so we should check if file exists or not not before deleting them
    if os.path.exists(filePath):
        os.remove(filePath)

    if os.path.exists(filePath2):
        os.remove(filePath2)
    
    face_encoding_file = weight_dir + "/" + str(pd.datetime.now().date()) + '_faces_encodings.dat'
    face_names_file = weight_dir + "/" + str(pd.datetime.now().date()) + '_faces_names.dat'
    
    with open(face_encoding_file, 'wb') as f:
        pickle.dump(known_face_encodings, f)

    with open(face_names_file, 'wb') as f:
        pickle.dump(known_face_names, f)


"""----------------------------- options -----------------------------"""
parser = argparse.ArgumentParser(description='model_training')
parser.add_argument('--training_dir', type=str, default='../training/',
                    help='location of training directory')
parser.add_argument('--weight_dir', type=str, default='../training/',
                    help='location of weight directory')

args = parser.parse_args()


if __name__ == "__main__":
    # train models
    training_model(args.training_dir, args.weight_dir)