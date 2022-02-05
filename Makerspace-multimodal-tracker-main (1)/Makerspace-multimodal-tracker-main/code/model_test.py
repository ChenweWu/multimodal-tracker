# install dependencies
import face_recognition
import numpy as np
import os
import pandas as pd
import math
import pickle
import argparse
import glob


def testing_model(test_path, face_encoding_input, face_names_input):
    # Load model training encodings
    with open(face_encoding_input, 'rb') as f:
        known_face_encodings = pickle.load(f)
    with open(face_names_input, 'rb') as f:
        known_face_names = pickle.load(f)

    # Input for testing directory
    image_filepaths = []
    csv = pd.read_csv(test_path)


    for filepath in list(csv['image_path']):
        image_filepaths.append(filepath)


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

    # list containing all images that a face was detected in, for prediction
    used_filepaths = []

    # list containing prediction confidence scores for each prediction, calculated from Euclidean distance
    distances = []

    # iterate through each image in test set and generate predictions
    no_images = len(image_filepaths)

    for test_image_num in range(no_images):
        unknown_image = face_recognition.load_image_file(image_filepaths[test_image_num])
        face_locations = face_recognition.face_locations(unknown_image, model="cnn")
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        name = "Unknown"
        distance = 0
        
        try:
            face_encoding = face_encodings[0]

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # 
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                used_filepaths.append(image_filepaths[test_image_num])
                distances.append(round(face_distance_to_conf(np.min(face_distances)),5))

        except:
            pass
                # update predictions and true labels lists
        predicted.append(name)
        if distance == 0:
            distances.append(distance)
        print('Predicted {} out of {}: {} with {}'.format(test_image_num+1,no_images,name,distance))

    csv["Predicted Result"] = np.array(predicted)
    csv["Confidence score"] = distances
    csv.to_csv(test_path)

def get_face_recognition():
    
    # obtain csv files from face_process
    faces_info_files = glob.glob(faces_new_info_folder+'*.csv')
    
    for face_csv in faces_info_files:
        
        testing_model(face_csv, face_encodings, face_names)

"""----------------------------- options -----------------------------"""
parser = argparse.ArgumentParser(description='model_training')
parser.add_argument('--face_encodings', type=str, default='./model_weights/2021-11-19_faces_encodings.dat',
                    help='location of face encoding file')
parser.add_argument('--face_names', type=str, default='./model_weights/2021-11-19_faces_names.dat',
                    help='location of face names file')
parser.add_argument('--output_dir', type=str, default='../outputs/',
                    help='location of output drive')

args = parser.parse_args()

if __name__ == "__main__":
    
    # create necessary variables
    face_names = args.face_names
    face_encodings = args.face_encodings
    
    code_dir = os.getcwd() +'/'
    os.chdir(args.output_dir)
    output_dir = os.getcwd() +'/'
    os.chdir(code_dir)
    
    info_dir = output_dir + 'info/'
    faces_new_info_folder = info_dir + 'faces_new_info/'
    
    get_face_recognition()