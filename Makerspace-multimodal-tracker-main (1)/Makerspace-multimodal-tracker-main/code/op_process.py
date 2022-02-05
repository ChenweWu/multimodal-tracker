import glob
import os
import argparse
import pandas as pd
import datetime
import subprocess
import shlex
import time
import re

def check_Nimages_match(path_input, path_processed):

    imgcountinput = len(glob.glob1(path_input, "*.jpg"))
    imgcountprocessed = len(glob.glob1(path_processed, "*.csv"))
    if imgcountinput == imgcountprocessed & imgcountinput != 0:
        print(imgcountinput, imgcountprocessed)
        return True
    print(imgcountinput, imgcountprocessed)
    return False

def get_csv_list(path_input, path_processed):
    list_input = glob.glob1(path_input, "*.jpg")
    list_csv = [x.replace("jpg", "csv") for x in list_input]
    list_actual = glob.glob1(path_processed, "*.csv")
    return list_csv, list_actual

def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

def df_from_file(list_files):
    list_files1 = []

    for f in list_files:
        if os.path.isfile(f):
            #df = pd.read_csv(f)
            df=pd.read_csv(f, header=0, usecols=['face', 'confidence', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z','gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y'])
            df['subject'] = f
            list_files1.append(df)
           # print(df.head(0))
        else:
           # print(f)
            df = pd.DataFrame(columns=['face', 'confidence', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                              'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y'])
            df.loc[0, 'subject'] = f
            list_files1.append(df)
          #  print(df.head(0))
    return list_files1


def merge_all_csv(path_input, path_processed):
    print("#all image have estimated gazes?: " +
          str(check_Nimages_match(path_input, path_processed)))
    code_dir = os.getcwd()+"/"
    abs_dir = code_dir+path_processed
    list_csv, list_actual = get_csv_list(path_input, path_processed)
    list_files = [abs_dir+x for x in list_csv]
    list_img_names = [x.replace(".csv", "") for x in list_csv]
    df_from_each_file = df_from_file(list_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    return concatenated_df


def run_commands(path_input):
    now = str(datetime.datetime.now().time())
    now2 = time.strftime("%Y%m%d-%H%M%S")
    path_output = "processed"+now2+'/'
    os.mkdir(path_output)
    cmdline = './build/bin/FaceLandmarkImg -fdir {} -out_dir {}'.format(
        path_input, path_output)
    run_command(cmdline)
    return path_output


def run_all(path_input, csv_path):
    path_processed = run_commands(path_input)
    m = merge_all_csv(path_input, path_processed)
    #now2 = time.strftime("%Y%m%d-%H%M%S")
    m2 = merge_with_curr(m, csv_path)
    output_location = faces_processed_folder + csv_path.split('/')[-1]
    m2.to_csv(output_location)
    print(output_location)


def merge_with_curr(m, csv_path):
    df_curr = pd.read_csv(csv_path)

    m['match'] = m['subject'].apply(lambda x: re.split(
        ' |/|\\\\', x)[-1].replace(".csv", ""))
    df_curr['match'] = df_curr['image_path'].apply(
        lambda x: re.split(' |/|\\\\', x)[-1].replace(".jpg", ""))
    out = df_curr.merge(m, left_on='match', right_on='match')
    return out

def gaze_detection():
    
    os.chdir(openface_dir)
    
    # obtain csv files from face_process
    faces_info_files = glob.glob(faces_new_info_folder+'*.csv')
    
    for face_csv in faces_info_files:
        
        df_face = pd.read_csv(face_csv)
        
        first_image_path = df_face.at[0,'image_path']
        len_char = len(first_image_path.split('/')[-1])
        face_dir = first_image_path[:-len_char]
        run_all(face_dir, face_csv)

"""----------------------------- options -----------------------------"""

parser = argparse.ArgumentParser(description='Gaze Detection')
parser.add_argument('--openface_dir', type=str, help='location of Open Face directory',
                    default='../../../OpenFace/')
parser.add_argument('--output_dir', type=str, default='../outputs/',
                    help='location of output drive')
args = parser.parse_args()


if __name__ == "__main__":
    
    # create necessary variables
    code_dir = os.getcwd() +'/'
    os.chdir(args.output_dir)
    output_dir = os.getcwd() +'/'
    os.chdir(args.openface_dir)
    openface_dir = os.getcwd() +'/'
    os.chdir(code_dir)
    
    info_dir = output_dir + 'info/'
    faces_new_info_folder = info_dir + 'faces_new_info/'    
    faces_processed_folder = info_dir + 'faces_processed_info/'
    
    # run
    gaze_detection()
    
