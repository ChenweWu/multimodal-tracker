import glob
import os
import argparse
import pandas as pd
import time
import re
import sys
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output

def load_model(code_dir,model_file):
    sys.path.append(code_dir)
    from model import GazeLSTM
    model = GazeLSTM()
    model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def get_outputs(model,img_dir,out_dir,alpha_csv):
    path=img_dir
    img_list=[]
    df_list=[]

    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img_list.append(file)
    for img in img_list:
      test_im=cv2.imread(os.path.join(img_dir,img))
      print(img, " has a shape of ",test_im.shape)
      new_im=test_im.copy()
      new_im=Image.fromarray(new_im,'RGB')
      init_h,init_w,init_channels=test_im.shape
      input_image = torch.zeros(7,3,init_h,init_w)
      for count in range(6):
        input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((init_h,init_w))(new_im)))
      input1=input_image.view(1,7,3,init_h,init_w).cuda()
      output_gaze,_ = model(input1)
      gaze = spherical2cartesial(output_gaze).detach().numpy()
      print(gaze)
      gaze_df=pd.DataFrame(data=gaze,columns=['gazex','gazey','gazez'])
      gaze_df['match']=img
      df_list.append(gaze_df)

      # resize image
      test_im = cv2.resize(test_im, (init_h,init_w), interpolation = cv2.INTER_AREA)
      height,width, channels=test_im.shape
      start_point = (int(height//2),int(width//2))
      
      # End coordinate
      end_point = (int(height//2-gaze[:,0]*50), int(width//2-gaze[:,1]*50))
      
      # Green color in BGR
      color = (255,0,0)
      
      # Line thickness of 9 px
      thickness = 5
      
      # Using cv2.arrowedLine() method
      # Draw a diagonal green arrow line
      # with thickness of 9 px
      out_image = cv2.arrowedLine(test_im, start_point, end_point,
                                          color, thickness)
      Image.fromarray(out_image,'RGB').save(os.path.join(out_dir,img))
    concatenated_df= pd.concat(df_list, ignore_index=True)
    
    alpha_csv['match']=alpha_csv['image_path'].apply(lambda x: re.split(' |/|\\\\' , x)[-1])
    out=alpha_csv.merge(concatenated_df, left_on='match',right_on='match')
    now=time.strftime("%Y%m%d-%H%M%S")
    out.to_csv(out_dir+now+".csv")

def gaze_detection():
    
    # obtain model
    model=load_model(gaze360_code_dir,gaze360_model_file)
    
    # obtain csv files from face_process
    faces_info_files = glob.glob(faces_new_info_folder+'*.csv')
    
    for face_csv in faces_info_files:
        
        df_face = pd.read_csv(face_csv)
        
        first_image_path = df_face.at[0,'image_path']
        len_char = len(first_image_path.split('/')[-1])
        face_dir = first_image_path[:-len_char]
        get_outputs(model,face_dir,faces_processed_folder,df_face)

        
"""----------------------------- options -----------------------------"""
parser = argparse.ArgumentParser(description='Gaze360')

parser.add_argument('--gaze360_dir', type=str, help='location of Gaze360 codes',
                    default='../../gaze360/')


parser.add_argument('--output_dir', type=str, default='../outputs/',
                    help='location of output drive')

args = parser.parse_args()


if __name__ == "__main__":
    
    # create necessary variables
    code_dir = os.getcwd() +'/'
    os.chdir(args.gaze360_dir)
    gaze360_dir = os.getcwd() +'/'
    os.chdir(args.output_dir)
    output_dir = os.getcwd() +'/'
    os.chdir(code_dir)
    
    gaze360_code_dir = gaze360_dir + 'code/'
    gaze360_model_file = gaze360_dir + 'model/gaze360_model.pth.tar'
    
    info_dir = output_dir + 'info/'
    faces_new_info_folder = info_dir + 'faces_new_info/'    
    faces_processed_folder = info_dir + 'faces_processed_info/'
    
    # run
    gaze_detection()
    

    
    
    
    
