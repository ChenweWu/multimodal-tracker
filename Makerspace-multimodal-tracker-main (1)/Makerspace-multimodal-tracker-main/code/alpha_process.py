import shleximport subprocessimport pandas as pdimport jsonimport datetimeimport osimport argparse# function to execute command in shelldef run_command(command):    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)    while True:        output = process.stdout.readline()        if process.poll() is not None:            break        if output:            print(output.strip())    rc = process.poll()    return rc# function to create directoriesdef create_dir(target_dir):    if not os.path.exists(target_dir):        try:            os.makedirs(target_dir)        except:            pass# helper function to create time folders in directorydef get_time_folder(base_dir,system_start,system_date=None,time_level='hr',setup=True):        if system_date == None:        dt_system_date = datetime.datetime.now()    else:        dt_system_date = datetime.datetime.strptime(system_date,'%Y-%m-%d')         dt_system_start = datetime.datetime.strptime(system_start,'%H:%M')         year = str(dt_system_date.year)    month = str(dt_system_date.month)    day = str(dt_system_date.day)        hour = str(dt_system_start.hour)    minute = str(dt_system_start.minute)    year_folder = base_dir + '{}/'.format(year)    month_folder = year_folder + '{}/'.format(month)    day_folder = month_folder + '{}/'.format(day)            if time_level == 'min':        time_folder = day_folder + '{}_{}/'.format(hour,minute)        time_string = '{}-{}-{}_{}-{}'.format(year,month,day,hour,minute)    elif time_level == 'hr':        time_folder = day_folder + '{}/'.format(hour)        time_string = '{}-{}-{}_{}'.format(year,month,day,hour)            if setup:        create_dir(year_folder)        create_dir(month_folder)        create_dir(day_folder)        create_dir(time_folder)            return time_folder, time_string# process to setup time foldersdef time_folders_setup(webcam_dir,system_start,system_date=None,time_level='hr',setup=True):        # get time folders    time_folder, _ = get_time_folder(webcam_dir,system_start,system_date=system_date,time_level=time_level,setup=setup)        # obtain output folders    video_store_folder = time_folder + 'video_store/'    video_alphapose_folder = time_folder + 'video_alphapose/'    alphapose_folder = time_folder + 'alphapose/'    faces_folder = time_folder + 'faces/'        # obtain info folders    info_dir = time_folder + 'info/'    video_info_folder = info_dir + 'video_info/'        alphapose_new_info_folder = info_dir + 'alphapose_new_info/'    alphapose_processed_folder = info_dir + 'alphapose_processed_info/'        faces_new_info_folder = info_dir + 'faces_new_info/'    faces_processed_folder = info_dir + 'faces_processed_info/'        # create folders    if setup:        create_dir(video_store_folder)        create_dir(video_alphapose_folder)        create_dir(alphapose_folder)        create_dir(faces_folder)        create_dir(info_dir)        create_dir(video_info_folder)        create_dir(alphapose_new_info_folder)        create_dir(alphapose_processed_folder)        create_dir(faces_new_info_folder)        create_dir(faces_processed_folder)        return video_store_folder,video_alphapose_folder,alphapose_folder,faces_folder,info_dir,video_info_folder,alphapose_new_info_folder,alphapose_processed_folder,faces_new_info_folder,faces_processed_folder# process to apply alphapose on input videodef alpha_process(process_time,alphapose_folder,poseconnect_time_folder,video_info_folder,alphapose_new_info_folder):        # obtain csv files from webcam_process    df_webcam_new = pd.read_csv(video_info_folder+'{}_info.csv'.format(process_time))    df_image_timestamps = pd.read_csv(video_info_folder+'{}_frame_ts.csv'.format(process_time))        # extract information    for i, row in df_webcam_new.iterrows():                # get information from df_webcam_new        video_id = row['video_id']        video_path = row['video_path']        camera_id = '{}_{}'.format(video_id.split('_')[0],video_id.split('_')[1])                # command for alphapose        os.chdir(alphapose_dir)        command_line = 'python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video "{}" --outdir "{}"'.format(video_path,alphapose_folder)        run_command(command_line)        os.chdir(code_dir)            # rename json file        old_json_file = alphapose_folder + 'alphapose-results.json'        json_file = alphapose_folder + '{}_alphapose-results.json'.format(video_id)        os.rename(old_json_file, json_file)                # convert data in json to dataframe        df_pose2d = pd.DataFrame(columns=['pose_2d_id','timestamp','camera_id','keypoint_coordinates_2d','keypoint_quality_2d','pose_quality_2d'])        df_faces = pd.DataFrame(columns=['video_id','image_id','keypoints','box'])        # read data from json file        with open(json_file) as json_data:                        data = json.load(json_data)            id_count = 0                        for row in data:                                # determine number of keypoints                keypoints_no = int(len(row['keypoints'])/3)                                if row['category_id'] == 1:                                        # obtain information for df_faces                    df_faces = df_faces.append({'video_id':video_id,                                                'image_id' : row['image_id'],                                                'keypoints': row['keypoints'],                                                'box' : row['box']},ignore_index=True)                                                            # obtain parameters for df_pose2d                    image_no = int(row['image_id'].split('.')[0])                    image_id = '{}_{}'.format(video_id,image_no)                    timestamps = df_image_timestamps[(df_image_timestamps['image_no']==image_no)&(df_image_timestamps['video_id']==video_id)]['timestamp'].values                    if len(timestamps)!=1:                        raise Exception('More than one timestamp obtained!')                    else:                        timestamp = timestamps[0]                    pose_2d_id = '{}_{}'.format(str(image_id),str(id_count))                     pose_quality_2d = row['score']                    keypoint_coordinates_2d = []                    keypoint_quality_2d = []                    keypoints_temp = row['keypoints']                    for j in range(keypoints_no):                        for k in range(3):                            if k == 0:                                keypoint_x  = keypoints_temp[j*3+k]                            elif k == 1:                                keypoint_y  = keypoints_temp[j*3+k]                            elif k == 2:                                keypoint_coordinates_2d.append([keypoint_x,keypoint_y])                                keypoint_quality_2d.append(keypoints_temp[j*3+k])                                        # obtain information for df_pose2d                    df_pose2d = df_pose2d.append({'pose_2d_id':pose_2d_id,                                                'timestamp':timestamp,                                                'camera_id':camera_id,                                                'keypoint_coordinates_2d':keypoint_coordinates_2d,                                                'keypoint_quality_2d':keypoint_quality_2d,                                                'pose_quality_2d':pose_quality_2d},ignore_index=True)                    id_count += 1                 # save output        df_pose2d.to_csv(poseconnect_time_folder+'pose2d_{}.csv'.format(video_id),index=0)        df_faces.to_csv(alphapose_new_info_folder+'faces_{}.csv'.format(video_id),index=0)    """----------------------------- options -----------------------------"""parser = argparse.ArgumentParser(description='Alpha Process')parser.add_argument('--alphapose_dir', type=str, default='../../AlphaPose',                    help='location of AlphaPose')parser.add_argument('--station_dir', type=str,                    help='location of station drive')parser.add_argument('--webcams', type=str,                    help='list of ids for webcam')parser.add_argument('--process_time', type=str,                    help='process time of webcam post processing (format %H:%M)')parser.add_argument('--process_date', type=str,                    help='process date of webcam post processing (format %Y-%m-%d)')args = parser.parse_args()if __name__ == "__main__":        # create necessary variables    webcam_ids = eval(args.webcams)    process_time = args.process_time    process_date = args.process_date    code_dir = os.getcwd() +'/'    os.chdir(args.alphapose_dir)    alphapose_dir = os.getcwd() +'/'    os.chdir(args.station_dir)    station_dir = os.getcwd() +'/'    os.chdir(code_dir)        # create poseconnect folders    poseconnect_dir = station_dir + 'poseconnect/'    create_dir(poseconnect_dir)    poseconnect_time_folder, _ = get_time_folder(poseconnect_dir,process_time,system_date=process_date,time_level='hr',setup=True)    for webcam_id in webcam_ids:                # obtain webcam folders          webcam_dir = station_dir + 'webcam_{}/'.format(webcam_id)                video_store_folder,video_alphapose_folder,alphapose_folder,faces_folder,info_dir,video_info_folder,alphapose_new_info_folder,alphapose_processed_folder,faces_new_info_folder,faces_processed_folder = time_folders_setup(webcam_dir,process_time,system_date=process_date,time_level='hr',setup=False)               # run alpha process        alpha_process(process_time,alphapose_folder,poseconnect_time_folder,video_info_folder,alphapose_new_info_folder)    