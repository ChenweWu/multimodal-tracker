import os
import smtplib 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import argparse
import schedule
import time
import datetime
import glob

# function to create directories
def create_dir(target_dir):
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except:
            pass
        
# function to write records
def write_record(record_file,output):
    with open(record_file,"a") as f:
        f.write(output+' \n')

# function to read records    
def read_record(record_file):
    with open(record_file,"r") as f:
        raw_video_ids = f.readlines()
        
    final_video_ids = []
    for video_id in raw_video_ids:
        final_video_ids.append(video_id.split(" ")[0])
        
    return final_video_ids

# helper function to create day folder
def create_day_folder(base_dir,system_date=None,setup=True):
    
    if system_date == None:
        dt_system_date = datetime.datetime.now()
    else:
        dt_system_date = datetime.datetime.strptime(system_date,'%Y-%m-%d') 
    
    year = str(dt_system_date.year)
    month = str(dt_system_date.month)
    day = str(dt_system_date.day)

    day_folder = base_dir + '{}-{}-{}/'.format(year,month,day)
    day_string = '{}-{}-{}'.format(year,month,day)
                
    if setup:
        create_dir(day_folder)
        
    return day_folder, day_string

def send(user, password, to_list, subject, text, attachment_list=None):
    
    # set up connection
    smtp_host = 'smtp.gmail.com'
    smtp_port = 587
    smtp_connection = smtplib.SMTP(smtp_host,smtp_port) #  initiate SMTP connection
    smtp_connection.ehlo() # send an EHLO (Extended Hello) command
    smtp_connection.starttls() # enable transport layer security (TLS) encryption
    smtp_connection.login(user, password) # login
    
    # write email
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg.attach(MIMEText(text))
    
    # attach files
    if attachment_list is not None:
        for attachment in attachment_list:
            with open(attachment, 'rb') as f:
                # Read in the attachment using MIMEApplication
                file = MIMEApplication(f.read(),name=os.path.basename(attachment))
            file['Content-Disposition'] = f'attachment;filename="{os.path.basename(attachment)}"'
              
            # Add the attachment to our message object
            msg.attach(file)
    
    # send
    smtp_connection.sendmail(from_addr=user,
                  to_addrs=to_list, msg=msg.as_string())
    
    # close connection
    smtp_connection.quit()
    
def check_and_record(upload_record,f_cloud_webcam_dir):
    
    # read record file
    try: 
        copied_videos = read_record(upload_record)
    except:
        write_record(upload_record,'')
        copied_videos = read_record(upload_record)
        
    last_record = copied_videos[-1]
    
    if last_record == 'stop':
        return False
    elif last_record == '':
        last_record = 0
        next_record = 0.5
    else:
        last_record = float(last_record)
        next_record = len(glob.glob(f_cloud_webcam_dir+'*.mp4'))
    
    write_record(upload_record,str(next_record))
    
    if next_record - last_record > 0:
        return False
    else:
        return True

def check_and_send():
    
     # check if there are any issues for each webcam
    for webcam_id in webcam_ids:
        
        webcam_dir = webcam_dirs[webcam_id]
        cloud_webcam_dir = cloud_webcam_dirs[webcam_id]
        upload_record = webcam_dir + 'email_record.txt'
        
        issue = check_and_record(upload_record,cloud_webcam_dir)
        
        # if not, we send an email
        if issue: 
            send(user='hgselitlab2@gmail.com', 
                password='litlab2019', 
                to_list=['chng_weimingedwin@g.harvard.edu'], 
                subject='No new upload for {}_{}'.format(station_id,webcam_id), 
                text='', 
                attachment_list=["{}/hdd_nextcloud_sync.log".format(station_dir)])
            write_record(upload_record,'stop')

"""----------------------------- options -----------------------------"""
parser = argparse.ArgumentParser(description='Email')
parser.add_argument('--station', type=int,
                    help='id for station')
parser.add_argument('--webcams', type=str,
                    help='list of ids for webcam')
parser.add_argument('--output_dir', type=str, default='../outputs/',
                    help='location of output drive')
parser.add_argument('--nextcloud_dir', type=str,
                    help='location of nextcloud drive')
parser.add_argument('--monitor_date', default=None,
                    help='which date to monitor, %Y-%m-%d')
args = parser.parse_args()


if __name__ == "__main__":
    
    # create necessary variables
    code_dir = os.getcwd() +'/'
    station_id = args.station
    webcam_ids = eval(args.webcams)
    monitor_date = args.monitor_date
    
    os.chdir(args.output_dir)
    output_dir = os.getcwd() +'/'
    os.chdir(args.nextcloud_dir)
    nextcloud_dir = os.getcwd() +'/'
    print('Next Cloud Directory - {}'.format(nextcloud_dir))
    os.chdir(code_dir)
    
    # create folders 
    station_dir = output_dir + 'station_{}/'.format(station_id)
    cloud_day_folder, _ = create_day_folder(nextcloud_dir,system_date=monitor_date,setup=False)   
    
    webcam_dirs = {}
    cloud_webcam_dirs = {}
    for webcam_id in webcam_ids:
        
        camera_id = '{}_{}'.format(station_id,webcam_id)
        
        webcam_dir = station_dir + 'webcam_{}/'.format(webcam_id)
        webcam_dirs[webcam_id] = webcam_dir
        
        cloud_webcam_dir = cloud_day_folder + '{}/'.format(camera_id)
        cloud_webcam_dirs[webcam_id] = cloud_webcam_dir

    # schedule jobs
    email_job = schedule.every(90).minutes.do(check_and_send)

    # start scheduler
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nEnding scheduler...')