import rosbag
from bagpy import bagreader
import pandas as pd
from matplotlib import pyplot as plt

#---------------------------------------------------------------------
#bag = rosbag.Bag('5_obj_1speed_linearrotate_lowlight_082height.bag')
# from std_msgs.msg import Int32, String
#bag = rosbag.Bag('5_obj_1speed_linearrotate_lowlight_082height.bag.bag') # didn't work
import bagpy
#print(bag) # didn't work
#print(bag.topic_table) # didn't work
#-------------------------------------------------------------------

#b = bagreader('5_obj_1speed_linearrotate_lowlight_082height.bag')
# get the list of topics
#print(b.topic_table)
# get all the messages of type velocity
#velmsgs   = b.vel_data()
#veldf = pd.read_csv(velmsgs[0])
#plt.plot(veldf['Time'], veldf['linear.x'])
#plt.show()
#images_in_ROS = b.compressed_images() #This is my try but did not work
import bagpy
from bagpy import bagreader
import pandas as pd
#b = bagreader('5_obj_1speed_linearrotate_lowlight_082height.bag')
# replace the topic name as per your need

#Depth_Info = b.message_by_topic('/camera/aligned_depth_to_color/image_raw')
#Color_Image_Info = b.message_by_topic('/camera/color/image_raw')
#Davis_left_events_Info = b.message_by_topic('/davis_left/events')
#Davis_left_image_Info = b.message_by_topic('/davis_left/image_raw')
#Davis_right_events_Info = b.message_by_topic('/davis_right/events')
#Davis_right_image_Info = b.message_by_topic('/davis_right/image_raw')
#velocity_info = b.message_by_topic('/tcp/vel')
#pose_info = b.message_by_topic('/tcp_pose')


#Depth_Info_df = pd.read_csv(Depth_Info)
#Color_Image_Info_df = pd.read_csv(Color_Image_Info)
#Davis_left_events_Info_df = pd.read_csv(Davis_left_events_Info)
#Davis_left_image_Info_df = pd.read_csv(Davis_left_image_Info)
#Davis_right_events_Info_df = pd.read_csv(Davis_right_events_Info)
#Davis_right_image_Info_df = pd.read_csv(Davis_right_image_Info)
#velocity_info_df = pd.read_csv(velocity_info)
#pose_info_df = pd.read_csv(pose_info)

#data_cl_colr_img = Color_Image_Info_df['data']
from cv_bridge import CvBridge
import cv2
#bridge = CvBridge()
#cv_image = bridge.imgmsg_to_cv2(Color_Image_Info_df, desired_encoding='passthrough')

#roscd image_view
#from cv_bridge import CvBridge
#bridge = CvBridge()
#cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
#cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#bag_file = rosbag.Bag()

#for topic, msg, t in bag_file.read_messages(topics=['/camera/color/image_raw']):
#    if topic == '/camera/color/image_raw':
#        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


#import cv2
#image = cv2.imread('Figure_1.jpeg')
#cv2.imshow('img', image)
#cv2.waitKey(0)
#print(image)

------------------------ Worked = to write save images in the folder  --------------------------
import h5py
hf = h5py.File('output.hdf5', 'r')
d435_frames = hf['d435_frames']
c=1
for l in range(82):
    img = d435_frames[l]
    name = 'img' + str(c) + '.png'
    cv2.imwrite(os.path.join(path, name), img)
    c += 1
    print(os.path.join(name))
    #img = cv2.imread('Figure_1.jpeg', 1)
    #plt.show()
hf.close()
------------------------ Create a video out of images --------------------------

import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('C:/Users/sanke/OneDrive/Desktop/Bag_Images/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project_5im_per_sec.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

-------------------------------------------------------------------
cap = cv2.VideoCapture('project.mp4')

cap = cv2.VideoCapture(0)

---------------------This one worked ------------------------------------
import os
c=1
img = cv2.imread('Figure_1.jpeg', 1)
path = 'C:/Users/sanke/OneDrive/Desktop/Bag_Images'
name = 'img' + str(c) + '.png'
cv2.imwrite(os.path.join(path ,name), img)
print(os.path.join(name))
cv2.waitKey(0)
------------------------------------------------------------
for n in range(0, len(onlyfiles)):
    # other things you need to do snipped
    cv2.imwrite(f'/path/to/destination/image_{n}.png',image)

img = cv2.imread('Figure_1.jpeg', 1)
img.shape

------------------------------------------------------
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()