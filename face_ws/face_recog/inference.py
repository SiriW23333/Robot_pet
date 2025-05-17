from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
import numpy as np
import cv2
import sys
sys.path.append(r"..\\client_sqlite")
from client_sqlite import get_favorability,add_client,find_similar_face


#create model object
models = dnn.load('./facenet.bin')
model_input_size = (160, 160)

def cam_init():
    # open usb camera: /dev/video8
    cap = cv2.VideoCapture(8)
    if(not cap.isOpened()):
        print(" Failed to open camera")
        exit(-1)
    print("Open usb camera successfully")
    # set the output of usb camera to MJPEG, solution 640 x 480
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# set input data format to RGB and input layout to NCHW
def preprocess_frame(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Change layout to NCHW
    nchw_frame = np.transpose(rgb_frame, (2, 0, 1))
    resized_frame = cv2.resize(nchw_frame, model_input_size)
    return resized_frame

def output(frame):
    outputs = models[0].forward(frame)
    
def postprocess(feature):
    if find_similar_face(feature, 0.6) == None:
        new_id=add_client(feature)
        return new_id
    else:
        return find_similar_face(feature,0.6)