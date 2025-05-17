from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'client_sqlite')))
from client_sqlite import add_client, find_similar_face
import serial 

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

def crop_and_resize_from_camera(cap, a, b, c, d):
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera")
        return None
    # Get bounding box coordinates
    x_min = int(min(a[0], b[0], c[0], d[0]))
    y_min = int(min(a[1], b[1], c[1], d[1]))
    x_max = int(max(a[0], b[0], c[0], d[0]))
    y_max = int(max(a[1], b[1], c[1], d[1]))
    # Crop the region of interest
    roi = frame[y_min:y_max, x_min:x_max]
    # Resize to 112x112
    roi_resized = cv2.resize(roi, (model_input_size))
    return roi_resized  #BGR格式的numpy数组

# set input data format to RGB and input layout to NCHW
def preprocess_frame(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Change layout to NCHW
    nchw_frame = np.transpose(rgb_frame, (2, 0, 1))
    # Convert to float32 and normalize if needed
    nchw_frame = nchw_frame.astype(np.float32) 
    # Add batch dimension
    nchw_frame = np.expand_dims(nchw_frame, axis=0)
    return nchw_frame

def output(frame):
    return models[0].forward(frame)
    
def postprocess(feature):
    if feature is not None:
        ser = None
        try:
            ser = serial.Serial('/dev/ttyS1', 9600, timeout=1)
            ser.write(bytes([0x43]))
            print("already indentify a people,say hello")
        except Exception as e:
            print("fail to send messeage to stm32")
        finally:
            if ser is not None and ser.is_open:
                ser.close()

        if find_similar_face(feature, 0.6) == None:
            new_id = add_client(feature)
            print(f"new id:{new_id}")
            return new_id
        else:
            id = find_similar_face(feature, 0.6)
            print(f"we find {id}")
            return id
