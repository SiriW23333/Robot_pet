# -*- coding: utf-8 -*-
from hobot_dnn import pyeasy_dnn as dnn

# ¼ÓÔØÄ£
models = dnn.load('/root/Robot_pet/face_ws/face_recog/model_output_dir/RF.bin')
input_tensor = models[0].inputs[0]

# ´òÓ¡ÊäÈëµÄÐÎ×´ºÍÀàÐÍ
print("ÊäÈëÐÎ×´£º", input_tensor.properties.shape)
print("ÊäÈëÊý¾ÝÀàÐÍ£º", input_tensor.properties.dtype)
