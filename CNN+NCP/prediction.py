#/*
# * Copyright 2016-2019. UISEE TECHNOLOGIES LTD. All rights reserved.
# * See LICENSE AGREEMENT file in the project root for full license information.
# */

# This demo code is based on Python3
# If Python2 is needed, please replace usimpy.so in root dirction with lib_x86/lib_py27/usimpy.so
import cv2
import os
import numpy as np
import usimpy
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras.models import load_model
import kerasncp as kncp
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

#NCP模型加载
path1 = 'linshi_model/test.h5'
model = load_model(path1, custom_objects={"LTCCell": kncp.LTCCell})
print('load model complate')

#模型预测
batch_imgs = []
def prediction(model,img):
    img = cv2.resize(np.asarray(img)[:, :, 1],(100,100))
    img = np.array(img,dtype = np.float32)
    batch_imgs.append(img)
    if len(batch_imgs)==5:
        predict_img = np.array(batch_imgs)
        predict_img = predict_img.reshape(-1,5,100,100,1)
        steering_angles,speeds = model.predict((predict_img),batch_size = 1)
        steering_angle = float(steering_angles[-1][-1])
        speed = float(speeds[-1][-1])
        batch_imgs.pop(0)
    else:
        steering_angle=0
        speed = 0
    steering_angle = float(steering_angle)
    speed = float(speed)
    steering_angle = steering_angle*60
    speed = speed*50
    return steering_angle,speed

