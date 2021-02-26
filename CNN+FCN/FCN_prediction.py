from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time 

#加载模型
path = "model/drivingMode.h5"
model = load_model(path)

imgs = []
# 加载测试赛道数据
for i in range(226):
    img = cv2.imread('test_imgs/%d.jpg'%i)
    img = cv2.resize(np.asarray(img)[:, :, 1],(100,100))
    imgs.append(img)

imgs = np.array(imgs)
print(imgs.shape)


#模型预测
res_st = []
res_sp = []

total_time = []

for i in range(imgs.shape[0]):
    predict_img = np.reshape(imgs[i],(1,100,100,1))
    steering_angle,speed = model.predict((predict_img),batch_size = 1)
    steering_angle = float(steering_angle)
    speed = float(speed)
    steering_angle = steering_angle*57.3
    speed = speed*5
    res_st.append(steering_angle)
    res_sp.append(speed)
plt.plot(res_st,'k',label='predict corner')
plt.plot(res_sp,'g',label='predict speed')
plt.legend()
plt.show()





