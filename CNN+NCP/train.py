import tensorflow as tf
from tensorflow import keras
import numpy as np
import kerasncp as kncp
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Lambda
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

#定义ncp网络结构
wiring1 = kncp.wirings.NCP(
    inter_neurons=12,  # Number of inter neurons
    command_neurons=6,  # Number of command neurons
    motor_neurons=1,  # Number of motor neurons
    sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
    inter_fanout=4,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=4,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=6,  # How many incomming syanpses has each motor neuron
)
wiring2 = kncp.wirings.NCP(
    inter_neurons=8,  # Number of inter neurons
    command_neurons=4,  # Number of command neurons
    motor_neurons=1,  # Number of motor neurons
    sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
    inter_fanout=3,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=3,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=4,  # How many incomming syanpses has each motor neuron
)

rnn_cell_streeing = kncp.LTCCell(wiring1)
rnn_cell_speed = kncp.LTCCell(wiring2)


inputs = keras.Input(shape=(5,100,100,1))
x = Lambda(lambda inputs:inputs / 127.5 -1.,input_shape=(5,100,100,1))(inputs)
#conv1
conv1 = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu"))(x)
pool1 = keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))(conv1)
#conv2
conv2 = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu"))(pool1)
pool2 = keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))(conv2)
#conv3
conv3 = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=128,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu"))(pool2)
pool3 = keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))(conv3)
#conv4
conv4 = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=512,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu"))(pool3)
pool4 = keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))(conv4)
#conv5
conv5 = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu"))(pool4)
pool5 = keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(3,3),strides=(3,3),padding="valid"))(conv5)
#flatten
flatten = keras.layers.TimeDistributed(keras.layers.Flatten())(pool5)
#ncp
streeing_output = keras.layers.RNN(rnn_cell_streeing,return_sequences=True)(flatten)
speed_output = keras.layers.RNN(rnn_cell_speed,return_sequences=True)(flatten)
#定义模型
model = keras.Model(inputs=inputs,outputs=[streeing_output,speed_output])
model.summary()
#设置优化器
model.compile(
    optimizer=keras.optimizers.Adam(0.0002), loss="mean_squared_error",
    loss_weights=[1,1],
)
filepath = 'min_Val_loss_model.h5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath,verbose=1,save_weights_only=True,save_best_only=False)

#绘制NCP神经元连接
sns.set_style("white")
plt.figure(figsize=(12,12))
legend_handles = rnn_cell_speed.draw_graph(layout='spiral',neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles,loc="upper center",bbox_to_anchor=(1,1))
sns.despine(left=True,bottom=True)
plt.tight_layout()
plt.show()

#加载数据
def load_dataset(img_path,steering_path,speed_path):
    with open(img_path,'rb') as f:
        imgs = np.array(pickle.load(f))
    with open(steering_path,'rb') as f:
        steerings = np.array(pickle.load(f))
    with open(speed_path,'rb') as f:
        speeds = np.array(pickle.load(f))
    return imgs,steerings,speeds

def augmentData(imgs, steerings,speeds):
    imgs = np.append(imgs, imgs[:,:,:,::-1], axis=0)
    steerings = np.append(steerings, -steerings, axis=0)
    speeds = np.append(speeds,speeds,axis=0)
    return imgs, steerings,speeds

def random_data(train_imgs,train_streeings,train_speeds,val_imgs,val_streeings,val_speeds):
    train_index = [i for i in range(train_imgs.shape[0])]
    val_index = [i for i in range(val_imgs.shape[0])]
    np.random.shuffle(train_index)
    np.random.shuffle(val_index)

    train_imgs = train_imgs[train_index]
    train_streeings = train_streeings[train_index]
    train_speeds = train_speeds[train_index]

    val_imgs = val_imgs[val_index]
    val_streeings = val_streeings[val_index]
    val_speeds = val_speeds[val_index]

    return train_imgs,train_streeings,train_speeds,val_imgs,val_streeings,val_speeds

imgs,streeings,speeds = load_dataset('dataset/Ncp_imgs','dataset/Ncp_streeings','dataset/Ncp_speeds')
# imgs,streeings,speeds = augmentData(imgs,streeings,speeds)
print(imgs.shape,streeings.shape,speeds.shape)
train_num = 3000

train_imgs = imgs[:train_num,:,:,:]
train_streeings= streeings[:train_num,:]
train_speeds = speeds[:train_num,:]

val_imgs = imgs[train_num:,:,:,:]
val_streeings = streeings[train_num:,:]
val_speeds = speeds[train_num:,:]

print('train:',train_imgs.shape,train_streeings.shape,train_speeds.shape)
print('val:',val_imgs.shape,val_streeings.shape,val_speeds.shape)

train_imgs = train_imgs.reshape(train_imgs.shape[0],5,100,100,1)
train_streeings = train_streeings.reshape(train_streeings.shape[0],5,1)
train_speeds = train_speeds.reshape(train_speeds.shape[0],5,1)

val_imgs = val_imgs.reshape(val_imgs.shape[0],5,100,100,1)
val_streeings = val_streeings.reshape(val_streeings.shape[0],5,1)
val_speeds = val_speeds.reshape(val_speeds.shape[0],5,1)

train_imgs,train_streeings,train_speeds,val_imgs,val_streeings,val_speeds=random_data(train_imgs,train_streeings,train_speeds,val_imgs,val_streeings,val_speeds)

history = model.fit(train_imgs,[train_streeings,train_speeds],validation_data=(val_imgs,[val_streeings,val_speeds]),
                    batch_size=32,epochs=50,callbacks=[checkpoint])
model.save("test.h5")

plt.plot(history.history['loss'],'r')
# plt.plot(history.history['val_loss'], 'b', label='val_loss')
# plt.legend()
plt.xlabel("epochs")
plt.show()

