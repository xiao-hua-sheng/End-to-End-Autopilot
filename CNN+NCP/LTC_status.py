from tensorflow import keras
import numpy as np
import kerasncp as kncp
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import cv2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


imgs = []
for i in range(930,935):
    img = cv2.imread('imgs/%d.jpg'%i)
    img = cv2.resize(np.asarray(img)[:, :, 1],(100,100))
    imgs.append(img)
imgs = np.array(imgs)
imgs = np.reshape(imgs,(-1,5,100,100,1))
# temp = []
# for i in range(948,1883):
    # img = cv2.imread('data_imgs/%d.jpg'%i)
    # img = cv2.resize(np.asarray(img)[:, :, 1],(100,100))
    # temp.append(img)
    # if len(temp)>=5:
        # imgs.append(temp[-5:])

# imgs = np.array(imgs)
# print("imgs shape:",imgs.shape)
# imgs = np.reshape(imgs,(-1,5,100,100,1))

# streeings = []
# speeds = []
# for line in open('data_imgs_label.txt','r'):
    # streeing = (float(line.split()[8])*0.001745-0.037)
    # speed = float(line.split()[9])/60
    # streeings.append(streeing)
    # speeds.append(speed)

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

print(rnn_cell_streeing.state_size)
print(rnn_cell_speed.state_size)

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
#加载模型权重
model.load_weights("model/min_Val_loss_model.h5")
print("load weight succeed")

#预测
# steering_angles,speeds = model.predict((imgs),batch_size = 1)
# steering_angle = float(steering_angles[-1][-1])
# # speed = float(speeds[-1][-1])
# print('streeing angle:',steering_angle)

input_single = tf.keras.Input(shape=(100,100,1))

input_state_1 = tf.keras.Input(shape=(rnn_cell_streeing.state_size,))
# input_state_2 = tf.keras.Input(shape=(rnn_cell_speed.state_size,))

x1 = Lambda(lambda inputs:inputs / 127.5 -1.,input_shape=(100,100,1))(input_single)
#conv1
c1 = keras.layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu")(x1)
p1 = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(c1)
#conv2
c2 = keras.layers.Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu")(p1)
p2 = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(c2)
#conv3
c3 = keras.layers.Conv2D(filters=128,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu")(p2)
p3 = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(c3)
#conv4
c4 = keras.layers.Conv2D(filters=512,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu")(p3)
p4 = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(c4)
#conv5
c5 = keras.layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation="relu")(p4)
p5 = keras.layers.MaxPool2D(pool_size=(3,3),strides=(3,3),padding="valid")(c5)
#flatten
fn = keras.layers.Flatten()(p5)


_,output_states_1 = rnn_cell_streeing(fn,input_state_1)
# _,output_states_2 = rnn_cell_speed(fn,input_state_2)

single_step_model_1 = tf.keras.Model([input_single,input_state_1],output_states_1)
# single_step_model_2 = tf.keras.Model([input_single,input_state_2],output_states_2)


def infer_hidden_states(single_step_model,state_size,data_x):
    """
        Infers the hidden states of a single-step RNN model
    Args:
        single_step_model: RNN model taking a pair (inputs,old_hidden_state) as input and outputting new_hidden_state
        state_size: Size of the RNN model (=number of units)
        data_x: Input data for which the hidden states should be inferred
    Returns:
        Tensor of shape (batch_size,sequence_length+1,state_size). The sequence starts with the initial hidden state
        (all zeros) and is therefore one time-step longer than the input sequence
    """
    batch_size = data_x.shape[0]
    seq_len = data_x.shape[1]
    hidden = tf.zeros((batch_size,state_size))
    hidden_states = [hidden]
    for t in range(seq_len):
        # Compute new hidden state from old hidden state + input at time t
        # print("hidden.shape",hidden)
        hidden = single_step_model([data_x[:,t],hidden])
        # print("all",hidden)
        # print("all",len(hidden))
        hidden_states.append(hidden)
    return tf.stack(hidden_states,axis=1)


# res = []
# for i in range(imgs.shape[0]):
    # imgss = np.reshape(imgs[i],(-1,5,100,100,1))
    # states = infer_hidden_states(single_step_model_1,rnn_cell_streeing.state_size,imgss)
    # res.append(states[0][-1][0])

# plt.plot(streeings,"r")
# plt.plot(res,"b")
# plt.show()
states = infer_hidden_states(single_step_model_1,rnn_cell_streeing.state_size,imgs)
# states = infer_hidden_states(single_step_model_2,rnn_cell_speed.state_size,imgs)
print("Hidden states of first example ",states[0])

for i in range(wiring1.units):
    print("Neuron {:0d} is a {:} neuron".format(i,wiring1.get_type_of_neuron(i)))
















