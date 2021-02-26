from keras.layers import Conv2D,MaxPool2D,Input,Dense,Dropout,Flatten,Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
from matplotlib import pyplot
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

IMG_H = 100
IMG_W = 100
IMG_C = 1


def modelNet():
    img_input = Input(shape=(100,100,1),name='img_input')
    Lam = Lambda(lambda img_input:img_input / 127.5 -1.,input_shape=(100,100,1))(img_input)

    conv_1 = Conv2D(32,(5,5),padding='same',activation='relu')(Lam)
    pool_1 = MaxPool2D((3,3),strides=(2,2),padding='valid')(conv_1)

    conv_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(pool_1)
    pool_2 = MaxPool2D((3, 3), strides=(2, 2), padding='valid')(conv_2)

    conv_3 = Conv2D(128, (5, 5), padding='same', activation='relu')(pool_2)
    pool_3 = MaxPool2D((3, 3), strides=(2, 2), padding='valid')(conv_3)

    conv_4 = Conv2D(512, (5, 5), padding='same', activation='relu')(pool_3)
    pool_4 = MaxPool2D((3, 3), strides=(2, 2), padding='valid')(conv_4)

    conv_5 = Conv2D(512, (5, 5), padding='same', activation='relu')(pool_4)
    pool_5 = MaxPool2D((3, 3), strides=(3, 3), padding='valid')(conv_5)

    flatten = Flatten()(pool_5)

    dropout = Dropout(0.3)(flatten)

    #prediction steering angle net
    fc_1 = Dense(512,activation='relu')(dropout)
    fc_2 = Dense(256,activation='relu')(fc_1)
    fc_3 = Dense(64,activation='relu')(fc_2)
    prediction_angle = Dense(1,name='output_angle')(fc_3)

    #prediction speed net
    fc_4 = Dense(512,activation='relu')(dropout)
    fc_5 = Dense(256,activation='relu')(fc_4)
    fc_6 = Dense(64,activation='relu')(fc_5)
    prediction_speed = Dense(1,name='output_speed')(fc_6)


    model = Model(inputs = img_input,outputs = [prediction_angle,prediction_speed])
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001),loss='mse',loss_weights=[1,1])
    filepath = 'drivingMode_v1.h5'
    checkpoint = ModelCheckpoint(filepath,verbose=1,save_weights_only=False,save_best_only=True)
    callback_list = [checkpoint]

    return model,callback_list

def load_dataset(img_path,steering_path,speed_path):
    with open(img_path,'rb') as f:
        imgs = np.array(pickle.load(f))

    with open(steering_path,'rb') as f:
        steerings = np.array(pickle.load(f))

    with open(speed_path,'rb') as f:
        speeds = np.array(pickle.load(f))

    return imgs,steerings,speeds


def augmentData(imgs, steerings,speeds):
    imgs = np.append(imgs, imgs[:, :, ::-1], axis=0)
    steerings = np.append(steerings, -steerings, axis=0)
    speeds = np.append(speeds,speeds,axis=0)
    return imgs, steerings,speeds

def train():
    imgs,steerings,speeds = load_dataset('dataset/imgs','dataset/streeings','dataset/speeds')
    # imgs,steerings,speeds = augmentData(imgs,steerings,speeds)

    train_img = imgs[:14000,:,:]
    train_steering = steerings[:14000]
    train_speed = speeds[:14000]

    val_img = imgs[14000:,:,:]
    val_steering = steerings[14000:]
    val_speed = speeds[14000:]

    train_index = [i for i in range(train_img.shape[0])]
    val_index = [i for i in range(val_img.shape[0])]
    np.random.shuffle(train_index)
    np.random.shuffle(val_index)

    train_img = train_img[train_index]
    train_steering = train_steering[train_index]
    train_speed = train_speed[train_index]

    val_img = val_img[val_index]
    val_steering = val_steering[val_index]
    val_speed = val_speed[val_index]

    train_img = train_img.reshape(train_img.shape[0],100,100,1)
    val_img = val_img.reshape(val_img.shape[0],100,100,1)

    model,callback_list = modelNet()
    history = model.fit(train_img,[train_steering,train_speed],validation_data=(val_img,[val_steering,val_speed]),
              epochs=40,batch_size=48,callbacks=callback_list)

    model.save('drivingMode.h5')
    pyplot.plot(history.history['loss'],'r',label='union_loss')
    pyplot.plot(history.history['output_angle_loss'], 'g', label='angle_loss')
    pyplot.plot(history.history['output_speed_loss'], 'k', label='speed_loss')
    # pyplot.plot(history.history['val_loss'], 'b', label='val_loss')
    pyplot.legend()
    pyplot.xlabel("epochs")
    pyplot.show()

train()