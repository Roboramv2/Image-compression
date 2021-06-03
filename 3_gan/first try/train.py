import pickle
from cv2 import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Conv2D, Dense, LeakyReLU, MaxPool2D
from keras.models import Input, Model, save_model, load_model
from keras.optimizers import Adam

progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

def dispimg(image):
    cv2.imshow('out',image)
    cv2.waitKey(0)

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

x_train = np.asarray(data[0])
y_train = np.asarray(data[1])
x_test = np.asarray(data[2])
y_test = np.asarray(data[3])

def savemod(model, name):
    save_model(model, name+'.h5')

def genimg(vector):
    vector = tf.convert_to_tensor(vector)
    image = progan(vector)['default']
    image = tf.constant(image)
    image = tf.image.convert_image_dtype(image, tf.uint8)[0]
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def testmod(name):
    model = load_model(name+'.h5')
    i = np.random.randint(0, 900)
    accvec = y_train[i]
    img = x_train[i]
    img = np.reshape(img, (1, 128, 128, 1))
    genvec = model.predict(img, batch_size=None)
    genvec = np.reshape(genvec, (512, ))
    image = genimg(genvec)
    dispimg(image)
    accimg = genimg(accvec)
    dispimg(accimg)

def definemodel():
    in_image = Input(shape=(128, 128, 1))
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = MaxPool2D(pool_size=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = MaxPool2D(pool_size=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    out = Dense(512)(d)
    model = Model(in_image, out)
    model.compile(loss='mse', optimizer=Adam(lr=0.008, beta_1=0.5), loss_weights=[0.5])
    return model

#main function
name = 'version4'
model = definemodel()
model.summary()
model.fit(x_train,y_train,epochs=25, batch_size=10)
savemod(model, name)
testmod(name)