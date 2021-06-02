import numpy as np
import keras
from keras import models, layers
from PIL import Image
from numpy import asarray

def preprocess(path):
    img = Image.open(path)
    image = np.asarray(img)
    image = image.astype('float32') / 255.
    img = np.expand_dims(image,axis=-1)
    img = np.expand_dims(image,axis=-0)
    return img
path = "__" #enter file path (28*28 jpg file)
img = preprocess(path)

encoder = keras.models.load_model("models/encoder.h5")
encoder.summary()

output = encoder.predict(img)
np.save("coded"+path.rstrip(".jpg"), output)