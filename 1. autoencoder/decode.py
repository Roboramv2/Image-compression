import numpy as np
import keras
from keras import models, layers
from numpy import asarray
from cv2 import cv2

def postprocess(img):
    out = np.reshape(img, (28, 28, 1))
    out = out * 255
    return out

path = "__"  #enter path of encoded file
comp = np.load(path)
decoder = keras.models.load_model("models/decoder.h5")

out = decoder.predict(comp)
out = postprocess(out)
cv2.imwrite("output.jpg", out)