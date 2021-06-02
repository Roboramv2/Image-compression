import pickle
import os
import numpy as np
from cv2 import cv2

name = 'bear'
filebase = os.listdir('.')
files = []
for x in filebase:
    if x.startswith(name):
        if x.split('.')[-1]=='pkl':
            files.append(x)

for fil in files:
    with open(fil, 'rb') as f:
        [U, s, V] = pickle.load(f)
    mid = np.dot(np.diag(s[:]),V[:,:])
    reconst_matrix = np.dot(U[:, :],mid)
    shap = (reconst_matrix.shape)
    newshap = (shap[0], int(shap[1]/3), 3)
    image = reconst_matrix.reshape(newshap)
    image = image/255
    imshape = (newshap[0], newshap[1])
    # if newshap[0]>500:
    #     rat = newshap[1]/newshap[0]
    #     w = int(500*rat)
    #     imshape = (w, 500)
    # image = cv2.resize(image, imshape)
    cv2.imshow('out', image)
    cv2.waitKey(0)

