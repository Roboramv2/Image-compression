import numpy as np
from numpy.linalg import svd
from cv2 import cv2
import pickle

images = ["bear.jpg"]

def compress_show_color_images_reshape(name, k):
    image = cv2.imread(name)
    image = np.asarray(image)
    original_shape = image.shape
    image_reshaped = image.reshape((original_shape[0],original_shape[1]*3))
    U,s,V = svd(image_reshaped,full_matrices=False)
    U = U[:,:k]
    s = s[:k]
    V = V[:k,:]
    lists = [U, s, V]
    return lists

ks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]

for i in images:
    for k in ks:
        lists = compress_show_color_images_reshape(i, k)
        nam = i.split('.')[0]
        if k>=10:
            with open(nam+str(k)+'.pkl', 'wb') as f:
                pickle.dump(lists, f)
        else:
            with open(nam+str(0)+str(k)+'.pkl', 'wb') as f:
                pickle.dump(lists, f)

