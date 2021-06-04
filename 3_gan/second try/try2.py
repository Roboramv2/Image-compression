import tensorflow as tf
import numpy as np
from cv2 import cv2
from numpy.linalg import svd
#Projects\compression\3_gan\"second try"
rose = cv2.imread('rose.jpg')
model = tf.keras.models.load_model('50SRResNet-generator.h5')
model.compile()

k=4
original_shape = rose.shape
image_reshaped = rose.reshape((original_shape[0],original_shape[1]*3))
U,s,V = svd(image_reshaped,full_matrices=False)
U = U[:,:k]
s = s[:k]
V = V[:k,:]
mid = np.dot(np.diag(s[:]),V[:,:])
reconst_matrix = np.dot(U[:, :],mid)
shap = (reconst_matrix.shape)
newshap = (shap[0], int(shap[1]/3), 3)
image = reconst_matrix.reshape(newshap)
image = image/255
cv2.imshow('out', image)
cv2.waitKey(0)
image = cv2.resize(image, (500, 500))
image = np.reshape(image, (1, 500, 500, 3))
new = model(image)
new = new.numpy()
new = np.reshape(new, (2000, 2000, 3))
new = cv2.resize(new, (500, 500))
cv2.imshow('out', new)
cv2.waitKey(0)