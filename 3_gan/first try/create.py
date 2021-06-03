from cv2 import cv2
import tensorflow as tf
import tensorflow_hub as hub
import pickle

progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

def genimg():
    vector = tf.random.normal([512])
    image = progan(vector)['default']
    image = tf.constant(image)
    image = tf.image.convert_image_dtype(image, tf.uint8)[0]
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vector = vector.numpy()
    return [vector, image]

#vec = tf.convert_to_tensor(vec)
#print(type(vec))

def dispimg(image):
    cv2.imshow('out',image)
    cv2.waitKey(0)

dataset = []
for i in range(1000):
    element = genimg()
    dataset.append(element)

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)