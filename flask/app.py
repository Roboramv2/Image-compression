
import os
import numpy as np
import keras
from keras import models, layers
from PIL import Image
from numpy import asarray
from cv2 import cv2
from keras.models import load_model
from os.path import join, dirname, realpath
from flask import Flask,render_template,request
import skimage
from skimage.transform import resize
from matplotlib import pyplot as plt


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"
app.static_folder = 'static'
f = ""
original_filename =""


UPLOADS_PATH = join(dirname(realpath(__file__)), 'static\\')


encoder = load_model("encoder.h5")
decoder = load_model("decoder.h5")
compressor = load_model("compressor.h5")

def preprocess(path):
    img = Image.open(path)
    image = np.asarray(img)
    image = image.astype('float32') / 255.
    img = np.expand_dims(image,axis=-1)
    img = np.expand_dims(image,axis=-0)
    return img

@app.route('/')
def upload_image():
    if os.path.exists("C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\decompressed.jpg"):
        os.remove("C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\decompressed.jpg")
    if os.path.exists("C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\intermediate.jpg"):
        os.remove("C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\intermediate.jpg")
    if os.path.exists("C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\compressed.npy"):
        os.remove("C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\compressed.npy")
    return render_template('index.html')

@app.route('/imageuploader',methods=['GET','POST'])
def image_upload():
    if request.method=='POST':
        f = request.files['image']
        f.save(os.path.join(UPLOADS_PATH,f.filename))
        original_filename = f.filename
      
        path = "C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\"+f.filename
        image = Image.open(path)
        img = preprocess(path)

        output = encoder.predict(img)
        path = "C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\compressed"
        np.save(path, output)


        img = np.resize(image, (28, 28,1)) 
        code = compressor.predict(img[None])[0]
        path = "C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\intermediate.jpg"
        plt.imsave(path,code.reshape([code.shape[-1]//2,-1]))
        return render_template('compress.html',filename = f.filename)

def postprocess(img):
    image = np.reshape(img, (28, 28, 1))
    image = image * 255
    return image


@app.route('/npyuploader',methods=['GET','POST'])
def npy_upload():
    if request.method=='POST':
        print(original_filename)
        f = request.files['npy']
        f.save(os.path.join(UPLOADS_PATH,f.filename))

        path = "C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\"+f.filename
        compressed_img = np.load(path)
        
        decompressed_img = decoder.predict(compressed_img)
        decompressed_img = postprocess(decompressed_img)
        
        path = "C:\\Users\\Sowmya\\Desktop\\LAB\\project\\static\\decompressed.jpg"
        cv2.imwrite(path, decompressed_img)

        return render_template('decompress.html',filename = original_filename)

if __name__ == '__main__':
    app.run()