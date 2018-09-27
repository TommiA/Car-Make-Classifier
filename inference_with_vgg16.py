# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input, decode_predictions

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time

image_path='/home/eletai/sda1/eletai/deep_learning/carreg/dataset/test/'
model_name='vgg16_1_VALACC85.h5'
model = load_model(model_name)
model.summary()
labels=pickle.load(open('labeltest.pickle','rb'))

imagePaths = sorted(list(paths.list_images(image_path)))

for imagePath in imagePaths:
    #print(imagePaths)

    img = image.load_img(imagePath, target_size=(400, 400))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    clazz = np.argmax(preds[0])
    predictedmake = ''
    for make in labels:
        if labels[make] == clazz:
            cmake=imagePath.split('-')[1]
            cmodel=imagePath.split('-')[2]
            print(make+" "+cmake+" "+cmodel+" "+imagePath)

        

