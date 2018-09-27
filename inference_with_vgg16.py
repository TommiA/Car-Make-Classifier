# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

from imutils import paths
import numpy as np
import pickle

image_path='/home/eletai/sda1/eletai/deep_learning/carreg/dataset/test/'
model_name='vgg16_1_VALACC85.h5'
model = load_model(model_name)
model.summary()
labels=pickle.load(open('labeltest.pickle','rb'))

imagePaths = sorted(list(paths.list_images(image_path)))

for imagePath in imagePaths:
    img = image.load_img(imagePath, target_size=(400, 400))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    clazz = np.argmax(preds[0])
    for make in labels:
        if labels[make] == clazz:
            print(make)