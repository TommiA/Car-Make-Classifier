# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 200
INIT_LR = 1e-4
BS = 150
IMAGE_DIMS = (400, 400, 3)
train_image_folder = '/mnt/datasets/carconv/dataset/train'
test_image_folder = '/mnt/datasets/carconv/dataset/test'

# construct the image generator for data augmentation
aug = ImageDataGenerator(rescale = 1./255, zoom_range=0.2, vertical_flip=True, fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = aug.flow_from_directory(
    train_image_folder,
    target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]),
    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    test_image_folder,
    target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]),
    class_mode="categorical")

# binarize the labels
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(train_generator.class_indices, protocol=4))
f.close()

#print(len(train_generator.class_indices))
train_data_length=len(train_generator.classes)
test_data_length=len(test_generator.classes)

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()
#for layer in model_vgg16_conv.layers:
#    layer.trainable = False

#Create your own input format (here 3x200x200)
input = Input(shape=IMAGE_DIMS,name = 'image_input')

#Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(256, activation='relu', name='fc3')(x)
x = Dense(81, activation='softmax', name='predictions')(x)

#Create your own model
my_model = Model(input=input, output=x)

my_model.summary()

print("[INFO] compiling model...")
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = SGD(lr=INIT_LR, momentum=0.9)
my_model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

# train the network
print("[INFO] training network...")
H = my_model.fit_generator(
    train_generator,
	steps_per_epoch=(train_data_length // BS),
    validation_data=test_generator,
    validation_steps=(test_data_length // BS),
    use_multiprocessing=True,
    workers=4,
    epochs=EPOCHS, verbose=1,
    callbacks = [checkpoint, early])

 
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy ("+str(IMAGE_DIMS[1])+"x"+str(IMAGE_DIMS[0])+") LR "+str(INIT_LR))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")

filename=str("plot_E"+str(EPOCHS)+"_IMG"+str(IMAGE_DIMS[1])+"_BS"+str(BS)+"_INVLR"+str(int(1/INIT_LR)))
plt.savefig(filename)

    
    

  
    
