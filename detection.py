import os
import random
import matplotlib.pyplot as plt
from cv2 import split
import keras,os,shutil
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, RandomFlip, RandomRotation, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import load_img, img_to_array, image_dataset_from_directory
import numpy as np
import tensorflow as tf

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.losses import CategoricalCrossentropy


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

dirname = os.path.dirname(__file__)

num_classes = 5

def genVGG16(num_classes):

    model = Sequential()
    #model.add(data_augmentation)
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))

    return model

def genVGG19(num_classes):

    model = Sequential()
    #model.add(data_augmentation)
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))

    return model

vgg16model = genVGG16(num_classes)
vgg19model = genVGG19(num_classes)

vgg16model.summary()

opt = SGD(lr=0.001, momentum=0.9)
vgg16model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
vgg19model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

vgg16keras = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
flat_layer = Flatten()(vgg16keras.output)
tail = Dense(num_classes, activation='softmax')(flat_layer)
newvgg16 = Model(inputs=vgg16keras.input, outputs=tail)
newvgg16.summary()

newvgg16.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

newvgg16.save("vgg16keras.h5")

#dataset = image_dataset_from_directory(os.path.join(dirname, "data"), validation_split=0.2, subset="both")
#print(dataset)

tr_data = ImageDataGenerator(validation_split=0.2, rescale=1/255)
training_data = tr_data.flow_from_directory(batch_size=32, directory=os.path.join(dirname, "data"), 
    class_mode='categorical', subset='training', target_size=(224,224))
#validation_data = tr_data.flow_from_directory(batch_size=32, directory=os.path.join(dirname, "data"), 
#    class_mode='categorical', subset='validation', target_size=(224,224))

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

#histvgg19 = vgg19model.fit_generator(dataset[0], steps_per_epoch=100, validation_data=dataset[1], validation_steps=10,epochs=5,callbacks=[checkpoint,early])
#histvgg19 = vgg19model.fit_generator(steps_per_epoch=100,generator=training_data, validation_steps=10,epochs=5,callbacks=[checkpoint,early])


