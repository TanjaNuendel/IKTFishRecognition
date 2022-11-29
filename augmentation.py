import os
import random
import matplotlib.pyplot as plt
from cv2 import split
import keras,os,shutil
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, RandomFlip, RandomRotation, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from tqdm import tqdm

dirname = os.path.dirname(__file__)
dir_to_clean = os.path.join(dirname, ".data", "fish_03")

'''
l = os.listdir(dir_to_clean)

#print(int(int(len(os.listdir(dir_to_clean))) / 11))

print(len(l[::1]))

with tqdm(total=len(l[::3])) as pbar:
    for n in l[::3]:
        target = dir_to_clean + '/' + n
        #print(target)
        if os.path.isfile(target):
            os.unlink(target)
        pbar.update(1)

'''


gen = ImageDataGenerator(rescale = 1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)

img_path = os.path.join(os.path.join(dirname, ".data"), "fish_10")

#chosen_image = random.choice(os.listdir(img_path))
#chosen_image_path = os.path.join(img_path,chosen_image)
i = 0
for img in os.listdir(img_path):
  print(i)
  i = i + 1
  chosen_image_path = os.path.join(img_path,img)
  pic = load_img(chosen_image_path)
  pic_array = img_to_array(pic)
  #print(pic_array.shape)
  pic_array = np.expand_dims(pic_array,0)
  #print(pic_array.shape)

  count = 0
  for batch in gen.flow(pic_array, batch_size=5, save_prefix="aug", save_to_dir=os.path.join(dirname, ".data", "fish_10")):
    count += 1
    #plotImages(batch)
    #plt.imshow(pic_array[0])
    #print(batch.shape)
    #plt.imshow(batch[0])
    #plotImages(batch[0])
    if count == 3:
      break
  print("augmentation for one image finished")

print("all files saved")


