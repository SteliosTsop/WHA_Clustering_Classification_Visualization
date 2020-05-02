%matplotlib inline

import os,sys
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt


from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.applications import vgg16

import cv2

from scipy.stats import mode

import itertools

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


model = vgg16.VGG16(include_top=False,weights='imagenet',pooling = 'avg')

# Create the paths for inserting the train images
ROOT_DIR = os.getcwd()
train_dir = os.path.join(ROOT_DIR, "Your_Datset_Dir\\train_images")
train_dirs = os.listdir(train_dir)
train_images = len(train_dirs)


# Create the paths for inserting  the test images
test_dir = os.path.join(ROOT_DIR, "Your_Datset_Dir\\test_images")
test_dirs = os.listdir(test_dir)
test_images = len(test_dirs)

# Total number of images
n_images = train_images + test_images

# Define a numpy.ndarray that stores the features that will be extracted by the VGG
features = np.empty([n_images,512])


# Crete an array for the targets and then define the targets
target = np.empty(n_images)

# For each tungsten composition there are 150 train images and 12 test images, except the 90wt% tungsten composition that contains 
# only 75 train images and 6 test images
train_imgs_per_class = 75
test_imgs_per_class = 6

# label each category
target[:train_imgs_per_class] = 0
target[train_imgs_per_class:3*train_imgs_per_class] = 1
target[3*train_imgs_per_class:5*train_imgs_per_class] = 2
target[5*train_imgs_per_class:7*train_imgs_per_class] = 3
target[7*train_imgs_per_class:train_images] = 4

target[train_images:train_images + test_imgs_per_class] = 0
target[train_images + test_imgs_per_class:train_images + 3*test_imgs_per_class] = 1
target[train_images + 3*test_imgs_per_class:train_images + 5*test_imgs_per_class] = 2
target[train_images + 5*test_imgs_per_class:train_images + 7*test_imgs_per_class] = 3
target[train_images + 7*test_imgs_per_class:train_images + test_images] = 4




# convert them into a list
targets = list(target)


i = 0
for item in train_dirs:
        img_path = os.path.join(train_dir,item) 
        if os.path.isfile(img_path):
            
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features[i] = model.predict(x)
            i+=1

for item in test_dirs:
        img_path = os.path.join(test_dir,item) 
        if os.path.isfile(img_path):
            
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features[i] = model.predict(x)
            i+=1            

            
