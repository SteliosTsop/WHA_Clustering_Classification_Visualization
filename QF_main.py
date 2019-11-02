%matplotlib inline

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from Model import *


model = VGG16()

# Create the paths for inserting images
ROOT_DIR = os.getcwd()
input_dir = os.path.join(ROOT_DIR, "Datasets\Image_test\images")
dirs = os.listdir(input_dir)
n_images = len(dirs)


# Define a numpy.ndarray that stores the features that will be extracted by the VGG
features = np.empty([n_images,4096]) 
# Crete an array for the targets and then define the targets
target = np.empty(n_images)

# for example
target[:117] = 0
target[117:] = 1
# convert them into a list
targets = list(target)

# Shuffle them 
combined = list(zip(dirs, targets))
shuffle(combined)

dirs, targets = zip(*combined)

i = 0
for item in dirs:
        img_path = os.path.join(input_dir,item) 
        if os.path.isfile(img_path):
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features[i] = model.predict(x)
            i+=1

                     
            
# Insert the extracted features to PCA
X_pca = PCA(n_components=50).fit_transform(features)
print(X_pca.shape)

# Plot the result with scatter
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(frameon=False)
plt.setp(ax, xticks=(), yticks=())
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                wspace=0.0, hspace=0.0)
plt.scatter(X_pca[:, 0], X_pca[:, 1],c=targets)


# Take the reduced features by PCA and insert them into t-sne
X_tsne= TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_pca)

# Plot the result with scatter
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(frameon=False)
plt.setp(ax, xticks=(), yticks=())
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                wspace=0.0, hspace=0.0)
plt.scatter(X_tsne[:, 0], X_pca[:, 1],c=targets)

