%matplotlib inline

import os,sys
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications.imagenet_utils import preprocess_input
from keras import models
import cv2

from sklearn.decomposition import PCA


model = vgg16.VGG16(include_top=False,weights='imagenet',pooling = 'avg')

# Create the paths for inserting images
ROOT_DIR = os.getcwd()
input_dir = os.path.join(ROOT_DIR, "Your_Dataset_Dir")
dirs = os.listdir(input_dir)
n_images = len(dirs)


# Define a numpy.ndarray that stores the features that will be extracted by the VGG
features = np.empty([n_images,512])

# Crete an array for the targets and then define the targets
target = np.empty(n_images)


imgs_per_class = 162

# label images according to the tungsten composition
target[:imgs_per_class] = 0
target[imgs_per_class:2*imgs_per_class] = 1
target[2*imgs_per_class:3*imgs_per_class] = 2
target[3*imgs_per_class:4*imgs_per_class] = 3
target[4*imgs_per_class:] = 4


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
            x = np.expand_dims(x, axis=0) # Adds a dimension to transform the array into a batch of size (1, h, w, 3)
            x = preprocess_input(x)   # converts pixels from [0,255] to [0,1]
            features[i] = model.predict(x)
            
            i+=1


# ---------------------------  VISUALIZATION OF ACTIVATION MAPS ---------------------------------------------------------


from keras import models

pca = PCA(n_components=50)
X_pca = pca.fit_transform(features)

PC_comp = 1 

e_values = pca.explained_variance_ # These are the eigenvalues of the principal components.
                                   # Each eigenvalue expresses the importance of  each principal component.
        
e_vectors = np.absolute(pca.components_) # This is a 50x512 matrix, where each row corresponds to the eigenvector of
                                         # of the most important principal components and expresses how much 
                                         # each of the features influences each principal component.First row for PC1, 2nd for PC2 ...

n_major_features = e_vectors[PC_comp,:].shape[0]


# Get the layer outputs from the last layers of the VGG15 architecture and create a new model that is called activation_model
layer_outputs = [layer.output for layer in model.layers[15:]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


for item in dirs:
    # load image
    img_path = os.path.join(input_dir,item)
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Get the activation maps for the last layers of the VGG16 network on the specific input image
    activations = activation_model.predict(x)
    # Then keep only the activations of the last Conv layer
    last_conv_layer_activation = activations[-2]
    heatmap = np.zeros(last_conv_layer_activation[0, :, :, 0].shape,dtype = np.float32)
    for i in range(n_major_features):
        # heatmap += last_conv_layer_activation[0,:,:,principal_features[i]]*e_vectors[0,principal_features[i]]
        heatmap += last_conv_layer_activation[0, :, :, i]*e_vectors[PC_comp,i]
        
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Use cv2 to load the original image
    img = cv2.imread(img_path)
    # Resize heatmap and make it RGB
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
        
    # Apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    # save it
    OUT_DIR =  os.path.join(ROOT_DIR, 'Your_results_Dir')
    out_file = os.path.join(OUT_DIR,item)
    cv2.imwrite(out_file, superimposed_img)
