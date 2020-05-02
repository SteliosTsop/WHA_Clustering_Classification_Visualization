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

from Utils import cmap_map


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

            
# K nearest neighbors parameters
neighbors = 15
weight_option = 'uniform' # 'distance'
#K means parameteres
clusters = 5


# Insert the extracted features to PCA, in order to reduce dimensions to 50 or 100
X_pca_50 = PCA(n_components=50).fit_transform(features)

target_array = np.asarray(targets,dtype=np.int8)


# For 5 classes
labels = [ 'WHA90 1xx', 'WHA92 2xx', 'WHA95 3xx', 'WHA97 4xx', 'WHA99 5xx']
colors = [0, 1, 2, 3, 4]



for p in range(10, 50, 5):
    
    print('\n')
    print('Perplexity = {}'.format(p))
    print('\n')
    
    # Take the reduced features by PCA and insert them into t-sne
    X_tsne= TSNE(n_components=2, perplexity=p, n_iter=4000, verbose=0).fit_transform(X_pca_50)
    
    # separate the train images and their ground truth labels
    X_tsne_train = X_tsne[:train_images]
    target_train = target_array[:train_images] 
    
    
    # separate the test images and their ground truth labels
    X_tsne_test = X_tsne[train_images:]
    target_test = target_array[train_images:]
    
    
    # -------------------------------------------------------------------------------------------------------------------------
    #   Kmeans labeling
    #--------------------------------------------------------------------------------------------------------------------------
    
    kmeans = KMeans(n_clusters=clusters, init = 'k-means++', n_init = 10, max_iter = 1000)
    kmeans.fit(X_tsne_train)
    labs = kmeans.labels_


    correspond_labels = np.zeros(labs.shape)
        
    correspond_labels[labs==0] = np.argmax(np.bincount(target_train[labs==0]))    
    correspond_labels[labs==1] = np.argmax(np.bincount(target_train[labs==1]))
    correspond_labels[labs==2] = np.argmax(np.bincount(target_train[labs==2]))
    correspond_labels[labs==3] = np.argmax(np.bincount(target_train[labs==3]))
    correspond_labels[labs==4] = np.argmax(np.bincount(target_train[labs==4])) 
    
    
    # find the errors
    l = [k if t==k else max(target_train)+1 for t,k in zip(target_train, correspond_labels)]
    l = np.asarray(l,dtype=np.int8)

    # correct indices
    cor_idx = l<=max(target_train)    
    cor_labels = l[cor_idx]
        
    X_tsne_train_c = X_tsne_train[cor_idx]  
    
    # -------------------------------------------------------------------------------------------------------------------------
    #   K Nearest Neighbors Classifier
    #--------------------------------------------------------------------------------------------------------------------------
    
        
    kmeans_neigh = KNeighborsClassifier(n_neighbors=neighbors,weights=weight_option)
    kmeans_neigh.fit(X_tsne_train_c, cor_labels)
 
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_tsne_train_c[:, 0].min() - 1, X_tsne_train_c[:, 0].max() + 1
    y_min, y_max = X_tsne_train_c[:, 1].min() - 1, X_tsne_train_c[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    xy = np.c_[xx.ravel(), yy.ravel()]
    # Predict the label of each mexh point with K-nearest neighbors
    knn_labels = kmeans_neigh.predict(xy)
    
    # Put the result into a color plot
    zz = knn_labels.reshape(xx.shape)
    
    light_brg = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.brg)
    dark_brg = cmap_map(lambda x: x*0.75, matplotlib.cm.brg)
    
    # Plot the result with scatter
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = plt.axes(frameon=False)
    plt.setp(ax1, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    mp1 = plt.pcolormesh(xx, yy, zz, cmap=light_brg )
    # sc = plt.scatter(X_tsne_test[:, 0], X_tsne_test[:, 1], c='k', marker="*")
    handles_1 = [plt.plot([],color=mp1.get_cmap()(mp1.norm(c)),ls="", marker="o")[0] for c in colors ]    
    legend_1 = ax1.legend(handles_1, labels, loc="lower right")
    ax1.add_artist(legend_1)

    # save the figure with the specific perplexity
    output_file1 =  os.path.join(ROOT_DIR, 'Your_Result_Dir\\perplexity_' + str(p) + '_knn.png')
    plt.savefig(output_file1)
