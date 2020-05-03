%matplotlib inline

import os,sys
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications import vgg16, inception_v3, resnet50
from keras.applications.vgg16 import preprocess_input

import cv2

from scipy.stats import mode

import itertools

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Initialize keras model for VGG16 architecture
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

# Ground truth labels for the 5 categories of tungsten composition
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

# Load images and extract features with VGG16
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
                     
            
# Insert the extracted features to PCA, in order to reduce dimensions to 50
X_pca_50 = PCA(n_components=50).fit_transform(features)

labels = ['WHA90 1xx', 'WHA92 2xx', 'WHA95 3xx', 'WHA97 4xx', 'WHA99 5xx']
colors = [0, 1, 2, 3, 4]


# Run t-SNE for different peerplexity values
for p in range(5, 60, 5):
    
    print('\n')
    print('Perplexity = {}'.format(p))
    print('\n')
    
    # Take the reduced features by PCA and insert them into t-sne
    X_tsne= TSNE(n_components=2, perplexity=p, n_iter=2000, verbose=1).fit_transform(X_pca_50)

    # -------------------------------------------------------------------------------------------------------------------------
    # Plot tsne results with ground truth
    # -------------------------------------------------------------------------------------------------------------------------
    
    
    # Plot the result with scatter
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = plt.axes(frameon=False)
    plt.setp(ax1, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    scatter1 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=targets, cmap="Spectral")
    handles1 = [plt.plot([],color=scatter1.get_cmap()(scatter1.norm(c)),ls="", marker="o")[0] for c in colors ]    
    legend1 = ax1.legend(handles1, labels, loc="lower right")
    ax1.add_artist(legend1)
    
    # save the figure with the specific perplexity
    output_file1 =  os.path.join(ROOT_DIR, 'Your_result_dir\\perp_' + str(p) + '_gt.png')
    plt.savefig(output_file1)



    
    # -------------------------------------------------------------------------------------------------------------------------
    #   Kmeans labeling
    #--------------------------------------------------------------------------------------------------------------------------
       
    
    kmeans = KMeans(n_clusters=5, init = 'k-means++', n_init = 10, max_iter = 1000)
    kmeans.fit(X_tsne)
      
    
    labs = kmeans.labels_
    target_array = np.asarray(targets,dtype=np.int8)
    correspond_labels = np.zeros(labs.shape)
    
    # Convert the k-Means labeling into the same colormap labeling as the ground truth labeling
    correspond_labels[labs==0] = np.argmax(np.bincount(target_array[labs==0]))    
    correspond_labels[labs==1] = np.argmax(np.bincount(target_array[labs==1]))
    correspond_labels[labs==2] = np.argmax(np.bincount(target_array[labs==2]))
    correspond_labels[labs==3] = np.argmax(np.bincount(target_array[labs==3]))
    correspond_labels[labs==4] = np.argmax(np.bincount(target_array[labs==4])) 
        
    
    # find the errors
    l = [k if t==k else max(targets)+1 for t,k in zip(targets, correspond_labels)]
    l = np.asarray(l,dtype=np.int8)

    cor_idx = l<=max(targets)
    error_idx = l==max(targets)+1

    list_labels = l[cor_idx]
    error_labels = l[error_idx]
  
    # -------------------------------------------------------------------------------------------------
    #  Plot kmeans labeling
    # -------------------------------------------------------------------------------------------------
    
    # Plot the result with scatter
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = plt.axes(frameon=False)
    plt.setp(ax2, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labs, cmap="Spectral")
    handles2 = [plt.plot([],color=scatter2.get_cmap()(scatter2.norm(c)),ls="", marker="o")[0] for c in colors ]    
    legend2 = ax2.legend(handles2, labels, loc="lower right")
    ax2.add_artist(legend2)
    
    # save the figure with the specific perplexity
    output_file2 =  os.path.join(ROOT_DIR, 'Your_results_dir\\perp_' + str(p) + '_kmeans.png')
    plt.savefig(output_file2)
    
    
    # Plot the result with scatter
    figx = plt.figure(figsize=(10, 10))
    axx = plt.axes(frameon=False)
    plt.setp(axx, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    scatterx = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=correspond_labels, cmap="Spectral")
    handlesx = [plt.plot([],color=scatterx.get_cmap()(scatterx.norm(c)),ls="", marker="o")[0] for c in colors ]    
    legendx = axx.legend(handlesx, labels, loc="lower right")
    axx.add_artist(legendx)
    
    # save the figure with the specific perplexity
    output_filex =  os.path.join(ROOT_DIR, 'Your_results_Dir\\perp_' + str(p) + '_kmeans_labeled.png')
    
    plt.savefig(output_filex)
       
    
    # -------------------------------------------------------------------------------------------------
    #  Plot kmeans labeling with erros
    # -------------------------------------------------------------------------------------------------
        
    # Plot the result with scatter
    fig3 = plt.figure(figsize=(10, 10))
    ax3 = plt.axes(frameon=False)
    plt.setp(ax3, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    sc1 = plt.scatter(X_tsne[cor_idx, 0], X_tsne[cor_idx, 1], c=list_labels, cmap="Spectral")
    sc2 = plt.scatter(X_tsne[error_idx, 0], X_tsne[error_idx, 1], c="k", marker = "x" )
    handles3 = [plt.plot([],color=sc1.get_cmap()(sc1.norm(c)),ls="", marker="o")[0] for c in colors ]    
    legend3 = ax3.legend(handles3, labels, loc="lower right")
    ax3.add_artist(legend3)

    # save the figure with the specific perplexity
    output_file3 =  os.path.join(ROOT_DIR, 'Your_results_Dir\\perp_' + str(p) + '_kmeans_errors.png')
    plt.savefig(output_file3)
    
    
    # -------------------------------------------------------------------------------------------------
    #  Confusion Matrix
    # -------------------------------------------------------------------------------------------------
        
    
    # Compute confusion matrix
    cm = confusion_matrix(target_array, correspond_labels)
    np.set_printoptions(precision=2)
        
    # Plot normalized confusion matrix
    plt.figure(figsize=(6,6)) 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    output_file =  os.path.join(ROOT_DIR, 'Your_results_Dir\\perp_' + str(p) + '_confusion_mat.png')
    plt.savefig(output_file)
    plt.close()
    
    output_txt =  os.path.join(ROOT_DIR, 'Your_results_Dir\\accuracy.txt')
    text_file = open(output_txt, "a")
    text_file.write('{:d} ,{:.3f} \n'.format(p,accuracy_score(target_array, correspond_labels)))
    text_file.close()
    
    
