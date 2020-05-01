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


