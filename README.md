# Clustering and Classification Unsupervised ML algorithms - Visualization of Activation Maps

This repository presents the computer algorithms developed for the open source publication: *"Unsupervised Machine Learning in Fractography: Evaluation and Interpretation"*

The main objective of this repository is to present Unsupervised ML data pipelines that enable the clustering and the classification of SEM fracture images of WHA samples, according to their tungsten composition. Additionally, aiming to interpret the functionality of these algorithms and acquire better understanding of the internal operations that enable the efficacy of the algorithms, another algorithm that visualizes the activation maps of the last convolution layer, according to their importance on the data pipelines, is developed.

The dataset that is used to evaluate the performance of the introduced algorithms is composed by 810 SEM fracture images with dimensions: 448x448. The SEM images are obtained after scanning the fracture surface of 5 different WHA samples with tungsten composition of: 90wt%, 92wt%, 95wt%, 97wt% and 99wt%. 

The entire WHA dataset and the corresponding Activation Maps are published in Materials Data Facility (MDF) with DOI: 

The source code of the Clustering and Classification algorithms builds upon the code published at [neu_vgg16](https://github.com/arkitahara/neu_vgg16) by [Andrew Kitahara](https://github.com/arkitahara).

## Clustering Data Pipeline

The main structure of the Clustering Data Pipeline is composed by 4 consecutive parts, where the output of each part is the input of the next (see the schematic flowchart of the pipeline in the figure below). 

<img src="Images/cluster_pipeline_comments.JPG">



## Classification Data Pipeline

Adding a *minimally supervision* algorithm at the end of the clustering pipeline and defining a different computational framework the previous data pipeline is converted to a classification algorithm that enables the classification of the input fracture images according to the tungsten composition. The structure of this classification algorithm and the definition of its computational framework is schematically presented in the mext figure.

<img src="Images/classification_pipeline_2.JPG">

The computational framework is composed of the following steps:

1. Initially, 90% of the WHA fracture images dataset is set as training images, while the rest 10% is set as test images.
2. The entire dataset is imported into the **VGG - PCA - t-SNE** pipeline and the result is a 2D scatter plot. This plot is separated into a plot that contains only the training data points and another one with the test data points.
3. The training data points, projected onto the specific locations on the 2D plot by the **VGG - PCA - t-SNE** pipeline, are imported into another pipeline constructed by k-Means and KNN algorithms.
4. A mesh grid with dimensions large enough to accommodate every training and test data point is created and KNN enables the classification of every grid point into one of the 5 tungsten composition labels, using the training data points and their annotations. The result of this step is a colormap, where each area is assigned to a different tungsten composition label.
5. The final step involves plotting the test data points, with the positions predicted by the **VGG - PCA - t-SNE** pipeline, onto the colormap. The label of the area that each test data point is placed defines the classification of the test point.
