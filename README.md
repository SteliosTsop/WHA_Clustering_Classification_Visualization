# Clustering and Classification Unsupervised ML algorithms - Visualization of Activation Maps

The main objective of this repository is to present Unsupervised ML data pipelines that enable the clustering and the classification of SEM fracture images of WHA samples, according to their tungsten composition. Additionally, aiming to interpret the functionality of these algorithms and acquire better understanding of the internal operations that enable the efficacy of the algorithms, another algorithm that visualizes the activation maps of the last convolution layer, according to their importance on the data pipelines, is developed.

The theoretical framework and the results of our research work are publically available in the open source publication: 

The dataset that is used to evaluate the performance of the introduced algorithms is composed by 810 SEM fracture images with dimensions: 448x448. The SEM images are obtained after scanning the fracture surface of 5 different WHA samples with tungsten composition of: 90wt%, 92wt%, 95wt%, 97wt% and 99wt%. 

The entire WHA dataset and the corresponding Activation Maps are published in Materials Data Facility(MDF) with DOI: 

## Clustering Data Pipeline

The main structure of the Clustering Data Pipeline is composed by 4 consecutive parts, where the output of each part is the input of the next.

