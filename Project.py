# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:10:54 2021

@author: Ali
"""
import sknetwork as skn
from IPython.display import SVG

import numpy as np
from scipy import sparse
import pandas as pd

################# Imports for the different parts of the project############
from sknetwork.utils import edgelist2adjacency
from sknetwork.data import convert_edge_list, load_edge_list
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph
from sknetwork.topology import connected_components
from sknetwork.utils.format import bipartite2undirected
from sknetwork.topology import CoreDecomposition
from sknetwork.classification import accuracy_score

################################## Show 3D data

def show3D(data):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax = plt.axes(projection='3d')

# Data for a three-dimensional line
    zline = data[:,2]
    xline = data[:,0]
    yline = data[:,1]
    ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
    ax.scatter3D(xline, yline, zline);
    return fig

def show2D(data):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes()

# Data for a three-dimensional line
    xline = data[:,0]
    yline = data[:,1]
   # ax.plot(xline, yline 'gray')

# Data for three-dimensional scattered points
    ax.scatter(xline, yline);
    return fig

############################ Loading You Tube Dataset####################
graph = load_edge_list('D:\PhdResearch\RepresentationLearning\Project\Dataset\YouTube\soc-youtube.mtx')
adjacency = graph.adjacency
names = graph.names
#position = graph.position



adjacency = graph.adjacency
#position = graph.position
k = 300
adjacency = adjacency[:k][:,:k]
#position = position[:k]

############################## Task 2: Embedding the network using PCA, Random Walk and SVD
from sknetwork.embedding import PCA,RandomProjection,SVD, GSVD
from sknetwork.ranking import PageRank,Diffusion
adjacency = graph.adjacency
k = 300
names = graph.names[:k]
adjacency = adjacency[:k][:,:k]
pagerank = PageRank()
position= np.random.rand(1500, 2)
scores = pagerank.fit_transform(adjacency)
degrees = adjacency.dot(np.ones(adjacency.shape[0]))
image = svg_graph(adjacency,names=names, position=position,edge_width=0.1,display_node_weight=True)
SVG(image)


#############Embedding Using PCA
pca = PCA(2,normalized=True,solver='auto')
embeddingPCA = pca.fit_transform(adjacency)
embeddingPCA.shape
image = svg_graph(adjacency,embeddingPCA,edge_width=0.08,display_node_weight=True)

SVG(image)

#############SHowing PCA 3D
#pca = PCA(3)
#embeddingPCA = pca.fit_transform(adjacency)
#show3D(embeddingPCA)  ###########Show embedding graph space 3D




#############Embedding Using Random Walk
#labels = graph.labels
projection = RandomProjection(2,random_walk=True,normalized=True)
embeddingRW = projection.fit_transform(adjacency)
embeddingRW.shape
image = svg_graph(adjacency, embeddingRW,edge_width=0.08,display_node_weight=True)
SVG(image)
show2D(embeddingRW)  ###########Show embedding graph space 3D
ad=embeddingRW



#############Embedding Using SVD
svd = SVD(2,normalized=True)
embeddingSVD = svd.fit_transform(adjacency)
embeddingSVD.shape
image = svg_graph(adjacency, embeddingSVD,edge_width=0.08,display_node_weight=True)
SVG(image)
show2D(embeddingSVD)
#show3D(embeddingSVD)





################################# Task 1: ink Prediction
from sknetwork.linkpred import JaccardIndex, AdamicAdar, is_edge, whitened_sigmoid

################# Link Prediction Using Adamic Adar
aa = AdamicAdar()
aa.fit(adjacency)

edges = [(0, 5), (0, 7), (32, 2), (0, 2),(108,2),(115,2),(0,32),(210,1),(0,160),(0,1)]
y_true = is_edge(adjacency, edges)

scores = aa.predict(edges)
y_pred = whitened_sigmoid(scores) > 0.6

accuracy_score(y_true, y_pred)


############## Link Prediction using Jacard Index
ji = JaccardIndex()
ji.fit(adjacency)
y_true = is_edge(adjacency, edges)

scores = ji.predict(edges)
y_pred = whitened_sigmoid(scores) > 0.6

accuracy_score(y_true, y_pred)





############################## Task 3: Using Cos. Modularity for evaluating the embedded graohs
from sknetwork.embedding import cosine_modularity
cosine_modularity(adjacency, embeddingPCA, weights='degree')
cosine_modularity(adjacency, embeddingRW, weights='degree')
cosine_modularity(adjacency, embeddingSVD, weights='degree')


############################## Task 4: Clustering of the network
from sknetwork.clustering import Louvain, modularity
from sknetwork.linalg import normalize

adjacency = graph.adjacency
k = 300
adjacency = adjacency[:k][:,:k]

################### Clustering Using Louvain
louvain = Louvain(resolution=1,return_membership=True)
labels = louvain.fit_transform(adjacency)
#scores = louvain.membership_[:,1].toarray().ravel()
labels_unique, counts = np.unique(labels, return_counts=True)
print(labels_unique, counts)
image = svg_graph(adjacency, position=embeddingRW, labels=labels,edge_width=0.08,node_size_min = 3, node_size_max = 20,display_node_weight=True)
SVG(image)
print("Modularity: ",modularity(adjacency, labels))
adj=louvain.adjacency_.toarray()
mem=louvain.adjacency_.toarray()


################### Clustering Using K-Means
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
kmeans = KMeans(n_clusters = 14, embedding_method=RandomProjection(2,random_walk=True,normalized=True))
labels = kmeans.fit_transform(adjacency)

unique_labels, counts = np.unique(labels, return_counts=True)
print(unique_labels, counts)
image = svg_graph(adjacency, position=embeddingRW, labels=labels,edge_width=0.08,node_size_min = 3, node_size_max = 20,display_node_weight=True)

SVG(image)
print("Modularity: ",modularity(adjacency, labels))
adj=kmeans.adjacency_

########################### Task 5: Finding Shortest path in the graph
from sknetwork.path import shortest_path,distance
src = 2 
dest = 290

path = shortest_path(adjacency, sources=src, targets=dest,unweighted=True)
edge_labels = [(path[k], path[k + 1],k) for k in range(len(path) - 1)]
names2=[k if (k==1 or k==3 or k==4 or k==90 or k==291) else "" for k in names]
image = svg_graph(adjacency,position=embeddingRW, edge_labels=edge_labels,edge_width=3,names=names2, display_edge_weight=False,scale=2)
SVG(image)




################################## Performing MLLE

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
X, _ = load_digits(return_X_y=True)
X.shape

embedding = LocallyLinearEmbedding(n_components=2,method='modified')
X_transformed = embedding.fit_transform(X[:100])
X_transformed.shape
##########################################

from IPython.display import SVG
import numpy as np
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.embedding import Spectral, BiSpectral, cosine_modularity
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph

graph = load_edge_list('D:\PhdResearch\RepresentationLearning\Project-Ali Abbasi Tadi\Dataset\YouTube\soc-youtube.mtx')
adjacency = graph.adjacency
names = graph.names
#position = graph.position



adjacency = graph.adjacency
#position = graph.position
k = 300
adjacency = adjacency[:k][:,:k]

from sknetwork.embedding import Spectral, BiSpectral, cosine_modularity, LaplacianEmbedding

from sknetwork.embedding import PCA,RandomProjection,SVD, GSVD
from sknetwork.ranking import PageRank,Diffusion
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
kmeans = KMeans(n_clusters = 14, embedding_method=RandomProjection(2,random_walk=True,normalized=True))
labels = kmeans.fit_transform(adjacency)


spectral = LaplacianEmbedding(2, normalized=False)
embedding = spectral.fit_transform(adjacency)
embedding.shape
image = svg_graph(adjacency, embedding, labels=labels)
SVG(image)



#####################################################


gsvd = GSVD(2, normalized=False)
embedding = gsvd.fit_transform(adjacency)
image = svg_graph(adjacency, embedding, labels=labels)
SVG(image)

#########################################
from sknetwork.embedding import Spring
from sknetwork.utils import KNNDense, CNNDense
knn = KNNDense(n_neighbors=3, undirected=True)
adjacency = knn.fit_transform(adjacency)
image = svg_graph(adjacency, labels=labels, display_edge_weight=False)
SVG(image)

data=adjacency.getcol(2)
data2=adjacency.getrow(299)

############################################