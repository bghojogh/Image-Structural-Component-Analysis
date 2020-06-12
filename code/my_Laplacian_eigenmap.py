import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors as KNN   # https://scikit-learn.org/stable/modules/neighbors.html
import math

# ----- python fast kernel matrix:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
# https://stackoverflow.com/questions/7391779/fast-kernel-matrix-computation-python
# https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# https://stackoverflow.com/questions/36324561/fast-way-to-calculate-kernel-matrix-python?rq=1

# ----- python fast scatter matrix:
# https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation

# ----- similarity and kernel matrix:
# https://scikit-learn.org/stable/modules/metrics.html#metrics
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html

class My_Laplacian_eigenmap:

    def __init__(self, n_components=None, n_neighbors=5, n_jobs=-1):
        self.n_components = n_components
        self.X = None
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs  # number of parallel jobs --> -1 means all processors --> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph

    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.X = X
        # W = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        W = self.calculate_adjacency_matrix(X)
        d = np.sum(W, axis=1)
        d = np.asarray(d)
        D = np.diag(d)
        L = D - W
        eig_val, eig_vec = LA.eigh(L)
        idx = eig_val.argsort()  # sort eigenvalues in ascending order (smallest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            X_transformed = eig_vec[:, 1:self.n_components+1] #--> note that first eigenvalue is zero
        else:
            X_transformed = eig_vec[:, 1:] #--> note that first eigenvalue is zero
        X_transformed = X_transformed.T  #--> the obtained Y in Laplacian eigenmap is row-wise vectors, so we transpose it
        return X_transformed

    def calculate_adjacency_matrix(self, X):
        n_samples = X.shape[1]
        adjacency_matrix = np.zeros((n_samples, n_samples))
        knn = KNN(n_neighbors=self.n_neighbors, algorithm='kd_tree', n_jobs=self.n_jobs)
        knn.fit(X=self.X.T)
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph
        # the following function gives n_samples*n_samples matrix, and puts 0 for where points are not connected directly in KNN graph
        connectivity_matrix = knn.kneighbors_graph(X=X.T, n_neighbors=self.n_neighbors+1, mode='connectivity')  #+1 because the point itself is also counted
        connectivity_matrix = connectivity_matrix.toarray()
        for point_index in range(connectivity_matrix.shape[0]):
            for point_index_2 in range(connectivity_matrix.shape[1]):
                if connectivity_matrix[point_index, point_index_2] == 1:
                    x1 = X[:, point_index]
                    x2 = X[:, point_index_2]
                    adjacency_matrix[point_index, point_index_2] = math.exp( - (LA.norm(x1 - x2)) ** 2 )
        return adjacency_matrix