import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors as KNN   # https://scikit-learn.org/stable/modules/neighbors.html
import math
import pickle

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

class My_Laplacian_eigenmap_SSIM:

    def __init__(self, n_components=None, n_neighbors=5, n_jobs=-1):
        self.n_components = n_components
        self.X = None
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs  # number of parallel jobs --> -1 means all processors --> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph

    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.X = X
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

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows,1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)
        return the_matrix

    def read_SSIM_distance_matrix(self):
        SSIM_distance_matrix = self.load_variable(name_of_variable="distance_index", path='./kernel_SSIM_2/')
        return SSIM_distance_matrix

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def calculate_adjacency_matrix(self, X):
        n_samples = X.shape[1]
        adjacency_matrix = np.zeros((n_samples, n_samples))
        SSIM_distance_matrix = self.read_SSIM_distance_matrix()
        SSIM_distance_matrix_centered = self.center_the_matrix(the_matrix=SSIM_distance_matrix, mode="double_center")
        SSIM_kernel = -0.5 * SSIM_distance_matrix_centered
        for row_index in range(SSIM_kernel.shape[0]):
            row = SSIM_kernel[row_index, :]
            idx = row.argsort()[::-1]  # sort in descending order (largest first)
            row_sorted = row[idx]
            for neighbor_index in range(self.n_neighbors+1):  #+1 because itself has the closest SSIM
                adjacency_matrix[row_index, idx[neighbor_index]] = row_sorted[neighbor_index]
                # x1 = X[:, row_index]
                # x2 = X[:, idx[neighbor_index]]
                # adjacency_matrix[row_index, idx[neighbor_index]] = math.exp( - (LA.norm(x1 - x2)) ** 2 )
        return adjacency_matrix