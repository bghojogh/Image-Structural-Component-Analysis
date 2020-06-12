import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
import os
import pickle


class My_dual_PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.X = None
        self.U = None
        self.S = None
        self.V = None
        self.mean_of_X = None

    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed

    def fit(self, X):
        # X: rows are features and columns are samples
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        X = X - self.mean_of_X
        self.X = X
        U, s, Vh = LA.svd(self.X, full_matrices=False)  #---> in dual PCA, the S should be square so --> full_matrices=False
        V = Vh.T
        if self.n_components != None:
            U = U[:,:self.n_components]
            s = s[:self.n_components]
            V = V[:,:self.n_components]
        s = np.asarray(s)
        S = np.diag(s)
        self.U = U
        self.S = S
        self.V = V

    def transform(self, X):
        # X: rows are features and columns are samples
        X_transformed = (self.S).dot(self.V.T)
        return X_transformed

    def transform_outOfSample(self, x):
        # x: a vector
        x = np.reshape(x,(-1,1))
        x = x - self.mean_of_X
        n = self.X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        x_transformed = (inv(self.S)).dot(self.V.T).dot(H).dot(self.X.T).dot(x)
        return x_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        X = X - self.mean_of_X
        if using_howMany_projection_directions != None:
            V = self.V[:, 0:using_howMany_projection_directions]
        else:
            V = self.V
        X_reconstructed = X.dot(V).dot(V.T)
        X_reconstructed = X_reconstructed + self.mean_of_X
        return X_reconstructed

    def reconstruct_outOfSample(self, x, using_howMany_projection_directions=None):
        # x: a vector
        x = np.reshape(x, (-1, 1))
        x = x - self.mean_of_X
        if using_howMany_projection_directions != None:
            V = self.V[:, 0:using_howMany_projection_directions]
            S = self.S[0:using_howMany_projection_directions, 0:using_howMany_projection_directions]
        else:
            V = self.V
            S = self.S
        x_reconstructed = (self.X).dot(V).dot(inv(S)).dot(inv(S)).dot(V.T).dot(self.X.T).dot(x)
        x_reconstructed = x_reconstructed + self.mean_of_X
        return x_reconstructed

    def classify_distortion_trainingSet(self):
        n_samples = self.X.shape[1]
        X_projected = (self.S).dot(self.V.T)
        # --- KNN:
        estimated_distortion_class = np.zeros((n_samples, 2))
        connectivity_matrix = KNN(X=X_projected.T, n_neighbors=1, mode='connectivity', include_self=False, n_jobs=-1)
        connectivity_matrix = connectivity_matrix.toarray()
        for image_index in range(n_samples):
            index_of_neighbor = int(np.argwhere(connectivity_matrix[image_index, :] == 1))
            if index_of_neighbor == 0:  # "original"
                estimated_distortion_class[image_index, 0] = 0
            elif index_of_neighbor >= 1 and index_of_neighbor <= 20:  # "contrast_stretched"
                estimated_distortion_class[image_index, 0] = 1
            elif index_of_neighbor >= 21 and index_of_neighbor <= 40:  # "Gaussian_noise"
                estimated_distortion_class[image_index, 0] = 2
            elif index_of_neighbor >= 41 and index_of_neighbor <= 60:  # "enhanced_luminance"
                estimated_distortion_class[image_index, 0] = 3
            elif index_of_neighbor >= 61 and index_of_neighbor <= 80:  # "Gaussian_blurring"
                estimated_distortion_class[image_index, 0] = 4
            elif index_of_neighbor >= 81 and index_of_neighbor <= 100:  # "impulse_noise"
                estimated_distortion_class[image_index, 0] = 5
            elif index_of_neighbor >= 101 and index_of_neighbor <= 120:  # "jpeg_distortion"
                estimated_distortion_class[image_index, 0] = 6
            estimated_distortion_class[image_index, 1] = image_index
        path_to_save = './output/dual_PCA/classification/'
        self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
        self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
        return estimated_distortion_class[:, 0]

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))