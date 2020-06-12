import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
import os
import pickle


class My_LLE:

    def __init__(self, X, n_neighbors=10, n_components=None, kernel=None):
        # X: rows are features and columns are samples
        self.n_components = n_components
        self.X = X
        self.n_training_images = self.X.shape[1]
        self.data_dimension = self.X.shape[0]
        self.n_neighbors = n_neighbors
        self.neighbor_indices = None
        self.w_linearReconstruction = None
        self.W_linearEmbedding = None
        self.kernel_of_images = None
        self.n_testing_images = None
        self.neighbor_indices_for_outOfSample = None
        self.w_linearReconstruction_outOfSample = None
        self.kernel_of_images_for_outOfSample = None
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'linear'

    def LLE_fit_transform(self):
        self.LLE_find_KNN()
        self.LLE_linear_reconstruction()
        X_transformed = self.LLE_linear_embedding()
        return X_transformed

    def kernel_LLE_fit_transform(self):
        self.kernel_LLE_find_KNN()
        self.kernel_LLE_linear_reconstruction()
        X_transformed = self.LLE_linear_embedding()
        return X_transformed

    def LLE_fit_transform_outOfSample(self, data_outOfSample, X_training_transformed):
        self.LLE_find_KNN_for_outOfSample(data_outOfSample=data_outOfSample, calculate_again=True)
        self.LLE_linear_reconstruction_outOfSample(data_outOfSample)
        data_outOfSample_transformed = self.LLE_linear_embedding_outOfSample(X_training_transformed=X_training_transformed)
        return data_outOfSample_transformed

    def kernel_LLE_fit_transform_outOfSample(self, data_outOfSample, X_training_transformed):
        self.kernel_LLE_find_KNN_for_outOfSample(data_outOfSample=data_outOfSample, calculate_again=True)
        self.kernel_LLE_linear_reconstruction_outOfSample()
        data_outOfSample_transformed = self.LLE_linear_embedding_outOfSample(X_training_transformed=X_training_transformed)
        return data_outOfSample_transformed

    def get_kernels(self):
        the_kernel = pairwise_kernels(X=self.X.T, Y=self.X.T, metric=self.kernel)
        the_kernel = self.normalize_the_kernel(kernel_matrix=the_kernel)
        the_kernel = self.center_the_matrix(the_matrix=the_kernel, mode="double_center")
        return the_kernel

    def normalize_the_kernel(self, kernel_matrix):
        diag_kernel = np.diag(kernel_matrix)
        k = (1 / np.sqrt(diag_kernel)).reshape((-1,1))
        normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
        return normalized_kernel_matrix

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

    def kernel_LLE_find_KNN(self, calculate_again=True):
        if calculate_again:
            self.kernel_of_images = self.get_kernels()
            self.neighbor_indices = np.zeros((self.n_training_images, self.n_neighbors))
            # --- KNN:
            for image_index_1 in range(self.n_training_images):
                distances_from_this_image = np.zeros((1, self.n_training_images))
                for image_index_2 in range(self.n_training_images):
                    distances_from_this_image[0, image_index_2] = self.distance_based_on_kernel(kernel_matrix=self.kernel_of_images, index1=image_index_1, index2=image_index_2)
                argsort_distances = np.argsort(distances_from_this_image.ravel())  # arg of ascending sort
                indices_of_neighbors_of_this_image = argsort_distances[:self.n_neighbors]
                self.neighbor_indices[image_index_1, :] = indices_of_neighbors_of_this_image
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices, name_of_variable="neighbor_indices", path_to_save="./LLE_settings/kernel_LLE/" + self.kernel + "/")
        else:
            self.kernel_of_blocks = self.get_kernels()
            self.neighbor_indices = self.load_variable(name_of_variable="neighbor_indices", path="./LLE_settings/kernel_LLE/" + self.kernel + "/")

    def distance_based_on_kernel(self, kernel_matrix, index1, index2):
        temp = kernel_matrix[index1, index1] - 2 * kernel_matrix[index1, index2] + kernel_matrix[index2, index2]
        if temp < 0:
            # might occur for imperfect software calculations
            temp = 0
        distance = temp ** 0.5
        return distance

    def kernel_LLE_linear_reconstruction(self):
        self.w_linearReconstruction = np.zeros((self.n_training_images, self.n_neighbors))
        for image_index in range(self.n_training_images):
            ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
            K = np.zeros((self.n_neighbors, self.n_neighbors))
            for row_index in range(self.n_neighbors):
                neighbor_index_1 = self.neighbor_indices[image_index, row_index].astype(int)
                for column_index in range(self.n_neighbors):
                    neighbor_index_2 = self.neighbor_indices[image_index, column_index].astype(int)
                    a = self.kernel_of_images[image_index, image_index]
                    b = self.kernel_of_images[image_index, neighbor_index_1]
                    c = self.kernel_of_images[image_index, neighbor_index_2]
                    d = self.kernel_of_images[neighbor_index_1, neighbor_index_2]
                    K[row_index, column_index] = a - b - c + d
            epsilon = 0.0000001
            K = K + (epsilon * np.eye(self.n_neighbors))
            numinator = (LA.inv(K)).dot(ones_vector)
            denominator = (ones_vector.T).dot(LA.inv(K)).dot(ones_vector)
            self.w_linearReconstruction[image_index, :] = ((1 / denominator) * numinator).ravel()

    def kernel_LLE_linear_reconstruction_outOfSample(self):
        self.w_linearReconstruction_outOfSample = np.zeros((self.n_testing_images, self.n_neighbors))
        for image_index in range(self.n_testing_images):
            ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
            K = np.zeros((self.n_neighbors, self.n_neighbors))
            for row_index in range(self.n_neighbors):
                neighbor_index_1 = self.neighbor_indices_for_outOfSample[image_index, row_index].astype(int)
                for column_index in range(self.n_neighbors):
                    neighbor_index_2 = self.neighbor_indices_for_outOfSample[image_index, column_index].astype(int)
                    # size of self.kernel_of_images_for_outOfSample was: (self.n_training_images+1, self.n_training_images+1, self.n_testing_images)
                    a = self.kernel_of_images_for_outOfSample[self.n_training_images, self.n_training_images, image_index]
                    b = self.kernel_of_images_for_outOfSample[neighbor_index_1, self.n_training_images, image_index]
                    c = self.kernel_of_images_for_outOfSample[neighbor_index_2, self.n_training_images, image_index]
                    d = self.kernel_of_images_for_outOfSample[neighbor_index_1, neighbor_index_2, image_index]
                    K[row_index, column_index] = a - b - c + d
            epsilon = 0.0000001
            K = K + (epsilon * np.eye(self.n_neighbors))
            numinator = (LA.inv(K)).dot(ones_vector)
            denominator = (ones_vector.T).dot(LA.inv(K)).dot(ones_vector)
            self.w_linearReconstruction_outOfSample[image_index, :] = ((1 / denominator) * numinator).ravel()

    def LLE_find_KNN(self, calculate_again=True):
        if calculate_again:
            self.neighbor_indices = np.zeros((self.n_training_images, self.n_neighbors))
            # --- KNN:
            connectivity_matrix = KNN(X=(self.X).T, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
            connectivity_matrix = connectivity_matrix.toarray()
            # --- store indices of neighbors:
            for image_index in range(self.n_training_images):
                self.neighbor_indices[image_index, :] = np.argwhere(connectivity_matrix[image_index, :] == 1).ravel()
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices, name_of_variable="neighbor_indices", path_to_save="./LLE_settings/LLE/")
        else:
            self.neighbor_indices = self.load_variable(name_of_variable="neighbor_indices", path="./LLE_settings/LLE/")

    def LLE_linear_reconstruction(self):
        self.w_linearReconstruction = np.zeros((self.n_training_images, self.n_neighbors))
        for image_index in range(self.n_training_images):
            neighbor_indices_of_this_image = self.neighbor_indices[image_index, :].astype(int)
            X_neighbors = self.X[:, neighbor_indices_of_this_image]
            image_vector = self.X[:, image_index].reshape((-1, 1))
            ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
            G = ((image_vector.dot(ones_vector.T) - X_neighbors).T).dot(image_vector.dot(ones_vector.T) - X_neighbors)
            epsilon = 0.0000001
            G = G + (epsilon * np.eye(self.n_neighbors))
            numinator = (LA.inv(G)).dot(ones_vector)
            denominator = (ones_vector.T).dot(LA.inv(G)).dot(ones_vector)
            self.w_linearReconstruction[image_index, :] = ((1 / denominator) * numinator).ravel()

    def LLE_linear_embedding(self):
        self.W_linearEmbedding = np.zeros((self.n_training_images, self.n_training_images))
        for image_index in range(self.n_training_images):
            neighbor_indices_of_this_image = self.neighbor_indices[image_index, :].astype(int)
            self.W_linearEmbedding[neighbor_indices_of_this_image, image_index] = self.w_linearReconstruction[image_index, :].ravel()
        temp = np.eye(self.n_training_images) - self.W_linearEmbedding
        M = (temp.T).dot(temp)
        eig_val, eig_vec = LA.eigh(M)
        idx = eig_val.argsort()  # sort eigenvalues in ascending order (smallest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            X_transformed = eig_vec[:, 1:self.n_components+1] #--> note that first eigenvalue is zero
        else:
            X_transformed = eig_vec[:, 1:] #--> note that first eigenvalue is zero
        X_transformed = X_transformed.T  #--> the obtained Y in Laplacian eigenmap is row-wise vectors, so we transpose it
        return X_transformed

    def classify_distortion_trainingSet(self, X_transformed, method, classify_again=True):
        if classify_again:
            class_of_distortion = np.zeros((self.n_training_images, 1))
            # --- KNN:
            connectivity_matrix = KNN(X=X_transformed.T, n_neighbors=1, mode='connectivity', include_self=False, n_jobs=-1)
            connectivity_matrix = connectivity_matrix.toarray()
            for image_index in range(self.n_training_images):
                index_of_neighbor = int(np.argwhere(connectivity_matrix[image_index, :] == 1))
                if index_of_neighbor == 0:  # "original"
                    class_of_distortion[image_index, 0] = 0
                elif index_of_neighbor >= 1 and index_of_neighbor <= 20:  # "contrast_stretched"
                    class_of_distortion[image_index, 0] = 1
                elif index_of_neighbor >= 21 and index_of_neighbor <= 40:  # "Gaussian_noise"
                    class_of_distortion[image_index, 0] = 2
                elif index_of_neighbor >= 41 and index_of_neighbor <= 60:  # "enhanced_luminance"
                    class_of_distortion[image_index, 0] = 3
                elif index_of_neighbor >= 61 and index_of_neighbor <= 80:  # "Gaussian_blurring"
                    class_of_distortion[image_index, 0] = 4
                elif index_of_neighbor >= 81 and index_of_neighbor <= 100:  # "impulse_noise"
                    class_of_distortion[image_index, 0] = 5
                elif index_of_neighbor >= 101 and index_of_neighbor <= 120:  # "jpeg_distortion"
                    class_of_distortion[image_index, 0] = 6
            # --- percentage of distortions:
            estimated_distortion_class = np.zeros((self.n_training_images, 2))
            true_distortion_class = np.zeros((self.n_training_images, 2))
            for image_index in range(self.n_training_images):
                # --- class of distortions:
                estimated_distortion_class[image_index, 0] = class_of_distortion[image_index, 0]
                estimated_distortion_class[image_index, 1] = image_index
                # --- true class of distortions:
                if image_index == 0:  # "original"
                    true_distortion_class[image_index, 0] = 0
                elif image_index >= 1 and image_index <= 20:  # "contrast_stretched"
                    true_distortion_class[image_index, 0] = 1
                elif image_index >= 21 and image_index <= 40:  # "Gaussian_noise"
                    true_distortion_class[image_index, 0] = 2
                elif image_index >= 41 and image_index <= 60:  # "enhanced_luminance"
                    true_distortion_class[image_index, 0] = 3
                elif image_index >= 61 and image_index <= 80:  # "Gaussian_blurring"
                    true_distortion_class[image_index, 0] = 4
                elif image_index >= 81 and image_index <= 100:  # "impulse_noise"
                    true_distortion_class[image_index, 0] = 5
                elif image_index >= 101 and image_index <= 120:  # "jpeg_distortion"
                    true_distortion_class[image_index, 0] = 6
                true_distortion_class[image_index, 1] = image_index
            # save:
            if method == "LLE":
                path_to_save = './output/LLE/classification/'
            elif method == "kernel_LLE":
                path_to_save = './output/kernel_LLE/' + self.kernel + '/classification/'
            self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_variable(variable=true_distortion_class, name_of_variable="true_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=true_distortion_class, name_of_variable="true_distortion_class", path_to_save=path_to_save)
        else:
            if method == "LLE":
                path_to_save = './output/LLE/classification/'
            elif method == "kernel_LLE":
                path_to_save = './output/kernel_LLE/' + self.kernel + '/classification/'
            estimated_distortion_class = self.load_variable(name_of_variable="estimated_distortion_class", path=path_to_save)
            true_distortion_class = self.load_variable(name_of_variable="true_distortion_class", path=path_to_save)
        return estimated_distortion_class[:, 0]

    def get_kernels_for_outOfSample(self, data_outOfSample):
        kernel_of_images_for_outOfSample = np.zeros((self.n_training_images+1, self.n_training_images+1, self.n_testing_images))
        for testing_image_index in range(self.n_testing_images):
            image_test = (data_outOfSample[:, testing_image_index]).reshape((-1, 1))
            trainX_and_the_test = np.hstack((self.X, image_test))
            temp = pairwise_kernels(X=trainX_and_the_test.T, Y=trainX_and_the_test.T, metric=self.kernel)
            kernel_train_with_testImage = self.normalize_the_kernel(kernel_matrix=temp)
            kernel_train_with_testImage = self.center_the_matrix(the_matrix=kernel_train_with_testImage, mode="double_center")
            kernel_of_images_for_outOfSample[:, :, testing_image_index] = kernel_train_with_testImage
        return kernel_of_images_for_outOfSample

    def kernel_LLE_find_KNN_for_outOfSample(self, data_outOfSample, calculate_again=True):
        # data_outOfSample --> rows: features, columns: samples
        self.n_testing_images = data_outOfSample.shape[1]
        self.kernel_of_images = self.get_kernels()
        self.kernel_of_images_for_outOfSample = self.get_kernels_for_outOfSample(data_outOfSample)
        if calculate_again:
            self.neighbor_indices_for_outOfSample = np.zeros((self.n_testing_images, self.n_neighbors))
            # --- KNN:
            for image_index_test in range(self.n_testing_images):
                distances_from_this_outOfSample_image = np.zeros((1, self.n_training_images))
                for image_index_train in range(self.n_training_images):
                    distances_from_this_outOfSample_image[0, image_index_train] = self.distance_based_on_kernel(kernel_matrix=self.kernel_of_images_for_outOfSample[:, :, image_index_test], index1=image_index_train, index2=self.n_training_images)
                argsort_distances = np.argsort(distances_from_this_outOfSample_image.ravel())  # arg of ascending sort
                indices_of_neighbors_of_this_outOfSample_image = argsort_distances[:self.n_neighbors]
                self.neighbor_indices_for_outOfSample[image_index_test, :] = indices_of_neighbors_of_this_outOfSample_image
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices_for_outOfSample, name_of_variable="neighbor_indices_for_outOfSample", path_to_save="./LLE_settings/kernel_LLE/"+self.kernel+"/")
        else:
            self.neighbor_indices_for_outOfSample = self.load_variable(name_of_variable="neighbor_indices_for_outOfSample", path="./LLE_settings/kernel_LLE/"+self.kernel+"/")

    def LLE_find_KNN_for_outOfSample(self, data_outOfSample, calculate_again=True):
        # data_outOfSample --> rows: features, columns: samples
        self.n_testing_images = data_outOfSample.shape[1]
        if calculate_again:
            self.neighbor_indices_for_outOfSample = np.zeros((self.n_testing_images, self.n_neighbors))
            # --- KNN:
            for image_index_1 in range(self.n_testing_images):
                test_image = data_outOfSample[:, image_index_1].reshape((-1, 1))
                distances_from_this_outOfSample_image = np.zeros((1, self.n_training_images))
                for image_index_2 in range(self.n_training_images):
                    training_image = self.X[:, image_index_2].reshape((-1, 1))
                    distances_from_this_outOfSample_image[0, image_index_2] = LA.norm(test_image - training_image)
                argsort_distances = np.argsort(distances_from_this_outOfSample_image.ravel())  # arg of ascending sort
                indices_of_neighbors_of_this_outOfSample_image = argsort_distances[:self.n_neighbors]
                self.neighbor_indices_for_outOfSample[image_index_1, :] = indices_of_neighbors_of_this_outOfSample_image
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices_for_outOfSample, name_of_variable="neighbor_indices_for_outOfSample", path_to_save="./LLE_settings/LLE/")
        else:
            self.neighbor_indices_for_outOfSample = self.load_variable(name_of_variable="neighbor_indices_for_outOfSample", path="./LLE_settings/LLE/")

    def LLE_linear_reconstruction_outOfSample(self, data_outOfSample):
        self.w_linearReconstruction_outOfSample = np.zeros((self.n_testing_images, self.n_neighbors))
        for image_index in range(self.n_testing_images):
            neighbor_indices_of_this_image = self.neighbor_indices_for_outOfSample[image_index, :].astype(int)
            X_neighbors = self.X[:, neighbor_indices_of_this_image]
            image_vector = data_outOfSample[:, image_index].reshape((-1, 1))
            ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
            G = ((image_vector.dot(ones_vector.T) - X_neighbors).T).dot(image_vector.dot(ones_vector.T) - X_neighbors)
            epsilon = 0.0000001
            G = G + (epsilon * np.eye(self.n_neighbors))
            numinator = (LA.inv(G)).dot(ones_vector)
            denominator = (ones_vector.T).dot(LA.inv(G)).dot(ones_vector)
            self.w_linearReconstruction_outOfSample[image_index, :] = ((1 / denominator) * numinator).ravel()

    def LLE_linear_embedding_outOfSample(self, X_training_transformed):
        Y_training = X_training_transformed.T   #--> Y_training: rows are samples and columns are features
        Y_outOfSample = np.zeros((self.n_testing_images, self.n_components))
        for outOfSample_image_index in range(self.n_testing_images):
            training_neighbor_indices_of_this_block = self.neighbor_indices_for_outOfSample[outOfSample_image_index, :].astype(int)
            Y_training_neighbors = Y_training[training_neighbor_indices_of_this_block, :].reshape((self.n_neighbors, self.n_components))
            w = self.w_linearReconstruction_outOfSample[outOfSample_image_index, :].ravel()
            summation = np.zeros((self.n_components, 1))
            for training_neighbor_image_index in range(self.n_neighbors):
                Y_neighbor = Y_training_neighbors[training_neighbor_image_index, :].ravel()
                summation = summation + (w[training_neighbor_image_index] * Y_neighbor).reshape((-1, 1))
            Y_outOfSample[outOfSample_image_index, :] = summation.ravel()
        data_outOfSample_transformed = Y_outOfSample.T
        return data_outOfSample_transformed

    def classify_distortion_testSet(self, data_outOfSample_transformed, X_training_transformed, method, classify_again=True, k=1):
        if classify_again:
            Y_training = X_training_transformed.T  # --> Y_training: rows are samples and columns are features
            Y_testing = data_outOfSample_transformed.T  # --> Y_testing: rows are samples and columns are features
            class_of_distortion = np.zeros((self.n_testing_images, 1))
            # --- KNN:
            X_train_and_the_test_image = np.hstack((Y_testing.T, Y_training.T))
            connectivity_matrix = KNN(X=X_train_and_the_test_image.T, n_neighbors=X_train_and_the_test_image.shape[1]-1, mode='distance', include_self=False, n_jobs=-1)
            connectivity_matrix = connectivity_matrix.toarray()
            for image_index in range(self.n_testing_images):
                a = connectivity_matrix[image_index, self.n_testing_images:]
                if k == 1:
                    index_of_neighbor = int(np.argmin(a))
                    if index_of_neighbor == 0:  # "original"
                        class_of_distortion[image_index, 0] = 0
                    elif index_of_neighbor >= 1 and index_of_neighbor <= 20:  # "contrast_stretched"
                        class_of_distortion[image_index, 0] = 1
                    elif index_of_neighbor >= 21 and index_of_neighbor <= 40:  # "Gaussian_noise"
                        class_of_distortion[image_index, 0] = 2
                    elif index_of_neighbor >= 41 and index_of_neighbor <= 60:  # "enhanced_luminance"
                        class_of_distortion[image_index, 0] = 3
                    elif index_of_neighbor >= 61 and index_of_neighbor <= 80:  # "Gaussian_blurring"
                        class_of_distortion[image_index, 0] = 4
                    elif index_of_neighbor >= 81 and index_of_neighbor <= 100:  # "impulse_noise"
                        class_of_distortion[image_index, 0] = 5
                    elif index_of_neighbor >= 101 and index_of_neighbor <= 120:  # "jpeg_distortion"
                        class_of_distortion[image_index, 0] = 6
                else:
                    indices_of_neighbor = np.argsort(a)[:k].ravel()  # first k sorted elements of a in ascending order
                    neighbor_distortion_count = [0] * 7
                    for i in range(k):
                        index_of_neighbor = indices_of_neighbor[i]
                        if index_of_neighbor == 0:  # "original"
                            neighbor_distortion_count[0] = neighbor_distortion_count[0] + 1
                        elif index_of_neighbor >= 1 and index_of_neighbor <= 20:  # "contrast_stretched"
                            neighbor_distortion_count[1] = neighbor_distortion_count[1] + 1
                        elif index_of_neighbor >= 21 and index_of_neighbor <= 40:  # "Gaussian_noise"
                            neighbor_distortion_count[2] = neighbor_distortion_count[2] + 1
                        elif index_of_neighbor >= 41 and index_of_neighbor <= 60:  # "enhanced_luminance"
                            neighbor_distortion_count[3] = neighbor_distortion_count[3] + 1
                        elif index_of_neighbor >= 61 and index_of_neighbor <= 80:  # "Gaussian_blurring"
                            neighbor_distortion_count[4] = neighbor_distortion_count[4] + 1
                        elif index_of_neighbor >= 81 and index_of_neighbor <= 100:  # "impulse_noise"
                            neighbor_distortion_count[5] = neighbor_distortion_count[5] + 1
                        elif index_of_neighbor >= 101 and index_of_neighbor <= 120:  # "jpeg_distortion"
                            neighbor_distortion_count[6] = neighbor_distortion_count[6] + 1
                    class_of_distortion[image_index, 0] = np.argmax(neighbor_distortion_count)
            estimated_distortion_class = np.zeros((self.n_testing_images, 2))
            for image_index in range(self.n_testing_images):
                # --- class of distortions:
                estimated_distortion_class[image_index, 0] = class_of_distortion[image_index, 0]
                estimated_distortion_class[image_index, 1] = image_index
            # save:
            if method == "LLE":
                path_to_save = './output/LLE/classification_test/'
            elif method == "kernel_LLE":
                path_to_save = './output/kernel_LLE/' + self.kernel + '/classification_test/'
            self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
        else:
            if method == "LLE":
                path_to_save = './output/LLE/classification_test/'
            elif method == "kernel_LLE":
                path_to_save = './output/kernel_LLE/' + self.kernel + '/classification_test/'
            estimated_distortion_class = self.load_variable(name_of_variable="estimated_distortion_class", path=path_to_save)
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