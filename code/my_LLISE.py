import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
import pickle
from scipy.signal import medfilt2d
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric


class My_LLISE:

    def __init__(self, X, image_height, image_width, n_neighbors=10, n_components=None, block_height=8, block_width=8, kernel="linear"):
        # X: rows are features and columns are samples
        # pixel intensity range: [0,1]
        self.X = X
        self.image_height = image_height
        self.image_width = image_width
        self.block_height = block_height
        self.block_width = block_width
        self.n_training_images = self.X.shape[1]  #--> n
        self.image_dimension = self.X.shape[0]  #--> d
        self.block_dimension = block_height * block_width  #--> q
        self.n_blocks = int(np.ceil(self.image_dimension / self.block_dimension))  #--> b
        self.blocks = None  #--> shape: (self.block_dimension, self.n_blocks, self.n_training_images)
        self.kernel = kernel
        self.kernel_of_blocks = None  #--> shape: (self.n_training_images, self.n_training_images, self.n_blocks)
        self.n_neighbors = n_neighbors
        self.neighbor_indices = None  #--> shape: (self.n_training_images, self.n_blocks, self.n_neighbors)
        if n_components != None:
            self.n_components = n_components  # --> p
        else:
            self.n_components = self.block_dimension
        self.blocks_meanRemoved = self.divide_images_to_their_blocks(X=self.X, remove_mean=True) #--> shape: (self.block_dimension, self.n_blocks, n_images)
        self.weights_linearReconstruction = None  #--> shape: (self.n_training_images, self.n_blocks, self.n_neighbors)
        self.n_testing_images = None
        self.neighbor_indices_for_outOfSample = None  # --> shape: (self.n_testing_images, self.n_blocks, self.n_neighbors)
        self.outOfSample_weights_linearReconstruction = None  # --> shape: (self.n_testing_images, self.n_blocks, self.n_neighbors)
        self.kernel_of_blocks_for_outOfSample = None  #--> shape: (self.n_training_images+1, self.n_training_images+1, self.n_testing_images, self.n_blocks)

    def LLISE_find_KNN(self, calculate_again=True):
        if calculate_again:
            self.neighbor_indices = np.zeros((self.n_training_images, self.n_blocks, self.n_neighbors))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " out of " + str(self.n_blocks) + " blocks...")
                the_block_amongst_images = self.blocks_meanRemoved[:, block_index, :].reshape((self.block_dimension, self.n_training_images))
                # --- KNN:
                # connectivity_matrix = KNN2(X=the_block_amongst_images.T, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
                connectivity_matrix = KNN(X=the_block_amongst_images.T, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
                connectivity_matrix = connectivity_matrix.toarray()
                # --- store indices of neighbors:
                for image_index in range(self.n_training_images):
                    self.neighbor_indices[image_index, block_index, :] = np.argwhere(connectivity_matrix[image_index, :] == 1).ravel()
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices, name_of_variable="neighbor_indices", path_to_save="./LLISE_settings/LLISE/")
        else:
            self.neighbor_indices = self.load_variable(name_of_variable="neighbor_indices", path="./LLISE_settings/LLISE/")

    def LLISE_find_KNN_for_outOfSample(self, data_outOfSample, calculate_again=True):
        # data_outOfSample --> rows: features, columns: samples
        self.n_testing_images = data_outOfSample.shape[1]
        self.blocks_meanRemoved_outOfSample = self.divide_images_to_their_blocks(X=data_outOfSample, remove_mean=True) #--> shape: (self.block_dimension, self.n_blocks, n_test_images)
        if calculate_again:
            self.neighbor_indices_for_outOfSample = np.zeros((self.n_testing_images, self.n_blocks, self.n_neighbors))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " out of " + str(self.n_blocks) + " blocks...")
                # --- KNN:
                for image_index_1 in range(self.n_testing_images):
                    block_outOfSample = self.blocks_meanRemoved_outOfSample[:, block_index, image_index_1].reshape((-1, 1))
                    distances_from_this_outOfSample_image = np.zeros((1, self.n_training_images))
                    for image_index_2 in range(self.n_training_images):
                        block_training = self.blocks_meanRemoved[:, block_index, image_index_2].reshape((-1, 1))
                        distances_from_this_outOfSample_image[0, image_index_2] = LA.norm(block_outOfSample - block_training)
                    argsort_distances = np.argsort(distances_from_this_outOfSample_image.ravel())  # arg of ascending sort
                    indices_of_neighbors_of_this_outOfSample_image = argsort_distances[:self.n_neighbors]
                    self.neighbor_indices_for_outOfSample[image_index_1, block_index, :] = indices_of_neighbors_of_this_outOfSample_image
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices_for_outOfSample, name_of_variable="neighbor_indices_for_outOfSample", path_to_save="./LLISE_settings/LLISE/")
        else:
            self.neighbor_indices_for_outOfSample = self.load_variable(name_of_variable="neighbor_indices_for_outOfSample", path="./LLISE_settings/LLISE/")

    def get_kernels(self, calculate_kernels_again=True, save_kernels=False):
        if calculate_kernels_again:
            self.blocks = self.divide_images_to_their_blocks(X=self.X)
            X_block_amongImages = np.zeros((self.block_dimension, self.n_training_images))
            kernel_of_blocks = np.zeros((self.n_training_images, self.n_training_images, self.n_blocks))
            for block_index in range(self.n_blocks):
                for image_index in range(self.n_training_images):
                    block = (self.blocks[:, block_index, image_index]).reshape((-1, 1))
                    # block = block - block.mean()  # --> remove mean of block
                    X_block_amongImages[:, image_index] = block.ravel()
                the_kernel = pairwise_kernels(X=X_block_amongImages.T, Y=X_block_amongImages.T, metric=self.kernel)
                kernel_of_blocks[:, :, block_index] = the_kernel
                kernel_of_blocks[:, :, block_index] = self.normalize_the_kernel(kernel_matrix=kernel_of_blocks[:, :, block_index])
                kernel_of_blocks[:, :, block_index] = self.center_the_matrix(the_matrix=kernel_of_blocks[:, :, block_index], mode="double_center")
            if save_kernels:
                self.save_variable(variable=kernel_of_blocks, name_of_variable="kernel_"+self.kernel, path_to_save='./LLISE_settings/kernels/')
        else:
            kernel_of_blocks = self.load_variable(name_of_variable="kernel_"+self.kernel, path='./LLISE_settings/kernels/')
        return kernel_of_blocks

    def normalize_the_kernel(self, kernel_matrix):
        diag_kernel = np.diag(kernel_matrix)
        k = (1 / np.sqrt(diag_kernel)).reshape((-1,1))
        normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
        return normalized_kernel_matrix

    def kernel_LLISE_find_KNN(self, calculate_again=True):
        if calculate_again:
            self.kernel_of_blocks = self.get_kernels(calculate_kernels_again=True, save_kernels=False)
            self.neighbor_indices = np.zeros((self.n_training_images, self.n_blocks, self.n_neighbors))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " out of " + str(self.n_blocks) + " blocks...")
                the_block_amongst_images = self.blocks_meanRemoved[:, block_index, :].reshape((self.block_dimension, self.n_training_images))
                # --- KNN:
                for image_index_1 in range(self.n_training_images):
                    distances_from_this_image = np.zeros((1, self.n_training_images))
                    for image_index_2 in range(self.n_training_images):
                        distances_from_this_image[0, image_index_2] = self.distance_based_on_kernel(kernel_matrix=self.kernel_of_blocks[:, :, block_index], index1=image_index_1, index2=image_index_2)
                    argsort_distances = np.argsort(distances_from_this_image.ravel())  # arg of ascending sort
                    indices_of_neighbors_of_this_image = argsort_distances[:self.n_neighbors]
                    self.neighbor_indices[image_index_1, block_index, :] = indices_of_neighbors_of_this_image
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices, name_of_variable="neighbor_indices", path_to_save="./LLISE_settings/kernel_LLISE/" + self.kernel + "/")
        else:
            self.kernel_of_blocks = self.get_kernels(calculate_kernels_again=True, save_kernels=False)
            self.neighbor_indices = self.load_variable(name_of_variable="neighbor_indices", path="./LLISE_settings/kernel_LLISE/" + self.kernel + "/")

    def distance_based_on_kernel(self, kernel_matrix, index1, index2):
        temp = kernel_matrix[index1, index1] - 2 * kernel_matrix[index1, index2] + kernel_matrix[index2, index2]
        if temp < 0:
            # might occur for imperfect software calculations
            temp = 0
        distance = temp ** 0.5
        return distance

    def kernel_LLISE_find_KNN_for_outOfSample(self, data_outOfSample, calculate_again=True):
        # data_outOfSample --> rows: features, columns: samples
        self.n_testing_images = data_outOfSample.shape[1]
        self.kernel_of_blocks = self.get_kernels(calculate_kernels_again=True, save_kernels=False)
        self.kernel_of_blocks_for_outOfSample = self.get_kernels_for_outOfSample(data_outOfSample, calculate_kernels_again=True, save_kernels=False)
        if calculate_again:
            self.neighbor_indices_for_outOfSample = np.zeros((self.n_testing_images, self.n_blocks, self.n_neighbors))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " out of " + str(self.n_blocks) + " blocks...")
                # --- KNN:
                for image_index_test in range(self.n_testing_images):
                    distances_from_this_outOfSample_image = np.zeros((1, self.n_training_images))
                    for image_index_train in range(self.n_training_images):
                        distances_from_this_outOfSample_image[0, image_index_train] = self.distance_based_on_kernel(kernel_matrix=self.kernel_of_blocks_for_outOfSample[:, :, image_index_test, block_index], index1=image_index_train, index2=self.n_training_images)
                    argsort_distances = np.argsort(distances_from_this_outOfSample_image.ravel())  # arg of ascending sort
                    indices_of_neighbors_of_this_outOfSample_image = argsort_distances[:self.n_neighbors]
                    self.neighbor_indices_for_outOfSample[image_index_test, block_index, :] = indices_of_neighbors_of_this_outOfSample_image
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices_for_outOfSample, name_of_variable="neighbor_indices_for_outOfSample", path_to_save="./LLISE_settings/kernel_LLISE/"+self.kernel+"/")
        else:
            self.neighbor_indices_for_outOfSample = self.load_variable(name_of_variable="neighbor_indices_for_outOfSample", path="./LLISE_settings/kernel_LLISE/"+self.kernel+"/")

    def get_kernels_for_outOfSample(self, data_outOfSample, calculate_kernels_again=True, save_kernels=False):
        if calculate_kernels_again:
            self.blocks = self.divide_images_to_their_blocks(X=self.X, remove_mean=False)
            blocks_of_testing_images = self.divide_images_to_their_blocks(X=data_outOfSample, remove_mean=False)
            X_block_among_training_Images = np.zeros((self.block_dimension, self.n_training_images))
            X_block_among_testing_Images = np.zeros((self.block_dimension, self.n_testing_images))
            kernel_of_blocks_for_outOfSample = np.zeros((self.n_training_images+1, self.n_training_images+1, self.n_testing_images, self.n_blocks))
            for block_index in range(self.n_blocks):
                for training_image_index in range(self.n_training_images):
                    block = (self.blocks[:, block_index, training_image_index]).reshape((-1, 1))
                    # block = block - block.mean()  # --> remove mean of block
                    X_block_among_training_Images[:, training_image_index] = block.ravel()
                for testing_image_index in range(self.n_testing_images):
                    block = (blocks_of_testing_images[:, block_index, testing_image_index]).reshape((-1, 1))
                    # block = block - block.mean()  # --> remove mean of block
                    X_block_among_testing_Images[:, testing_image_index] = block.ravel()
                for testing_image_index in range(self.n_testing_images):
                    block_test = (X_block_among_testing_Images[:, testing_image_index]).reshape((-1, 1))
                    trainX_and_the_test = np.hstack((X_block_among_training_Images, block_test))
                    temp = pairwise_kernels(X=trainX_and_the_test.T, Y=trainX_and_the_test.T, metric=self.kernel)
                    kernel_train_with_testBlock = self.normalize_the_kernel(kernel_matrix=temp)
                    kernel_train_with_testBlock = self.center_the_matrix(the_matrix=kernel_train_with_testBlock, mode="double_center")
                    # kernel_train_with_testBlock = kernel_train_with_testBlock[:-1, -1].reshape((-1, 1))
                    # kernel_of_blocks_for_outOfSample[:, testing_image_index, block_index] = kernel_train_with_testBlock.ravel()
                    kernel_of_blocks_for_outOfSample[:, :, testing_image_index, block_index] = kernel_train_with_testBlock
            if save_kernels:
                self.save_variable(variable=kernel_of_blocks_for_outOfSample, name_of_variable="kernel_outOfSample_"+self.kernel, path_to_save='./LLISE_settings/kernels/')
        else:
            kernel_of_blocks_for_outOfSample = self.load_variable(name_of_variable="kernel_outOfSample_"+self.kernel, path='./LLISE_settings/kernels/')
        return kernel_of_blocks_for_outOfSample

    def OutOfSample_linear_reconstruction_ADMM(self, method="LLISE", calculate_again=True, max_epochs=None, step_checkpoint=10):
        if calculate_again == False:
            if method == "LLISE":
                self.outOfSample_weights_linearReconstruction = self.load_variable(name_of_variable="w_58", path='./LLISE_settings/LLISE/linear_recons_test/w/')
            if method == "kernel_LLISE":
                if self.kernel == "rbf":
                    temp = "w_49"
                if self.kernel == "sigmoid":
                    temp = "w_2"
                if self.kernel == "polynomial":
                    temp = "w_49"
                self.outOfSample_weights_linearReconstruction = self.load_variable(name_of_variable=temp, path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_recons_test/w/')
            return
        self.outOfSample_weights_linearReconstruction = np.zeros((self.n_testing_images, self.n_blocks, self.n_neighbors))
        zeta_linearReconstruction = np.zeros((self.n_testing_images, self.n_blocks, self.n_neighbors))
        j_dual_linearReconstruction = np.zeros((self.n_testing_images, self.n_blocks, self.n_neighbors))
        print("initialize weights...")
        for image_index in range(self.n_testing_images):
            for block_index in range(self.n_blocks):
                random_weights = np.random.rand(self.n_neighbors, 1).ravel()  # --> rand is in range [0,1)
                self.outOfSample_weights_linearReconstruction[image_index, block_index, :] = random_weights / sum(random_weights)
                random_weights = np.random.rand(self.n_neighbors, 1).ravel()  # --> rand is in range [0,1)
                zeta_linearReconstruction[image_index, block_index, :] = random_weights / sum(random_weights)
                random_weights = np.random.rand(self.n_neighbors, 1).ravel()  # --> rand is in range [0,1)
                j_dual_linearReconstruction[image_index, block_index, :] = random_weights / sum(random_weights)
        iteration_index = -1
        changeOfWeight_average_iters = np.zeros((self.n_testing_images, self.n_blocks))
        reconstruction_error_iters = np.zeros((step_checkpoint, 1))
        if method == "LLISE":
            rho = 0.1
            eta = 0.1
        if method == "kernel_LLISE":
            rho = 0.01
            eta = 0.1
        while True:
            iteration_index = iteration_index + 1
            print("----- iteration #" + str(iteration_index))
            reconstruction_error = 0
            changeOfWeight_for_block = 0
            # max_w = -np.inf
            # min_w = np.inf
            # max_zeta = -np.inf
            # min_zeta = np.inf
            for image_index in range(self.n_testing_images):
                # print("image #" + str(image_index))
                for block_index in range(self.n_blocks):
                    w = self.outOfSample_weights_linearReconstruction[image_index, block_index, :].reshape((-1, 1))
                    zeta = zeta_linearReconstruction[image_index, block_index, :].reshape((-1, 1))
                    j_dual = j_dual_linearReconstruction[image_index, block_index, :].reshape((-1, 1))
                    w_previous = w.copy()
                    if method == "LLISE":
                        gradient_f = self.gradient_f_for_outOfSample(outOfSample_image_index=image_index, block_index=block_index, w=w)
                    elif method == "kernel_LLISE":
                        gradient_f = self.gradient_f_kernel_for_outOfSample(outOfSample_image_index=image_index, block_index=block_index, w=w)
                    w = w - eta*gradient_f - eta*(rho * (w - zeta + j_dual))
                    # max_w = max(max_w, np.max(w))
                    # min_w = min(min_w, np.min(w))
                    # zeta = (w + j_dual) / (np.sum(w + j_dual))
                    # zeta = (w + j_dual) / (LA.norm(w + j_dual))
                    zeta = self.constraint_project_1(vector = w + j_dual)
                    # max_zeta = max(max_zeta, np.max(zeta))
                    # min_zeta = min(min_zeta, np.min(zeta))
                    j_dual = j_dual + w - zeta
                    self.outOfSample_weights_linearReconstruction[image_index, block_index, :] = w.ravel()
                    zeta_linearReconstruction[image_index, block_index, :] = zeta.ravel()
                    j_dual_linearReconstruction[image_index, block_index, :] = j_dual.ravel()
                    if method == "LLISE":
                        reconstruction_error = reconstruction_error + self.f_for_outOfSample(outOfSample_image_index=image_index, block_index=block_index, w=w)
                    elif method == "kernel_LLISE":
                        reconstruction_error = reconstruction_error + self.f_kernel_for_outOfSample(outOfSample_image_index=image_index, block_index=block_index, w=w)
                    changeOfWeight_for_block = changeOfWeight_for_block + LA.norm(w - w_previous)
            reconstruction_error = reconstruction_error / (self.n_training_images * self.n_blocks)     #--> taking average
            changeOfWeight_for_block = changeOfWeight_for_block / (self.n_training_images * self.n_blocks)     #--> taking average
            # print(max_w)
            # print(min_w)
            # print(max_zeta)
            # print(min_zeta)
            print("----- average reconstruction error in iteration #" + str(iteration_index) + ": " + str(reconstruction_error))
            print("----- average change of weights in iteration #" + str(iteration_index) + ": " + str(changeOfWeight_for_block))
            index_to_save = iteration_index % step_checkpoint
            reconstruction_error_iters[index_to_save] = reconstruction_error
            changeOfWeight_average_iters[index_to_save] = changeOfWeight_for_block
            # save the information at checkpoints:
            if (iteration_index+1) % step_checkpoint == 0:
                if method == "LLISE":
                    path_to_save = './LLISE_settings/LLISE/linear_recons_test/'
                if method == "kernel_LLISE":
                    path_to_save = './LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_recons_test/'
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / step_checkpoint))
                self.save_variable(variable=reconstruction_error_iters, name_of_variable="reconstruction_error_iters_"+str(checkpoint_index), path_to_save=path_to_save+'reconstruction_error/')
                self.save_np_array_to_txt(variable=reconstruction_error_iters, name_of_variable="reconstruction_error_iters_"+str(checkpoint_index), path_to_save=path_to_save+'reconstruction_error/')
                self.save_variable(variable=changeOfWeight_average_iters, name_of_variable="changeOfWeight_average_iters_"+str(checkpoint_index), path_to_save=path_to_save+'w_error/')
                self.save_np_array_to_txt(variable=changeOfWeight_average_iters, name_of_variable="changeOfWeight_average_iters_"+str(checkpoint_index), path_to_save=path_to_save+'w_error/')
                self.save_variable(variable=self.outOfSample_weights_linearReconstruction, name_of_variable="w_"+str(checkpoint_index), path_to_save=path_to_save+'w/')
                self.save_variable(variable=zeta_linearReconstruction, name_of_variable="zeta_"+str(checkpoint_index), path_to_save=path_to_save+'zeta/')
                self.save_variable(variable=j_dual_linearReconstruction, name_of_variable="j_dual_"+str(checkpoint_index), path_to_save=path_to_save+'j_dual/')
            # termination check:
            if max_epochs != None:
                if iteration_index >= max_epochs:
                    break

    def linear_reconstruction_ADMM(self, method="LLISE", calculate_again=True, max_epochs=None, step_checkpoint=10):
        if calculate_again == False:
            if method == "LLISE":
                self.weights_linearReconstruction = self.load_variable(name_of_variable="w_22", path='./LLISE_settings/LLISE/linear_recons/w/')
            if method == "kernel_LLISE":
                if self.kernel == "rbf":
                    temp = "w_15"
                if self.kernel == "sigmoid":
                    temp = "w_6"
                if self.kernel == "polynomial":
                    temp = "w_14"
                self.weights_linearReconstruction = self.load_variable(name_of_variable=temp, path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_recons/w/')
            return
        self.weights_linearReconstruction = np.zeros((self.n_training_images, self.n_blocks, self.n_neighbors))
        zeta_linearReconstruction = np.zeros((self.n_training_images, self.n_blocks, self.n_neighbors))
        j_dual_linearReconstruction = np.zeros((self.n_training_images, self.n_blocks, self.n_neighbors))
        print("initialize weights...")
        for image_index in range(self.n_training_images):
            for block_index in range(self.n_blocks):
                random_weights = np.random.rand(self.n_neighbors, 1).ravel()  # --> rand is in range [0,1)
                self.weights_linearReconstruction[image_index, block_index, :] = random_weights / sum(random_weights)
                random_weights = np.random.rand(self.n_neighbors, 1).ravel()  # --> rand is in range [0,1)
                zeta_linearReconstruction[image_index, block_index, :] = random_weights / sum(random_weights)
                random_weights = np.random.rand(self.n_neighbors, 1).ravel()  # --> rand is in range [0,1)
                j_dual_linearReconstruction[image_index, block_index, :] = random_weights / sum(random_weights)
        iteration_index = -1
        changeOfWeight_average_iters = np.zeros((self.n_training_images, self.n_blocks))
        reconstruction_error_iters = np.zeros((step_checkpoint, 1))
        if method == "LLISE":
            rho = 0.1
            eta = 0.1
        if method == "kernel_LLISE":
            rho = 0.01
            eta = 0.1
        while True:
            iteration_index = iteration_index + 1
            print("----- iteration #" + str(iteration_index))
            reconstruction_error = 0
            changeOfWeight_for_block = 0
            # max_w = -np.inf
            # min_w = np.inf
            # max_zeta = -np.inf
            # min_zeta = np.inf
            for image_index in range(self.n_training_images):
                # print("image #" + str(image_index))
                for block_index in range(self.n_blocks):
                    w = self.weights_linearReconstruction[image_index, block_index, :].reshape((-1, 1))
                    zeta = zeta_linearReconstruction[image_index, block_index, :].reshape((-1, 1))
                    j_dual = j_dual_linearReconstruction[image_index, block_index, :].reshape((-1, 1))
                    w_previous = w.copy()
                    if method == "LLISE":
                        gradient_f = self.gradient_f(image_index, block_index, w)
                    elif method == "kernel_LLISE":
                        gradient_f = self.gradient_f_kernel(image_index, block_index, w)
                    w = w - eta*gradient_f - eta*(rho * (w - zeta + j_dual))
                    # max_w = max(max_w, np.max(w))
                    # min_w = min(min_w, np.min(w))
                    # zeta = (w + j_dual) / (np.sum(w + j_dual))
                    # zeta = (w + j_dual) / (LA.norm(w + j_dual))
                    zeta = self.constraint_project_1(vector = w + j_dual)
                    # max_zeta = max(max_zeta, np.max(zeta))
                    # min_zeta = min(min_zeta, np.min(zeta))
                    j_dual = j_dual + w - zeta
                    self.weights_linearReconstruction[image_index, block_index, :] = w.ravel()
                    zeta_linearReconstruction[image_index, block_index, :] = zeta.ravel()
                    j_dual_linearReconstruction[image_index, block_index, :] = j_dual.ravel()
                    # reconstruction_error = reconstruction_error + self.f(image_index, block_index, w)
                    if method == "LLISE":
                        reconstruction_error = reconstruction_error + self.f(image_index, block_index, w)
                    elif method == "kernel_LLISE":
                        reconstruction_error = reconstruction_error + self.f_kernel(image_index, block_index, w)
                    changeOfWeight_for_block = changeOfWeight_for_block + LA.norm(w - w_previous)
            reconstruction_error = reconstruction_error / (self.n_training_images * self.n_blocks)     #--> taking average
            changeOfWeight_for_block = changeOfWeight_for_block / (self.n_training_images * self.n_blocks)     #--> taking average
            # print(max_w)
            # print(min_w)
            # print(max_zeta)
            # print(min_zeta)
            print("----- average reconstruction error in iteration #" + str(iteration_index) + ": " + str(reconstruction_error))
            print("----- average change of weights in iteration #" + str(iteration_index) + ": " + str(changeOfWeight_for_block))
            index_to_save = iteration_index % step_checkpoint
            reconstruction_error_iters[index_to_save] = reconstruction_error
            changeOfWeight_average_iters[index_to_save] = changeOfWeight_for_block
            # save the information at checkpoints:
            if (iteration_index+1) % step_checkpoint == 0:
                if method == "LLISE":
                    path_to_save = './LLISE_settings/LLISE/linear_recons/'
                if method == "kernel_LLISE":
                    path_to_save = './LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_recons/'
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / step_checkpoint))
                self.save_variable(variable=reconstruction_error_iters, name_of_variable="reconstruction_error_iters_"+str(checkpoint_index), path_to_save=path_to_save+'reconstruction_error/')
                self.save_np_array_to_txt(variable=reconstruction_error_iters, name_of_variable="reconstruction_error_iters_"+str(checkpoint_index), path_to_save=path_to_save+'reconstruction_error/')
                self.save_variable(variable=changeOfWeight_average_iters, name_of_variable="changeOfWeight_average_iters_"+str(checkpoint_index), path_to_save=path_to_save+'w_error/')
                self.save_np_array_to_txt(variable=changeOfWeight_average_iters, name_of_variable="changeOfWeight_average_iters_"+str(checkpoint_index), path_to_save=path_to_save+'w_error/')
                self.save_variable(variable=self.weights_linearReconstruction, name_of_variable="w_"+str(checkpoint_index), path_to_save=path_to_save+'w/')
                self.save_variable(variable=zeta_linearReconstruction, name_of_variable="zeta_"+str(checkpoint_index), path_to_save=path_to_save+'zeta/')
                self.save_variable(variable=j_dual_linearReconstruction, name_of_variable="j_dual_"+str(checkpoint_index), path_to_save=path_to_save+'j_dual/')
            # termination check:
            if max_epochs != None:
                if iteration_index >= max_epochs:
                    break

    def LLISE_linear_reconstruction_NewtonMethod(self, calculate_again=True, max_epochs=None, step_checkpoint=10):
        if calculate_again == False:
            return
        self.weights_linearReconstruction = np.zeros((self.n_training_images, self.n_blocks, self.n_neighbors))
        print("initialize weights...")
        for image_index in range(self.n_training_images):
            for block_index in range(self.n_blocks):
                random_weights = np.random.rand(self.n_neighbors, 1).ravel()  # --> rand is in range [0,1)
                self.weights_linearReconstruction[image_index, block_index, :] = random_weights / sum(random_weights)
        iteration_index = -1
        changeOfWeight_average_iters = np.zeros((self.n_training_images, self.n_blocks))
        reconstruction_error_iters = np.zeros((step_checkpoint, 1))
        while True:
            iteration_index = iteration_index + 1
            print("----- iteration #" + str(iteration_index))
            reconstruction_error = 0
            changeOfWeight_for_block = 0
            for image_index in range(115,120):# range(self.n_training_images):
                print("image #" + str(image_index))
                for block_index in range(self.n_blocks):
                    w = self.weights_linearReconstruction[image_index, block_index, :].reshape((-1, 1))
                    w_previous = w.copy()
                    gradient_f = self.gradient_f(image_index, block_index, w)
                    # Hessian_f = self.Hessian_f(image_index, block_index, w)
                    Hessian_f = np.eye(self.n_neighbors)
                    # Hessian_f = self.numerical_Hessian_f(image_index, block_index, w)
                    delta_step, _ = self.Newton_systemOfEquations(gradient_f, Hessian_f)
                    w_updated = w.copy() + delta_step
                    self.weights_linearReconstruction[image_index, block_index, :] = w_updated.ravel()
                    reconstruction_error = reconstruction_error + self.f(image_index, block_index, w_updated)
                    changeOfWeight_for_block = changeOfWeight_for_block + LA.norm(w_updated - w_previous)
            reconstruction_error = reconstruction_error #/ (self.n_training_images * self.n_blocks)     #--> taking average
            changeOfWeight_for_block = changeOfWeight_for_block #/ (self.n_training_images * self.n_blocks)     #--> taking average
            print("----- average reconstruction error in iteration #" + str(iteration_index) + ": " + str(reconstruction_error))
            print("----- average change of weights in iteration #" + str(iteration_index) + ": " + str(changeOfWeight_for_block))
            index_to_save = iteration_index % step_checkpoint
            reconstruction_error_iters[index_to_save] = reconstruction_error
            changeOfWeight_average_iters[index_to_save] = changeOfWeight_for_block

    def classify_distortion_testSet(self, X, method, classify_again=True, k=1):
        n_test_images = X.shape[1]
        if classify_again:
            if method == "LLISE":
                Y_outOfSample = self.load_variable(name_of_variable="Y_outOfSample", path='./LLISE_settings/LLISE/linear_embed_test/Y/')
                Y_training = self.load_variable(name_of_variable="Y_iters_5", path='./LLISE_settings/LLISE/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
            elif method == "kernel_LLISE":
                Y_outOfSample = self.load_variable(name_of_variable="Y_outOfSample", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed_test/Y/')
                if self.kernel == "rbf":
                    Y_training = self.load_variable(name_of_variable="Y_iters_10", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
                if self.kernel == "sigmoid":
                    Y_training = self.load_variable(name_of_variable="Y_iters_19", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
                if self.kernel == "polynomial":
                    Y_training = self.load_variable(name_of_variable="Y_iters_26", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
            class_of_distortion = np.zeros((n_test_images, self.n_blocks))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " / " + str(self.n_blocks-1) + "...")
                #--- test (out of sample) images:
                test_X_projected_for_a_block = np.zeros((self.n_components, n_test_images))
                for image_index in range(n_test_images):
                    if method == "LLISE":
                        test_X_projected_for_a_block[:, image_index] = Y_outOfSample[image_index, :, block_index].ravel()
                    elif method == "kernel_LLISE":
                        test_X_projected_for_a_block[:, image_index] = Y_outOfSample[image_index, :, block_index].ravel()
                #--- training images:
                train_X_projected_for_a_block = np.zeros((self.n_components, self.n_training_images))
                for image_index in range(self.n_training_images):
                    if method == "LLISE":
                        train_X_projected_for_a_block[:, image_index] = Y_training[image_index, :, block_index].ravel()
                    elif method == "kernel_LLISE":
                        train_X_projected_for_a_block[:, image_index] = Y_training[image_index, :, block_index].ravel()
                # --- KNN:
                X_train_and_the_test_image = np.hstack((test_X_projected_for_a_block, train_X_projected_for_a_block))
                connectivity_matrix = KNN(X=X_train_and_the_test_image.T, n_neighbors=X_train_and_the_test_image.shape[1]-1, mode='distance', include_self=False, n_jobs=-1)
                connectivity_matrix = connectivity_matrix.toarray()
                for image_index in range(n_test_images):
                    a = connectivity_matrix[image_index, n_test_images:]
                    if k == 1:
                        index_of_neighbor = int(np.argmin(a))
                        if index_of_neighbor == 0:  # "original"
                            class_of_distortion[image_index, block_index] = 0
                        elif index_of_neighbor >= 1 and index_of_neighbor <= 20:  # "contrast_stretched"
                            class_of_distortion[image_index, block_index] = 1
                        elif index_of_neighbor >= 21 and index_of_neighbor <= 40:  # "Gaussian_noise"
                            class_of_distortion[image_index, block_index] = 2
                        elif index_of_neighbor >= 41 and index_of_neighbor <= 60:  # "enhanced_luminance"
                            class_of_distortion[image_index, block_index] = 3
                        elif index_of_neighbor >= 61 and index_of_neighbor <= 80:  # "Gaussian_blurring"
                            class_of_distortion[image_index, block_index] = 4
                        elif index_of_neighbor >= 81 and index_of_neighbor <= 100:  # "impulse_noise"
                            class_of_distortion[image_index, block_index] = 5
                        elif index_of_neighbor >= 101 and index_of_neighbor <= 120:  # "jpeg_distortion"
                            class_of_distortion[image_index, block_index] = 6
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
                        class_of_distortion[image_index, block_index] = np.argmax(neighbor_distortion_count)
            # --- percentage of distortions:
            votes = np.zeros((n_test_images, 7+1))
            estimated_distortion_class = np.zeros((n_test_images, 2))
            for image_index in range(n_test_images):
                print("processing image #" + str(image_index) + " / " + str(n_test_images-1) + "...")
                votes[image_index, 0] = (sum(class_of_distortion[image_index, :] == 0) / self.n_blocks) * 100
                votes[image_index, 1] = (sum(class_of_distortion[image_index, :] == 1) / self.n_blocks) * 100
                votes[image_index, 2] = (sum(class_of_distortion[image_index, :] == 2) / self.n_blocks) * 100
                votes[image_index, 3] = (sum(class_of_distortion[image_index, :] == 3) / self.n_blocks) * 100
                votes[image_index, 4] = (sum(class_of_distortion[image_index, :] == 4) / self.n_blocks) * 100
                votes[image_index, 5] = (sum(class_of_distortion[image_index, :] == 5) / self.n_blocks) * 100
                votes[image_index, 6] = (sum(class_of_distortion[image_index, :] == 6) / self.n_blocks) * 100
                votes[image_index, 7] = image_index
                # --- class of distortions:
                estimated_distortion_class[image_index, 0] = np.argmax(votes[image_index, :-1].ravel())
                estimated_distortion_class[image_index, 1] = image_index
            # save:
            if method == "LLISE":
                path_to_save = './output/LLISE/classification_test/'
            elif method == "kernel_LLISE":
                path_to_save = './output/kernel_LLISE/' + self.kernel + '/classification_test/'
            self.save_variable(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
        else:
            if method == "LLISE":
                path_to_save = './output/LLISE/classification_test/'
            elif method == "kernel_LLISE":
                path_to_save = './output/kernel_LLISE/' + self.kernel + '/classification_test/'
            votes = self.load_variable(name_of_variable="votes", path=path_to_save)
            estimated_distortion_class = self.load_variable(name_of_variable="estimated_distortion_class", path=path_to_save)
        return estimated_distortion_class[:, 0]

    def OutOfSample_linear_embedding_ADMM(self, method="LLISE", classify_again=True):
        if classify_again == False:
            return
        if method == "LLISE":
            Y_training = self.load_variable(name_of_variable="Y_iters_5", path='./LLISE_settings/LLISE/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
        if method == "kernel_LLISE":
            if self.kernel == "rbf":
                Y_training = self.load_variable(name_of_variable="Y_iters_10", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
            if self.kernel == "sigmoid":
                Y_training = self.load_variable(name_of_variable="Y_iters_19", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
            if self.kernel == "polynomial":
                Y_training = self.load_variable(name_of_variable="Y_iters_26", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')  #--> similar index with the "classify_distortion_trainingSet" function
        Y_outOfSample = np.zeros((self.n_testing_images, self.n_components, self.n_blocks))
        for outOfSample_image_index in range(self.n_testing_images):
            print("embedding the blocks in the out-of-sample image #" + str(outOfSample_image_index))
            for block_index in range(self.n_blocks):
                training_neighbor_indices_of_this_block = self.neighbor_indices_for_outOfSample[outOfSample_image_index, block_index, :].astype(int)
                Y_training_neighbors = Y_training[training_neighbor_indices_of_this_block, :, block_index].reshape((self.n_neighbors, self.n_components))
                w = self.outOfSample_weights_linearReconstruction[outOfSample_image_index, block_index, :].ravel()
                summation = np.zeros((self.n_components, 1))
                for training_neighbor_image_index in range(self.n_neighbors):
                    Y_neighbor = Y_training_neighbors[training_neighbor_image_index, :].ravel()
                    summation = summation + (w[training_neighbor_image_index] * Y_neighbor).reshape((-1, 1))
                Y_outOfSample[outOfSample_image_index, :, block_index] = summation.ravel()
        if method == "LLISE":
            path_to_save = './LLISE_settings/LLISE/linear_embed_test/'
        if method == "kernel_LLISE":
            path_to_save = './LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed_test/'
        self.save_variable(variable=Y_outOfSample, name_of_variable="Y_outOfSample", path_to_save=path_to_save+'Y/')

    def linear_embedding_ADMM(self, method="LLISE", calculate_again=True, max_epochs=None, step_checkpoint=10):
        if calculate_again == False:
            return
        Y = np.random.rand(self.n_training_images, self.n_components, self.n_blocks)  # --> rand in [0,1)
        V = np.random.rand(self.n_training_images, self.n_components, self.n_blocks)  # --> rand in [0,1)
        J = np.random.rand(self.n_training_images, self.n_components, self.n_blocks)  # --> rand in [0,1)
        if method == "LLISE":
            path_to_save = './LLISE_settings/LLISE/linear_embed/'
        if method == "kernel_LLISE":
            path_to_save = './LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/'
        self.save_variable(variable=Y, name_of_variable="Y_initial", path_to_save=path_to_save)
        self.save_variable(variable=V, name_of_variable="V_initial", path_to_save=path_to_save)
        self.save_variable(variable=J, name_of_variable="J_initial", path_to_save=path_to_save)
        rho = 0.01
        eta = 0.01
        iteration_index = -1
        change_of_Y = np.zeros((self.n_blocks, 1))
        reconstruction_error_iters = np.zeros((step_checkpoint, 1))
        average_change_of_Y_iters = np.zeros((step_checkpoint, 1))
        while True:
            iteration_index = iteration_index + 1
            print("----- iteration #" + str(iteration_index))
            Y_previous_iteration = Y.copy()
            reconstruction_error_blocks = np.zeros((self.n_blocks, 1))
            for block_index in range(self.n_blocks):
                # print("processing block " + str(block_index) + "/" + str(self.n_blocks-1) + " in iteration #" + str(iteration_index))
                sum_gradient_of_f = np.zeros((self.n_training_images, self.n_components))
                for image_index in range(self.n_training_images):  # --> iterating on images
                    gradient_of_f = self.gradient_theta(image_index=image_index, block_index=block_index, Y=Y[:, :, block_index])
                    sum_gradient_of_f = sum_gradient_of_f + gradient_of_f
                Y[:, :, block_index] = Y[:, :, block_index] - eta*sum_gradient_of_f - eta*(rho * (Y[:, :, block_index] - V[:, :, block_index] + J[:, :, block_index]))
                V[:, :, block_index] = self.constraint_project_2(matrix = Y[:, :, block_index] + J[:, :, block_index])
                J[:, :, block_index] = J[:, :, block_index] + Y[:, :, block_index] - V[:, :, block_index]
                for image_index in range(self.n_training_images):  # --> iterating on images
                    reconstruction_error_blocks[block_index] = reconstruction_error_blocks[block_index] + self.theta(image_index=image_index, block_index=block_index, Y=Y[:, :, block_index])
            # epoch error 1:
            reconstruction_error_blocks_meanOfImages = reconstruction_error_blocks / self.n_training_images
            reconstruction_error = reconstruction_error_blocks_meanOfImages.mean()  #--> reconstruction_error_meanOfBlocks_meanOfImages
            index_to_save = iteration_index % step_checkpoint
            reconstruction_error_iters[index_to_save] = reconstruction_error
            print("----- average reconstruction error of iteration #" + str(iteration_index) + ": " + str(reconstruction_error))
            # epoch error 2:
            for block_index in range(self.n_blocks):
                change_of_Y[block_index] = LA.norm(Y[:, :, block_index] - Y_previous_iteration[:, :, block_index], ord="fro")
            average_change_of_Y_iters[index_to_save] = change_of_Y.mean()
            print("----- average change of Y in iteration #" + str(iteration_index) + ": " + str(average_change_of_Y_iters[index_to_save]))
            # save the information at checkpoints:
            if (iteration_index+1) % step_checkpoint == 0:
                if method == "LLISE":
                    path_to_save = './LLISE_settings/LLISE/linear_embed/'
                if method == "kernel_LLISE":
                    path_to_save = './LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/'
                print("Saving the checkpoint in epoch #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / step_checkpoint))
                self.save_variable(variable=reconstruction_error_iters, name_of_variable="reconstruction_error_iters_"+str(checkpoint_index), path_to_save=path_to_save+'recons_error/')
                self.save_np_array_to_txt(variable=reconstruction_error_iters, name_of_variable="reconstruction_error_iters_"+str(checkpoint_index), path_to_save=path_to_save+'recons_error/')
                self.save_variable(variable=average_change_of_Y_iters, name_of_variable="average_change_of_Y_iters_"+str(checkpoint_index), path_to_save=path_to_save+'Y_change/')
                self.save_np_array_to_txt(variable=average_change_of_Y_iters, name_of_variable="average_change_of_Y_iters_"+str(checkpoint_index), path_to_save=path_to_save+'Y_change/')
                self.save_variable(variable=Y, name_of_variable="Y_iters_"+str(checkpoint_index), path_to_save=path_to_save+'Y/')
                self.save_variable(variable=V, name_of_variable="V_iters_"+str(checkpoint_index), path_to_save=path_to_save+'V/')
                self.save_variable(variable=J, name_of_variable="J_iters_"+str(checkpoint_index), path_to_save=path_to_save+'J/')
            # termination check:
            if max_epochs != None:
                if iteration_index >= max_epochs:
                    break

    def classify_distortion_trainingSet(self, method, classify_again=True):
        if classify_again:
            if method == "LLISE":
                Y = self.load_variable(name_of_variable="Y_iters_5", path='./LLISE_settings/LLISE/linear_embed/Y/')
            if method == "kernel_LLISE":
                if self.kernel == "rbf":
                    Y = self.load_variable(name_of_variable="Y_iters_10", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')
                if self.kernel == "sigmoid":
                    Y = self.load_variable(name_of_variable="Y_iters_19", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')
                if self.kernel == "polynomial":
                    Y = self.load_variable(name_of_variable="Y_iters_26", path='./LLISE_settings/kernel_LLISE/' + self.kernel + '/linear_embed/Y/')
            class_of_distortion = np.zeros((self.n_training_images, self.n_blocks))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " / " + str(self.n_blocks-1) + "...")
                X_embedded_for_a_block = np.zeros((self.n_components, self.n_training_images))
                for image_index in range(self.n_training_images):
                    X_embedded_for_a_block[:, image_index] = Y[image_index, :, block_index].ravel()
                # --- KNN:
                connectivity_matrix = KNN(X=X_embedded_for_a_block.T, n_neighbors=1, mode='connectivity', include_self=False, n_jobs=-1)
                connectivity_matrix = connectivity_matrix.toarray()
                for image_index in range(self.n_training_images):
                    index_of_neighbor = int(np.argwhere(connectivity_matrix[image_index, :] == 1))
                    if index_of_neighbor == 0:  # "original"
                        class_of_distortion[image_index, block_index] = 0
                    elif index_of_neighbor >= 1 and index_of_neighbor <= 20:  # "contrast_stretched"
                        class_of_distortion[image_index, block_index] = 1
                    elif index_of_neighbor >= 21 and index_of_neighbor <= 40:  # "Gaussian_noise"
                        class_of_distortion[image_index, block_index] = 2
                    elif index_of_neighbor >= 41 and index_of_neighbor <= 60:  # "enhanced_luminance"
                        class_of_distortion[image_index, block_index] = 3
                    elif index_of_neighbor >= 61 and index_of_neighbor <= 80:  # "Gaussian_blurring"
                        class_of_distortion[image_index, block_index] = 4
                    elif index_of_neighbor >= 81 and index_of_neighbor <= 100:  # "impulse_noise"
                        class_of_distortion[image_index, block_index] = 5
                    elif index_of_neighbor >= 101 and index_of_neighbor <= 120:  # "jpeg_distortion"
                        class_of_distortion[image_index, block_index] = 6
            # --- percentage of distortions:
            votes = np.zeros((self.n_training_images, 7+1))
            estimated_distortion_class = np.zeros((self.n_training_images, 2))
            true_distortion_class = np.zeros((self.n_training_images, 2))
            for image_index in range(self.n_training_images):
                print("processing image #" + str(image_index) + " / " + str(self.n_training_images-1) + "...")
                votes[image_index, 0] = (sum(class_of_distortion[image_index, :] == 0) / self.n_blocks) * 100
                votes[image_index, 1] = (sum(class_of_distortion[image_index, :] == 1) / self.n_blocks) * 100
                votes[image_index, 2] = (sum(class_of_distortion[image_index, :] == 2) / self.n_blocks) * 100
                votes[image_index, 3] = (sum(class_of_distortion[image_index, :] == 3) / self.n_blocks) * 100
                votes[image_index, 4] = (sum(class_of_distortion[image_index, :] == 4) / self.n_blocks) * 100
                votes[image_index, 5] = (sum(class_of_distortion[image_index, :] == 5) / self.n_blocks) * 100
                votes[image_index, 6] = (sum(class_of_distortion[image_index, :] == 6) / self.n_blocks) * 100
                votes[image_index, 7] = image_index
                # --- class of distortions:
                estimated_distortion_class[image_index, 0] = np.argmax(votes[image_index, :-1].ravel())
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
            if method == "LLISE":
                path_to_save = './output/LLISE/classification/'
            elif method == "kernel_LLISE":
                path_to_save = './output/kernel_LLISE/' + self.kernel + '/classification/'
            self.save_variable(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_variable(variable=true_distortion_class, name_of_variable="true_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=true_distortion_class, name_of_variable="true_distortion_class", path_to_save=path_to_save)
        else:
            if method == "LLISE":
                path_to_save = './output/LLISE/classification/'
            elif method == "kernel_LLISE":
                path_to_save = './output/kernel_LLISE/' + self.kernel + '/classification/'
            votes = self.load_variable(name_of_variable="votes", path=path_to_save)
            estimated_distortion_class = self.load_variable(name_of_variable="estimated_distortion_class", path=path_to_save)
            true_distortion_class = self.load_variable(name_of_variable="true_distortion_class", path=path_to_save)
        return estimated_distortion_class[:, 0]

    def constraint_project_2(self, matrix):
        matrix = self.center_the_matrix(the_matrix=matrix, mode="remove_mean_of_rows_from_rows")
        Q, Sigma, Omega_h = LA.svd(matrix, full_matrices=False)
        Sigma = np.diag(self.n_training_images * np.ones((self.n_components, 1)).ravel())
        matrix_projected = Q.dot(Sigma).dot(Omega_h)
        return matrix_projected

    def constraint_project_1(self, vector):
        # gamma = 2
        # vector = vector / (np.sum(vector))
        # vector[vector < -gamma] = -gamma
        # vector[vector > gamma] = gamma
        # vector[vector <= 0] = 1e-20
        # vector[vector >= 1] = 1
        # vector = vector / (np.sum(vector))
        vector = vector / (LA.norm(vector))
        return vector

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

    def theta(self, image_index, block_index, Y):
        # Y: n * p matrix
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        w = np.zeros((self.n_training_images, 1))
        w[neighbor_indices_of_this_block, 0] = self.weights_linearReconstruction[image_index, block_index, :]
        one_j_vector = np.zeros(self.n_training_images).reshape((-1, 1))
        one_j_vector[image_index, 0] = 1
        M = one_j_vector.dot(one_j_vector.T) + w.dot(w.T) - 2 * (one_j_vector.dot(w.T))
        Psi = one_j_vector.dot(one_j_vector.T) + w.dot(w.T)
        numinator = np.trace((Y.T).dot(M).dot(Y))
        q = self.block_dimension
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        denominator = np.trace((Y.T).dot(Psi).dot(Y)) + c
        theta = numinator / denominator
        theta = float(theta)
        return theta

    def gradient_theta(self, image_index, block_index, Y):
        # Y: n * p matrix
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        w = np.zeros((self.n_training_images, 1))
        w[neighbor_indices_of_this_block, 0] = self.weights_linearReconstruction[image_index, block_index, :]
        one_j_vector = np.zeros(self.n_training_images).reshape((-1, 1))
        one_j_vector[image_index, 0] = 1
        M = one_j_vector.dot(one_j_vector.T) + w.dot(w.T) - 2 * (one_j_vector.dot(w.T))
        Psi = one_j_vector.dot(one_j_vector.T) + w.dot(w.T)
        q = self.block_dimension
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        part1 = 2 / (np.trace((Y.T).dot(Psi).dot(Y)) + c)
        theta = self.theta(image_index, block_index, Y)
        part2 = (M - (theta * Psi)).dot(Y)
        gradient_theta = part1 * part2
        return gradient_theta

    def f_kernel_for_outOfSample(self, outOfSample_image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices_for_outOfSample[outOfSample_image_index, block_index, :].astype(int)
        q = self.block_dimension
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        kernel_with_itself = self.kernel_of_blocks_for_outOfSample[self.n_training_images, self.n_training_images, outOfSample_image_index, block_index]
        temp = self.kernel_of_blocks_for_outOfSample[neighbor_indices_of_this_block, :, outOfSample_image_index, block_index]
        kernel_neighbors_with_neighbors = temp[:, neighbor_indices_of_this_block]
        kernel_neighbors_with_it = self.kernel_of_blocks_for_outOfSample[neighbor_indices_of_this_block, self.n_training_images, outOfSample_image_index, block_index].reshape((-1, 1))
        numinator = kernel_with_itself + (w.T).dot(kernel_neighbors_with_neighbors).dot(w) - (2 * (w.T).dot(kernel_neighbors_with_it))
        denominator = kernel_with_itself + (w.T).dot(kernel_neighbors_with_neighbors).dot(w) + c
        f = numinator / denominator
        f = float(f)
        return f

    def gradient_f_kernel_for_outOfSample(self, outOfSample_image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices_for_outOfSample[outOfSample_image_index, block_index, :].astype(int)
        q = self.block_dimension
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        f_kernel_for_outOfSample = self.f_kernel_for_outOfSample(outOfSample_image_index, block_index, w)
        kernel_with_itself = self.kernel_of_blocks_for_outOfSample[self.n_training_images, self.n_training_images, outOfSample_image_index, block_index]
        temp = self.kernel_of_blocks_for_outOfSample[neighbor_indices_of_this_block, :, outOfSample_image_index, block_index]
        kernel_neighbors_with_neighbors = temp[:, neighbor_indices_of_this_block]
        kernel_neighbors_with_it = self.kernel_of_blocks_for_outOfSample[neighbor_indices_of_this_block, self.n_training_images, outOfSample_image_index, block_index].reshape((-1, 1))
        numinator = ((1 - f_kernel_for_outOfSample) * kernel_neighbors_with_neighbors.dot(w)) - kernel_neighbors_with_it
        numinator = 2 * numinator
        denominator = kernel_with_itself + (w.T).dot(kernel_neighbors_with_neighbors).dot(w) + c
        gradient_f_kernel = (1 / denominator) * numinator
        return gradient_f_kernel

    def f_kernel(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block = self.blocks_meanRemoved[:, block_index, image_index].reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        q = block.shape[0]
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        kernel_with_itself = self.kernel_of_blocks[image_index, image_index, block_index]
        temp = self.kernel_of_blocks[neighbor_indices_of_this_block, :, block_index]
        kernel_neighbors_with_neighbors = temp[:, neighbor_indices_of_this_block]
        kernel_neighbors_with_it = self.kernel_of_blocks[neighbor_indices_of_this_block, image_index, block_index].reshape((-1, 1))
        numinator = kernel_with_itself + (w.T).dot(kernel_neighbors_with_neighbors).dot(w) - (2 * (w.T).dot(kernel_neighbors_with_it))
        denominator = kernel_with_itself + (w.T).dot(kernel_neighbors_with_neighbors).dot(w) + c
        f = numinator / denominator
        f = float(f)
        return f

    def gradient_f_kernel(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block = self.blocks_meanRemoved[:, block_index, image_index].reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        q = block.shape[0]
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        f_kernel = self.f_kernel(image_index, block_index, w)
        kernel_with_itself = self.kernel_of_blocks[image_index, image_index, block_index]
        temp = self.kernel_of_blocks[neighbor_indices_of_this_block, :, block_index]
        kernel_neighbors_with_neighbors = temp[:, neighbor_indices_of_this_block]
        kernel_neighbors_with_it = self.kernel_of_blocks[neighbor_indices_of_this_block, image_index, block_index].reshape((-1, 1))
        numinator = ((1 - f_kernel) * kernel_neighbors_with_neighbors.dot(w)) - kernel_neighbors_with_it
        numinator = 2 * numinator
        denominator = kernel_with_itself + (w.T).dot(kernel_neighbors_with_neighbors).dot(w) + c
        gradient_f_kernel = (1 / denominator) * numinator
        return gradient_f_kernel

    def f_for_outOfSample(self, outOfSample_image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block_outOfSample = self.blocks_meanRemoved_outOfSample[:, block_index, outOfSample_image_index].reshape((-1, 1))
        training_neighbor_indices_of_this_block = self.neighbor_indices_for_outOfSample[outOfSample_image_index, block_index, :].astype(int)
        X_training_neighbors = self.blocks_meanRemoved[:, block_index, training_neighbor_indices_of_this_block]
        q = self.block_dimension
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        numinator = (block_outOfSample.T).dot(block_outOfSample) + (w.T).dot(X_training_neighbors.T).dot(X_training_neighbors).dot(w) - (block_outOfSample.T).dot(X_training_neighbors).dot(w) - (w.T).dot(X_training_neighbors.T).dot(block_outOfSample)
        denominator = (block_outOfSample.T).dot(block_outOfSample) + (w.T).dot(X_training_neighbors.T).dot(X_training_neighbors).dot(w) + c
        f = numinator / denominator
        f = float(f)
        return f

    def gradient_f_for_outOfSample(self, outOfSample_image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block_outOfSample = self.blocks_meanRemoved_outOfSample[:, block_index, outOfSample_image_index].reshape((-1, 1))
        training_neighbor_indices_of_this_block = self.neighbor_indices_for_outOfSample[outOfSample_image_index, block_index, :].astype(int)
        X_training_neighbors = self.blocks_meanRemoved[:, block_index, training_neighbor_indices_of_this_block]
        q = self.block_dimension
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        f_for_outOfSample = self.f_for_outOfSample(outOfSample_image_index, block_index, w)
        part1_numinator = 2 * (X_training_neighbors.T)
        part1_denominator = (block_outOfSample.T).dot(block_outOfSample) + (w.T).dot(X_training_neighbors.T).dot(X_training_neighbors).dot(w) + c
        part1 = (1 / part1_denominator) * part1_numinator
        part2 = ((1 - f_for_outOfSample) * (X_training_neighbors.dot(w))) - block_outOfSample
        gradient_f = part1.dot(part2)
        return gradient_f

    def f(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block = self.blocks_meanRemoved[:, block_index, image_index].reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        X_neighbors = self.blocks_meanRemoved[:, block_index, neighbor_indices_of_this_block]
        q = block.shape[0]
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        numinator = (block.T).dot(block) + (w.T).dot(X_neighbors.T).dot(X_neighbors).dot(w) - (block.T).dot(X_neighbors).dot(w) - (w.T).dot(X_neighbors.T).dot(block)
        denominator = (block.T).dot(block) + (w.T).dot(X_neighbors.T).dot(X_neighbors).dot(w) + c
        f = numinator / denominator
        f = float(f)
        return f

    def gradient_f(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block = self.blocks_meanRemoved[:, block_index, image_index].reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        X_neighbors = self.blocks_meanRemoved[:, block_index, neighbor_indices_of_this_block]
        q = block.shape[0]
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        f = self.f(image_index, block_index, w)
        part1_numinator = 2 * (X_neighbors.T)
        part1_denominator = (block.T).dot(block) + (w.T).dot(X_neighbors.T).dot(X_neighbors).dot(w) + c
        part1 = (1 / part1_denominator) * part1_numinator
        part2 = ((1 - f) * (X_neighbors.dot(w))) - block
        gradient_f = part1.dot(part2)
        return gradient_f

    def f_wrong(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block = self.blocks_meanRemoved[:, block_index, image_index].reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        X_neighbors = self.blocks_meanRemoved[:, block_index, neighbor_indices_of_this_block]
        ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
        G = ((block.dot(ones_vector.T) - X_neighbors).T).dot(block.dot(ones_vector.T) - X_neighbors)
        Gamma = ones_vector.dot(block.T).dot(block).dot(ones_vector.T) + (X_neighbors.T).dot(X_neighbors)
        q = block.shape[0]
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        numinator = (w.T).dot(G).dot(w)
        denominator = (w.T).dot(Gamma).dot(w) + c
        f = numinator / denominator
        f = float(f)
        return f

    def gradient_f_wrong(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block = self.blocks_meanRemoved[:, block_index, image_index].reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        X_neighbors = self.blocks_meanRemoved[:, block_index, neighbor_indices_of_this_block]
        ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
        G = ((block.dot(ones_vector.T) - X_neighbors).T).dot(block.dot(ones_vector.T) - X_neighbors)
        Gamma = ones_vector.dot(block.T).dot(block).dot(ones_vector.T) + (X_neighbors.T).dot(X_neighbors)
        q = block.shape[0]
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        f = self.f(image_index, block_index, w)
        part1 = 2 / ((w.T).dot(Gamma).dot(w) + c)
        part2 = G - (f * Gamma)
        part3 = w
        gradient_f = part1 * part2.dot(part3)
        return gradient_f

    def numerical_Hessian_f(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        epsilon = 0.0001
        gradient_f_beforeChange = self.gradient_f(image_index, block_index, w)
        Hessian = np.zeros((self.n_neighbors, self.n_neighbors))
        for w_component_index in range(self.n_neighbors):  # iteration on columns of Hessian
            w_changed_component = w.copy()
            w_changed_component[w_component_index, 0] = w[w_component_index, 0] + epsilon
            gradient_f_afterChange = self.gradient_f(image_index, block_index, w_changed_component)
            for gradientOf_f_component_index in range(self.n_neighbors):  # iteration on rows of Hessian
                derivative_approximate = (gradient_f_afterChange[gradientOf_f_component_index, 0] - gradient_f_beforeChange[gradientOf_f_component_index, 0]) / epsilon
                Hessian[gradientOf_f_component_index, w_component_index] = derivative_approximate
        return Hessian

    def Hessian_f(self, image_index, block_index, w):
        # w: k * 1 vector
        w = w.reshape((-1, 1))
        block = self.blocks_meanRemoved[:, block_index, image_index].reshape((-1, 1))
        neighbor_indices_of_this_block = self.neighbor_indices[image_index, block_index, :].astype(int)
        X_neighbors = self.blocks_meanRemoved[:, block_index, neighbor_indices_of_this_block]
        ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
        G = ((block.dot(ones_vector.T) - X_neighbors).T).dot(block.dot(ones_vector.T) - X_neighbors)
        Gamma = ones_vector.dot(block.T).dot(block).dot(ones_vector.T) + (X_neighbors.T).dot(X_neighbors)
        q = block.shape[0]
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        f = self.f(image_index, block_index, w)
        gradient_f = self.gradient_f(image_index, block_index, w)
        # part1 = 2 / ((w.T).dot(Gamma).dot(w) + c)
        # # part2 = G.T - gradient_f.dot(w.T).dot(Gamma.T) - (f * Gamma.T)
        # part2 = G - gradient_f.dot(w.T).dot(Gamma) - (f * Gamma)
        # # part3_numinator = 2 * Gamma.dot(w).dot(w.T).dot((f * Gamma.T) - G.T)
        # part3_numinator = 2 * Gamma.dot(w).dot(w.T).dot((f * Gamma) - G)
        # part3_denominator = part1 / 2
        # part3 = (1 / part3_denominator) * part3_numinator
        # Hessian_f = part1 * (part2 + part3)
        part1 = 2 / ((w.T).dot(Gamma).dot(w) + c)
        part2 = G.T - (Gamma.T).dot(w).dot(gradient_f.T) - (f * Gamma.T) - (2 * gradient_f.dot(w.T).dot(Gamma.T))
        Hessian_f = part1 * part2
        return Hessian_f

    def Newton_systemOfEquations(self, gradient_f, Hessian_f):
        #------ method 1: (using Schur complement)
        # ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
        # # epsilon = 0.00001
        # epsilon = 0.000000001
        # Hessian_inverse = LA.inv(Hessian_f + epsilon*np.eye(self.n_neighbors))
        # S = 0 - (ones_vector.T).dot(Hessian_inverse).dot(ones_vector)
        # b_tilde = 0 - (ones_vector.T).dot(Hessian_inverse).dot(- gradient_f)
        # lambda_lagrangeMultiplier = LA.inv(S).dot(b_tilde)
        # b_tilde_2 = - gradient_f - ones_vector.dot(lambda_lagrangeMultiplier)
        # delta_step = Hessian_inverse.dot(b_tilde_2)
        # ------ method 2:
        ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
        A = np.zeros((self.n_neighbors+1, self.n_neighbors+1))
        A[:-1, :-1] = Hessian_f
        A[:-1, -1] = ones_vector.ravel()
        A[-1, :-1] = ones_vector.T
        A[-1, -1] = 0
        b = np.zeros((self.n_neighbors+1, 1))
        b[:-1, 0] = (-1 * gradient_f).ravel()
        b[-1, 0] = 0
        epsilon = 0.000000001
        x = LA.inv(A + + epsilon*np.eye(self.n_neighbors+1)).dot(b).reshape((-1, 1))
        delta_step = x[:-1, 0].reshape((-1, 1))
        lambda_lagrangeMultiplier = x[-1, 0]
        return delta_step, lambda_lagrangeMultiplier

    def divide_images_to_their_blocks(self, X, remove_mean=False):
        n_images = X.shape[1]
        blocks = np.zeros((self.block_dimension, self.n_blocks, n_images))
        n_blocks_in_height = np.ceil(self.image_height / self.block_height)
        n_blocks_in_width = np.ceil(self.image_width / self.block_width)
        for image_index in range(n_images):
            image = X[:, image_index]
            image_2D = image.reshape((self.image_height, self.image_width))
            for block_index in range(self.n_blocks): #--> moving left to right and then next row, ...
                row_of_block = int(np.floor(block_index / n_blocks_in_width))
                column_of_block = int(block_index % n_blocks_in_width)
                rowOfImage_start = row_of_block * self.block_height
                rowOfImage_end = rowOfImage_start + self.block_height - 1
                columnOfImage_start = column_of_block * self.block_width
                columnOfImage_end = columnOfImage_start + self.block_width - 1
                block = image_2D[rowOfImage_start:rowOfImage_end+1, columnOfImage_start:columnOfImage_end+1]
                block = block.reshape((-1,1))
                if remove_mean:
                    block = block - block.mean()  # --> remove mean of block
                blocks[:, block_index, image_index] = block.ravel()
        return blocks

    def train_U_by_ADMM(self, max_epochs=None, step_checkpoint=10):
        self.blocks = self.divide_images_to_their_blocks(X=self.X)
        U = np.random.rand(self.block_dimension, self.n_components, self.n_blocks)  # --> rand in [0,1)
        V = np.random.rand(self.block_dimension, self.n_components, self.n_blocks)  # --> rand in [0,1)
        J = np.random.rand(self.block_dimension, self.n_components, self.n_blocks)  # --> rand in [0,1)
        self.save_variable(variable=U, name_of_variable="U_epochs_initial", path_to_save='./ISCA_settings/ADMM/weights/')
        self.save_variable(variable=V, name_of_variable="V_epochs_initial", path_to_save='./ISCA_settings/ADMM/weights/')
        self.save_variable(variable=J, name_of_variable="J_epochs_initial", path_to_save='./ISCA_settings/ADMM/weights/')
        self.save_image_of_weights(U, path_to_save="./ISCA_settings/ADMM/U_figs/initial/")
        self.save_image_of_weights(V, path_to_save="./ISCA_settings/ADMM/V_figs/initial/")
        self.save_image_of_weights(J, path_to_save="./ISCA_settings/ADMM/J_figs/initial/")
        rho = 1
        alpha = 0.1
        epoch_index = -1
        error_of_block = np.zeros((self.n_blocks, 1))
        reconstruction_error_epochs = np.zeros((step_checkpoint, 1))
        average_error_U_epochs = np.zeros((step_checkpoint, 1))
        # U_epochs = [None] * step_checkpoint
        while True:
            epoch_index = epoch_index + 1
            print("----- epoch #" + str(epoch_index))
            U_previous_epoch = U.copy()
            reconstruction_error_blocks = np.zeros((self.n_blocks, 1))
            for image_index in range(self.n_training_images): #--> iterating on images (an epoch)
                # print("processing image " + str(image_index) + "/" + str(self.n_training_images-1) + " in epoch #" + str(epoch_index))
                for block_index in range(self.n_blocks):
                    block = (self.blocks[:, block_index, image_index]).reshape((-1,1))
                    block = block - block.mean()  #--> remove mean of block
                    gradient_of_f = self.G(block, U[:, :, block_index])
                    U[:, :, block_index] = U[:, :, block_index] - alpha*gradient_of_f - alpha*(rho * (U[:, :, block_index] - V[:, :, block_index] + J[:, :, block_index]))
                    V[:, :, block_index] = self.set_singular_values_one(matrix =U[:, :, block_index] + J[:, :, block_index])
                    J[:, :, block_index] = J[:, :, block_index] + U[:, :, block_index] - V[:, :, block_index]
                    reconstruction_error_blocks[block_index] = reconstruction_error_blocks[block_index] + self.f(block, U[:, :, block_index])
            # epoch error 1:
            reconstruction_error_blocks_meanOfImages = reconstruction_error_blocks / self.n_training_images
            # reconstruction_error_blocks_meanOfImages = reconstruction_error_blocks / 1
            reconstruction_error = reconstruction_error_blocks_meanOfImages.mean()  #--> reconstruction_error_meanOfBlocks_meanOfImages
            index_to_save = epoch_index % step_checkpoint
            reconstruction_error_epochs[index_to_save] = reconstruction_error
            print("----- average reconstruction error of epoch #" + str(epoch_index) + ": " + str(reconstruction_error))
            # epoch error 2:
            for block_index in range(self.n_blocks):
                error_of_block[block_index] = LA.norm(U[:, :, block_index] - U_previous_epoch[:, :, block_index], ord="fro")
            average_error_U = error_of_block.mean()
            average_error_U_epochs[index_to_save] = average_error_U
            print("----- average U error of epoch #" + str(epoch_index) + ": " + str(average_error_U))
            # save U (weights):
            # U_epochs[index_to_save] = U
            # save the information at checkpoints:
            if (epoch_index+1) % step_checkpoint == 0:
                print("Saving the checkpoint in epoch #" + str(epoch_index))
                checkpoint_index = int(np.floor(epoch_index / step_checkpoint))
                self.save_variable(variable=reconstruction_error_epochs, name_of_variable="reconstruction_error_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM/reconstruction_error/')
                self.save_np_array_to_txt(variable=reconstruction_error_epochs, name_of_variable="reconstruction_error_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM/reconstruction_error/')
                self.save_variable(variable=average_error_U_epochs, name_of_variable="average_error_U_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM/U_error/')
                self.save_np_array_to_txt(variable=average_error_U_epochs, name_of_variable="average_error_U_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM/U_error/')
                self.save_variable(variable=U, name_of_variable="U_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM/U/')
                self.save_variable(variable=V, name_of_variable="V_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM/V/')
                self.save_variable(variable=J, name_of_variable="J_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM/J/')
                self.save_image_of_weights(U, path_to_save="./ISCA_settings/ADMM/U_figs/epoch_"+str(epoch_index)+"/")
                self.save_image_of_weights(V, path_to_save="./ISCA_settings/ADMM/V_figs/epoch_"+str(epoch_index)+"/")
                self.save_image_of_weights(J, path_to_save="./ISCA_settings/ADMM/J_figs/epoch_"+str(epoch_index)+"/")
            # termination check:
            if max_epochs != None:
                if epoch_index >= max_epochs:
                    break

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