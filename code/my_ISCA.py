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


class My_ISCA:

    def __init__(self, X, image_height, image_width, n_components=None, block_height=8, block_width=8, kernel="linear"):
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
        self.blocks = None
        self.kernel = kernel
        self.kernel_of_blocks = None
        if n_components != None:
            self.n_components = n_components  # --> p
        else:
            self.n_components = self.block_dimension

    def classify_distortion_testSet(self, X, method, classify_again=True):
        n_test_images = X.shape[1]
        if classify_again:
            if method == "ISCA":
                which_U_or_Theta_epoch = 199
                U = self.load_variable(name_of_variable="U_epochs_"+str(int((which_U_or_Theta_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/U/')
                blocks_test = self.divide_images_to_their_blocks(X=X)
                blocks_train = self.divide_images_to_their_blocks(X=self.X)
            elif method == "kernel_ISCA":
                which_U_or_Theta_epoch = 99
                Theta = self.load_variable(name_of_variable="Theta_epochs_"+str(int((which_U_or_Theta_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM_kernel/' + self.kernel + '/Theta/')
                blocks_test = self.divide_images_to_their_blocks(X=X)
                blocks_train = self.divide_images_to_their_blocks(X=self.X)
                kernel_of_blocks_train_NormalizedandCentered = self.get_kernels(calculate_kernels_again=True, save_kernels=False)
                train_X_block_amongImages = np.zeros((self.block_dimension, self.n_training_images, self.n_blocks))
                kernel_of_blocks_train_NotNormalizedandCentered = np.zeros((self.n_training_images, self.n_training_images, self.n_blocks))
                for block_index in range(self.n_blocks):
                    for image_index in range(self.n_training_images):
                        block_train = (blocks_train[:, block_index, image_index]).reshape((-1, 1))
                        train_X_block_amongImages[:, image_index, block_index] = block_train.ravel()
                    kernel_of_blocks_train_NotNormalizedandCentered[:, :, block_index] = pairwise_kernels(X=train_X_block_amongImages[:, :, block_index].T, Y=train_X_block_amongImages[:, :, block_index].T, metric=self.kernel)
                # print(kernel_of_blocks_train_NotNormalizedandCentered[:, :, 0])
                # # print(np.diag(kernel_of_blocks_train_NotNormalizedandCentered[:, :, block_index]))
                # input("jj")
            class_of_distortion = np.zeros((n_test_images, self.n_blocks))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " / " + str(self.n_blocks-1) + "...")
                #--- test (out of sample) images:
                test_X_projected_for_a_block = np.zeros((self.n_components, n_test_images))
                for image_index in range(n_test_images):
                    if method == "ISCA":
                        block_test = (blocks_test[:, block_index, image_index]).reshape((-1, 1))
                        block_test = block_test - block_test.mean()  # --> remove mean of block
                        test_X_projected_for_a_block[:, image_index] = (U[:, :, block_index].T).dot(block_test.reshape(-1, 1)).ravel()
                    elif method == "kernel_ISCA":
                        block_test = (blocks_test[:, block_index, image_index]).reshape((-1, 1))
                        #--- normalizing the kernel (method 1): --> both methods result the same in normalization but not in centering
                        # kernel_train_with_testBlock = pairwise_kernels(X=train_X_block_amongImages[:, :, block_index].T, Y=block_test.T, metric=self.kernel)
                        # kernel_testBlock_with_itself = pairwise_kernels(X=block_test.T, Y=block_test.T, metric=self.kernel)[0][0]
                        # for train_image_index in range(self.n_training_images):
                        #     kernel_train_with_testBlock[train_image_index, 0] = kernel_train_with_testBlock[train_image_index, 0] / ((kernel_of_blocks_train_NotNormalizedandCentered[train_image_index, train_image_index, block_index] * kernel_testBlock_with_itself) ** 0.5)
                        # --- centering the kernel (method 1):
                        # kernel_train_with_testBlock = self.center_the_matrix(the_matrix=kernel_train_with_testBlock, mode="remove_mean_of_rows_from_rows")
                        # print(kernel_of_blocks_train_NormalizedandCentered)
                        #--- normalizing the kernel (method 2): --> both methods result the same in normalization but not in centering
                        trainX_and_the_test = np.hstack((train_X_block_amongImages[:, :, block_index], block_test))
                        temp = pairwise_kernels(X=trainX_and_the_test.T, Y=trainX_and_the_test.T, metric=self.kernel)
                        kernel_train_with_testBlock = self.normalize_the_kernel(kernel_matrix=temp)
                        # kernel_train_with_testBlock = kernel_train_with_testBlock[:-1, -1].reshape((-1,1))
                        # --- centering the kernel (method 2):
                        kernel_train_with_testBlock = self.center_the_matrix(the_matrix=kernel_train_with_testBlock, mode="double_center")
                        kernel_train_with_testBlock = kernel_train_with_testBlock[:-1, -1].reshape((-1, 1))
                        #--- projection:
                        test_X_projected_for_a_block[:, image_index] = (Theta[:, :, block_index].T).dot(kernel_train_with_testBlock).ravel()
                # print(test_X_projected_for_a_block[:, 0])
                #--- training images:
                train_X_projected_for_a_block = np.zeros((self.n_components, self.n_training_images))
                for image_index in range(self.n_training_images):
                    if method == "ISCA":
                        block_train = (blocks_train[:, block_index, image_index]).reshape((-1, 1))
                        block_train = block_train - block_train.mean()  # --> remove mean of block
                        train_X_projected_for_a_block[:, image_index] = (U[:, :, block_index].T).dot(block_train.reshape(-1, 1)).ravel()
                    elif method == "kernel_ISCA":
                        train_X_projected_for_a_block[:, image_index] = (Theta[:, :, block_index].T).dot(kernel_of_blocks_train_NormalizedandCentered[:, image_index, block_index].reshape(-1, 1)).ravel()
                # print(blocks_train[:, block_index, image_index])
                # print(train_X_projected_for_a_block[:, 0])
                # --- KNN:
                X_train_and_the_test_image = np.hstack((test_X_projected_for_a_block, train_X_projected_for_a_block))
                connectivity_matrix = KNN(X=X_train_and_the_test_image.T, n_neighbors=X_train_and_the_test_image.shape[1]-1, mode='distance', include_self=False, n_jobs=-1)
                connectivity_matrix = connectivity_matrix.toarray()
                for image_index in range(n_test_images):
                    a = connectivity_matrix[image_index, n_test_images:]
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
                    # print(class_of_distortion[image_index, block_index])
                    # input("hi")
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
            if method == "ISCA":
                path_to_save = './output/ISCA/classification_test/'
            elif method == "kernel_ISCA":
                path_to_save = './output/kernel_ISCA/' + self.kernel + '/classification_test/'
            self.save_variable(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
        else:
            if method == "ISCA":
                path_to_save = './output/ISCA/classification_test/'
            elif method == "kernel_ISCA":
                path_to_save = './output/kernel_ISCA/' + self.kernel + '/classification_test/'
            votes = self.load_variable(name_of_variable="votes", path=path_to_save)
            estimated_distortion_class = self.load_variable(name_of_variable="estimated_distortion_class", path=path_to_save)
        return estimated_distortion_class[:, 0]

    def classify_distortion_trainingSet(self, method, classify_again=True):
        if classify_again:
            if method == "ISCA":
                which_U_or_Theta_epoch = 199
                U = self.load_variable(name_of_variable="U_epochs_"+str(int((which_U_or_Theta_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/U/')
                blocks = self.divide_images_to_their_blocks(X=self.X)
            elif method == "kernel_ISCA":
                which_U_or_Theta_epoch = 99
                Theta = self.load_variable(name_of_variable="Theta_epochs_"+str(int((which_U_or_Theta_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM_kernel/' + self.kernel + '/Theta/')
                kernel_of_blocks = self.get_kernels(calculate_kernels_again=True, save_kernels=False)
            class_of_distortion = np.zeros((self.n_training_images, self.n_blocks))
            for block_index in range(self.n_blocks):
                print("processing block #" + str(block_index) + " / " + str(self.n_blocks-1) + "...")
                X_projected_for_a_block = np.zeros((self.n_components, self.n_training_images))
                for image_index in range(self.n_training_images):
                    if method == "ISCA":
                        block = (blocks[:, block_index, image_index]).reshape((-1, 1))
                        block = block - block.mean()  # --> remove mean of block
                        X_projected_for_a_block[:, image_index] = (U[:, :, block_index].T).dot(block.reshape(-1, 1)).ravel()
                    elif method == "kernel_ISCA":
                        X_projected_for_a_block[:, image_index] = (Theta[:, :, block_index].T).dot(kernel_of_blocks[:, image_index, block_index].reshape(-1, 1)).ravel()
                # --- KNN:
                connectivity_matrix = KNN(X=X_projected_for_a_block.T, n_neighbors=1, mode='connectivity', include_self=False, n_jobs=-1)
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
            if method == "ISCA":
                path_to_save = './output/ISCA/classification/'
            elif method == "kernel_ISCA":
                path_to_save = './output/kernel_ISCA/' + self.kernel + '/classification/'
            self.save_variable(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=votes, name_of_variable="votes", path_to_save=path_to_save)
            self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
            self.save_variable(variable=true_distortion_class, name_of_variable="true_distortion_class", path_to_save=path_to_save)
            self.save_np_array_to_txt(variable=true_distortion_class, name_of_variable="true_distortion_class", path_to_save=path_to_save)
        else:
            if method == "ISCA":
                path_to_save = './output/ISCA/classification/'
            elif method == "kernel_ISCA":
                path_to_save = './output/kernel_ISCA/' + self.kernel + '/classification/'
            votes = self.load_variable(name_of_variable="votes", path=path_to_save)
            estimated_distortion_class = self.load_variable(name_of_variable="estimated_distortion_class", path=path_to_save)
            true_distortion_class = self.load_variable(name_of_variable="true_distortion_class", path=path_to_save)
        return estimated_distortion_class[:, 0]

    def ISCA_transform(self):
        U_prime = self.load_variable(name_of_variable="U_prime", path='./ISCA_settings/ADMM/unified/')
        self.blocks = self.divide_images_to_their_blocks(X=self.X)
        X_transformed = np.zeros((self.n_components, self.n_training_images))
        for image_index in range(self.n_training_images):
            s = 0
            for block_index in range(self.n_blocks):
                block = (self.blocks[:, block_index, image_index]).reshape((-1, 1))
                block = block - block.mean()  # --> remove mean of block
                s = s + (U_prime[:, :, block_index].T).dot(block.reshape(-1, 1))
                # s = s + (U_prime[:, :, 0].T).dot(block.reshape(-1, 1))
            X_transformed[:, image_index] = s.ravel()
        return X_transformed

    def reconstruct_test(self, which_U_epoch, X):
        n_test_images = X.shape[1]
        U = self.load_variable(name_of_variable="U_epochs_"+str(int((which_U_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/U/')
        blocks_test = self.divide_images_to_their_blocks(X=X)
        X_reconstructed = np.zeros((self.image_dimension, n_test_images))
        for image_index in range(n_test_images):
            print("Reconstructing test image #" + str(image_index) + "...")
            image_reconstructed = np.zeros((self.image_height, self.image_width))
            for block_index in range(self.n_blocks):
                block_test = (blocks_test[:, block_index, image_index]).reshape((-1, 1))
                block_test_mean = block_test.mean()
                block_test = block_test - block_test_mean  # --> remove mean of block
                block_reconstructed = U[:, :, block_index].dot(U[:, :, block_index].T).dot(block_test.reshape(-1, 1))
                block_reconstructed = block_reconstructed + block_test_mean
                #--- put reconstructed block in image:
                n_blocks_in_column = int(np.ceil(self.image_width / self.block_width))
                columnOfImage_start = int(block_index % n_blocks_in_column) * self.block_width
                columnOfImage_end = columnOfImage_start + self.block_width - 1
                rowOfImage_start = int(np.floor(block_index / n_blocks_in_column)) * self.block_height
                rowOfImage_end = rowOfImage_start + self.block_height - 1
                image_reconstructed[rowOfImage_start:rowOfImage_end+1, columnOfImage_start:columnOfImage_end+1] = block_reconstructed.reshape((self.block_height, self.block_width))
            X_reconstructed[:, image_index] = image_reconstructed.reshape((-1,1)).ravel()
        return X_reconstructed

    def reconstruct(self, which_U_epoch):
        U = self.load_variable(name_of_variable="U_epochs_"+str(int((which_U_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/U/')
        if self.blocks == None:
            self.blocks = self.divide_images_to_their_blocks(X=self.X)
        X_reconstructed = np.zeros((self.image_dimension, self.n_training_images))
        for image_index in range(self.n_training_images):
            print("Reconstructing image #" + str(image_index) + "...")
            image_reconstructed = np.zeros((self.image_height, self.image_width))
            for block_index in range(self.n_blocks):
                block = (self.blocks[:, block_index, image_index]).reshape((-1, 1))
                block_mean = block.mean()
                block = block - block_mean  # --> remove mean of block
                block_reconstructed = U[:, :, block_index].dot(U[:, :, block_index].T).dot(block.reshape(-1, 1))
                block_reconstructed = block_reconstructed + block_mean
                #--- put reconstructed block in image:
                n_blocks_in_column = int(np.ceil(self.image_width / self.block_width))
                columnOfImage_start = int(block_index % n_blocks_in_column) * self.block_width
                columnOfImage_end = columnOfImage_start + self.block_width - 1
                rowOfImage_start = int(np.floor(block_index / n_blocks_in_column)) * self.block_height
                rowOfImage_end = rowOfImage_start + self.block_height - 1
                image_reconstructed[rowOfImage_start:rowOfImage_end+1, columnOfImage_start:columnOfImage_end+1] = block_reconstructed.reshape((self.block_height, self.block_width))
            X_reconstructed[:, image_index] = image_reconstructed.reshape((-1,1)).ravel()
        return X_reconstructed


    def unify_subspaces(self, U=None, which_U_epoch=None):
        self.blocks = self.divide_images_to_their_blocks(X=self.X)
        # initializations:
        upsilon = np.random.rand(self.block_dimension, 1)  # --> rand in [0,1)
        if U != None:
            U_prime = U
        else:
            U = self.load_variable(name_of_variable="U_epochs_"+str(int((which_U_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/U/')
            U_prime = U
        # parameters:
        eta = 0.1
        gamma1 = 1
        gamma2 = 1
        # iterations:
        sum_blocks_over_images = np.zeros((self.block_dimension, self.n_blocks))
        iteration = -1
        while True:
            iteration = iteration + 1
            if iteration == 0:  #--> calculate it only once
                for block_index in range(self.n_blocks):
                    for image_index in range(self.n_training_images):
                        block = (self.blocks[:, block_index, image_index]).reshape((-1, 1))
                        block = block - block.mean()  # --> remove mean of block
                        sum_blocks_over_images[:, block_index] = sum_blocks_over_images[:, block_index] + block.ravel()
            # update upsilon:
            for update_upsilon_iteration in range(100):
                temp = np.zeros((self.block_dimension, 1))
                for block_index in range(self.n_blocks):
                    r_i = (np.eye(self.block_dimension, self.block_dimension) - U_prime[:, :, block_index].dot(U_prime[:, :, block_index].T)).dot(sum_blocks_over_images[:, block_index].reshape(-1,1))
                    temp = temp + r_i
                # upsilon = ((1 - (2 * eta * gamma1)) * upsilon) + (eta * temp)
                upsilon = upsilon + (eta * temp)
                upsilon = upsilon / LA.norm(upsilon)
            for block_index in range(self.n_blocks):
                for update_U_prime_iteration in range(100):
                    U_prime[:, :, block_index] = ((1 - (2 * eta * (upsilon.T).dot(sum_blocks_over_images[:, block_index])) + (2 * eta * gamma2)) * U_prime[:, :, block_index]) - (2 * eta * gamma2 * U[:, :, block_index])
                    U_prime[:, :, block_index] = self.set_singular_values_one(matrix=U_prime[:, :, block_index])
            # costs:
            sum_upsilon_transpose_temp = 0
            sum_upsilon_transpose_r = 0
            cost_U_prime_blocks = np.zeros((self.n_blocks, 1))
            for block_index in range(self.n_blocks):
                r_i = (np.eye(self.block_dimension, self.block_dimension) - U_prime[:, :, block_index].dot(U_prime[:, :, block_index].T)).dot(sum_blocks_over_images[:, block_index].reshape(-1,1))
                sum_upsilon_transpose_r = sum_upsilon_transpose_r + (upsilon.T).dot(r_i.reshape(-1,1))
                temp = U_prime[:, :, block_index].dot(U_prime[:, :, block_index].T).dot(sum_blocks_over_images[:, block_index].reshape(-1,1))
                sum_upsilon_transpose_temp = sum_upsilon_transpose_temp + (upsilon.T).dot(temp.reshape(-1, 1))
                cost_U_prime_blocks[block_index, 0] = (upsilon.T).dot(temp.reshape(-1,1)) + (gamma2 * (LA.norm(U_prime[:, :, block_index] - U[:, :, block_index]) ** 2))
            cost_upsilon = sum_upsilon_transpose_r - (gamma1 * (LA.norm(upsilon) ** 2))
            cost_U_prime_average = np.mean(cost_U_prime_blocks)
            print("----- upsilon cost of iteration #" + str(iteration) + ": " + str(cost_upsilon.ravel()))
            print("----- average U prime cost of iteration #" + str(iteration) + ": " + str(cost_U_prime_average))
            # save the U_prime and upsilon:
            self.save_variable(variable=U_prime, name_of_variable="U_prime", path_to_save='./ISCA_settings/ADMM/unified/')
            self.save_variable(variable=upsilon, name_of_variable="upsilon", path_to_save='./ISCA_settings/ADMM/unified/')


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

                # eig_val, eig_vec = LA.eigh(the_kernel)
                # print(eig_val)

                # kernel_of_blocks[:, :, block_index] = self.center_the_matrix(the_matrix=the_kernel, mode="double_center")
                kernel_of_blocks[:, :, block_index] = the_kernel

                # print(np.diag(kernel_of_blocks[:, :, block_index]))
                # input("hi3")

                kernel_of_blocks[:, :, block_index] = self.normalize_the_kernel(kernel_matrix=kernel_of_blocks[:, :, block_index])

                # print(kernel_of_blocks[:, :, block_index])
                # input("hi3")

                kernel_of_blocks[:, :, block_index] = self.center_the_matrix(the_matrix=kernel_of_blocks[:, :, block_index], mode="double_center")

                # eig_val, eig_vec = LA.eigh(kernel_of_blocks[:, :, block_index])
                # print(eig_val)
                # input("hi")

            # input("hi333")

            if save_kernels:
                self.save_variable(variable=kernel_of_blocks, name_of_variable="kernel_"+self.kernel, path_to_save='./ISCA_settings/kernels/')
        else:
            kernel_of_blocks = self.load_variable(name_of_variable="kernel_"+self.kernel, path='./ISCA_settings/kernels/')
        return kernel_of_blocks

    def normalize_the_kernel(self, kernel_matrix):
        diag_kernel = np.diag(kernel_matrix)
        # print(diag_kernel)
        # input("hi")
        k = (1 / np.sqrt(diag_kernel)).reshape((-1,1))
        normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
        return normalized_kernel_matrix

    def train_Theta_by_ADMM_kernel(self, max_epochs=None, step_checkpoint=10):
        self.kernel_of_blocks = self.get_kernels(calculate_kernels_again=True, save_kernels=False)
        Theta = np.random.rand(self.n_training_images, self.n_components, self.n_blocks)  # --> rand in [0,1)
        W = np.random.rand(self.n_training_images, self.n_components, self.n_blocks)  # --> rand in [0,1)
        J = np.random.rand(self.n_training_images, self.n_components, self.n_blocks)  # --> rand in [0,1)
        self.save_variable(variable=Theta, name_of_variable="Theta_epochs_initial", path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/weights/')
        self.save_variable(variable=W, name_of_variable="W_epochs_initial", path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/weights/')
        self.save_variable(variable=J, name_of_variable="J_epochs_initial", path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/weights/')
        #---> decompose kernel: (K = Delta^T Delta)
        Delta = np.zeros((self.n_training_images, self.n_training_images, self.n_blocks))
        for block_index in range(self.n_blocks):
            # print(np.isinf(self.kernel_of_blocks[:, :, block_index]).any())
            # Q, omega, Qh = LA.svd(self.kernel_of_blocks[:, :, block_index], full_matrices=True)
            # if block_index == 1062:
            #     aa = self.kernel_of_blocks[:, :, block_index]
            #     aa[aa < 0] = 0
            #     self.kernel_of_blocks[:, :, block_index] = aa

            # if block_index == 1:
            #     print(self.kernel_of_blocks[:, :, block_index].ravel())

            # aa = self.kernel_of_blocks[:, :, block_index]
            # aa[aa < 0] = 0
            # self.kernel_of_blocks[:, :, block_index] = aa

            # print(block_index)
            # print(min(self.kernel_of_blocks[:, :, block_index].ravel()))
            # print(max(self.kernel_of_blocks[:, :, block_index].ravel()))
            # eig_val, eig_vec = LA.eigh(self.kernel_of_blocks[:, :, block_index])
            # print(eig_val)
            # print(np.linalg.eigvals(self.kernel_of_blocks[:, :, block_index]))
            # print(np.all(np.linalg.eigvals(self.kernel_of_blocks[:, :, block_index]) >= 0))
            # print(LA.norm(self.kernel_of_blocks[:, :, block_index]))
            Q, omega, Qh = LA.svd(self.kernel_of_blocks[:, :, block_index], full_matrices=True)
            omega = np.asarray(omega)
            a = omega ** 0.5
            a = np.nan_to_num(a)
            Omega_square_root = np.diag(a)
            Delta[:, :, block_index] = Omega_square_root.dot(Q.T)

        # print(np.max(Delta[:, :, :]))
        # print(np.min(Delta[:, :, :]))
            # input("hiii")
            # print(LA.norm(self.kernel_of_blocks[:, :, block_index]))
            # print(LA.norm(Delta[:, :, block_index]))
            # print(LA.norm(Q.dot(np.diag(omega)).dot(Qh) - self.kernel_of_blocks[:, :, block_index]))
            # print(LA.norm(Q.dot(np.diag(omega**0.5)).dot(np.diag(omega**0.5)).dot(Qh) - self.kernel_of_blocks[:, :, block_index]))
            # print(LA.norm(Q.dot(np.diag(omega**0.5)).dot(np.diag(omega**0.5)).dot(Q.T) - self.kernel_of_blocks[:, :, block_index]))
            #
            # print(LA.norm(Delta[:, :, block_index].T.dot(Delta[:, :, block_index]) - self.kernel_of_blocks[:, :, block_index]))
            # print((Delta[:, :, block_index].T.dot(Delta[:, :, block_index]) - Q.dot(np.diag(omega)).dot(Qh)))
            # input("hi")
            # print("***********")
        # print(Delta)
        # print(LA.norm(Delta[:, :, 1]))
        # input("hi")
        rho = 0.1
        alpha = 0.1
        epoch_index = -1
        error_of_block = np.zeros((self.n_blocks, 1))
        reconstruction_error_epochs = np.zeros((step_checkpoint, 1))
        average_error_Theta_epochs = np.zeros((step_checkpoint, 1))
        while True:
            epoch_index = epoch_index + 1
            print("----- epoch #" + str(epoch_index))
            Theta_previous_epoch = Theta.copy()
            reconstruction_error_blocks = np.zeros((self.n_blocks, 1))
            for image_index in range(self.n_training_images): #--> iterating on images (an epoch) #range(116,117): #
                # print("processing image " + str(image_index) + "/" + str(self.n_training_images-1) + " in epoch #" + str(epoch_index))
                for block_index in range(self.n_blocks):
                    gradient_of_f = self.G_kernel(image_index, block_index, Theta[:, :, block_index])
                    # if image_index >= 116:
                    #     print(gradient_of_f)
                    #     print(Delta[:, :, block_index])
                    #     print(Theta[:, :, block_index])
                    #     print(J[:, :, block_index])
                    # print(Theta[:, :, block_index])
                    # print(LA.norm(Delta[:, :, block_index]))

                    # print(LA.norm(gradient_of_f))

                    Theta[:, :, block_index] = Theta[:, :, block_index] - alpha*gradient_of_f - alpha * rho * (Delta[:, :, block_index].T).dot(Delta[:, :, block_index].dot(Theta[:, :, block_index]) - W[:, :, block_index] + J[:, :, block_index])
                    W[:, :, block_index] = self.set_singular_values_one(matrix =Delta[:, :, block_index].dot(Theta[:, :, block_index]) + J[:, :, block_index])
                    J[:, :, block_index] = J[:, :, block_index] + Delta[:, :, block_index].dot(Theta[:, :, block_index]) - W[:, :, block_index]
                    reconstruction_error_blocks[block_index] = reconstruction_error_blocks[block_index] + self.f_kernel(image_index, block_index, Theta[:, :, block_index])

                # print(max(Theta[:, :, :].ravel()))
                # print(min(Theta[:, :, :].ravel()))
                # print(max(J[:, :, :].ravel()))
                # print(min(J[:, :, :].ravel()))
            # print(Theta)
            # epoch error 1:
            reconstruction_error_blocks_meanOfImages = reconstruction_error_blocks / self.n_training_images
            # reconstruction_error_blocks_meanOfImages = reconstruction_error_blocks / 1
            reconstruction_error = reconstruction_error_blocks_meanOfImages.mean()  #--> reconstruction_error_meanOfBlocks_meanOfImages
            index_to_save = epoch_index % step_checkpoint
            reconstruction_error_epochs[index_to_save] = reconstruction_error
            print("----- average reconstruction error of epoch #" + str(epoch_index) + ": " + str(reconstruction_error))
            # epoch error 2:
            for block_index in range(self.n_blocks):
                error_of_block[block_index] = LA.norm(Theta[:, :, block_index] - Theta_previous_epoch[:, :, block_index], ord="fro")
            average_error_Theta = error_of_block.mean()
            average_error_Theta_epochs[index_to_save] = average_error_Theta
            print("----- average U error of epoch #" + str(epoch_index) + ": " + str(average_error_Theta))
            # save the information at checkpoints:
            if (epoch_index+1) % step_checkpoint == 0:
                print("Saving the checkpoint in epoch #" + str(epoch_index))
                checkpoint_index = int(np.floor(epoch_index / step_checkpoint))
                self.save_variable(variable=reconstruction_error_epochs, name_of_variable="reconstruction_error_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/reconstruction_error/')
                self.save_np_array_to_txt(variable=reconstruction_error_epochs, name_of_variable="reconstruction_error_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/reconstruction_error/')
                self.save_variable(variable=average_error_Theta_epochs, name_of_variable="average_error_Theta_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/Theta_error/')
                self.save_np_array_to_txt(variable=average_error_Theta_epochs, name_of_variable="average_error_Theta_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/Theta_error/')
                self.save_variable(variable=Theta, name_of_variable="Theta_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/Theta/')
                self.save_variable(variable=W, name_of_variable="W_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/W/')
                self.save_variable(variable=J, name_of_variable="J_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/ADMM_kernel/'+self.kernel+'/J/')
            # termination check:
            if max_epochs != None:
                if epoch_index >= max_epochs:
                    break

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


    def train_U_by_subgradient(self, max_epochs=None, step_checkpoint=1):
        self.blocks = self.divide_images_to_their_blocks(X=self.X)
        U = np.random.rand(self.block_dimension, self.n_components, self.n_blocks) #--> rand in [0,1)
        self.save_variable(variable=U, name_of_variable="U_epochs_initial", path_to_save='./ISCA_settings/subgradient/weights/')
        self.save_image_of_weights(U, path_to_save="./ISCA_settings/subgradient/weight_figs/initial/")
        alpha = 10
        h = 10
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
                print("processing image " + str(image_index) + "/" + str(self.n_training_images-1) + " in epoch #" + str(epoch_index))
                for block_index in range(self.n_blocks):
                    block = (self.blocks[:, block_index, image_index]).reshape((-1,1))
                    block = block - block.mean()  #--> remove mean of block
                    U_old = U[:, :, block_index]
                    gradient_of_f = self.G(block, U_old)
                    alpha = h / ((epoch_index + 1) ** 0.5)
                    U_before_projection = U_old - (alpha * gradient_of_f)
                    U[:, :, block_index] = self.set_singular_values_one(matrix=U_before_projection)
                    reconstruction_error_blocks[block_index] = reconstruction_error_blocks[block_index] + self.f(block, U[:, :, block_index])
            # epoch error 1:
            reconstruction_error_blocks_meanOfImages = reconstruction_error_blocks / self.n_training_images
            reconstruction_error = reconstruction_error_blocks_meanOfImages.mean()  #--> reconstruction_error_meanOfBlocks_meanOfImages
            if epoch_index % step_checkpoint != 0:
                index_to_save = (epoch_index % step_checkpoint) - 1
            else:
                index_to_save = step_checkpoint - 1
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
            if epoch_index % step_checkpoint == 0:
                print("Saving the checkpoint in epoch #" + str(epoch_index))
                checkpoint_index = int(np.floor(epoch_index / step_checkpoint))
                self.save_variable(variable=reconstruction_error_epochs, name_of_variable="reconstruction_error_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/subgradient/reconstruction_error/')
                self.save_np_array_to_txt(variable=reconstruction_error_epochs, name_of_variable="reconstruction_error_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/subgradient/reconstruction_error/')
                self.save_variable(variable=average_error_U_epochs, name_of_variable="average_error_U_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/subgradient/U_error/')
                self.save_np_array_to_txt(variable=average_error_U_epochs, name_of_variable="average_error_U_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/subgradient/U_error/')
                self.save_variable(variable=U, name_of_variable="U_epochs_"+str(checkpoint_index), path_to_save='./ISCA_settings/subgradient/weights/')
                self.save_image_of_weights(U, path_to_save="./ISCA_settings/subgradient/weight_figs/epoch_"+str(epoch_index)+"/")
            # termination check:
            if max_epochs != None:
                if epoch_index >= max_epochs:
                    break

    def contrast_strech(self, image, r_min, r_max):
        # reasonable range of r_min --> (0, 255/2)
        # reasonable range of r_max --> (255/2, 255)
        # note r_max should be greater than r_min
        contrast_streched_image = np.around(1 * (image - r_min) / (r_max - r_min))
        return contrast_streched_image

    def save_image_of_weights(self, U, path_to_save="./"):
        # U --> self.block_dimension * self.n_components * self.n_blocks
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        for component_index in range(self.n_components):
            U_in_image_form = self.make_U_in_image_form(U)
            plt.imshow(U_in_image_form[:, :, component_index], cmap='gray')
            plt.axis('off')
            # plt.colorbar()
            # plt.show()
            plt.savefig(path_to_save+"u"+str(component_index)+".png")

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

    def make_U_in_image_form(self, U):
        U_in_image_form = np.zeros((self.image_height, self.image_width, self.n_components))
        for block_index in range(U.shape[2]):
            U_of_block = U[:, :, block_index]
            for component_index in range(U.shape[1]):
                U_of_block_oneComponent = U_of_block[:, component_index]
                U_of_block_oneComponent = U_of_block_oneComponent.reshape((self.block_height, self.block_width))
                n_blocks_in_column = int(np.ceil(self.image_width / self.block_width))
                columnOfImage_start = int(block_index % n_blocks_in_column) * self.block_width
                columnOfImage_end = columnOfImage_start + self.block_width - 1
                rowOfImage_start = int(np.floor(block_index / n_blocks_in_column)) * self.block_height
                rowOfImage_end = rowOfImage_start + self.block_height - 1
                U_in_image_form[rowOfImage_start:rowOfImage_end+1, columnOfImage_start:columnOfImage_end+1, component_index] = U_of_block_oneComponent
        return U_in_image_form

    def set_singular_values_one(self, matrix):
        # if np.isnan(matrix).any():
        #     print("hhhhh")
        # if np.isinf(matrix).any():
        #     print("kkkkk")
        Q, Sigma, Omega_h = LA.svd(matrix, full_matrices=False)
        Sigma = np.diag(np.ones((self.n_components, 1)).ravel())
        matrix_projected = Q.dot(Sigma).dot(Omega_h)
        return matrix_projected

    def f(self, block, U):
        # block: q * 1 vector
        # U: q * p matrix
        q = block.shape[0]
        numinator = (block.T).dot(np.eye(q) - (U.dot(U.T))).dot(block)
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        denominator = (block.T).dot(np.eye(q) + (U.dot(U.T))).dot(block) + c
        f = numinator / denominator
        return f

    def G(self, block, U):
        # block: q * 1 vector
        # U: q * p matrix
        q = block.shape[0]
        numinator = -2 * (1 + self.f(block, U))
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        denominator = (LA.norm(block))**2 + (LA.norm(U.dot(U.T).dot(block)))**2 + c
        G = (numinator / denominator) * (block.dot(block.T).dot(U))
        return G

    def f_kernel(self, image_index, block_index, Theta):
        # Theta: n * p matrix
        kernel_matrix = self.kernel_of_blocks[:, :, block_index]
        kernel_vector = (kernel_matrix[:, image_index]).reshape((-1,1))
        kernel_scalar = kernel_matrix[image_index, image_index]
        q = self.block_dimension
        numinator = kernel_scalar - (kernel_vector.T).dot(Theta).dot(Theta.T).dot(kernel_vector)
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        denominator = kernel_scalar + (kernel_vector.T).dot(Theta).dot(Theta.T).dot(kernel_vector) + c
        f = numinator / denominator
        return f

    def G_kernel(self, image_index, block_index, Theta):
        # Theta: n * p matrix
        kernel_matrix = self.kernel_of_blocks[:, :, block_index]
        kernel_vector = (kernel_matrix[:, image_index]).reshape((-1, 1))
        kernel_scalar = kernel_matrix[image_index, image_index]
        q = self.block_dimension
        numinator = -2 * (1 + self.f_kernel(image_index, block_index, Theta))
        L = 1
        c2 = (0.03 * L) ** 2
        c = (q - 1) * c2
        denominator = kernel_scalar + (kernel_vector.T).dot(Theta).dot(Theta.T).dot(kernel_vector) + c
        G = (numinator / denominator) * (kernel_vector.dot(kernel_vector.T).dot(Theta))
        return G

    def divide_images_to_their_blocks(self, X):
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
                blocks[:, block_index, image_index] = block.ravel()
        return blocks

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

    # def transform(self, X):
    #     # X: rows are features and columns are samples
    #     self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
    #     X = X - self.mean_of_X
    #     X_transformed = (self.U.T).dot(X)
    #     return X_transformed
    #
    # def transform_outOfSample(self, x):
    #     # x: a vector
    #     x = np.reshape(x,(-1,1))
    #     x = x - self.mean_of_X
    #     x_transformed = (self.U.T).dot(x)
    #     return x_transformed
    #
    # def get_projection_directions(self):
    #     return self.U
    #
    # def reconstruct(self, X, using_howMany_projection_directions=None):
    #     # X: rows are features and columns are samples
    #     if using_howMany_projection_directions != None:
    #         U = self.U[:, 0:using_howMany_projection_directions]
    #     else:
    #         U = self.U
    #     X = X - self.mean_of_X
    #     X_transformed = (U.T).dot(X)
    #     X_reconstructed = U.dot(X_transformed)
    #     X_reconstructed = X_reconstructed + self.mean_of_X
    #     return X_reconstructed
    #
    # def reconstruct_outOfSample(self, x, using_howMany_projection_directions=None):
    #     # x: a vector
    #     x = np.reshape(x, (-1, 1))
    #     x = x - self.mean_of_X
    #     if using_howMany_projection_directions != None:
    #         U = self.U[:, 0:using_howMany_projection_directions]
    #     else:
    #         U = self.U
    #     x_transformed = (U.T).dot(x)
    #     x_reconstructed = U.dot(x_transformed)
    #     x_reconstructed = x_reconstructed + self.mean_of_X
    #     return x_reconstructed



