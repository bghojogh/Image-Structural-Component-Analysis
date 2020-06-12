import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_kernels
from my_SSIM import My_SSIM
import os
import pickle
import h5py

# ----- python fast kernel matrix:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
# https://stackoverflow.com/questions/7391779/fast-kernel-matrix-computation-python
# https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# https://stackoverflow.com/questions/36324561/fast-way-to-calculate-kernel-matrix-python?rq=1

# ----- python fast scatter matrix:
# https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation


class My_kernel_PCA_SSIM:

    def __init__(self, image_height, image_width, n_components=None):
        self.n_components = n_components
        self.image_height = image_height
        self.image_width = image_width
        self.X = None
        self.S = None
        self.V = None

    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed

    def fit(self, X):
        # X: rows are features and columns are samples
        self.X = X
        # kernel_X_X = self.SSIM_kernel_2(matrix1=X, matrix2=X)
        SSIM_distance_matrix = self.read_SSIM_distance_matrix()
        SSIM_distance_matrix_centered = self.center_the_matrix(the_matrix=SSIM_distance_matrix, mode="double_center")
        SSIM_kernel = -0.5 * SSIM_distance_matrix_centered
        eig_val, eig_vec = LA.eigh(SSIM_kernel)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            V = eig_vec[:, :self.n_components]
            s = eig_val[:self.n_components]
        else:
            V = eig_vec
            s = eig_val
        s = np.asarray(s)
        s = s ** 0.5
        s = np.nan_to_num(s)
        S = np.diag(s)
        self.S = S
        self.V = V

    def transform(self, X):
        # X: rows are features and columns are samples
        X_transformed = (self.S).dot(self.V.T)
        return X_transformed

    def transform_outOfSample(self, x, test_image_index):
        # x: a vector
        x = np.reshape(x,(-1,1))
        SSIM_distance_matrix = self.load_variable(name_of_variable="distance_index", path='./kernel_SSIM_3/')
        SSIM_distance_vector = SSIM_distance_matrix[:, test_image_index].reshape((-1,1))
        SSIM_distance_vector_centered = self.center_the_matrix(the_matrix=SSIM_distance_vector, mode="remove_mean_of_rows_from_rows")
        SSIM_kernel = -0.5 * SSIM_distance_vector_centered
        diag_S = np.diag(self.S) + 0.00001
        self.S = np.diag(diag_S)
        x_transformed = inv(self.S).dot(self.V.T).dot(SSIM_kernel)
        return x_transformed

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

    def SSIM_kernel(self, matrix1, matrix2):
        # matrix1: rows are features and columns are samples
        # matrix2: rows are features and columns are samples
        n_samples_matrix1 = matrix1.shape[1]
        n_samples_matrix2 = matrix2.shape[1]
        kernel = np.zeros((n_samples_matrix1, n_samples_matrix2))
        my_SSIM = My_SSIM(window_size=8)
        for matrix1_sample_index in range(n_samples_matrix1):
            for matrix2_sample_index in range(matrix1_sample_index, n_samples_matrix2):
                image1 = matrix1[:, matrix1_sample_index].reshape((self.image_height, self.image_width))
                image2 = matrix2[:, matrix2_sample_index].reshape((self.image_height, self.image_width))
                SSIM_index, _ = my_SSIM.SSIM_index(image1=image1, image2=image2)
                print("SSIM index between image " + str(matrix1_sample_index) + " and " + str(matrix2_sample_index) + " = " + str(SSIM_index))
                kernel[matrix1_sample_index, matrix2_sample_index] = SSIM_index
                kernel[matrix2_sample_index, matrix1_sample_index] = SSIM_index
        self.save_variable(variable=kernel, name_of_variable="kernel", path_to_save='./kernel_SSIM/')
        self.save_np_array_to_txt(variable=kernel, name_of_variable="kernel", path_to_save='./kernel_SSIM/')
        self.load_variable(name_of_variable="kernel", path='./kernel_SSIM/')
        input("hi")
        return kernel

    def SSIM_kernel_2(self, matrix1, matrix2):
        # matrix1: rows are features and columns are samples
        # matrix2: rows are features and columns are samples
        n_samples_matrix1 = matrix1.shape[1]
        n_samples_matrix2 = matrix2.shape[1]
        n_features = matrix1.shape[0]
        n_nonRepeated_elements = int(np.ceil((n_samples_matrix1 * (n_samples_matrix1+1)) / 2))
        SSIM_index = np.zeros((n_samples_matrix1, n_samples_matrix2))
        SSIM = np.zeros((n_nonRepeated_elements, 2 + n_features))
        distance_index = np.zeros((n_samples_matrix1, n_samples_matrix2))
        distance = np.zeros((n_nonRepeated_elements, 2 + n_features))
        distance_index_MeanRemoved = np.zeros((n_samples_matrix1, n_samples_matrix2))
        distance_MeanRemoved = np.zeros((n_nonRepeated_elements, 2 + n_features))
        # luminance_score = np.zeros((n_nonRepeated_elements, 2 + n_features))
        # contrast_score = np.zeros((n_nonRepeated_elements, 2 + n_features))
        # structure_score = np.zeros((n_nonRepeated_elements, 2 + n_features))
        my_SSIM = My_SSIM(window_size=8)
        counter = -1
        for matrix1_sample_index in range(n_samples_matrix1):
            for matrix2_sample_index in range(matrix1_sample_index, n_samples_matrix2):
                counter = counter + 1
                image1 = matrix1[:, matrix1_sample_index].reshape((self.image_height, self.image_width))
                image2 = matrix2[:, matrix2_sample_index].reshape((self.image_height, self.image_width))
                SSIM_index_, SSIM_, distance_index_, distance_, distance_index_MeanRemoved_, distance_MeanRemoved_ = my_SSIM.SSIM_index_2(image1=image1, image2=image2)
                print("SSIM index between image " + str(matrix1_sample_index) + " and " + str(matrix2_sample_index) + " = " + str(SSIM_index_))
                print("Distance index between image " + str(matrix1_sample_index) + " and " + str(matrix2_sample_index) + " = " + str(distance_index_))
                print("Distance index (mean removed) between image " + str(matrix1_sample_index) + " and " + str(matrix2_sample_index) + " = " + str(distance_index_MeanRemoved_))
                SSIM_index[matrix1_sample_index, matrix2_sample_index] = SSIM_index_
                SSIM_index[matrix2_sample_index, matrix1_sample_index] = SSIM_index_
                SSIM[counter, 0] = matrix1_sample_index
                SSIM[counter, 1] = matrix2_sample_index
                SSIM[counter, 2:] = SSIM_.ravel()
                distance_index[matrix1_sample_index, matrix2_sample_index] = distance_index_
                distance_index[matrix2_sample_index, matrix1_sample_index] = distance_index_
                distance[counter, 0] = matrix1_sample_index
                distance[counter, 1] = matrix2_sample_index
                distance[counter, 2:] = distance_.ravel()
                distance_index_MeanRemoved[matrix1_sample_index, matrix2_sample_index] = distance_index_MeanRemoved_
                distance_index_MeanRemoved[matrix2_sample_index, matrix1_sample_index] = distance_index_MeanRemoved_
                distance_MeanRemoved[counter, 0] = matrix1_sample_index
                distance_MeanRemoved[counter, 1] = matrix2_sample_index
                distance_MeanRemoved[counter, 2:] = distance_MeanRemoved_.ravel()
                # luminance_score[counter, 0] = matrix1_sample_index
                # luminance_score[counter, 1] = matrix2_sample_index
                # luminance_score[counter, 2:] = luminance_score_.ravel()
                # contrast_score[counter, 0] = matrix1_sample_index
                # contrast_score[counter, 1] = matrix2_sample_index
                # contrast_score[counter, 2:] = contrast_score_.ravel()
                # structure_score[counter, 0] = matrix1_sample_index
                # structure_score[counter, 1] = matrix2_sample_index
                # structure_score[counter, 2:] = structure_score_.ravel()
        self.save_variable(variable=SSIM_index, name_of_variable="SSIM_index", path_to_save='./kernel_SSIM_2/')
        self.save_np_array_to_txt(variable=SSIM_index, name_of_variable="SSIM_index", path_to_save='./kernel_SSIM_2/')
        print("SSIM_index" + " saved...")
        self.save_variable(variable=distance_index, name_of_variable="distance_index", path_to_save='./kernel_SSIM_2/')
        self.save_np_array_to_txt(variable=distance_index, name_of_variable="distance_index", path_to_save='./kernel_SSIM_2/')
        print("distance_index" + " saved...")
        self.save_variable(variable=distance_index_MeanRemoved, name_of_variable="distance_index_MeanRemoved", path_to_save='./kernel_SSIM_2/')
        self.save_np_array_to_txt(variable=distance_index_MeanRemoved, name_of_variable="distance_index_MeanRemoved", path_to_save='./kernel_SSIM_2/')
        print("distance_index_MeanRemoved" + " saved...")
        first_third_length = int(np.floor(SSIM.shape[0] / 3))
        second_third_length = int(np.floor(2 * SSIM.shape[0] / 3))
        self.save_variable_large(variable=SSIM[:first_third_length, :], name_of_variable="SSIM_1", path_to_save='./kernel_SSIM_2/')
        input("stop..., enter a key:")
        self.save_variable_large(variable=SSIM[first_third_length:second_third_length, :], name_of_variable="SSIM_2", path_to_save='./kernel_SSIM_2/')
        input("stop..., enter a key:")
        self.save_variable_large(variable=SSIM[second_third_length:, :], name_of_variable="SSIM_3", path_to_save='./kernel_SSIM_2/')
        print("SSIM" + " saved...")
        input("stop..., enter a key:")
        self.save_variable_large(variable=distance[:first_third_length, :], name_of_variable="distance_1", path_to_save='./kernel_SSIM_2/')
        input("stop..., enter a key:")
        self.save_variable_large(variable=distance[first_third_length:second_third_length, :], name_of_variable="distance_2", path_to_save='./kernel_SSIM_2/')
        input("stop..., enter a key:")
        self.save_variable_large(variable=distance[second_third_length:, :], name_of_variable="distance_3", path_to_save='./kernel_SSIM_2/')
        print("distance" + " saved...")
        input("stop..., enter a key:")
        self.save_variable_large(variable=distance_MeanRemoved[:first_third_length, :], name_of_variable="distance_MeanRemoved_1", path_to_save='./kernel_SSIM_2/')
        input("stop..., enter a key:")
        self.save_variable_large(variable=distance_MeanRemoved[first_third_length:second_third_length, :], name_of_variable="distance_MeanRemoved_2", path_to_save='./kernel_SSIM_2/')
        input("stop..., enter a key:")
        self.save_variable_large(variable=distance_MeanRemoved[second_third_length:, :], name_of_variable="distance_MeanRemoved_3", path_to_save='./kernel_SSIM_2/')
        print("distance_MeanRemoved" + " saved...")
        input("stop..., enter a key:")
        self.load_variable(name_of_variable="SSIM_index", path='./kernel_SSIM_2/')
        self.load_variable(name_of_variable="distance_index", path='./kernel_SSIM_2/')
        self.load_variable(name_of_variable="distance_index_MeanRemoved", path='./kernel_SSIM_2/')
        self.load_variable_large(name_of_variable="SSIM_1", path='./kernel_SSIM_2/')
        self.load_variable_large(name_of_variable="distance", path='./kernel_SSIM_2/')
        self.load_variable_large(name_of_variable="distance_MeanRemoved", path='./kernel_SSIM_2/')
        input("stop..., enter a key:")
        return SSIM_index

    def SSIM_kernel_3(self, matrix1, matrix2):
        # matrix1: rows are features and columns are samples
        # matrix2: rows are features and columns are samples
        n_samples_matrix1 = matrix1.shape[1]
        n_samples_matrix2 = matrix2.shape[1]
        distance_index = np.zeros((n_samples_matrix1, n_samples_matrix2))
        my_SSIM = My_SSIM(window_size=8)
        counter = -1
        for matrix1_sample_index in range(n_samples_matrix1):
            for matrix2_sample_index in range(n_samples_matrix2):
                counter = counter + 1
                image1 = matrix1[:, matrix1_sample_index].reshape((self.image_height, self.image_width))
                image2 = matrix2[:, matrix2_sample_index].reshape((self.image_height, self.image_width))
                distance_index_ = my_SSIM.SSIM_index_3(image1=image1, image2=image2)
                print("Distance index between image " + str(matrix1_sample_index) + " and " + str(matrix2_sample_index) + " = " + str(distance_index_))
                distance_index[matrix1_sample_index, matrix2_sample_index] = distance_index_
        self.save_variable(variable=distance_index, name_of_variable="distance_index", path_to_save='./kernel_SSIM_3/')
        self.save_np_array_to_txt(variable=distance_index, name_of_variable="distance_index", path_to_save='./kernel_SSIM_3/')
        print("distance_index" + " saved...")
        input("stop..., enter a key:")
        self.load_variable(name_of_variable="distance_index", path='./kernel_SSIM_3/')
        input("stop..., enter a key:")
        return distance_index

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def save_variable_large(self, variable, name_of_variable, path_to_save='./'):
        # I googled: python save large array
        # https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py/20938742#20938742
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        # https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
        with h5py.File(path_to_save + name_of_variable + ".h5", 'w') as hf:
            hf.create_dataset(path_to_save + name_of_variable, data=variable)

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def load_variable_large(self, name_of_variable, path='./'):
        # I googled: python save large array
        # https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py/20938742#20938742
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        # https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
        with h5py.File(path + name_of_variable + ".h5", 'r') as hf:
            variable = hf[path + name_of_variable][:]
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(
                path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))