import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from my_PCA import My_PCA
from my_dual_PCA import My_dual_PCA
from my_kernel_PCA import My_kernel_PCA
from my_supervised_PCA import My_supervised_PCA
from my_dual_supervised_PCA import My_dual_supervised_PCA
from my_kernel_supervised_PCA_UsingDual import My_kernel_supervised_PCA_UsingDual
from my_kernel_supervised_PCA_UsingDirect import My_kernel_supervised_PCA_UsingDirect
from my_MDS import My_MDS
from my_Isomap import My_Isomap
from my_Laplacian_eigenmap import My_Laplacian_eigenmap
from my_kernel_PCA_SSIM import My_kernel_PCA_SSIM
from my_MDS_SSIM import My_MDS_SSIM
from sklearn.metrics.pairwise import pairwise_kernels
from my_dual_supervised_PCA_SSIM import My_dual_supervised_PCA_SSIM
from my_kernel_supervised_PCA_UsingDual_SSIM import My_kernel_supervised_PCA_UsingDual_SSIM
from my_kernel_supervised_PCA_UsingDirect_SSIM import My_kernel_supervised_PCA_UsingDirect_SSIM
from my_Laplacian_eigenmap_SSIM import My_Laplacian_eigenmap_SSIM
from my_ISCA import My_ISCA
from my_LLISE import My_LLISE
from my_LLE import My_LLE
from my_SNE import My_SNE
from sklearn.decomposition import PCA

from add_distortion import Add_distortion
import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import pandas as pd
import scipy.io
import csv
import scipy.misc
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import itertools
from struct import *




def main():
    # ---- settings:
    dataset = "MNIST"  #--> Frey, ATT, dataset_distorted
    create_noisy_dataset_again = False
    create_noisy_test_dataset_again = False
    calculate_SSIM_distance_matrix_again = False
    manifold_learning_method = "SNE" #--> PCA, dual_PCA, kernel_PCA, supervised_PCA, dual_supervised_PCA, kernel_SPCA_UsingDual, kernel_SPCA_UsingDirect, MDS, Isomap, Laplacian_eigenmap, kernel_PCA_SSIM, MDS_SSIM, dual_supervised_PCA_SSIM
    kernel = "linear"  #kernel over data (X) --> ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’ --> if None, it is linear
    kernel_on_labels_in_SPCA = "linear"  #kernel over labels (Y) --> ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’ --> if None, it is linear
    n_neighbors_in_KNN = 5
    convert_mat2csv_again = False
    show_an_image = False
    save_image_dataset_again = False
    save_projection_directions_again = False
    n_projection_directions_to_save = 10 #--> an integer >= 1, if None: save all "specified" directions when creating the python class
    save_reconstructed_images_again = False
    indices_reconstructed_images_to_save = [0,120]  #--> list of two indices (start and end), e.g. [100,120] --> if None, save all of them
    reconstruct_using_howMany_projection_directions = None #--> an integer >= 1, if None: using all "specified" directions when creating the python class
    plot_projected_pointsAndImages_again = True
    which_dimensions_to_plot_inpointsAndImagesPlot = [0,1] #--> list of two indices (start and end), e.g. [1,3] or [0,1]
    project_out_of_samples = False
    classify_distortions = False
    classify_test_distortions = False
    save_reconstructed_test_images_again = False


    if create_noisy_dataset_again == True and dataset == "dataset_distorted":
        # reading original image:
        image_original = load_image(address_image="./lena_gray.gif")
        dataset, MSE_of_images, distortion_type_of_images = create_noisy_dataset(image_original=image_original, n_images_per_distortion_type=20)
        # save dataset:
        save_variable(variable=MSE_of_images, name_of_variable="MSE_of_images", path_to_save="./dataset_distorted_MSE/")
        save_np_array_to_txt(variable=MSE_of_images, name_of_variable="MSE_of_images", path_to_save="./dataset_distorted_MSE/")
        save_variable(variable=distortion_type_of_images, name_of_variable="distortion_type_of_images", path_to_save="./dataset_distorted_MSE/")
        save_np_array_to_txt(variable=distortion_type_of_images, name_of_variable="distortion_type_of_images", path_to_save="./dataset_distorted_MSE/")
        image_height = image_original.shape[0]
        image_width = image_original.shape[1]
        n_samples = dataset.shape[1]
        for image_index in range(n_samples):
            sample = dataset[:, image_index].reshape((image_height, image_width))
            save_image(image_array=sample, path_without_file_name="./dataset_distorted/", file_name=str(image_index) + ".tif")

    if create_noisy_test_dataset_again == True and dataset == "dataset_distorted":
        # reading original image:
        image_original = load_image(address_image="./lena_gray.gif")
        dataset_test, MSE_of_images, distortion_type_of_images = create_noisy_test_dataset(image_original=image_original, desired_MSE=500, n_images_per_distortion_type=1)
        # save dataset:
        save_variable(variable=MSE_of_images, name_of_variable="MSE_of_images", path_to_save="./test_dataset_distorted_MSE/")
        save_np_array_to_txt(variable=MSE_of_images, name_of_variable="MSE_of_images", path_to_save="./test_dataset_distorted_MSE/")
        save_variable(variable=distortion_type_of_images, name_of_variable="distortion_type_of_images", path_to_save="./test_dataset_distorted_MSE/")
        save_np_array_to_txt(variable=distortion_type_of_images, name_of_variable="distortion_type_of_images", path_to_save="./test_dataset_distorted_MSE/")
        image_height = image_original.shape[0]
        image_width = image_original.shape[1]
        n_samples = dataset_test.shape[1]
        for image_index in range(n_samples):
            sample = dataset_test[:, image_index].reshape((image_height, image_width))
            save_image(image_array=sample, path_without_file_name="./test_dataset_distorted/", file_name=str(image_index) + ".tif")

    if dataset == "Frey":
        path_dataset = "./Frey_dataset/frey_rawface"
        # ---- read mat dataset and convert to csv:
        if convert_mat2csv_again:
            convert_mat_to_csv(path_mat=path_dataset, path_to_save=path_dataset)
        # ---- read the csv dataset:
        data = read_csv_file(path=path_dataset+".csv")
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- change range of images from [0,255] to [0,1]:
        data = data/255
        # # studentize:
        # scaler = StandardScaler().fit(data.T)
        # data = (scaler.transform(data.T)).T
        # ---- image settings:
        n_samples = data.shape[1]
        image_height = 28
        image_width = 20
        # ---- show one of the images:
        if show_an_image:
            an_image = data[:, 0].reshape((image_height, image_width))
            plt.imshow(an_image, cmap='gray')
            plt.colorbar()
            plt.show()
        # ---- save the images in a folder:
        if save_image_dataset_again:
            for image_index in range(n_samples):
                an_image = data[:, image_index].reshape((image_height, image_width))
                # scale (resize) image array:
                an_image = scipy.misc.imresize(arr=an_image, size=500)  #--> 5 times bigger
                # save image:
                save_image(image_array=an_image, path_without_file_name="./Frey_dataset/images/", file_name=str(image_index)+".png")
    elif dataset == "ATT":
        path_dataset = "./Att_dataset/"
        n_samples = 400
        image_height = 112
        image_width = 92
        data = np.zeros((image_height*image_width, n_samples))
        labels = np.zeros((1, n_samples))
        for image_index in range(n_samples):
            img = load_image(address_image=path_dataset+str(image_index+1)+".jpg")
            data[:, image_index] = img.ravel()
            labels[:, image_index] = math.floor(image_index/10) + 1
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- change range of images from [0,255] to [0,1]:
        data = data / 255
        # ---- show one of the images:
        if show_an_image:
            an_image = data[:, 0].reshape((image_height, image_width))
            plt.imshow(an_image, cmap='gray')
            plt.colorbar()
            plt.show()
    elif dataset == 'MNIST':
        load_dataset_again = False
        subset_of_MNIST = True
        pick_subset_of_MNIST_again = True
        MNIST_subset_cardinality_training = 100
        MNIST_subset_cardinality_testing = 10
        path_dataset = './input/mnist/'
        path_dataset_save = './MNIST_dataset/'
        if load_dataset_again:
            training_data = list(read_MNIST_dataset(dataset = "training", path = path_dataset))
            testing_data = list(read_MNIST_dataset(dataset = "testing", path = path_dataset))

            number_of_training_samples = len(training_data)
            dimension_of_data = 28 * 28
            X_train = np.empty((0, dimension_of_data))
            y_train = np.empty((0, 1))
            for sample_index in range(number_of_training_samples):
                if np.mod(sample_index, 1) == 0:
                    print('sample ' + str(sample_index) + ' from ' + str(number_of_training_samples) + ' samples...')
                label, pixels = training_data[sample_index]
                pixels_reshaped = np.reshape(pixels, (1, 28*28))
                X_train = np.vstack([X_train, pixels_reshaped])
                y_train = np.vstack([y_train, label])
            y_train = y_train.ravel()

            number_of_testing_samples = len(testing_data)
            dimension_of_data = 28 * 28
            X_test = np.empty((0, dimension_of_data))
            y_test = np.empty((0, 1))
            for sample_index in range(number_of_testing_samples):
                if np.mod(sample_index, 1) == 0:
                    print('sample ' + str(sample_index) + ' from ' + str(number_of_testing_samples) + ' samples...')
                label, pixels = testing_data[sample_index]
                pixels_reshaped = np.reshape(pixels, (1, 28*28))
                X_test = np.vstack([X_test, pixels_reshaped])
                y_test = np.vstack([y_test, label])
            y_test = y_test.ravel()

            save_variable(X_train, 'X_train', path_to_save=path_dataset_save)
            save_variable(y_train, 'y_train', path_to_save=path_dataset_save)
            save_variable(X_test, 'X_test', path_to_save=path_dataset_save)
            save_variable(y_test, 'y_test', path_to_save=path_dataset_save)
        else:
            file = open(path_dataset_save+'X_train.pckl','rb')
            X_train = pickle.load(file); file.close()
            file = open(path_dataset_save+'y_train.pckl','rb')
            y_train = pickle.load(file); file.close()
            file = open(path_dataset_save+'X_test.pckl','rb')
            X_test = pickle.load(file); file.close()
            file = open(path_dataset_save+'y_test.pckl','rb')
            y_test = pickle.load(file); file.close()

        if subset_of_MNIST:
            if pick_subset_of_MNIST_again:
                dimension_of_data = 28 * 28
                X_train_picked = np.empty((0, dimension_of_data))
                y_train_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_train[y_train == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_training, :]
                    X_train_picked = np.vstack((X_train_picked, X_class_picked))
                    y_class = y_train[y_train == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_training].reshape((-1, 1))
                    y_train_picked = np.vstack((y_train_picked, y_class_picked))
                y_train_picked = y_train_picked.ravel()
                X_test_picked = np.empty((0, dimension_of_data))
                y_test_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_test[y_test == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_testing, :]
                    X_test_picked = np.vstack((X_test_picked, X_class_picked))
                    y_class = y_test[y_test == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_testing].reshape((-1, 1))
                    y_test_picked = np.vstack((y_test_picked, y_class_picked))
                y_test_picked = y_test_picked.ravel()
                # X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                # X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                # y_train_picked = y_train[0:MNIST_subset_cardinality_training]
                # y_test_picked = y_test[0:MNIST_subset_cardinality_testing]
                save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset_save)
                save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset_save)
                save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset_save)
                save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset_save)
            else:
                file = open(path_dataset_save+'X_train_picked.pckl','rb')
                X_train_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+'X_test_picked.pckl','rb')
                X_test_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+'y_train_picked.pckl','rb')
                y_train_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+'y_test_picked.pckl','rb')
                y_test_picked = pickle.load(file); file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            y_train = y_train_picked
            y_test = y_test_picked
        data = X_train.T / 255
        data_test = X_test.T / 255
        labels = y_train.reshape((1, -1))
        n_samples = data.shape[1]
        image_height = 28
        image_width = 28
    elif dataset == "dataset_distorted":
        path_dataset = "./dataset_distorted/"
        n_samples = 121
        image_height = 512
        image_width = 512
        data = np.zeros((image_height * image_width, n_samples))
        labels = np.zeros((1, n_samples))
        for image_index in range(n_samples):
            img = load_image(address_image=path_dataset + str(image_index) + ".tif")
            data[:, image_index] = img.ravel()
        # ---- labels:
        if manifold_learning_method == "supervised_PCA" or manifold_learning_method == "dual_supervised_PCA" \
            or manifold_learning_method == "kernel_SPCA_UsingDual" or manifold_learning_method == "kernel_SPCA_UsingDirect":
            kernel_X_X = pairwise_kernels(X=data.T, Y=data.T, metric=kernel)
            labels[:, :] = kernel_X_X[0, :]
        elif manifold_learning_method == "dual_supervised_PCA_SSIM":
            SSIM_distance_matrix = read_SSIM_distance_matrix()
            SSIM_distance_matrix_centered = center_the_matrix(the_matrix=SSIM_distance_matrix, mode="double_center")
            SSIM_kernel = -0.5 * SSIM_distance_matrix_centered
            labels[:, :] = SSIM_kernel[0, :]
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- change range of images from [0,255] to [0,1]:
        data = data / 255
        # ---- show one of the images:
        if show_an_image:
            an_image = data[:, 0].reshape((image_height, image_width))
            plt.imshow(an_image, cmap='gray')
            plt.colorbar()
            plt.show()
        # ---- test dataset:
        path_dataset = "./test_dataset_distorted/"
        n_samples_test = 12
        data_test = np.zeros((image_height * image_width, n_samples_test))
        for image_index in range(n_samples_test):
            img = load_image(address_image=path_dataset + str(image_index) + ".tif")
            data_test[:, image_index] = img.ravel()
        # ---- cast dataset from string to float:
        data_test = data_test.astype(np.float)
        # ---- change range of images from [0,255] to [0,1]:
        data_test = data_test / 255
        # ---- standardize all dataset:
        # data_train_and_test = np.hstack((data, data_test))
        # scaler = StandardScaler().fit(data_train_and_test.T)
        # data_train_and_test = (scaler.transform(data_train_and_test.T)).T
        # data = data_train_and_test[:, :data.shape[1]]
        # data_test = data_train_and_test[:, data.shape[1]:]

    if calculate_SSIM_distance_matrix_again:
        my_kernel_PCA_SSIM = My_kernel_PCA_SSIM(image_height=image_height, image_width=image_width, n_components=None)
        my_kernel_PCA_SSIM.SSIM_kernel_3(matrix1=data, matrix2=data_test)

    # ---- apply manifold learning:
    if manifold_learning_method == "PCA":
        my_manifold_learning = My_PCA(n_components=None)
        data_transformed = my_manifold_learning.fit_transform(X=data)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "dual_PCA":
        my_manifold_learning = My_dual_PCA(n_components=None)
        data_transformed = my_manifold_learning.fit_transform(X=data)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "kernel_PCA":
        my_manifold_learning = My_kernel_PCA(n_components=None, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=data)
    elif manifold_learning_method == "supervised_PCA":
        my_manifold_learning = My_supervised_PCA(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA)
        data_transformed = my_manifold_learning.fit_transform(X=data, Y=labels)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "dual_supervised_PCA":
        my_manifold_learning = My_dual_supervised_PCA(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA)
        data_transformed = my_manifold_learning.fit_transform(X=data, Y=labels)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "kernel_SPCA_UsingDual":
        my_manifold_learning = My_kernel_supervised_PCA_UsingDual(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=data, Y=labels)
    elif manifold_learning_method == "kernel_SPCA_UsingDirect":
        my_manifold_learning = My_kernel_supervised_PCA_UsingDirect(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=data, Y=labels)
    elif manifold_learning_method == "MDS":
        my_manifold_learning = My_MDS(n_components=None, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=data)
    elif manifold_learning_method == "Isomap":
        my_manifold_learning = My_Isomap(n_components=None, n_neighbors=n_neighbors_in_KNN, n_jobs=-1)
        data_transformed = my_manifold_learning.fit_transform(X=data)
    elif manifold_learning_method == "Laplacian_eigenmap":
        my_manifold_learning = My_Laplacian_eigenmap(n_components=None, n_neighbors=5)
        data_transformed = my_manifold_learning.fit_transform(X=data)
    elif manifold_learning_method == "kernel_PCA_SSIM":
        my_manifold_learning = My_kernel_PCA_SSIM(image_height=image_height, image_width=image_width, n_components=None)
        data_transformed = my_manifold_learning.fit_transform(X=data)
    elif manifold_learning_method == "MDS_SSIM":
        my_manifold_learning = My_MDS_SSIM(n_components=None)
        data_transformed = my_manifold_learning.fit_transform(X=data)
    elif manifold_learning_method == "dual_supervised_PCA_SSIM":
        my_manifold_learning = My_dual_supervised_PCA_SSIM(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA)
        data_transformed = my_manifold_learning.fit_transform(X=data, Y=labels)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "kernel_SPCA_UsingDual_SSIM":
        my_manifold_learning = My_kernel_supervised_PCA_UsingDual_SSIM(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=data, Y=labels)
    elif manifold_learning_method == "kernel_SPCA_UsingDirect_SSIM":
        my_manifold_learning = My_kernel_supervised_PCA_UsingDirect_SSIM(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=data, Y=labels)
    elif manifold_learning_method == "Laplacian_eigenmap_SSIM":
        my_manifold_learning = My_Laplacian_eigenmap_SSIM(n_components=None, n_neighbors=5)
        data_transformed = my_manifold_learning.fit_transform(X=data)
    elif manifold_learning_method == "ISCA":
        my_manifold_learning = My_ISCA(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
        # my_manifold_learning.train_U_by_ADMM()
        # my_manifold_learning.unify_subspaces(U=None, which_U_epoch=199)
        # data_transformed = my_manifold_learning.ISCA_transform()
        pass
    elif manifold_learning_method == "kernel_ISCA":
        # my_manifold_learning = My_ISCA(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
        # my_manifold_learning.train_Theta_by_ADMM_kernel()
        pass
    elif manifold_learning_method == "LLISE":
        my_manifold_learning = My_LLISE(X=data, image_height=image_height, image_width=image_width, n_neighbors=10, n_components=4, block_height=8, block_width=8, kernel=kernel)
        my_manifold_learning.LLISE_find_KNN(calculate_again=False)
        my_manifold_learning.linear_reconstruction_ADMM(method="LLISE", calculate_again=False, max_epochs=None, step_checkpoint=10)
        my_manifold_learning.linear_embedding_ADMM(method="LLISE", calculate_again=False, max_epochs=None, step_checkpoint=10)
        # input("hi...")
    elif manifold_learning_method == "kernel_LLISE":
        my_manifold_learning = My_LLISE(X=data, image_height=image_height, image_width=image_width, n_neighbors=10, n_components=4, block_height=8, block_width=8, kernel=kernel)
        my_manifold_learning.kernel_LLISE_find_KNN(calculate_again=False)
        my_manifold_learning.linear_reconstruction_ADMM(method="kernel_LLISE", calculate_again=False, max_epochs=None, step_checkpoint=10)
        my_manifold_learning.linear_embedding_ADMM(method="kernel_LLISE", calculate_again=False, max_epochs=None, step_checkpoint=10)
        # input("hi")
    elif manifold_learning_method == "LLE":
        my_manifold_learning = My_LLE(X=data, n_neighbors=10, n_components=4, kernel=kernel)
        data_transformed = my_manifold_learning.LLE_fit_transform()
        # input("hi")
    elif manifold_learning_method == "kernel_LLE":
        my_manifold_learning = My_LLE(X=data, n_neighbors=10, n_components=4, kernel=kernel)
        data_transformed = my_manifold_learning.kernel_LLE_fit_transform()
        # input("hi")
    elif manifold_learning_method == "SNE":
        # clf = PCA(n_components=50)
        # clf.fit(X=data.T)
        # X_train_projected = (clf.transform(X=data.T)).T
        # X_test_projected = (clf.transform(X=data_test.T)).T
        # my_manifold_learning = My_SNE(X=X_train_projected, n_components=2, kernel=kernel)
        my_manifold_learning = My_SNE(X=data, n_components=2, kernel=kernel)
        data_transformed = my_manifold_learning.SNE_embed(max_iterations=None, step_checkpoint=100, calculate_again=False)
        # input("hi")

    if classify_distortions:
        if manifold_learning_method == "ISCA":
            my_manifold_learning__ = My_ISCA(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            estimated_distortion_class = my_manifold_learning__.classify_distortion_trainingSet(method="ISCA", classify_again=True)
        elif manifold_learning_method == "kernel_ISCA":
            my_manifold_learning__ = My_ISCA(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            estimated_distortion_class = my_manifold_learning__.classify_distortion_trainingSet(method="kernel_ISCA", classify_again=True)
        elif manifold_learning_method == "dual_PCA":
            my_manifold_learning__ = My_dual_PCA(n_components=4)
            _ = my_manifold_learning__.fit_transform(X=data)
            estimated_distortion_class = my_manifold_learning__.classify_distortion_trainingSet()
        elif manifold_learning_method == "kernel_PCA":
            my_manifold_learning__ = My_kernel_PCA(n_components=4, kernel=kernel)
            _ = my_manifold_learning__.fit_transform(X=data)
            estimated_distortion_class = my_manifold_learning__.classify_distortion_trainingSet()
        elif manifold_learning_method == "LLISE":
            my_manifold_learning__ = My_LLISE(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            estimated_distortion_class = my_manifold_learning__.classify_distortion_trainingSet(method="LLISE", classify_again=True)
        elif manifold_learning_method == "kernel_LLISE":
            my_manifold_learning__ = My_LLISE(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            estimated_distortion_class = my_manifold_learning__.classify_distortion_trainingSet(method="kernel_LLISE", classify_again=False)
        elif manifold_learning_method == "LLE":
            estimated_distortion_class = my_manifold_learning.classify_distortion_trainingSet(X_transformed=data_transformed, method="LLE", classify_again=True)
        elif manifold_learning_method == "kernel_LLE":
            estimated_distortion_class = my_manifold_learning.classify_distortion_trainingSet(X_transformed=data_transformed, method="kernel_LLE", classify_again=True)
        true_distortion_class = np.zeros((n_samples, 1))
        for image_index in range(n_samples):
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
        cnf_matrix = confusion_matrix(y_true=true_distortion_class[:].ravel(), y_pred=estimated_distortion_class[:].ravel())
        cnf_matrix = cnf_matrix[1:, :]
        print("confusion matrix:")
        print(cnf_matrix)
        # class_names = ["contrast stretched", "Gaussian noise", "enhanced luminance", "Gaussian blurring", "impulse noise", "jpeg distortion"]
        class_names = [0, 1, 2, 3, 4, 5, 6]
        plot_confusion_matrix(confusion_matrix=cnf_matrix, class_names=class_names, normalize=False, cmap="gray_r")
        input("enter a key to continue: ")

    if classify_test_distortions:
        if manifold_learning_method == "ISCA":
            my_manifold_learning__ = My_ISCA(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            estimated_distortion_class_test = my_manifold_learning__.classify_distortion_testSet(X=data_test, method="ISCA", classify_again=True)
        elif manifold_learning_method == "kernel_ISCA":
            my_manifold_learning__ = My_ISCA(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            estimated_distortion_class_test = my_manifold_learning__.classify_distortion_testSet(X=data_test, method="kernel_ISCA", classify_again=True)
        elif manifold_learning_method == "dual_PCA":
            my_manifold_learning__ = My_dual_PCA(n_components=4)
            _ = my_manifold_learning__.fit_transform(X=data)
            my_manifold_learning__.classify_distortion_testSet(X=data_test)
        elif manifold_learning_method == "LLISE":
            my_manifold_learning__ = My_LLISE(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            my_manifold_learning__.LLISE_find_KNN_for_outOfSample(data_outOfSample=data_test, calculate_again=False)
            my_manifold_learning__.OutOfSample_linear_reconstruction_ADMM(method="LLISE", calculate_again=False, max_epochs=None, step_checkpoint=10)
            my_manifold_learning__.OutOfSample_linear_embedding_ADMM(method="LLISE", classify_again=False)
            my_manifold_learning__.classify_distortion_testSet(X=data_test, method="LLISE", classify_again=True)
        elif manifold_learning_method == "kernel_LLISE":
            my_manifold_learning__ = My_LLISE(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            my_manifold_learning__.kernel_LLISE_find_KNN_for_outOfSample(data_outOfSample=data_test, calculate_again=False)
            my_manifold_learning__.OutOfSample_linear_reconstruction_ADMM(method="kernel_LLISE", calculate_again=False, max_epochs=None, step_checkpoint=10)
            my_manifold_learning__.OutOfSample_linear_embedding_ADMM(method="kernel_LLISE", classify_again=False)
            my_manifold_learning__.classify_distortion_testSet(X=data_test, method="kernel_LLISE", classify_again=True, k=1)
        elif manifold_learning_method == "LLE":
            data_outOfSample_transformed = my_manifold_learning.LLE_fit_transform_outOfSample(data_outOfSample=data_test, X_training_transformed=data_transformed)
            my_manifold_learning.classify_distortion_testSet(data_outOfSample_transformed=data_outOfSample_transformed, X_training_transformed=data_transformed, method="LLE", classify_again=True, k=1)
        elif manifold_learning_method == "kernel_LLE":
            data_outOfSample_transformed = my_manifold_learning.kernel_LLE_fit_transform_outOfSample(data_outOfSample=data_test, X_training_transformed=data_transformed)
            my_manifold_learning.classify_distortion_testSet(data_outOfSample_transformed=data_outOfSample_transformed, X_training_transformed=data_transformed, method="kernel_LLE", classify_again=True, k=1)
        input("enter a key to continue: ")

    if project_out_of_samples:
        data_test_transformed = np.zeros((data_transformed.shape[0], n_samples_test))
        if manifold_learning_method == "PCA":
            for test_sample_index in range(data_test.shape[1]):
                data_test_transformed[:, test_sample_index] = (my_manifold_learning.transform_outOfSample(x=data_test[:, test_sample_index])).ravel()
        elif manifold_learning_method == "dual_PCA":
            for test_sample_index in range(data_test.shape[1]):
                data_test_transformed[:, test_sample_index] = (my_manifold_learning.transform_outOfSample(x=data_test[:, test_sample_index])).ravel()
        elif manifold_learning_method == "kernel_PCA":
            for test_sample_index in range(data_test.shape[1]):
                data_test_transformed[:, test_sample_index] = (my_manifold_learning.transform_outOfSample(x=data_test[:, test_sample_index])).ravel()
            # data_test_transformed[:, :] = my_manifold_learning.transform_outOfSample_matrix(X=data_test)
        elif manifold_learning_method == "kernel_PCA_SSIM":
            for test_sample_index in range(data_test.shape[1]):
                data_test_transformed[:, test_sample_index] = (my_manifold_learning.transform_outOfSample(x=data_test[:, test_sample_index], test_image_index=test_sample_index)).ravel()
        elif manifold_learning_method == "dual_supervised_PCA":
            for test_sample_index in range(data_test.shape[1]):
                data_test_transformed[:, test_sample_index] = (my_manifold_learning.transform_outOfSample(x=data_test[:, test_sample_index])).ravel()

    # ---- save projection directions:
    if save_projection_directions_again:
        if manifold_learning_method != "ISCA" and manifold_learning_method != "kernel_ISCA":
            if n_projection_directions_to_save == None:
                n_projection_directions_to_save = projection_directions.shape[1]
            for projection_direction_index in range(n_projection_directions_to_save):
                an_image = projection_directions[:, projection_direction_index].reshape((image_height, image_width))
                # scale (resize) image array:
                an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
                # save image:
                save_image(image_array=an_image, path_without_file_name="./output/"+manifold_learning_method+"/directions/", file_name=str(projection_direction_index)+".png")
        elif manifold_learning_method == "ISCA":
            my_manifold_learning__ = My_ISCA(X=data, image_height=image_height, image_width=image_width, n_components=4, block_height=8, block_width=8, kernel=kernel)
            which_U_or_Theta_epoch = 199
            component = 0
            # J:
            J = load_variable(name_of_variable="J_epochs_"+str(int((which_U_or_Theta_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/J/')
            J_in_image_form = my_manifold_learning__.make_U_in_image_form(J)
            plt.imshow(J_in_image_form[:, :, component], cmap='seismic') #--> cmap='gray', 'gray_r' --> https://matplotlib.org/examples/color/colormaps_reference.html
            plt.axis('off')
            plt.clim(-0.3, 0.3)
            plt.colorbar()
            plt.show()
            # U:
            U = load_variable(name_of_variable="U_epochs_"+str(int((which_U_or_Theta_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/U/')
            U_in_image_form = my_manifold_learning__.make_U_in_image_form(U)
            plt.imshow(U_in_image_form[:, :, component], cmap='seismic') #--> cmap='gray', 'gray_r'
            plt.axis('off')
            plt.clim(-0.3, 0.3)
            plt.colorbar()
            plt.show()
            # V:
            V = load_variable(name_of_variable="V_epochs_"+str(int((which_U_or_Theta_epoch+1)/10 - 1)), path='./ISCA_settings/ADMM/V/')
            V_in_image_form = my_manifold_learning__.make_U_in_image_form(V)
            plt.imshow(V_in_image_form[:, :, component], cmap='seismic') #--> cmap='gray', 'gray_r'
            plt.axis('off')
            plt.clim(-0.3, 0.3)
            plt.colorbar()
            plt.show()


    # ---- save reconstructed images:
    if save_reconstructed_images_again:
        if manifold_learning_method == "ISCA":
            X_reconstructed = my_manifold_learning.reconstruct(which_U_epoch=199)
        else:
            X_reconstructed = my_manifold_learning.reconstruct(X=data, using_howMany_projection_directions=reconstruct_using_howMany_projection_directions)
        if indices_reconstructed_images_to_save == None:
            indices_reconstructed_images_to_save = [0, X_reconstructed.shape[1]]
        for image_index in range(indices_reconstructed_images_to_save[0], indices_reconstructed_images_to_save[1]):
            an_image = X_reconstructed[:, image_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=100)  # --> size=500 means 5 times bigger
            # save image:
            save_image(image_array=an_image, path_without_file_name="./output/"+manifold_learning_method+"/reconstructed/", file_name=str(image_index)+".tif")

    # ---- save test reconstructed images:
    if save_reconstructed_test_images_again:
        if manifold_learning_method == "ISCA":
            X_test_reconstructed = my_manifold_learning.reconstruct_test(which_U_epoch=199, X=data_test)
        for image_index in range(data_test.shape[1]):
            an_image = X_test_reconstructed[:, image_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=100)  # --> 1 times bigger
            # save image:
            save_image(image_array=an_image, path_without_file_name="./output/"+manifold_learning_method+"/reconstructed_test/", file_name=str(image_index)+".tif")
        input("enter a key to continue: ")

    # Plotting the scatter plot of transformed distorted images:
    if dataset == "dataset_distorted":
        if project_out_of_samples:
            show_projected_test = True
        else:
            show_projected_test = False
            data_test_transformed = None
        # scatter_plot_transformed_distorted_images(data_transformed, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot)
        scatter_plot_transformed_distorted_images(data_transformed, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
                                                  data_test_transformed=data_test_transformed, show_projected_test=show_projected_test)

    # Plotting the embedded data:
    if dataset == "Frey":
        scale = 5
    elif dataset == "ATT":
        scale = 1
    elif dataset == "MNIST":
        scale = 3
    elif dataset == "dataset_distorted":
        scale = 1
    if plot_projected_pointsAndImages_again:
        dataset_notReshaped = np.zeros((n_samples, image_height*scale, image_width*scale))
        for image_index in range(n_samples):
            image = data[:, image_index]
            image_not_reshaped = image.reshape((image_height, image_width))
            image_not_reshaped_scaled = scipy.misc.imresize(arr=image_not_reshaped, size=scale*100)
            dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
        fig, ax = plt.subplots(figsize=(10, 10))
        # only take two dimensions to plot:
        # plot_components(X_projected=data_transformed, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
        #                 images=255-dataset_notReshaped, ax=ax, image_scale=0.3, markersize=10, thumb_frac=0.03, cmap='gray_r')
        # plot_components(X_projected=data_transformed,
        #                 which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
        #                 images=dataset_notReshaped, ax=ax, image_scale=0.3, markersize=10, thumb_frac=0.001,
        #                 cmap='gray_r')
        if dataset == "MNIST":
            scatter_plot_MNIST(data_transformed, which_dimensions_to_plot=[0,1], labels=labels.ravel(), data_test_transformed=None, show_projected_test=False)
            # plot_embedding(X=data_transformed, y=labels.ravel(), images=dataset_notReshaped, image_scale=0.2, title=None)

    # out-of-sample projection:
    if manifold_learning_method == "MDS_SSIM":
        data_transformed = my_manifold_learning.fit_transform(X=data)


def plot_confusion_matrix(confusion_matrix, class_names, normalize=False, cmap="gray"):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names, rotation=45)
    plt.xticks(tick_marks, class_names, rotation=0)
    # plt.yticks(tick_marks, class_names)
    tick_marks = np.arange(len(class_names)-1)
    plt.yticks(tick_marks, class_names[1:])
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.ylabel('true distortion type')
    plt.xlabel('predicted distortion type')
    plt.tight_layout()
    plt.show()

def read_SSIM_distance_matrix():
    SSIM_distance_matrix = load_variable(name_of_variable="distance_index", path='./kernel_SSIM_2/')
    return SSIM_distance_matrix

def center_the_matrix(the_matrix, mode="double_center"):
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

def scatter_plot_transformed_distorted_images(data_transformed, which_dimensions_to_plot, data_test_transformed=None, show_projected_test=False):
    for sample_index in range(data_transformed.shape[1]):
        # list of colors --> https: // matplotlib.org / examples / color / named_colors.html
        if sample_index >= 1 and sample_index <= 20:
            color = "green"
            if sample_index % 20 != 0:
                alpha = 1 - ((sample_index % 20) * (1.0 / 20.0)) + 0.01
            else:
                alpha = 1 - (20 * (1.0 / 20.0)) + 0.01
            if sample_index == 1:
                label = "contrast stretched"
            else:
                label = None
        if sample_index >= 21 and sample_index <= 40:
            color = "blue"
            if sample_index % 20 != 0:
                alpha = 1 - ((sample_index % 20) * (1.0 / 20.0)) + 0.01
            else:
                alpha = 1 - (20 * (1.0 / 20.0)) + 0.01
            if sample_index == 21:
                label = "Gaussian noise"
            else:
                label = None
        if sample_index >=41  and sample_index <= 60:
            color = "black"
            if sample_index % 20 != 0:
                alpha = 1 - ((sample_index % 20) * (1.0 / 20.0)) + 0.01
            else:
                alpha = 1 - (20 * (1.0 / 20.0)) + 0.01
            if sample_index == 41:
                label = "enhanced luminance"
            else:
                label = None
        if sample_index >= 61 and sample_index <= 80:
            color = "fuchsia"
            if sample_index % 20 != 0:
                alpha = 1 - ((sample_index % 20) * (1.0 / 20.0)) + 0.01
            else:
                alpha = 1 - (20 * (1.0 / 20.0)) + 0.01
            if sample_index == 61:
                label = "Gaussian blurring"
            else:
                label = None
        if sample_index >= 81 and sample_index <= 100:
            color = "y"
            if sample_index % 20 != 0:
                alpha = 1 - ((sample_index % 20) * (1.0 / 20.0)) + 0.01
            else:
                alpha = 1 - (20 * (1.0 / 20.0)) + 0.01
            if sample_index == 81:
                label = "impulse noise"
            else:
                label = None
        if sample_index >= 101 and sample_index <= 120:
            color = "orange"
            if sample_index % 20 != 0:
                alpha = 1 - ((sample_index % 20) * (1.0 / 20.0)) + 0.01
            else:
                alpha = 1 - (20 * (1.0 / 20.0)) + 0.01
            if sample_index == 101:
                label = "jpeg distortion"
            else:
                label = None
        if sample_index == 0:
            alpha = 1
            color = "red"
            label = "original"
        plt.scatter(data_transformed[which_dimensions_to_plot[0], sample_index], data_transformed[which_dimensions_to_plot[1], sample_index], label=label, color=color, marker="o", s=50, alpha=alpha)
    plt.scatter(data_transformed[which_dimensions_to_plot[0], 0], data_transformed[which_dimensions_to_plot[1], 0], label=None, color="red", marker="o", s=50)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    # plt.legend()
    g = ( max(data_transformed[which_dimensions_to_plot[0], :]) - min(data_transformed[which_dimensions_to_plot[0], :]) ) * 0.1
    plt.xlim([min(data_transformed[which_dimensions_to_plot[0], :]) - g, max(data_transformed[which_dimensions_to_plot[0], :]) + g])
    g = (max(data_transformed[which_dimensions_to_plot[1], :]) - min(data_transformed[which_dimensions_to_plot[1], :])) * 0.1
    plt.ylim([min(data_transformed[which_dimensions_to_plot[1], :]) - g, max(data_transformed[which_dimensions_to_plot[1], :]) + g])
    plt.xticks([])
    plt.yticks([])
    if show_projected_test:
        min_x_axis = min(data_transformed[which_dimensions_to_plot[0], :])
        max_x_axis = max(data_transformed[which_dimensions_to_plot[0], :])
        min_y_axis = min(data_transformed[which_dimensions_to_plot[1], :])
        max_y_axis = max(data_transformed[which_dimensions_to_plot[1], :])
        for test_sample_index in range(data_test_transformed.shape[1]):
            dim1 = data_test_transformed[which_dimensions_to_plot[0], test_sample_index]
            dim2 = data_test_transformed[which_dimensions_to_plot[1], test_sample_index]
            plt.scatter(dim1, dim2, label=label, color="magenta", marker="s", s=80, alpha=1)
            plt.annotate(test_sample_index+1, (dim1, dim2)) #https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
            min_x_axis = min(dim1, min_x_axis)
            max_x_axis = max(dim1, max_x_axis)
            min_y_axis = min(dim2, min_y_axis)
            max_y_axis = max(dim2, max_y_axis)
        plt.xlim([min_x_axis, max_x_axis])
        plt.ylim([min_y_axis, max_y_axis])
    plt.show()

def convert_mat_to_csv(path_mat, path_to_save):
    # https://gist.github.com/Nixonite/bc2f69b0c4430211bcad
    data = scipy.io.loadmat(path_mat)
    for i in data:
        if '__' not in i and 'readme' not in i:
            np.savetxt((path_to_save + i + ".csv"), data[i], delimiter=',')

def read_csv_file(path):
    # https://stackoverflow.com/questions/46614526/how-to-import-a-csv-file-into-a-data-array
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    # convert to numpy array:
    data = np.asarray(data)
    return data

def create_noisy_dataset(image_original, n_images_per_distortion_type=10):
    image_height = image_original.shape[0]
    image_width = image_original.shape[1]
    # create dataset of noisy images:
    distortion_types = ["contrast_strech", "gaussian_noise", "mean_shift", "gaussian_blurring", "salt_and_pepper", "jpeg_distortion"]
    n_features = image_height * image_width
    n_samples = 1 + (n_images_per_distortion_type * len(distortion_types))  # +1 because the first image is the original image
    dataset = np.zeros(shape=(n_features, n_samples))
    MSE_of_images = np.zeros(shape=(1, n_samples))
    distortion_type_of_images = []
    dataset[:, 0] = image_original.ravel()  # original image as the first image
    MSE_of_images[:, 0] = 0  # original image as the first image
    distortion_type_of_images.append("original")  # original image as the first image
    add_distortion = Add_distortion(image_original=image_original)
    max_MSE = 900
    for distortion_type_index, distortion_type in enumerate(distortion_types):
        print("==== Distortion type: " + distortion_type)
        for image_index in range(1, n_images_per_distortion_type+1):
            print("---- image index: " + str(image_index) + " out of " + str(n_images_per_distortion_type) + " images")
            desired_MSE = (max_MSE / n_images_per_distortion_type) * (image_index)
            if distortion_type == "contrast_strech":
                initial_distortion = 255/4
            elif distortion_type == "gaussian_noise":
                initial_distortion = 10
            elif distortion_type == "mean_shift":
                initial_distortion = 0
            elif distortion_type == "gaussian_blurring":
                initial_distortion = 1
            elif distortion_type == "salt_and_pepper":
                initial_distortion = 0.5
            elif distortion_type == "jpeg_distortion":
                initial_distortion = 10
            distorted_image, MSE = add_distortion.add_distrotion_for_an_MSE_level(desired_MSE=desired_MSE, distrotion_type=distortion_type, initial_distortion=initial_distortion)
            print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
            dataset[:, (distortion_type_index*n_images_per_distortion_type)+image_index] = distorted_image.ravel()
            MSE_of_images[:, (distortion_type_index*n_images_per_distortion_type)+image_index] = MSE
            distortion_type_of_images.append(distortion_type)
    return dataset, MSE_of_images, distortion_type_of_images

def create_noisy_test_dataset(image_original, desired_MSE, n_images_per_distortion_type=1):
    image_height = image_original.shape[0]
    image_width = image_original.shape[1]
    # create dataset of noisy images:
    distortion_types = ["contrast_strech", "gaussian_noise", "mean_shift", "gaussian_blurring", "salt_and_pepper", "jpeg_distortion"]
    n_features = image_height * image_width
    n_samples = n_images_per_distortion_type * len(distortion_types)
    dataset = np.zeros(shape=(n_features, n_samples))
    MSE_of_images = np.zeros(shape=(1, n_samples))
    distortion_type_of_images = []
    add_distortion = Add_distortion(image_original=image_original)
    for distortion_type_index, distortion_type in enumerate(distortion_types):
        print("==== Distortion type: " + distortion_type)
        for image_index in range(1, n_images_per_distortion_type+1):
            print("---- image index: " + str(image_index) + " out of " + str(n_images_per_distortion_type) + " images")
            if distortion_type == "contrast_strech":
                initial_distortion = 255/4
            elif distortion_type == "gaussian_noise":
                initial_distortion = 10
            elif distortion_type == "mean_shift":
                initial_distortion = 0
            elif distortion_type == "gaussian_blurring":
                initial_distortion = 1
            elif distortion_type == "salt_and_pepper":
                initial_distortion = 0.5
            elif distortion_type == "jpeg_distortion":
                initial_distortion = 10
            distorted_image, MSE = add_distortion.add_distrotion_for_an_MSE_level(desired_MSE=desired_MSE, distrotion_type=distortion_type, initial_distortion=initial_distortion)
            print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
            dataset[:, (distortion_type_index*n_images_per_distortion_type)+image_index-1] = distorted_image.ravel()
            MSE_of_images[:, (distortion_type_index*n_images_per_distortion_type)+image_index-1] = MSE
            distortion_type_of_images.append(distortion_type)
    # mixture of distortions:
    distrotion_type_1, distrotion_type_2 = "gaussian_blurring", "gaussian_noise"
    add_distortion = Add_distortion(image_original=image_original)
    distorted_image, MSE1 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=50, distrotion_type=distrotion_type_1, initial_distortion=1)
    add_distortion = Add_distortion(image_original=distorted_image)
    distorted_image, MSE2 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=desired_MSE-50, distrotion_type=distrotion_type_2, initial_distortion=10)
    MSE = MSE1 + MSE2
    dataset = np.hstack((dataset, distorted_image.reshape((-1,1))))
    MSE_of_images = np.hstack((MSE_of_images, MSE.reshape((-1,1))))
    distortion_type_of_images.append(distrotion_type_1 + " + " + distrotion_type_2)
    print("==== Distortion type: " + distrotion_type_1 + " + " + distrotion_type_2)
    print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
    # mixture of distortions:
    distrotion_type_1, distrotion_type_2 = "gaussian_blurring", "mean_shift"
    add_distortion = Add_distortion(image_original=image_original)
    distorted_image, MSE1 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=50, distrotion_type=distrotion_type_1, initial_distortion=1)
    add_distortion = Add_distortion(image_original=distorted_image)
    distorted_image, MSE2 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=desired_MSE-50, distrotion_type=distrotion_type_2, initial_distortion=10)
    MSE = MSE1 + MSE2
    dataset = np.hstack((dataset, distorted_image.reshape((-1,1))))
    MSE_of_images = np.hstack((MSE_of_images, MSE.reshape((-1,1))))
    distortion_type_of_images.append(distrotion_type_1 + " + " + distrotion_type_2)
    print("==== Distortion type: " + distrotion_type_1 + " + " + distrotion_type_2)
    print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
    # mixture of distortions:
    distrotion_type_1, distrotion_type_2 = "salt_and_pepper", "mean_shift"
    add_distortion = Add_distortion(image_original=image_original)
    distorted_image, MSE1 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=30, distrotion_type=distrotion_type_1, initial_distortion=1)
    add_distortion = Add_distortion(image_original=distorted_image)
    distorted_image, MSE2 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=desired_MSE-30, distrotion_type=distrotion_type_2, initial_distortion=10)
    MSE = MSE1 + MSE2
    dataset = np.hstack((dataset, distorted_image.reshape((-1,1))))
    MSE_of_images = np.hstack((MSE_of_images, MSE.reshape((-1,1))))
    distortion_type_of_images.append(distrotion_type_1 + " + " + distrotion_type_2)
    print("==== Distortion type: " + distrotion_type_1 + " + " + distrotion_type_2)
    print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
    # mixture of distortions:
    distrotion_type_1, distrotion_type_2 = "jpeg_distortion", "gaussian_noise"
    add_distortion = Add_distortion(image_original=image_original)
    distorted_image, MSE1 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=80, distrotion_type=distrotion_type_1, initial_distortion=1)
    add_distortion = Add_distortion(image_original=distorted_image)
    distorted_image, MSE2 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=desired_MSE-80, distrotion_type=distrotion_type_2, initial_distortion=10)
    MSE = MSE1 + MSE2
    dataset = np.hstack((dataset, distorted_image.reshape((-1,1))))
    MSE_of_images = np.hstack((MSE_of_images, MSE.reshape((-1,1))))
    distortion_type_of_images.append(distrotion_type_1 + " + " + distrotion_type_2)
    print("==== Distortion type: " + distrotion_type_1 + " + " + distrotion_type_2)
    print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
    # mixture of distortions:
    distrotion_type_1, distrotion_type_2 = "jpeg_distortion", "mean_shift"
    add_distortion = Add_distortion(image_original=image_original)
    distorted_image, MSE1 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=80, distrotion_type=distrotion_type_1, initial_distortion=1)
    add_distortion = Add_distortion(image_original=distorted_image)
    distorted_image, MSE2 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=desired_MSE-80, distrotion_type=distrotion_type_2, initial_distortion=10)
    MSE = MSE1 + MSE2
    dataset = np.hstack((dataset, distorted_image.reshape((-1,1))))
    MSE_of_images = np.hstack((MSE_of_images, MSE.reshape((-1,1))))
    distortion_type_of_images.append(distrotion_type_1 + " + " + distrotion_type_2)
    print("==== Distortion type: " + distrotion_type_1 + " + " + distrotion_type_2)
    print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
    # mixture of distortions:
    distrotion_type_1, distrotion_type_2 = "jpeg_distortion", "contrast_strech"
    add_distortion = Add_distortion(image_original=image_original)
    distorted_image, MSE1 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=80, distrotion_type=distrotion_type_1, initial_distortion=1)
    add_distortion = Add_distortion(image_original=distorted_image)
    distorted_image, MSE2 = add_distortion.add_distrotion_for_an_MSE_level_2(desired_MSE=desired_MSE-80, distrotion_type=distrotion_type_2, initial_distortion=10)
    MSE = MSE1 + MSE2
    dataset = np.hstack((dataset, distorted_image.reshape((-1,1))))
    MSE_of_images = np.hstack((MSE_of_images, MSE.reshape((-1,1))))
    distortion_type_of_images.append(distrotion_type_1 + " + " + distrotion_type_2)
    print("==== Distortion type: " + distrotion_type_1 + " + " + distrotion_type_2)
    print("------------ image created with MSE = " + str(MSE) + " | desired MSE was " + str(desired_MSE))
    return dataset, MSE_of_images, distortion_type_of_images

def load_image(address_image):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    img_arr = np.array(img)
    return img_arr

def save_image(image_array, path_without_file_name, file_name):
    if not os.path.exists(path_without_file_name):
        os.makedirs(path_without_file_name)
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.fromarray(image_array)
    img = img.convert("L")
    img.save(path_without_file_name + file_name)

def show_image(img):
    plt.imshow(img)
    plt.gray()
    plt.show()

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    if type(variable) is list:
        variable = np.asarray(variable)
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def plot_components(X_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # plot the first (original) image once more to be on top of other images:
        # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    # plt.xticks([])
    # plt.yticks([])
    plt.show()

def scatter_plot_MNIST(data_transformed, which_dimensions_to_plot, labels, data_test_transformed=None, show_projected_test=False):
    colors_of_labels = ["green", "blue", "black", "fuchsia", "y", "orange", "red", "slategrey", "slateblue", "olive"]
    used_labels = []
    for sample_index in range(data_transformed.shape[1]):
        # list of colors --> https: // matplotlib.org / examples / color / named_colors.html
        label_numeric = (labels[sample_index]).astype(int)
        # print(label_numeric)
        color = colors_of_labels[label_numeric]
        label = str(label_numeric)
        if label in used_labels:
            label = ""
        else:
            used_labels.append(label)
        plt.scatter(data_transformed[which_dimensions_to_plot[0], sample_index], data_transformed[which_dimensions_to_plot[1], sample_index], label=label, color=color, marker="o", s=50)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    g = ( max(data_transformed[which_dimensions_to_plot[0], :]) - min(data_transformed[which_dimensions_to_plot[0], :]) ) * 0.1
    plt.xlim([min(data_transformed[which_dimensions_to_plot[0], :]) - g, max(data_transformed[which_dimensions_to_plot[0], :]) + g])
    g = (max(data_transformed[which_dimensions_to_plot[1], :]) - min(data_transformed[which_dimensions_to_plot[1], :])) * 0.1
    plt.ylim([min(data_transformed[which_dimensions_to_plot[1], :]) - g, max(data_transformed[which_dimensions_to_plot[1], :]) + g])
    plt.xticks([])
    plt.yticks([])
    if show_projected_test:
        min_x_axis = min(data_transformed[which_dimensions_to_plot[0], :])
        max_x_axis = max(data_transformed[which_dimensions_to_plot[0], :])
        min_y_axis = min(data_transformed[which_dimensions_to_plot[1], :])
        max_y_axis = max(data_transformed[which_dimensions_to_plot[1], :])
        for test_sample_index in range(data_test_transformed.shape[1]):
            dim1 = data_test_transformed[which_dimensions_to_plot[0], test_sample_index]
            dim2 = data_test_transformed[which_dimensions_to_plot[1], test_sample_index]
            plt.scatter(dim1, dim2, label=label, color="magenta", marker="s", s=80, alpha=1)
            plt.annotate(test_sample_index+1, (dim1, dim2)) #https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
            min_x_axis = min(dim1, min_x_axis)
            max_x_axis = max(dim1, max_x_axis)
            min_y_axis = min(dim2, min_y_axis)
            max_y_axis = max(dim2, max_y_axis)
        plt.xlim([min_x_axis, max_x_axis])
        plt.ylim([min_y_axis, max_y_axis])
    plt.show()

def read_MNIST_dataset(dataset = "training", path = "."):
    # https://gist.github.com/akesling/5358964
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        print('error.....')
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    get_img = lambda idx: (lbl[idx], img[idx])
    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def plot_embedding(X, y, images=None, image_scale=None,  title=None):
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    # X: rows are features, columns are samples
    # y: a row vector (vector of labels)
    X = X.T
    y = y.ravel()
    y = y.astype(int)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if image_scale != None:
        images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()