import numpy as np
from numpy import linalg as LA


class My_SSIM:

    def __init__(self, window_size=8):
        self.window_size = window_size

    def __padding_the_array(self, array_2D, how_many_pixels, padding_type="reflect"):
        padded_array = np.pad(array=array_2D, pad_width=((how_many_pixels, how_many_pixels), (how_many_pixels, how_many_pixels)), mode=padding_type)
        return padded_array

    def SSIM_index(self, image1, image2):
        if np.max(image1) - np.min(image1) <= 1:
            # range of image is [0,1] --> change it to [0,255]:
            image1 = np.round(image1 * 255)
        if np.max(image2) - np.min(image2) <= 1:
            # range of image is [0,1] --> change it to [0,255]:
            image2 = np.round(image2 * 255)
        L = 255
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        C3 = C2 / 2
        SSIM = np.zeros(image1.shape)
        # ---- pad the images:
        padding_width = (np.floor(self.window_size / 2)).astype(int)
        image1 = self.__padding_the_array(array_2D=image1, how_many_pixels=padding_width, padding_type="reflect")
        image2 = self.__padding_the_array(array_2D=image2, how_many_pixels=padding_width, padding_type="reflect")
        # ---- calculate SSIM:
        for row_index in range(0 + padding_width, image1.shape[0] - padding_width):
            for column_index in range(0 + padding_width, image1.shape[1] - padding_width):
                if self.window_size % 2 != 0:
                    pixels1_in_window = image1[row_index - padding_width:row_index + padding_width, column_index - padding_width:column_index + padding_width]
                    pixels2_in_window = image2[row_index - padding_width:row_index + padding_width, column_index - padding_width:column_index + padding_width]
                else:
                    pixels1_in_window = image1[row_index - padding_width:row_index + padding_width - 1, column_index - padding_width:column_index + padding_width - 1]
                    pixels2_in_window = image2[row_index - padding_width:row_index + padding_width - 1, column_index - padding_width:column_index + padding_width - 1]
                N = self.window_size ** 2
                mean1 = np.mean(pixels1_in_window)
                mean2 = np.mean(pixels1_in_window)
                # print(mean1)
                # print(mean2)
                # print("=====")
                std1 = ((1 / (N - 1)) * sum((pixels1_in_window.ravel() - mean1) ** 2)) ** 0.5
                std2 = ((1 / (N - 1)) * sum((pixels2_in_window.ravel() - mean2) ** 2)) ** 0.5
                # print(std1)
                # print(std2)
                # print("=====")
                cross_correlation = (1 / (N - 1)) * sum((pixels1_in_window.ravel() - mean1) * (pixels2_in_window.ravel() - mean2))
                # print(cross_correlation)
                # print(std1 * std2)
                # print(std1)
                # print(std2)
                # print("=====")
                luminance_score = ((2 * mean1 * mean2) + C1) / ((mean1 ** 2) + (mean2 ** 2) + C1)
                contrast_score = ((2 * std1 * std2) + C2) / ((std1 ** 2) + (std2 ** 2) + C2)
                structure_score = (cross_correlation + C3) / ((std1 * std2) + C3)
                # print(structure_score)
                # print("=====")
                SSIM[row_index - padding_width, column_index - padding_width] = luminance_score * contrast_score * structure_score
        SSIM_index = np.mean(SSIM)
        return SSIM_index, SSIM








    def SSIM_index_2(self, image1, image2):
        if np.max(image1) - np.min(image1) <= 1:
            # range of image is [0,1] --> change it to [0,255]:
            image1 = np.round(image1 * 255)
        if np.max(image2) - np.min(image2) <= 1:
            # range of image is [0,1] --> change it to [0,255]:
            image2 = np.round(image2 * 255)
        L = 255
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        C3 = C2 / 2
        SSIM = np.zeros(image1.shape)
        distance = np.zeros(image1.shape)
        distance_MeanRemoved = np.zeros(image1.shape)
        # luminance_score = np.zeros(image1.shape)
        # contrast_score = np.zeros(image1.shape)
        # structure_score = np.zeros(image1.shape)
        # ---- pad the images:
        padding_width = (np.floor(self.window_size / 2)).astype(int)
        image1 = self.__padding_the_array(array_2D=image1, how_many_pixels=padding_width, padding_type="reflect")
        image2 = self.__padding_the_array(array_2D=image2, how_many_pixels=padding_width, padding_type="reflect")
        # ---- calculate SSIM:
        for row_index in range(0 + padding_width, image1.shape[0] - padding_width):
            for column_index in range(0 + padding_width, image1.shape[1] - padding_width):
                if self.window_size % 2 != 0:
                    pixels1_in_window = image1[row_index - padding_width:row_index + padding_width, column_index - padding_width:column_index + padding_width]
                    pixels2_in_window = image2[row_index - padding_width:row_index + padding_width, column_index - padding_width:column_index + padding_width]
                else:
                    pixels1_in_window = image1[row_index - padding_width:row_index + padding_width - 1, column_index - padding_width:column_index + padding_width - 1]
                    pixels2_in_window = image2[row_index - padding_width:row_index + padding_width - 1, column_index - padding_width:column_index + padding_width - 1]
                N = self.window_size ** 2
                mean1 = np.mean(pixels1_in_window)
                mean2 = np.mean(pixels1_in_window)
                std1 = ((1 / (N - 1)) * sum((pixels1_in_window.ravel() - mean1) ** 2)) ** 0.5
                std2 = ((1 / (N - 1)) * sum((pixels2_in_window.ravel() - mean2) ** 2)) ** 0.5
                cross_correlation = (1 / (N - 1)) * sum((pixels1_in_window.ravel() - mean1) * (pixels2_in_window.ravel() - mean2))
                # luminance_score_ = ((2 * mean1 * mean2) + C1) / ((mean1 ** 2) + (mean2 ** 2) + C1)
                # contrast_score_ = ((2 * std1 * std2) + C2) / ((std1 ** 2) + (std2 ** 2) + C2)
                # structure_score_ = (cross_correlation + C3) / ((std1 * std2) + C3)
                # luminance_score[row_index - padding_width, column_index - padding_width] = luminance_score_
                # contrast_score[row_index - padding_width, column_index - padding_width] = contrast_score_
                # structure_score[row_index - padding_width, column_index - padding_width] = structure_score_
                S1 = ((2 * mean1 * mean2) + C1) / ((mean1 ** 2) + (mean2 ** 2) + C1)
                S2 = ((2 * cross_correlation) + C2) / ((std1 ** 2) + (std2 ** 2) + C2)
                SSIM[row_index - padding_width, column_index - padding_width] = S1 * S2
                distance[row_index - padding_width, column_index - padding_width] = (abs(2 - S1 - S2)) ** 0.5
                # ------ distance for MeanRemoved:
                x1 = pixels1_in_window.reshape((-1, 1))
                x2 = pixels2_in_window.reshape((-1, 1))
                x1 = x1 - np.mean(x1)
                x2 = x2 - np.mean(x2)
                C = (N - 1) * C2
                distance_MeanRemoved[row_index - padding_width, column_index - padding_width] = ((LA.norm(x1-x2)) ** 2) / ((LA.norm(x1))**2 + (LA.norm(x2))**2 + C)
        SSIM_index = np.mean(SSIM)
        distance_index = LA.norm(distance, ord="fro")
        distance_index_MeanRemoved = LA.norm(distance_MeanRemoved, ord="fro")
        SSIM = SSIM.reshape((-1, 1))

        distance = distance.reshape((-1, 1))
        distance_MeanRemoved = distance_MeanRemoved.reshape((-1, 1))
        # luminance_score = luminance_score.reshape((-1, 1))
        # contrast_score = contrast_score.reshape((-1, 1))
        # structure_score = structure_score.reshape((-1, 1))
        return SSIM_index, SSIM, distance_index, distance, distance_index_MeanRemoved, distance_MeanRemoved

    def SSIM_index_3(self, image1, image2):
        if np.max(image1) - np.min(image1) <= 1:
            # range of image is [0,1] --> change it to [0,255]:
            image1 = np.round(image1 * 255)
        if np.max(image2) - np.min(image2) <= 1:
            # range of image is [0,1] --> change it to [0,255]:
            image2 = np.round(image2 * 255)
        L = 255
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        C3 = C2 / 2
        distance = np.zeros(image1.shape)
        # ---- pad the images:
        padding_width = (np.floor(self.window_size / 2)).astype(int)
        image1 = self.__padding_the_array(array_2D=image1, how_many_pixels=padding_width, padding_type="reflect")
        image2 = self.__padding_the_array(array_2D=image2, how_many_pixels=padding_width, padding_type="reflect")
        # ---- calculate SSIM:
        for row_index in range(0 + padding_width, image1.shape[0] - padding_width):
            for column_index in range(0 + padding_width, image1.shape[1] - padding_width):
                if self.window_size % 2 != 0:
                    pixels1_in_window = image1[row_index - padding_width:row_index + padding_width, column_index - padding_width:column_index + padding_width]
                    pixels2_in_window = image2[row_index - padding_width:row_index + padding_width, column_index - padding_width:column_index + padding_width]
                else:
                    pixels1_in_window = image1[row_index - padding_width:row_index + padding_width - 1, column_index - padding_width:column_index + padding_width - 1]
                    pixels2_in_window = image2[row_index - padding_width:row_index + padding_width - 1, column_index - padding_width:column_index + padding_width - 1]
                N = self.window_size ** 2
                mean1 = np.mean(pixels1_in_window)
                mean2 = np.mean(pixels1_in_window)
                std1 = ((1 / (N - 1)) * sum((pixels1_in_window.ravel() - mean1) ** 2)) ** 0.5
                std2 = ((1 / (N - 1)) * sum((pixels2_in_window.ravel() - mean2) ** 2)) ** 0.5
                cross_correlation = (1 / (N - 1)) * sum((pixels1_in_window.ravel() - mean1) * (pixels2_in_window.ravel() - mean2))
                S1 = ((2 * mean1 * mean2) + C1) / ((mean1 ** 2) + (mean2 ** 2) + C1)
                S2 = ((2 * cross_correlation) + C2) / ((std1 ** 2) + (std2 ** 2) + C2)
                distance[row_index - padding_width, column_index - padding_width] = (abs(2 - S1 - S2)) ** 0.5
        distance_index = LA.norm(distance, ord="fro")
        return distance_index

    