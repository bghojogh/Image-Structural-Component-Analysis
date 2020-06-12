import numpy as np
import scipy.ndimage.filters as scipy_filters
from PIL import Image
from scipy import ndimage
import random
from random import randint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class Add_distortion:

    def __init__(self, image_original):
        self.image_original = image_original

    def add_distrotion_for_an_MSE_level(self, desired_MSE, distrotion_type, initial_distortion):
        if distrotion_type == "contrast_strech":
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(self.image_original, distrotion_type, desired_MSE, initial_distortion)
        elif distrotion_type == "gaussian_noise":
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(self.image_original, distrotion_type, desired_MSE, initial_distortion)
        elif distrotion_type == "mean_shift":
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(self.image_original, distrotion_type, desired_MSE, initial_distortion)
        elif distrotion_type == "gaussian_blurring":
            max_MSE = 900  # it is set in the main.py file, too.
            min_gap = 20
            max_gap = 50
            gap = ((max_gap - min_gap) * (desired_MSE / max_MSE)) + min_gap
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(self.image_original, "contrast_strech", desired_MSE-gap, 255 / 4)
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(best_found_distorted_image, distrotion_type, desired_MSE, initial_distortion)
        elif distrotion_type == "salt_and_pepper":
            max_MSE = 900  # it is set in the main.py file, too.
            min_gap = 20
            max_gap = 90
            gap = ((max_gap - min_gap) * (desired_MSE / max_MSE)) + min_gap
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(self.image_original, "contrast_strech", desired_MSE - gap, 255 / 4)
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(best_found_distorted_image, distrotion_type, desired_MSE, initial_distortion)
        elif distrotion_type == "jpeg_distortion":
            max_MSE = 900  # it is set in the main.py file, too.
            min_gap = 20
            max_gap = 90
            gap = ((max_gap - min_gap) * (desired_MSE / max_MSE)) + min_gap
            best_found_distorted_image, best_found_MSE_1 = self.search_for_bestMatch_distortion(self.image_original, distrotion_type, gap, initial_distortion)
            best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(best_found_distorted_image, "contrast_strech", desired_MSE - gap, 255 / 4)
            best_found_MSE = best_found_MSE + best_found_MSE_1
        return best_found_distorted_image, best_found_MSE

    def add_distrotion_for_an_MSE_level_2(self, desired_MSE, distrotion_type, initial_distortion):
        best_found_distorted_image, best_found_MSE = self.search_for_bestMatch_distortion(self.image_original, distrotion_type, desired_MSE, initial_distortion)
        return best_found_distorted_image, best_found_MSE

    def search_for_bestMatch_distortion(self, image, distrotion_type, desired_MSE, initial_distortion):
        previous_error_of_MSE_errors = np.inf
        epsilon = 10 ** (-1)
        distortion = initial_distortion
        iteration_index = 0
        step_for_gaussianNoise = 1
        while True:
            iteration_index = iteration_index + 1
            if distrotion_type == "contrast_strech":
                distorted_image = self.contrast_strech(image=image,
                                                       r_min=255 / 2 - (255 / 2 - distortion),
                                                       r_max=255 / 2 + (255 / 2 + distortion))
            elif distrotion_type == "gaussian_noise":
                distorted_image = self.gaussian_noise(image=image, std=distortion)
            elif distrotion_type == "mean_shift":
                distorted_image = self.mean_shift(image=image, shift_amount=distortion)
            elif distrotion_type == "gaussian_blurring":
                distorted_image = self.gaussian_blurring(image=image, std=distortion)
            elif distrotion_type == "salt_and_pepper":
                distorted_image = self.salt_and_pepper(image=image, noise_amount=distortion)
            elif distrotion_type == "jpeg_distortion":
                distorted_image = self.jpeg_distortion(image=image, quality_jpeg=20 - distortion)
            # =================== calculate obtained MSE:
            MSE = mean_squared_error(image.ravel(), distorted_image.ravel())
            # =================== update distortion amount:
            if distrotion_type == "contrast_strech":
                if MSE < desired_MSE:
                    distortion = min(distortion + 1 + ((-1) ** (randint(1, 2)) * random.uniform(0, 5)), 255 / 2 - 0.1)
                else:
                    distortion = max(distortion - 1 + ((-1) ** (randint(1, 2)) * random.uniform(0, 5)), 0)
            elif distrotion_type == "gaussian_noise":
                step_for_gaussianNoise = max(step_for_gaussianNoise - 0.1, 0.1)
                if MSE < desired_MSE:
                    distortion = distortion + (step_for_gaussianNoise * distortion)
                else:
                    distortion = distortion - (step_for_gaussianNoise * distortion)
            elif distrotion_type == "mean_shift":
                if MSE < desired_MSE:
                    distortion = min(distortion + random.uniform(0, 5), 255)
                else:
                    distortion = max(distortion - random.uniform(0, 5), -255)
            elif distrotion_type == "gaussian_blurring":
                step_for_gaussianNoise = max(step_for_gaussianNoise - 0.1, 0.1)
                if MSE < desired_MSE:
                    distortion = distortion + (step_for_gaussianNoise * distortion)
                else:
                    distortion = distortion - (step_for_gaussianNoise * distortion)
            elif distrotion_type == "salt_and_pepper":
                if MSE < desired_MSE:
                    distortion = min(distortion + 0.01 + random.uniform(0, 0.03), 1)
                else:
                    distortion = max(distortion - 0.01 - random.uniform(0, 0.03), 0)
            elif distrotion_type == "jpeg_distortion":
                if MSE < desired_MSE:
                    distortion = min(distortion + 1, 20)
                else:
                    distortion = max(distortion - 1, 1)
            # save the best found distorted image:
            if abs(MSE - desired_MSE) < previous_error_of_MSE_errors:
                previous_error_of_MSE_errors = abs(MSE - desired_MSE)
                best_found_distorted_image = distorted_image
                best_found_MSE = MSE
            # termination condition of search:
            # print(str(MSE) + ", " + str(best_found_MSE) + ", " + str(desired_MSE))
            if abs(MSE - desired_MSE) < epsilon:
                break
            if iteration_index > 200:
                break
        return best_found_distorted_image, best_found_MSE

    def gaussian_noise(self, image, std):
        # https://stackoverflow.com/questions/46385212/adding-noise-to-numpy-array
        # I googled: add noise to 2d array python
        noisy_image = np.random.normal(image, std)
        noisy_image[noisy_image > 255] = 255
        noisy_image[noisy_image < 0] = 0
        return noisy_image

    def gaussian_blurring(self, image, std):
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html
        blurred_image = scipy_filters.gaussian_filter(input=image, sigma=std)
        #------------------------ another approach:
        # https://www.graceunderthesea.com/thesis/blur-image-in-python-without-opencv
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
        # filter_kernel_size = (5, 5)
        # filter_kernel = np.ones(filter_kernel_size) / float(filter_kernel_size[0] * filter_kernel_size[1])
        # blurred_image = ndimage.convolve(input=self.image_original, weights=filter_kernel, mode='mirror')
        return blurred_image

    def gamma_transform(self, image, gamma=1, c=1):
        # extreme range of gamma --> almost (0.04, 25)
        # reasonable range of gamma -> almost (0.7, 1.5)
        gamma_transformed_image = c * (image ** gamma)
        gamma_transformed_image[gamma_transformed_image > 225] = 255
        return gamma_transformed_image

    def contrast_strech(self, image, r_min, r_max):
        # reasonable range of r_min --> (0, 255/2)
        # reasonable range of r_max --> (255/2, 255)
        # note r_max should be greater than r_min
        contrast_streched_image = np.around(255 * (image - r_min) / (r_max - r_min))
        return contrast_streched_image

    def mean_shift(self, image, shift_amount):
        # reasonable range for shift_amount --> (-255/2, 255/2)
        mean_shifted_image = image + shift_amount
        mean_shifted_image[mean_shifted_image > 255] = 255
        mean_shifted_image[mean_shifted_image < 0] = 0
        return mean_shifted_image

    def salt_and_pepper(self, image, noise_amount):
        # range of noise_amount --> (0, 1) but range (0, 0.6) is reasonable
        s_vs_p = 0.5
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(noise_amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(noise_amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    def add_poisson_noise(self, image, lambda_differenceFromMean):
        # test lambda_differenceFromMean with range (-100,100)
        lambda_poisson = image + lambda_differenceFromMean
        lambda_poisson[lambda_poisson < 0] = 0.000001
        noisy_image = np.random.poisson(lam=lambda_poisson)
        noisy_image[noisy_image > 255] = 255
        noisy_image[noisy_image < 0] = 0
        return noisy_image

    def jpeg_distortion(self, image, quality_jpeg=10):
        # https://stackoverflow.com/questions/30771652/how-to-perform-jpeg-compression-in-python-without-writing-reading
        # I googled: python jpeg distortion
        # good range for quality_jpeg --> random.randint(1, 20) --> should be integer
        # high values of quality_jpeg may give error: https://stackoverflow.com/questions/10318732/python-pil-jpeg-quality
        img_pil = Image.fromarray(image)
        address = "./temp_files/temp.jpeg"
        img_pil.save(address, "JPEG", quality=quality_jpeg)
        img_pil = Image.open(address).convert('L')
        img_arr = np.array(img_pil)
        return img_arr

    def noisy(self, noise_typ, image):
        # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        if noise_typ == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy

    # SNR:
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.signaltonoise.html