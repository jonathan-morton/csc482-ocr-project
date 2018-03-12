__author__ = 'Jonathan Morton'
import numpy as np
import skimage.filters as filters
import skimage.util
from skimage.transform import resize as sk_resize

IMAGE_RESIZE = 32


def process_image(image: np.ndarray, image_size=IMAGE_RESIZE, make_sparse=True):
    if make_sparse:
        image = _invert_make_sparse_(image)

    image = _resize_(image, size=image_size)

    if make_sparse:
        final_image = _threshold_(image)
    else:
        final_image = image.astype(np.float32) / 255

    return final_image


def _resize_(image, size=IMAGE_RESIZE):
    image_size = image.shape[0]
    if (image_size < size):
        return image

    scale = size / image.shape[0]  # working only with square images
    sigma = (1 - scale) / 2.0
    image = filters.gaussian(image, sigma=sigma, preserve_range=True)
    image = sk_resize(image, (size, size), preserve_range=True)
    return image


def _invert_make_sparse_(image):
    image = skimage.util.invert(image)  #
    threshold = filters.threshold_otsu(image)
    threshold_index = image > threshold
    image[threshold_index] = 255
    return image


def _threshold_(image):
    final_threshold_index = image > 10
    image[final_threshold_index] = 255
    image[~final_threshold_index] = 0
    image = image.astype(np.float32) / 255
    image = np.ceil(image)
    return image


def add_salt_pepper_noise(orig_image, threshold_prob):
    image = orig_image.copy()
    max = np.max(image)
    min = np.min(image)
    random_image = np.random.rand(*image.shape)
    salt_index = random_image >= 1 - threshold_prob
    pepper_index = random_image < threshold_prob
    image[salt_index] = max
    image[pepper_index] = min
    return image
