__author__ = 'Jonathan Morton'
import numpy as np
from skimage.transform import resize as sk_resize
import skimage.filters as filters
import skimage.util
IMAGE_RESIZE = 32

def process_image(image: np.ndarray):
    image = skimage.util.invert(image) #
    threshold = filters.threshold_otsu(image)
    threshold_index = image > threshold
    image[threshold_index] = 255
    image = _resize_(image)

    final_threshold_index = image > 10
    image[final_threshold_index] = 255
    image[~final_threshold_index] = 0
    return image.astype(np.float32) / 255


def _resize_(image):
    image_size = image.shape[0]
    if (image_size < IMAGE_RESIZE):
        return image

    scale = IMAGE_RESIZE / image.shape[0]  # working only with square images
    sigma = (1 - scale) / 2.0
    image = filters.gaussian(image, sigma=sigma, preserve_range=True)
    image = sk_resize(image, (IMAGE_RESIZE, IMAGE_RESIZE), preserve_range=True)
    return image

