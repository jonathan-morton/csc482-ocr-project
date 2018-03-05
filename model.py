__author__ = 'Jonathan Morton'
from keras import models
from keras import layers

def get_cnn_model(image_size, classification_num):
    model = models.Sequential()
    model.add(layers.Conv2D())
