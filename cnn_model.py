__author__ = 'Jonathan Morton'
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import regularizers

def get_cnn_model(image_size, classification_num):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(classification_num, activation='softmax'))

    # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=1.9)
    model.compile(optimizers.Adam(lr=1e-5), 'categorical_crossentropy', metrics=['accuracy'])

    return model

# def get_cnn_model(image_size, classification_num):
#     model = models.Sequential()
#     model.add(layers.Flatten(input_shape=(image_size, image_size, 1)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dense(classification_num, W_regularizer=regularizers.l2(0.02)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('softmax'))
#
#     model.compile(optimizers.Adam(lr=1e-5), 'categorical_crossentropy', metrics=['accuracy'])
#
#     return model
