__author__ = 'Jonathan Morton'
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score


# From: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f" — val_precision: {_val_precision} — val_recall {_val_recall}")
        return


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
    metrics = Metrics()
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
