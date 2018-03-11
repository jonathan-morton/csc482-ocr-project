__author__ = 'Jonathan Morton'
import pickle
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model as keras_load
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import cnn_model as cnnm

size_record = 8199
file_count = 33
root_filename = "ETL8G/ETL8G_"
codes_filename = "jis208toUnicode.csv"
#


class JisData:
    def __init__(self, codes_filename):
        self.df = self._get_jis_unicode_data_frame_(codes_filename)

    @staticmethod
    def _get_jis_unicode_data_frame_(codes_filename):
        df = pd.read_csv(codes_filename, na_values=[''])
        return df

    def get_jis_code(self, japanese_char):
        kanji_query = f'KANJI == "{japanese_char}"'
        data = self.df.query(kanji_query)
        jis_code = data.iloc[0]['JIS']
        return '0x' + jis_code

    def get_character(self, jis_code):
        jis_code = jis_code.upper()
        jis_query = f'JIS == "{jis_code}"'
        data = self.df.query(jis_query)
        kanji = data.iloc[0]['KANJI']
        return kanji

def get_all_jis_codes(japanese_chars):
    codes = []
    jis_data = JisData(codes_filename)
    for char in japanese_chars:
        codes.append(jis_data.get_jis_code(char))
    return codes

codes = get_all_jis_codes(["月","火","水","木","金","土","日"])


# codes = get_all_jis_codes(["月","火"])

#%%
def read_record_ETL8G(f):
    s = f.read(size_record)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)

def read_kanji(jis_codes):
    # Character type = 72, person = 160, y = 127, x = 128
    records = []
    count = 0
    jis_data = JisData(codes_filename)
    # charSet = set()
    last_file_count = 33 #33 files
    ary = np.zeros([len(jis_codes), 160, 127, 128], dtype=np.uint8)
    for i in range(1,last_file_count+1):
        filename = root_filename + '{:02d}'.format(i)
        with open(filename, 'rb') as f:
            if i == last_file_count:
                id_range = range(1)
            else:
                id_range = range(5)
            for id_dataset in id_range:
                f.seek(id_dataset * 956 * size_record)
                for j in range(956):
                    record = read_record_ETL8G(f)
                    #print(jis_data.get_character(hex(record[1])[2:]))
                    hex_code = "0x" + hex(record[1])[2:].upper()
                    if(hex_code in jis_codes):
                        # charSet.add(hex(record[1])[2:])
                        record_list = list(record)
                        kanji_image = Image.eval(record[-1], lambda x: 255 - x * 16)
                        kanji_np_array = np.array(kanji_image)
                        record_list[-1] = kanji_np_array
                        new_record = tuple(record_list)
                        records.append(new_record)
    return records


full_records = read_kanji(codes)
# %%
import imageprocessing


def preprocess_images(records, make_sparse=False):
    new_records = []
    for record in records:
        list_record = list(record)
        list_record[-1] = imageprocessing.process_image(record[-1], make_sparse=make_sparse)
        new_records.append(tuple(list_record))
    return new_records


def add_noise(records):
    new_records = []
    for record in records:
        list_record = list(record)
        list_record[-1] = imageprocessing.add_salt_pepper_noise(record[-1], threshold_prob=0.01)
        new_records.append(tuple(list_record))
    return new_records


resize_records = preprocess_images(full_records, make_sparse=False)
sparse_records = preprocess_images(full_records, make_sparse=True)
noisy_sparse_records = add_noise(sparse_records)

all_mixed_records = resize_records + sparse_records + noisy_sparse_records
np.random.shuffle(all_mixed_records)
#%%
plt.interactive(True)
plt.imshow(sparse_records[70][-1],
           cmap='gray')  # https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000143610-Problems-with-Interactive-Plotting-in-Debug-Mode-in-PyCharm-Version-2017-1

plt.interactive(True)
plt.imshow(resize_records[69][-1],
           cmap='gray')

from importlib import reload

reload(imageprocessing)
noisy_img = imageprocessing.add_salt_pepper_noise(resize_records[69][-1], 0.01)
plt.interactive(True)
plt.imshow(noisy_img,
           cmap='gray')
# %%

def test_train_data(records):
    #np.random.shuffle(records)
    X = [record[-1] for record in records]
    X = np.asarray(X)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = X.astype('float32')

    Y = [hex(record[1])[2:].upper() for record in records]
    Y = np.array(Y)
    num_classes = np.unique(Y).size
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    return num_classes, x_train, x_test, y_train, y_test

def one_hot_encode(y_labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return label_encoder, onehot_encoded

def look_up_label(label_encoder, encoded_row, get_kanji=False):
    inverted_hex = label_encoder.inverse_transform([argmax(encoded_row)])
    if not get_kanji:
        return inverted_hex[0]
    jis_data = JisData(codes_filename)
    return jis_data.get_character(inverted_hex[0].upper())


def generate_model_parameters(records, model_name):
    num_classes, x_train, x_test, y_train, y_test = test_train_data(records)

    y_train_encoder, y_train_categorical = one_hot_encode(y_train)
    y_test_encoder, y_test_categorical = one_hot_encode(y_test)
    model = cnnm.get_cnn_model(imageprocessing.IMAGE_RESIZE, num_classes)
    return {
        "model": model,
        "model_name": model_name,
        "num_classes": num_classes,
        "x_train": x_train,
        "x_test": x_test,
        "y_train_encoder": y_train_encoder,
        "y_train_categorical": y_train_categorical,
        "y_test_encoder": y_test_encoder,
        "y_test_categorical": y_test_categorical
    }

def fit_model(model_parameters, epochs=200, use_datagen=False):
    metrics = cnnm.Metrics()
    model = model_parameters["model"]

    x_train = model_parameters["x_train"]
    y_train_categorical = model_parameters["y_train_categorical"]
    x_test = model_parameters["x_test"]
    y_test_categorical = model_parameters["y_test_categorical"]

    batch_size = 8
    if use_datagen:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            fill_mode="constant"
        )
        model.fit_generator(datagen.flow(x_train, y_train_categorical, batch_size=batch_size),
                            steps_per_epoch=len(x_train) / batch_size,
                            validation_data=(x_test, y_test_categorical),
                            epochs=epochs,
                            callbacks=[metrics]
                            )
    else:
        model.fit(x_train, y_train_categorical,
                  validation_data=(x_test, y_test_categorical), epochs=epochs, batch_size=batch_size,
                  callbacks=[metrics]
                  )

    model_parameters["metrics"] = metrics


def evaluate_model(model_parameters):
    x_test = model_parameters["x_test"]
    y_test_categorical = model_parameters["y_test_categorical"]
    model = model_parameters["model"]
    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical)

    return test_loss, test_accuracy


def save_model(model_parameters):
    model = model_parameters["model"]
    name = model_parameters["model_name"]
    metrics = model_parameters["metrics"]
    model.save(f"model-{name}.h5")

    with open(f"model-{name}-metrics.p", 'wb') as fp:
        metrics_to_save = {"precisions": metrics.val_precisions, "recalls": metrics.val_recalls}
        pickle.dump(metrics_to_save, fp)


def load_model_data(model_name):
    loaded_model = keras_load(f"{model_name}.h5")
    with open(f"{model_name}-metrics.p", 'rb') as fp:
        metrics = pickle.load(fp)

    return loaded_model, metrics


resize_model_params = generate_model_parameters(resize_records, "resize")
sparse_model_params = generate_model_parameters(sparse_records, "sparse")
noisy_sparse_model_params = generate_model_parameters(noisy_sparse_records, "noisy_sparse")
all_mixed_model_params = generate_model_parameters(all_mixed_records, "all_mixed")
all_augmented_model_params = generate_model_parameters(all_mixed_records, "all_augmented")


#%%
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

total_epochs = 2
# total_epochs = 200

fit_model(resize_model_params, epochs=total_epochs)
fit_model(sparse_model_params, epochs=total_epochs)
fit_model(noisy_sparse_model_params, epochs=total_epochs)
fit_model(all_mixed_model_params, epochs=total_epochs)
fit_model(all_augmented_model_params, epochs=total_epochs, use_datagen=True)

save_model(resize_model_params)
save_model(sparse_model_params)
save_model(noisy_sparse_model_params)
save_model(all_mixed_model_params)
save_model(all_augmented_model_params)

# %%

# # %%
# # cnn_model.evaluate(x_test, y_test_categorical)
# test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test_categorical)
#
# print(f'accuracy = {test_accuracy}, loss = {test_loss}\n')
# print(f'precision = {metrics.val_precisions}\n, recalls = {metrics.val_recalls}\n')
# cnn_model.save('sparse_model.h5')
