__author__ = 'Jonathan Morton'
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import cnn_model as cnnm
import imageprocessing

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

records = read_kanji(codes)
#%%
def preprocess_images(records):
    new_records = []
    for record in records:
        list_record = list(record)
        list_record[-1] = imageprocessing.process_image(record[-1])
        new_records.append(tuple(list_record))
    return new_records

records = preprocess_images(records)
#%%
plt.interactive(True)
plt.imshow(records[70][-1],
           cmap='gray')  # https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000143610-Problems-with-Interactive-Plotting-in-Debug-Mode-in-PyCharm-Version-2017-1
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

# %%
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

num_classes, x_train, x_test, y_train, y_test  = test_train_data(records)

y_train_encoder, y_train_categorical = one_hot_encode(y_train)
y_test_encoder, y_test_categorical = one_hot_encode(y_test)
#%%
cnn_model = cnnm.get_cnn_model(imageprocessing.IMAGE_RESIZE, num_classes)  # TODO Fix number

#%%
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

metrics = cnnm.Metrics()
epochs = 200
cnn_model.fit(x_train, y_train_categorical,
              validation_data=(x_test, y_test_categorical), epochs=epochs, batch_size=8, callbacks=[metrics])
# %%
# cnn_model.evaluate(x_test, y_test_categorical)
test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test_categorical)

print(f'accuracy = {test_accuracy}, loss = {test_loss}\n')
print(f'precision = {metrics.val_precisions}\n, recalls = {metrics.val_recalls}\n')
#test_etlcdb()