__author__ = 'Jonathan Morton'
import struct
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
import imageprocessing
import model
#%%
large_image = records[30][-1]
small_image = imageprocessing.process_image(large_image)
plt.interactive(True)
plt.imshow(small_image, cmap='gray') # https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000143610-Problems-with-Interactive-Plotting-in-Debug-Mode-in-PyCharm-Version-2017-1
#%%
# TODO not working yet
from keras.utils import to_categorical

def test_train_data(records):
    np.random.shuffle(records)
    X = [record[-1] for record in records]
    X = np.asarray(X)

    Y = [record[1] for record in records]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test  = test_train_data(records)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#%%

#test_etlcdb()