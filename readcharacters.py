__author__ = 'Jonathan Morton'
import struct
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#%%
size_record = 8199
file_count = 33
root_filename = "ETL8G/ETL8G_"
codes_filename = "jis208toUnicode.csv"
#

#%%
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
        jis_query = f'JIS == "{jis_code}"'
        data = self.df.query(jis_query)
        kanji = data.iloc[0]['KANJI']
        return kanji
#%%
def get_all_jis_codes(japanese_chars):
    codes = []
    jis_data = JisData(codes_filename)
    for char in japanese_chars:
        codes.append(jis_data.get_jis_code(char))
    return codes
#%%
codes = get_all_jis_codes(["三","一","二","四","五","六","七","八", "九","十",])

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
    last_file_count = 33 #33 files
    ary = np.zeros([len(jis_codes), 160, 127, 128], dtype=np.uint8)
    for i in range(1,last_file_count):
        filename = root_filename + '{:02d}'.format(i)
        with open(filename, 'rb') as f:
            for id_dataset in range(5):
                for j in range(956):
                    record = read_record_ETL8G(f)
                    if(hex(record[1]) in jis_codes):
                        record_list = list(record)
                        kanji_image = Image.eval(record[-1], lambda x: 255 - x * 16)
                        kanji_np_array = np.array(kanji_image)
                        record_list[-1] = kanji_np_array
                        new_record = tuple(record_list)
                        records.append(new_record)
    return records

records = read_kanji(codes)
#%%
def test_train_data(records):
    np.random.shuffle(records)
    X = [record[-1] for record in records]
    Y = [record[1] for record in records]
    train_x, train_y, test_x, test_y = train_test_split(X, Y, test_size=0.25)
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = test_train_data(records)
#%%
def test_etlcdb():
    filename = 'ETL8G/ETL8G_01'
    id_record = 0

    with open(filename, 'rb') as f:
        r = read_record_ETL8G(f)

    print(r[0:-2], hex(r[1]))
    iE = Image.eval(r[-1], lambda x: 255 - x * 16)
    fn = 'ETL8G_{:d}_{:s}.png'.format((r[0] - 1) % 20 + 1, hex(r[1])[-4:])
    iE.save(fn, 'PNG')

def read_hiragana():
    # Character type = 72, person = 160, y = 127, x = 128
    ary = np.zeros([72, 160, 127, 128], dtype=np.uint8)

    for j in range(1, 33):
        filename = 'ETL8G/ETL8G_{:02d}'.format(j)
        with open(filename, 'rb') as f:
            for id_dataset in range(5):
                moji = 0
                for i in range(956):
                    r = read_record_ETL8G(f)
                    if b'.HIRA' in r[2]:
                        ary[moji, (j - 1) * 5 + id_dataset] = np.array(r[-1])
                        moji += 1
    print(ary)
    #np.savez_compressed("hiragana.npz", ary)


read_hiragana()

#test_etlcdb()