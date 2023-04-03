import datetime,time
import numpy as np
import torch
import os
import re
from tqdm import tqdm


def prepro(d_path_train):

    filenames_train = os.listdir(d_path_train)
    print('训练数据读取...')

    def capture(original_train_path):
        train_src = []
        train_trg = []
        file_entity_data = {}
        file_entity_trg = {}
        for i in tqdm(filenames_train):
            trg = re.findall("\d+", i)
            trg_s = np.array(list(map(int, trg)))-1
            file_path = os.path.join(d_path_train, i)
            filenames_train_data = os.listdir(file_path)
            for j in filenames_train_data:
                file_data_path = os.path.join(file_path, j)
                file_data = np.loadtxt(file_data_path, dtype=np.float32, delimiter='\n')
                train_data = file_data.reshape(-1, file_data.shape[0])
                train_src.append(train_data)
                train_trg.append(trg_s)
            file_entity_data[i] = train_src.copy()
            file_entity_trg[i] = train_trg.copy()
            train_src.clear()
            train_trg.clear()
            time.sleep(0.05)
        return file_entity_data, file_entity_trg

    def add_labels(train_src, train_trg):  # train_test 为字典
        X = []
        for i in filenames_train:
            x = train_src[i]
            X += x
        Y = []
        for j in filenames_train:
            y = train_trg[j]
            Y += y

        return X, Y

    data_src, data_trg = capture(original_train_path=d_path_train)
    Train_src, Train_trg = add_labels(data_src, data_trg)
    Train_src = np.asarray(Train_src)
    Train_trg = np.asarray(Train_trg)
    Train_src = torch.from_numpy(Train_src)
    Train_trg = torch.from_numpy(Train_trg)
    print('训练数据读取完成...')
    return Train_src, Train_trg


def prepro_valid(d_path_validation):

    filenames_validation = os.listdir(d_path_validation)
    print('验证数据读取...')

    def capture(original_train_path):
        train_src = []
        train_trg = []
        file_entity_data = {}
        file_entity_trg = {}
        for i in tqdm(filenames_validation):
            trg = re.findall("\d+", i)
            trg_s = np.array(list(map(int, trg)))-1
            file_path = os.path.join(d_path_validation, i)
            filenames_train_data = os.listdir(file_path)
            for j in filenames_train_data:
                file_data_path = os.path.join(file_path, j)
                file_data = np.loadtxt(file_data_path, dtype=np.float32, delimiter='\n')
                train_data = file_data.reshape(-1, file_data.shape[0])
                train_src.append(train_data)
                train_trg.append(trg_s)
            file_entity_data[i] = train_src.copy()
            file_entity_trg[i] = train_trg.copy()
            train_src.clear()
            train_trg.clear()
            time.sleep(0.05)
        return file_entity_data, file_entity_trg

    def add_labels(train_src, train_trg):  # train_test 为字典
        X = []
        for i in filenames_validation:
            x = train_src[i]
            X += x
        Y = []
        for j in filenames_validation:
            y = train_trg[j]
            Y += y

        return X, Y

    data_src, data_trg = capture(original_train_path=d_path_validation)
    Valid_src, Valid_trg = add_labels(data_src, data_trg)
    Valid_src = np.asarray(Valid_src)
    Valid_trg = np.asarray(Valid_trg)
    Valid_src = torch.from_numpy(Valid_src)
    Valid_trg = torch.from_numpy(Valid_trg)
    print('验证数据读取完成...')
    return Valid_src, Valid_trg

def prepro_test(d_path_test):

    filenames_test = os.listdir(d_path_test)
    filenames_test_data = []
    for strings in filenames_test:
        if 'input' in strings:
            filenames_test_data.append(strings)

    re_digits = re.compile(r'(\d+)')

    def embedded_numbers(s):
        pieces = re_digits.split(s)
        pieces[1::2] = map(int, pieces[1::2])
        return pieces

    def sort_string(lst):
        return sorted(lst, key=embedded_numbers)
    filenames_test_data = sort_string(filenames_test_data)

    def capture(original_test_path):
        test_src = []
        file_entity_data = {}

        for i in filenames_test_data:
            file_path = os.path.join(d_path_test, i)
            file_data = np.loadtxt(file_path, dtype=np.float32, delimiter='\n')
            test_data = file_data.reshape(-1, file_data.shape[0])
            test_src.append(test_data)
            file_entity_data[i] = test_src.copy()
            test_src.clear()
        return file_entity_data

    def add_labels(test_src):
        X = []
        for i in filenames_test_data:
            x = test_src[i]
            X += x
        return X

    data_src= capture(original_test_path=d_path_test)
    Test_src = add_labels(data_src)
    Test_src = np.asarray(Test_src)
    Test_src = torch.from_numpy(Test_src)

    return Test_src

if __name__ == '__main__':
    data_path_train = r'F:\LYY\dataset\Elec_iden\个体识别数据集\train\train'
    data_path_validation = r'F:\LYY\dataset\Elec_iden\个体识别数据集\validation\validation'
    data_path_test = r'F:\LYY\dataset\Elec_iden\个体识别数据集\test-lyy2'
    normal = False
    seq_len_in = 3000
    seq_len_out = 8
    prepro(data_path_train)
    prepro_valid(data_path_validation)
    prepro_test(data_path_test)

