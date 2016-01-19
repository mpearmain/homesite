# -*- coding: utf-8 -*-
import pickle
import csv

def csv2dicts(csvfile):
    data = []
    keys = []
    for row_index, row in enumerate(csvfile):
        if row_index == 0:
            keys = row
            print(row)
            continue
        if row_index % 100000 == 0:
            print(row_index)
        data.append({key: value for key, value in zip(keys, row)})
    return data


train_data = "./input/xtrain_mp4.csv"
test_data = "./input/xtest_mp4.csv"

with open(train_data) as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    with open('./input/xtrain_mp4_data.pickle', 'wb') as f:
        data = csv2dicts(data)
        data = data[::-1]
        pickle.dump(data, f, -1)
        print(data[:3])

with open(test_data) as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    with open('./input/xtest_mp4_data.pickle', 'wb') as f:
        data = csv2dicts(data)
        pickle.dump(data, f, -1)
        print(data[0])

