#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:14:31 2018

@author: andrey
"""
# %%
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

work_dir = '/home/andrey/yandexML/test_intern/'
name_data = 'test_ml.txt'
name_bin_features = 'bin_feat.txt'
name_con_features = 'con_feat.txt'
name_save_data = 'data.txt'
name_save_data_gain = 'data_gain.txt'
name_save_data_cor = 'data_cor.txt'


def check_to_equal(a, b):
    eps = 1e-14
    return abs(a - b) < eps


def get_stat_moments(data, count_features, flag_print=False):
    del_ind = []
    for i in range(count_features):  # если начинать  отсчет с единицы, то 24 - нулевой.
        var = np.var(data[i])
        if flag_print == True:
            print(i)
            print('mean =', np.mean(data[i]))
            print('var =', var)
        if check_to_equal(var, 0.0):  # delete const values
            del_ind.append(i)

    data = np.delete(data, del_ind, axis=0)
    count_features -= len(del_ind)
    return count_features, data


def show_distributuon(data, count_features, bins=30):
    sns.set(color_codes=True)
    for i in range(count_features):
        plt.figure()
        sns.distplot(data[i], kde=False, rug=True, bins=bins);


def divided_features(data, count_samples, count_features):  # split on binary and continuous; need to optimize
    bin_feat = np.empty((0, count_samples))
    con_feat = np.empty((0, count_samples))
    for i in range(count_features):
        flag_bin = True
        for j in range(count_samples):
            if ((not check_to_equal(data[i, j], 0.0)) and (not check_to_equal(data[i, j], 1.0))):
                con_feat = np.append(con_feat, np.expand_dims(data[i], axis=0), axis=0)
                flag_bin = False
                break
        if flag_bin == True:
            bin_feat = np.append(bin_feat, np.expand_dims(data[i], axis=0), axis=0)

    print('shape bin: ', bin_feat.shape)
    print('shape con: ', con_feat.shape)
    return bin_feat, con_feat


def delete_extremal_values(data):
    ind = np.where(data[-1] > 0.7)  # three values in y  greater than 0.7( see boxplot in R); hardcode
    print('extreme ind:', ind)
    return np.delete(data, ind, axis=1)


# %% main

if __name__ == "__main__":
    data = np.genfromtxt(join(work_dir, name_data), delimiter='\t')
    data = np.transpose(data)
    data = np.delete(data, (0), axis=1)  # delete headers
    data = np.delete(data, (14), axis=0)  # 14 feature's has only two non zero values
    data = delete_extremal_values(data)
    print('shape: ', data.shape)
    count_features, count_samples = data.shape  # count_features include y
    count_features, data = get_stat_moments(data, count_features)
    print('shape without const values: ', data.shape)
    bin_feat, con_feat = divided_features(data, count_samples, count_features)
    np.savetxt(join(work_dir, name_bin_features), np.transpose(bin_feat), fmt='%.2f', delimiter='\t')
    np.savetxt(join(work_dir, name_con_features), np.transpose(con_feat), fmt='%.2f', delimiter='\t')
    np.savetxt(join(work_dir, name_save_data), data, fmt='%.2f', delimiter='\t')

#    show_distributuon(bin_feat, bin_feat.shape[0])


# %% первая выборка признаков
# con_ind = [1, 3, 4, 5, 6, 8, 15, 21, 17, 7, 12, 25]
# data1 = np.concatenate( (bin_feat, con_feat[con_ind]))
# np.savetxt(join(work_dir, name_save_data_cor ), data1, fmt='%.2f', delimiter = '\t')

# %% вторая выборка признаков
# con_ind = [0, 1, 3, 6, 8, 10, 11, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25 ]
# data2 = np.concatenate( (bin_feat, con_feat[con_ind]))
# np.savetxt(join(work_dir, name_save_data_gain), data2, fmt='%.2f', delimiter = '\t')
#
