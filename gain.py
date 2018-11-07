#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 01:04:25 2018

@author: andrey
"""


def get_information_gain(X, y, count_bin_feat=5, num_iter=100):
    X = np.transpose(X)
    y = np.transpose(y)
    print(X.shape)
    print(y.shape)
    mi = np.zeros(X.shape[1])
    count_zero = np.zeros(X.shape[1])
    for i in range(num_iter):
        loc_mi = mutual_info_regression(X, y, discrete_features=list(range(count_bin_feat)))
        loc_mi /= np.max(loc_mi)
        mi += loc_mi
        for j in range(len(count_zero)):
            if check_to_equal(loc_mi[j], 0.0):
                count_zero[j] += 1

    mi = mi / num_iter
    print(mi)
    print(count_zero)
    return np.where(count_zero < 0.2 * num_iter)
