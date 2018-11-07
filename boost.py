#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:52:18 2018

@author: andrey
"""

# %%
import xgboost as xgb

# import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def get_metrics(bst, dtrain, dtest, y_train, y_test):
    y_pred = bst.predict(dtrain)
    mae_train = mean_absolute_error(y_train, y_pred)
    print('mae train: ', mae_train)

    y_pred = bst.predict(dtest)
    mae_test = mean_absolute_error(y_test, y_pred)
    print('mae test: ', mae_test)

    r2 = r2_score(y_test, y_pred)
    #    print('mae train: ', mean_absolute_error(y_train, y_test))
    return mae_train, mae_test, r2


def xgb_regres(X_train, y_train, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'max_depth': 4, 'eta': 0.3, 'silent': 1, 'booster': 'gbtree'}
    param['nthread'] = 4
    param['eval_metric'] = 'mae'
    evallist = [(dtest, 'eval'), (dtrain, 'train')]  # ?

    num_round = 100
    bst = xgb.train(param, dtrain, num_round, evallist)

    get_metrics(bst, dtrain, dtest, y_train, y_test)
    #    feat_im = bst.get_score(importance_type='gain')
    #    print(feat_im)
    #    plt.bar(range(len(feat_im)), list(feat_im.values()), align='center')
    #    plt.xticks(range(len(feat_im)), list(feat_im.keys()), fontsize=10, rotation = 90)
    #
    return get_metrics(bst, dtrain, dtest, y_train, y_test)
#   plt.show()
