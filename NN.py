#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:27:54 2018

@author: andrey
"""

# %%
import numpy as np
from utils import name_save_data, name_save_data_gain, name_save_data_cor, work_dir
from os.path import join
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from boost import xgb_regres
from sklearn.metrics import r2_score

epochs = 100
batch_size = 4
num_iter = 20
count_train = 500

TREE = 'tree'
LIN = 'linear_regres'
FC = 'fc_nn'
method = TREE


def divided_train_test(data, count_features, count_samples):
    per = np.random.permutation(range(count_samples))
    X = data[:count_features - 1]
    y = data[-1]
    X = X[:, per]
    y = y[per]
    X = np.transpose(X)
    return X[:count_train], y[:count_train], X[count_train:], y[count_train:]


def get_FC_model(input_shape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model


def linear_regres_model(input_shape):
    model = Sequential()
    model.add(Dense(1, input_dim=input_shape))
    return model


def train(model, X_train, y_train, X_test, y_test):
    model.compile(loss='mse', optimizer='nadam', metrics=['mae'])
    # train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                        validation_data=(X_test, y_test))
    # plot metrics
    #    pyplot.plot(history.history['mean_squared_error'])
    pyplot.plot(history.history['mean_absolute_error'])
    #    pyplot.show()

    score = model.evaluate(X_test, y_test, verbose=0)
    mae_train = history.history['mean_absolute_error'][-1]
    mae_test = score[1]
    print('score', score)
    print('mae train=', mae_train)
    print('mae_test', mae_test)

    return mae_train, mae_test


# %%
if __name__ == "__main__":
    mae_train = 0
    mae_test = 0
    r2 = 0

    for i in range(num_iter):
        print('finish iter:', i)
        data = np.genfromtxt(join(work_dir, name_save_data_gain), delimiter='\t')
        print(data.shape)
        count_features = data.shape[0]
        count_samples = data.shape[1]
        print(count_features)
        print(count_samples)
        X_train, y_train, X_test, y_test = divided_train_test(data, count_features, count_samples)
        if method == TREE:
            loc_mae_train, loc_mae_test, loc_r2 = xgb_regres(X_train, y_train.reshape(-1, 1), X_test,
                                                             y_test.reshape(-1, 1))
        else:
            if method == LIN:
                model = linear_regres_model(count_features - 1)
            if method == FC:
                model = get_FC_model(count_features - 1)
            print(model.summary())
            loc_mae_train, loc_mae_test = train(model, X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test, batch_size=1)
            y_pred = np.around(y_pred, 2)
            loc_r2 = r2_score(y_test, y_pred)

        mae_train += loc_mae_train
        mae_test += loc_mae_test
        r2 += loc_r2

    print('mae train:', mae_train / num_iter)
    print('mae test: ', mae_test / num_iter)
    print('r2: ', r2 / num_iter)
