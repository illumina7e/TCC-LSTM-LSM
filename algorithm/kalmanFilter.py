import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics as metrics
import filterpy
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def kf(dataset):
    PATH2 = r"D:\DataSet" #数据集地址
    FILE1 = PATH2 + os.sep + "train" + os.sep + dataset  #训练集path
    FILE2 = PATH2 + os.sep + "test" + os.sep + dataset  #测试集path

    attr = 'Lane 1 Flow (Veh/5 Minutes)'
    df1 = pd.read_csv(FILE1, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(FILE2, encoding='utf-8').fillna(0)

    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    measurements = flow2

    numsteps = len(measurements)
    init_state = 0.
    state_noise = 0.6
    measure_noise = 0.3

    my_filter = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
    f = my_filter
    f.x = np.array([[init_state]])
    f.F = np.array([[1]])
    f.H = np.array([[1]])
    f.P = state_noise   # covariance matrix
    f.R = np.array([[measure_noise]])  # measurement noise
    f.Q = np.array([[state_noise]]) # state uncertainty

    estimates = []
    num_obs = numsteps
    for n in range(num_obs):
        my_filter.predict()
        x = my_filter.x
        estimates.append(x[0])
        my_filter.update(measurements[n])
        
        


    measurements = np.array(measurements)
    estimates = np.array(estimates)
    estimates = scaler.inverse_transform(estimates.reshape(-1, 1)).reshape(1, -1)[0]
    measurements = scaler.inverse_transform(measurements.reshape(-1, 1)).reshape(1, -1)[0]

    mape = MAPE(measurements,  estimates)
    mse = metrics.mean_squared_error(measurements, estimates)
    s1 = dataset.split('.')[0] + '- KalmanFilter' 
    s2 = 'rmse:%f' % math.sqrt(mse)
    s3 = 'mape:%f%%' % mape
    print(s1)
    print(s2)
    print(s3)
    print('\n------------------')
    with open(r'C:\Users\illum\Desktop\result.txt', 'a+') as res:
        res.writelines(s1 + '\n')
        res.writelines(s2 + '\n')
        res.writelines(s3 + '\n')
        res.writelines('\n------------------' + '\n')

names1 = ['1108299.csv', '1108380.csv', '1108439.csv', '1108599.csv', '1111514.csv','1111565.csv', '1114254.csv', '1114515.csv', '1117857.csv', '1117945.csv']
names2 = ['es088d_5min.csv', 'es088d_10min.csv', 'es645d_5min.csv', 'es645d_10min.csv', 'es708d_5min.csv', 'es708d_10min.csv', 'es855d_5min.csv', 'es855d_10min.csv']

for name in names1:
    kf(name)