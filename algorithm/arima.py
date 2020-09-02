import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")

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

def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction
"""
ARIMA
"""
def arima(dataset):
    PATH2 = r"../input" #数据集地址
    DATASET = dataset    #数据集名称
    FILE1 = PATH2 + os.sep + "traffic-train2" + os.sep + dataset  #训练集path
    FILE2 = PATH2 + os.sep + "trafic-test2" + os.sep + dataset  #测试集path

    attr = 'Lane 1 Flow (Veh/5 Minutes)'
    df1 = pd.read_csv(FILE1, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(FILE2, encoding='utf-8').fillna(0)

    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]


    TrainingData = flow1[-288:]
    TestData = flow2

    #new arrays to store actual and predictions
    Actual = [x for x in TrainingData]
    Predictions = list()


    #in a for loop, predict values using ARIMA model
    for timepoint in range(len(TestData)):
        ActualValue =  TestData[timepoint]
        #forcast value
        Prediction = StartARIMAForecasting(Actual, 2,1,0)
        #add it in the list
        Predictions.append(Prediction)
        Actual.append(ActualValue)
        Actual.pop(0)

    Predictions = np.array(Predictions)
    TestData = np.array(TestData)
    Predictions = scaler.inverse_transform(Predictions.reshape(-1, 1)).reshape(1, -1)[0]
    TestData = scaler.inverse_transform(TestData.reshape(-1, 1)).reshape(1, -1)[0]
