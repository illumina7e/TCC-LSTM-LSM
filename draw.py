"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
from algorithm.random_walk import RW
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from dbn import SupervisedDBNRegression

from keras.utils.generic_utils import get_custom_objects
from correntropy.correntropy import correntropy
loss = correntropy
get_custom_objects().update({"correntropy": loss})  #解决BUG：Unknown loss function:correntropy

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

    y = [x for x in y_true if (x>[0,0]).all()]
    y_pred = [y_pred[i] for i in range(len(y_true)) if (y_pred[i]>[0,0]).all()]

    num = len(144)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_true[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return np.mean(mape)



def eva_regress(dataset, name, y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    mape = MAPE(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    s1 = dataset.split('.')[0] + '-' + name
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
    


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    #d = '2016-3-4 00:00'
    x = range(144)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow(Vehs/h)')

    #date_format = mpl.dates.DateFormatter("%H:%M")
    #ax.xaxis.set_major_formatter(date_format)
    #fig.autofmt_xdate()

    plt.show()


def main(dataset, lag):
    PATH1 = r"C:\Users\illum\Downloads\model\groud"  #模型存储路径
    PATH2 = r"D:\DataSet" #数据集地址
    DATASET = dataset    #数据集名称
    FILE1 = PATH2 + os.sep + "train" + os.sep + DATASET  #训练集path
    FILE2 = PATH2 + os.sep + "test" + os.sep + DATASET   #测试集path

    #dnn = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'dnn.h5')
    #dbn = SupervisedDBNRegression.load(PATH1 + os.sep + dataset.split(".")[0] + 'dbn.pkl')
    #saes = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'saes.h5')
    tcn = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'tcn.h5')
    #lstm = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'lstm.h5')
    #tcnlstm = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'tcnlstm.h5')

    tcn_lstm = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'tcnlstmcorr.h5')
    
    models = [tcn, tcn_lstm]
    names = ["TCN", 'TC-cimLSTM']

    _, _, X_test, y_test, scaler = process_data(FILE1, FILE2, lag)
    y_test = scaler.inverse_transform(y_test)

    y_preds = []
    for name, model in zip(names, models):
        if name == 'DBN':
            pass
        elif name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted = model.predict(X_test[:144])
        predicted = scaler.inverse_transform(predicted)
        y_preds.append(predicted[:144])
        #print(name)
        #eva_regress(dataset, name, y_test, predicted)
        
    obs = y_test[:144][:,0]
    with open(r'C:\Users\illum\Desktop\result.csv', 'a+') as res:
        res.writelines(names[0] + "," + names[1] + ',' + "observation" + "\n")
        for i in range(144):
            res.writelines(str(y_preds[0][i][0]) + "," + str(y_preds[1][i][0]) + ',' + str(obs[i]) + "\n")
    #plot_results(y_test[144:288][:,0], y_preds, names)



lag = 48
op = ['es855d_10min.csv']


for name in op:
    main(name, lag)

#for name in names1:
#    main(name, lag)

