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

    y = [x for x in y_true if (x > 0)]
    y_pred = [y_pred[i] for i in range(len(y_true)) if (y_pred[i]> 0)]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
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
    s2 =  '%f' % math.sqrt(mse)
    s3 =  '%f%%' % mape
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
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow(Vehs/h)')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def main(dataset, lag):
    PATH1 = r"C:\Users\illum\Downloads\model\lag=48_com" #模型存储路径
    PATH2 = r"D:\DataSet" #数据集地址
    DATASET = dataset    #数据集名称
    FILE1 = PATH2 + os.sep + "train" + os.sep + DATASET  #训练集path
    FILE2 = PATH2 + os.sep + "test" + os.sep + DATASET   #测试集path


    #dbn = SupervisedDBNRegression.load(PATH1 + os.sep + dataset.split(".")[0] + 'dbn.pkl')
    #saes = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'saes.h5')
    #dnn = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'dnn.h5')
    #tcn = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'tcn.h5')
    #lstm = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'lstm.h5')
    tcnlstm = load_model(PATH1 + os.sep + DATASET.split(".")[0] + 'tcnlstm.h5')
    
    #models = [saes, tcn, lstm, dnn, tcnlstm]
    #names = ["SAEs", 'TCN', 'LSTM', "DNN", 'TCN-LSTM']

    #models = [dbn]
    #names = ["DBN"]

    #models = [tcn]
    #names = ["TCN"]

    models = [tcnlstm]
    names = ["TCC-LSTM"]

    _, _, X_test, y_test, scaler = process_data(FILE1, FILE2, lag)
    y_test = scaler.inverse_transform(y_test)

    for name, model in zip(names, models):
        if name == 'DBN':
            pass
        elif name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted)
        predicted = predicted[:,-1]
        print(name)
        eva_regress(dataset, name, y_test[:,-1], predicted)
        

    #plot_results(y_test[: 288], y_preds, names)



lag = 48
names1 = ['1108299.csv', '1108380.csv', '1108439.csv', '1108599.csv', '1114254.csv', '1117857.csv']
names2 = ['es088d_10min.csv', 'es645d_10min.csv', 'es708d_10min.csv', 'es855d_10min.csv']

op2 = ['1117857.csv']
op = ['1114254.csv']

for name in op:
    main(name, lag)


#for name in names1:
#    main(name, lag)





