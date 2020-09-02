import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn.tensorflow import SupervisedDBNRegression

PATH1 = r"C:\Users\illum\Downloads\model\lag=48_com" #模型存储路径

dataset = 'es088d_10min.csv'
dbn = SupervisedDBNRegression.load(PATH1 + os.sep + dataset.split(".")[0] + 'dbn.pkl')
