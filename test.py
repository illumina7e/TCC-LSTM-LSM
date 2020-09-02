import os
import sys
import warnings
import argparse

import pandas as pd

from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import plot_model

from correntropy.correntropy import correntropy
from keras.utils.generic_utils import get_custom_objects
from correntropy.correntropy import correntropy
loss = correntropy
get_custom_objects().update({"correntropy": loss})  #解决BUG：Unknown loss function:correntropy

PATH1 = r"D:\Trafficflow_predicting\model"  #模型存储路径
PATH2 = r"D:\DataSet" #数据集地址


dataset = '1108299.csv'
model = load_model(PATH1 + os.sep + 'pre' + dataset.split(".")[0] + os.sep + 'tcnlstm.h5')
plot_model(model, to_file = r'C:\Users\illum\Desktop\Saves\model.png')