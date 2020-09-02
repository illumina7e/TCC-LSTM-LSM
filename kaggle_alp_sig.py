###
"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import regularizers
from keras.models import Input, Model


def get_tcnlstm(units):
    """TCN-LSTM(Temporal Convolution Network LSTM)
    Build TCN-LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    i = Input(batch_shape=(None, units[0], 1))
    o = TCN(return_sequences=True)(i)  # The TCN layers are here.
    o = LSTM(units[1],input_shape=(units[0], 128), return_sequences=True)(o)
    o = LSTM(units[2])(o)
    o = Dropout(0.2)(o)
    o = Dense(units[3], activation='sigmoid')(o)
    model = Model(inputs=[i], outputs=[o])

    return model

###
"""
Train the NN model.
"""
import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")

PATH2 = r"../input" #数据集地址


def train_model(model, X_train, y_train, name, config, dataset):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="MSE", optimizer="adam", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        shuffle=False)

    model.save(dataset.split(".")[0] + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(dataset.split(".")[0] + name + ' loss.csv', encoding='utf-8', index=False)


def main(model_con, dataset, lag, batch, epoch):
    FILE1 = PATH2 + os.sep + "traffic-train" + os.sep + dataset  #训练集path
    FILE2 = PATH2 + os.sep + "traffic-test" + os.sep + dataset  #测试集path
    
    X_train, y_train, _, _, _ = process_data(FILE1, FILE2, lag) #input, process data
    config = {"batch": batch, "epochs": epoch}
    
    if model_con == 'tcnlstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = get_tcnlstm([lag, 128, 128, 1])
        train_model(m, X_train, y_train, model_con, config, dataset)
        
def train(model, dataset, lag, batch=576, epoch=20):
    main(model, dataset,lag, batch, epoch)


###
"""
Train the NN model.
"""
import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model




PATH2 = r"../input" #数据集地址
ALPHA0 = 2
SIGMA0 = 0.1

def correntropy(y_true, y_pred): 
    alpha = ALPHA0 #shape
    sigma = SIGMA0  #bandwidth
    lamda = - (1 / (np.power(sigma, alpha)))
    e = K.abs(y_true - y_pred)
    
    return (1 - K.exp(lamda * np.power(e, alpha)))

from keras.utils.generic_utils import get_custom_objects
loss = correntropy
get_custom_objects().update({"correntropy": loss})  #解决BUG：Unknown loss function:correntropy
warnings.filterwarnings("ignore")

def train_model_corr(model, X_train, y_train, name, config, dataset, alp, sig):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    ALPHA0 = alp
    SIGMA0 = sig
    model.compile(loss=correntropy, optimizer="adam", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        shuffle=False)

    model.save("alpha sigma_" + str(int(alp * 100)) + " " + str(int(sig * 100)) + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv("alpha sigma_" + str(int(alp * 100)) + " " + str(int(sig * 100)) +  ' loss.csv', encoding='utf-8', index=False)


def main_corr(model_con, dataset, lag, batch, epoch, alp, sig):
    FILE1 = PATH2 + os.sep + "traffic-train" + os.sep + dataset  #训练集path
    FILE2 = PATH2 + os.sep + "traffic-test" + os.sep + dataset  #测试集path
    
    config = {"batch": batch, "epochs": epoch}
    X_train, y_train, _, _, _ = process_data(FILE1, FILE2, lag) #input, process data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if model_con == 'tcnlstm':
        m = load_model('pre' + dataset.split(".")[0] + 'tcnlstm.h5')
        train_model_corr(m, X_train, y_train, model_con, config, dataset, alp, sig)
        
def train_corr(model, dataset, lag, batch=576, epoch=20, alp=2, sig=0.1): 
    main_corr(model, dataset, lag, batch, epoch, alp, sig)


###
lag = 48
alphs = [1.5, 2, 2.5] #
sigms = [0.01, 0.05, 0.1, 0.25, 0.5]
#TCC-LSTM
#for name in TCCLSTM:
#    train('tcnlstm', name, lag=lag, epoch=80)

#TCC-cimLSTM
train('tcnlstm', ('pre' + "es855d_10min.csv"), lag=lag, epoch=10)
for alph in alphs:
    for sigm in sigms:
        train_corr('tcnlstm', "es855d_10min.csv", lag=lag, epoch=60, alp=alph, sig=sigm)
