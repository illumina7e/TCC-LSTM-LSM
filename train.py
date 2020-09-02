"""
Train the NN model.
"""
import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")

PATH1 = r"D:\Trafficflow_predicting\model"  #模型存储路径
PATH2 = r"D:\DataSet" #数据集地址

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

    model.save(PATH1 + os.sep + dataset.split(".")[0] + os.sep + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(PATH1 + os.sep + dataset.split(".")[0] + os.sep + name + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config, dataset):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="MSE", optimizer="adam", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config, dataset)


def main(model_con, dataset, batch, epoch):
    #文件处理
    FILE1 = PATH2 + os.sep + "train" + os.sep + dataset  #训练集path
    FILE2 = PATH2 + os.sep + "test" + os.sep + dataset  #测试集path
    if not os.path.exists(PATH1 + os.sep + dataset.split(".")[0]):
        os.makedirs(PATH1 + os.sep + dataset.split(".")[0]) 
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default = model_con,
        help="Model to train.")
    args = parser.parse_args()
    lag = 288
    X_train, y_train, _, _, _ = process_data(FILE1, FILE2, lag) #input, process data
    config = {"batch": batch, "epochs": epoch}
    
    if args.model == 'dnn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_dnn([288, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, dataset)
    if args.model == 'SimpleRNN':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_SimpleRNN([288, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, dataset)
    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([288, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, dataset)
    if args.model == 'tcnlstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_tcnlstm([288, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, dataset)
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([288, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, dataset)
    if args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([288, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config, dataset)

def train(model, dataset, batch=288, epoch=20):
    main(model, dataset, batch, epoch)





    

