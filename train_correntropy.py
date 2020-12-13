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
from keras.models import load_model

from correntropy.correntropy import correntropy
from keras.utils.generic_utils import get_custom_objects
from correntropy.correntropy import correntropy
loss = correntropy
get_custom_objects().update({"correntropy": loss})

warnings.filterwarnings("ignore")

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
    model.compile(loss=correntropy, optimizer="adam", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        shuffle=False)

    model.save('tcclstmlsm.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('tcclstmlsm.csv', encoding='utf-8', index=False)

def main(model_con, dataset, batch, epoch):
    if not os.path.exists(PATH1 + os.sep + dataset.split(".")[0]):
        os.makedirs(PATH1 + os.sep + dataset.split(".")[0]) 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=model_con,
        help="Model to train.")
    args = parser.parse_args()

    lag = 288
    config = {"batch": batch, "epochs": epoch}
    X_train, y_train, _, _, _ = process_data(FILE1, FILE2, lag) #input, process data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if args.model == 'tcclstmlsm':
        m = load_model('tcclstmlsm.h5')
        #m = model.get_tcnlstm([288, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, dataset)






