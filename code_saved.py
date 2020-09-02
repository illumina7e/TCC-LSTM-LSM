###
!pip install git+git://github.com/albertbup/deep-belief-network.git

###
"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Sequential
from keras import regularizers
from keras.models import Input, Model

from dbn import SupervisedDBNRegression




def get_dnn(units):
    """DNN
    Build DNN Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    model.add(Dense(units[1], input_shape=(units[0], 1), activation='relu'))
    model.add(Dense(units[2], activation='relu'))
    model.add(Dense(units[2], activation='relu'))
    model.add(Dense(units[2], activation='relu'))
    model.add(Dense(units[2], activation='relu'))
    model.add(Dense(units[2], activation='relu'))
    model.add(Flatten())
    model.add(Dense(units[3], activation='linear'))

    return model

def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model

def get_tcn(units):
    """TCN(Temporal Convolution Network )
    Build TCN Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    i = Input(batch_shape=(None, units[0], 1))
    o = TCN(return_sequences=False)(i)  # The TCN layers are here.
    o = Dense(units[3])(o)
    model = Model(inputs=[i], outputs=[o])

    return model

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

def get_dbn(batch, epoch):
    """DBN(Deep Belief Network)
    Build DBN Model.

    # Returns
        model: Model, nn model.
    """
    regressor = SupervisedDBNRegression(hidden_layers_structure=[120],
                                    learning_rate_rbm=0.025,
                                    learning_rate=0.05,
                                    n_epochs_rbm=epoch,
                                    n_iter_backprop=200,
                                    batch_size=batch,
                                    activation_function='relu',
                                    verbose=False)
    return regressor


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model

def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models


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

def train_dbn(regressor, X_train, y_train, name, dataset):
    regressor.fit(X_train, y_train)
    regressor.save(dataset.split(".")[0] + name + '.pkl')
    
def main(model_con, dataset, lag, batch, epoch):
    FILE1 = PATH2 + os.sep + "traffic-train" + os.sep + dataset  #训练集path
    FILE2 = PATH2 + os.sep + "traffic-test" + os.sep + dataset  #测试集path
    

    X_train, y_train, _, _, _ = process_data(FILE1, FILE2, lag) #input, process data
    config = {"batch": batch, "epochs": epoch}
    
    if model_con == 'dnn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = get_dnn([lag, 128, 128, 1])
        train_model(m, X_train, y_train, model_con, config, dataset)
    if model_con == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = get_lstm([lag, 128, 128, 1])
        train_model(m, X_train, y_train, model_con, config, dataset)
    if model_con == 'tcn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = get_tcn([lag, 128, 128, 1])
        train_model(m, X_train, y_train, model_con, config, dataset)
    if model_con == 'tcnlstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = get_tcnlstm([lag, 128, 128, 1])
        train_model(m, X_train, y_train, model_con, config, dataset)
    if model_con == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = get_saes([lag, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, model_con, config, dataset)
    if model_con == 'dbn':
        #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = get_dbn(batch, epoch)
        train_dbn(m, X_train, y_train, model_con, dataset)
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
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects
loss = correntropy
get_custom_objects().update({"correntropy": loss})  #解决BUG：Unknown loss function:correntropy
warnings.filterwarnings("ignore")


PATH2 = r"../input" #数据集地址

def train_model_corr(model, X_train, y_train, name, config, dataset):
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

    model.save(dataset.split(".")[0] + name + 'corr.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(dataset.split(".")[0] + name + ' losscorr.csv', encoding='utf-8', index=False)


def main_corr(model_con, dataset, lag, batch, epoch):
    FILE1 = PATH2 + os.sep + "traffic-train" + os.sep + dataset  #训练集path
    FILE2 = PATH2 + os.sep + "traffic-test" + os.sep + dataset  #测试集path
    
    config = {"batch": batch, "epochs": epoch}
    X_train, y_train, _, _, _ = process_data(FILE1, FILE2, lag) #input, process data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if model_con == 'tcnlstm':
        m = load_model('pre' + dataset.split(".")[0] + 'tcnlstm.h5')
        train_model_corr(m, X_train, y_train, model_con, config, dataset)
        
def train_corr(model, dataset, lag, batch=576, epoch=20): 
    main_corr(model, dataset, lag, batch, epoch)

