from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Sequential
from keras import regularizers
from keras.models import Input, Model
from tcn.tcn import TCN

from keras.utils.generic_utils import get_custom_objects
from correntropy.correntropy import correntropy
loss = correntropy
get_custom_objects().update({"correntropy": loss}) 

warnings.filterwarnings("ignore")

def get_tcclstm(units):
    """TCC-LSTM
    Build TCC-LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    i = Input(batch_shape=(None, units[0], 1))
    o = TCN(return_sequences=True)(i)  # The TCN layers are here.
    o = Dense(12)(o)
    o = LSTM(units[1],input_shape=(12, 1), return_sequences=True)(o)
    o = LSTM(units[2])(o)
    o = Dropout(0.2)(o)
    o = Dense(units[3], activation='sigmoid')(o)
    model = Model(inputs=[i], outputs=[o])

    return model

