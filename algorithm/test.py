from keras.layers import Dense
from keras.models import Input, Model
from keras.layers.recurrent import LSTM

from TCN import TCN

batch_size, timesteps, input_dim = None, 144, 1


def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train


i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(return_sequences=True)(i)  # The TCN layers are here.
o = Dense(12)(o)
o = LSTM(64,input_shape=(12, 1))(o)
o = Dense(1)(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')

x, y = get_x_y()
m.fit(x, y, epochs=10, validation_split=0.2)